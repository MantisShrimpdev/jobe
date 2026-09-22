# -*- coding: utf-8 -*-
"""Split a run by how sure the readout was, because that is where everything happens.

Two thirds of the public tasks come back above 0.85 confidence and are right
96% of the time; nothing a LoRA or a thinking pass does moves them. One task in
three lands in the uncertain band, where accuracy is 64% and every intervention
measured here - training, routed reasoning - does all of its work, good and bad.
A change reported as a single tier number averages those two populations
together and hides which one it touched.

With `--candidate`, it also reports how many answers the second run changed in
each band, which is the diagnostic that says whether an intervention went where
it was aimed.

  python bench/confidence_bands.py --results bench/runs/<a>/results.jsonl \
      [--candidate bench/runs/<b>/results.jsonl] [--bands 0.45 0.85]
"""

from __future__ import annotations

import argparse
import json


def read(path: str) -> dict[str, dict]:
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    return {r["task_id"]: r for r in rows}


def confidence(row: dict) -> float:
    """The readout's top probability — its own statement of how sure it was."""
    probs = row.get("probs") or {}
    return max(probs.values()) if probs else 0.0


def bands_of(cuts: list[float]) -> list[tuple[float, float]]:
    edges = [0.0, *sorted(cuts), 1.0]
    return list(zip(edges, edges[1:]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", required=True)
    ap.add_argument("--candidate", default="")
    ap.add_argument("--bands", type=float, nargs="*", default=[0.45, 0.85],
                    help="the cut points between bands; the ends are 0 and 1")
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)

    base = read(args.results)
    cand = read(args.candidate) if args.candidate else {}
    shared = sorted(set(base) & set(cand)) if cand else []
    out = []

    head = f"{'baseline confidence':22s} {'n':>4s} {'share':>6s} {'accuracy':>9s}"
    if cand:
        head += f" {'changed':>9s} {'fixed':>6s} {'broken':>7s}"
    print(head)
    for lo, hi in bands_of(args.bands):
        top = hi >= 1.0
        group = [r for r in base.values() if lo <= confidence(r) < hi or (top and confidence(r) >= 1.0)]
        if not group:
            continue
        acc = sum(bool(r["correct"]) for r in group) / len(group)
        row = {"lo": lo, "hi": hi, "n": len(group), "share": len(group) / len(base), "accuracy": acc}
        line = f"{f'{lo:.2f} - {hi:.2f}':22s} {len(group):4d} {len(group)/len(base):5.0%} {acc:9.3f}"
        if cand:
            ids = [r["task_id"] for r in group if r["task_id"] in shared]
            fixed = sum(1 for i in ids if not base[i]["correct"] and cand[i]["correct"])
            broken = sum(1 for i in ids if base[i]["correct"] and not cand[i]["correct"])
            n = fixed + broken
            line += f" {n:4d} {n/len(ids) if ids else 0:4.0%} {fixed:+6d} {-broken:+7d}"
            row |= {"changed": n, "fixed": fixed, "broken": broken, "n_shared": len(ids)}
        print(line)
        out.append(row)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"results": args.results, "candidate": args.candidate, "bands": out}, fh, indent=1)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

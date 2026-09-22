# -*- coding: utf-8 -*-
"""Split a run by how sure the readout was, because that is where everything happens.

Two thirds of the public tasks come back above 0.85 confidence and are right
96% of the time; nothing a LoRA or a thinking pass does moves them. One task in
three lands in the uncertain band, where accuracy is 64% and every intervention
measured here - training, routed reasoning - does all of its work, good and bad.
A change reported as a single tier number averages those two populations
together and hides which one it touched.

With `--candidate`, it also reports how many answers the second run MOVED in
each band, alongside how many it fixed and broke. The move count is the primary
one: a wrong answer replaced by a different wrong answer leaves accuracy
untouched and is still the model being unstable, which is exactly what an
order study or a perturbation study is asking about.

  python bench/confidence_bands.py --results bench/runs/<a>/results.jsonl \
      [--candidate bench/runs/<b>/results.jsonl] [--bands 0.45 0.85]
"""

from __future__ import annotations

import argparse
import json


def _rows(path: str) -> list[dict]:
    """Read a results file, and say something useful if it is the wrong one.

    Two runners write results here. `bench/jobe_direct.py` under the official
    CLI writes the scored per-item schema these tools read; `bench/run_jevbench.py`
    writes a richer unscored file keyed on `id`. Handed the second, every tool
    downstream used to die on `KeyError: 'task_id'`, which names the symptom
    and not the cause.
    """
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    if out and "task_id" not in out[0]:
        keys = ", ".join(sorted(out[0])[:6])
        raise SystemExit(
            f"{path} is a run_jevbench.py results file (keys: {keys}...), not a "
            "scored one. Convert it first with:" + chr(10) +
            "  python bench/to_official_records.py --jevbench <clone> "
            f"--tasks <tasks.jsonl> --results {path} --out <scored>.jsonl"
        )
    return out


def read(path: str) -> dict[str, dict]:
    return {r["task_id"]: r for r in _rows(path)}


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
        head += f" {'answer moved':>14s} {'fixed':>6s} {'broken':>7s}"
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
            # The answer moving is the primary quantity: a wrong answer replaced by a
            # different wrong answer changed nothing in the accuracy columns and is
            # still the model being unstable, which is what an order or a perturbation
            # study is actually asking about.
            moved = sum(1 for i in ids if base[i].get("predicted") != cand[i].get("predicted"))
            line += f" {moved:5d} {moved/len(ids) if ids else 0:8.0%} {fixed:+6d} {-broken:+7d}"
            row |= {"moved": moved, "fixed": fixed, "broken": broken, "n_shared": len(ids)}
        print(line)
        out.append(row)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"results": args.results, "candidate": args.candidate, "bands": out}, fh, indent=1)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

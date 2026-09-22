# -*- coding: utf-8 -*-
"""Two harness runs, side by side: which tiers moved, which families paid for it.

The tier table is what a leaderboard sees. The family table is what tells you
whether a change is a gain or a trade, and it is the one that decided against
lora-3fam-v1: that run's hard tier fell by two items while its trained families
rose, because judge_hard dropped four. A tier delta of -2 and a tier delta of
-2 that is really +4/-6 are different results, and only the second table
separates them.

Reads the `results.jsonl` the official harness writes, so it compares anything
that ran through it — a trained adapter against its baseline, one backbone
against another, one prompt version against the next.

  python bench/compare_runs.py --baseline bench/runs/<a>/results.jsonl \
      --candidate bench/runs/<b>/results.jsonl --labels v0.1.0 anchored
"""

from __future__ import annotations

import argparse
import json

TIERS = {"easy": "easy", "original": "standard", "hard": "hard"}
ORDER = ("easy", "standard", "hard")


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


def tier_of(task_id: str) -> str:
    return TIERS.get(task_id.split("-", 1)[0], "?")


def cell(rows: dict[str, dict], ids: list[str]) -> tuple[int, int]:
    """Correct and scored, over the ids both runs share."""
    present = [i for i in ids if i in rows]
    return sum(bool(rows[i].get("correct")) for i in present), len(present)


def table(title: str, groups: list[tuple[str, list[str]]], a, b, labels) -> list[str]:
    out = [f"\n{title}", f"{'':24s} {labels[0]:>13s} {labels[1]:>13s} {'Δ':>6s}"]
    for name, ids in groups:
        ca, na = cell(a, ids)
        cb, nb = cell(b, ids)
        if not na:
            continue
        out.append(f"{name:24s} {ca:5d}/{na:<3d} {ca/na:5.3f} {cb:5d}/{nb:<3d} {cb/nb:5.3f} {cb - ca:+6d}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--labels", nargs=2, default=["baseline", "candidate"])
    ap.add_argument("--json", default="", help="also write the comparison here")
    args = ap.parse_args(argv)

    a, b = read(args.baseline), read(args.candidate)
    shared = sorted(set(a) & set(b))
    only = (len(a) - len(shared), len(b) - len(shared))
    print(f"{len(shared)} tasks in both runs" + (f"; {only[0]} only in {args.labels[0]}, "
          f"{only[1]} only in {args.labels[1]} (excluded)" if any(only) else ""))

    tiers = [(t, [i for i in shared if tier_of(i) == t]) for t in ORDER]
    lines = table("BY TIER", [(t, ids) for t, ids in tiers] + [("all", shared)], a, b, args.labels)

    for tier, ids in tiers:
        fams: dict[str, list[str]] = {}
        for i in ids:
            fams.setdefault(a[i].get("family") or "?", []).append(i)
        moved = [(f, v) for f, v in fams.items() if cell(a, v)[0] != cell(b, v)[0]]
        if moved:
            moved.sort(key=lambda fv: cell(b, fv[1])[0] - cell(a, fv[1])[0])
            lines += table(f"BY FAMILY — {tier} tier (only the families that moved)", moved, a, b, args.labels)
    print("\n".join(lines))

    fixed = [i for i in shared if not a[i].get("correct") and b[i].get("correct")]
    broke = [i for i in shared if a[i].get("correct") and not b[i].get("correct")]
    print(f"\n{len(fixed)} fixed, {len(broke)} broken, net {len(fixed) - len(broke):+d}")
    for label, group in ((f"fixed by {args.labels[1]}", fixed), (f"broken by {args.labels[1]}", broke)):
        for tier in ORDER:
            ids = [i for i in group if tier_of(i) == tier]
            if ids:
                print(f"  {label} [{tier}]: " + ", ".join(ids[:10]) + (f" +{len(ids) - 10} more" if len(ids) > 10 else ""))

    if args.json:
        payload = {
            "baseline": args.baseline, "candidate": args.candidate, "labels": args.labels,
            "n_shared": len(shared),
            "tiers": {t: {"baseline": cell(a, ids), "candidate": cell(b, ids)} for t, ids in tiers},
            "fixed": fixed, "broken": broke,
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

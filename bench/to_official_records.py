# -*- coding: utf-8 -*-
"""Score a `run_jevbench.py` run with JevBench's own scoring, without its runner.

Two runners exist here and they are not interchangeable. `bench/jobe_direct.py`
is the submission adapter, driven by `python -m jevbench.cli run`, and it writes
the official per-item record schema. `bench/run_jevbench.py` is our own driver;
it writes a richer file (logits, per-item flip rate) in its own schema, which
`score_official.py` cannot read - and feeding one to the other fails with a
KeyError on `task_id` rather than with anything that explains itself.

This converts the second into the first. It does not re-derive correctness: it
hands each distribution to `jevbench.scoring.score_task`, which is the same
function the official runner calls, so `valid`, `strict_valid`, `renormalized`,
`predicted`, `correct` and `ordinal_ev` come from their code. Verified against
both archived official runs: the rule reproduces 186/231 and 182/231 with zero
prediction mismatches.

WHAT IT CANNOT RECOVER. `run_jevbench.py` does not record per-item latency, and
the Speed axis needs it. A composite from these records would be missing a
quarter of itself, so this writes accuracy and calibration only and says so. For
a composite, run the official adapter.

  python bench/to_official_records.py --jevbench <clone> --tasks all231.jsonl \
      --results <run>/results.jsonl --out <run>/official_records.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys

TIERS = {"easy": "easy", "original": "standard", "hard": "hard"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--jevbench", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--compare", default="", help="an official results.jsonl to diff against")
    args = ap.parse_args(argv)

    sys.path.insert(0, args.jevbench)
    from jevbench.scoring import score_task  # noqa: E402
    from jevbench.summarize import summarize  # noqa: E402
    from jevbench.tasks import load_jsonl  # noqa: E402

    tasks = load_jsonl(args.tasks)
    by_id = {t.id: t for t in tasks}
    records = []
    with open(args.results, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            tid = row.get("task_id") or row["id"]
            task = by_id[tid]
            scored = score_task(row["probs"], task)
            records.append({
                "task_id": tid, "family": task.family, "split": task.split,
                "group": task.group, "status": 200 if row.get("ok") else 500,
                "ok": bool(row.get("ok")), "probs_source": "native",
                "probs_as_returned": row["probs"], **scored,
            })

    overall = summarize(tasks, records)
    print(f"{overall['n_correct']}/{overall['n_scorable']} = {overall['accuracy']:.4f}   "
          f"valid {overall['n_valid']}/{overall['n_attempted']}   "
          f"Brier {overall['brier_mean']:.4f}   ECE {overall['ece']['ece']:.4f}")

    by_tier: dict[str, list[int]] = {}
    for r in records:
        t = TIERS[r["task_id"].split("-", 1)[0]]
        v = by_tier.setdefault(t, [0, 0])
        v[0] += 1
        v[1] += bool(r["correct"])
    print(f"\n{'tier':9s} {'n':>4s} {'correct':>8s} {'accuracy':>9s}")
    for t in ("easy", "standard", "hard"):
        if t in by_tier:
            n, c = by_tier[t]
            print(f"{t:9s} {n:4d} {c:8d} {c / n:9.3f}")

    if args.compare:
        other = {}
        with open(args.compare, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    other[r["task_id"]] = r
        mine = {r["task_id"]: r for r in records}
        shared = sorted(set(mine) & set(other))
        moved = [i for i in shared if mine[i]["predicted"] != other[i].get("predicted")]
        fixed = [i for i in shared if mine[i]["correct"] and not other[i].get("correct")]
        broke = [i for i in shared if not mine[i]["correct"] and other[i].get("correct")]
        print(f"\nagainst {args.compare}: {len(moved)} answers moved, "
              f"{len(fixed)} gained, {len(broke)} lost, net {len(fixed) - len(broke):+d}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

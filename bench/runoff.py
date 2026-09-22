# -*- coding: utf-8 -*-
"""Re-ask the contested decisions with only the readout's top two options.

In the band where this model is unsure - top probability between 0.45 and 0.85,
30% of the public tasks - the correct option is its FIRST choice 61% of the
time and inside its top TWO 87% of the time. The answer is nearly always
already in hand; what is failing is the ordering of two candidates.

Renormalising the existing distribution over those two would change nothing,
because the ordering is what it is. This re-asks: a fresh prompt offering only
the two candidates, scored by the same letter readout. That is a different
computation, and there is a measured reason to expect it to do better - menu
size is this family's documented weakness (Verdict 97% at K=3 against 72% at
K=25; Laya 0.425 on 77 labels), so a five-way question asked as a two-way
question is being asked in the shape the model is best at.

It is also cheap. A runoff costs one extra forward pass on a shorter prompt,
and only for the third of decisions that are contested.

ORDINAL QUESTIONS ARE EXCLUDED. A "rate this 1-5" is a scale, not a menu: the
harness grades it by the expected value over the levels, and offering two of
five levels changes what is being asked rather than how it is asked. They are
counted in the header so the exclusion is visible.

  python bench/runoff.py --tasks <all231.jsonl> --results <results.jsonl> \
      --model D:/Coding/models/qwen35-4b --band 0.45 0.85 --out runoff.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from jobe import Decision, Option, load, score  # noqa: E402


def options_for(task: dict) -> list[Option]:
    q = task["question"]
    crit = q.get("criteria")
    if q["type"] == "noul":
        c = crit or {}
        pairs = [(k, c.get(k) or f"The proposition is {k}.") for k in ("true", "false")]
    elif q["type"] == "choice":
        pairs = [(k, v or k) for k, v in (crit or {}).items()]
    else:
        pairs = [(str(i), lvl) for i, lvl in enumerate(crit or [])]
    return [Option(id=k, description=f"{k}: {v}") for k, v in pairs]


def canonical(kind: str, scores: dict) -> dict:
    """The harness reports yes/no; the prompt uses true/false."""
    if kind != "noul":
        return scores
    return {"yes": scores.get("true", 0.0), "no": scores.get("false", 0.0)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--results", required=True, help="the first-pass results.jsonl this re-asks from")
    ap.add_argument("--model", required=True)
    ap.add_argument("--band", type=float, nargs=2, default=[0.45, 0.85])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    tasks = {}
    with open(args.tasks, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                t = json.loads(line)
                tasks[t["id"]] = t
    first = {}
    with open(args.results, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                first[r["task_id"]] = r

    lo, hi = args.band
    contested, ordinal = [], 0
    for tid, r in first.items():
        probs = r.get("probs") or {}
        if not probs or tid not in tasks:
            continue
        if not (lo <= max(probs.values()) < hi) or len(probs) <= 2:
            continue
        # An ordinal question is not a menu that can be truncated. Its levels are
        # a scale, the harness grades it by the expected value over that scale,
        # and offering two of five levels changes what is being asked rather than
        # how it is asked. So they are excluded, and counted rather than dropped
        # quietly.
        if tasks[tid]["question"]["type"] == "score":
            ordinal += 1
            continue
        contested.append(tid)
    contested.sort()
    print(f"{len(contested)} contested choice decisions with more than two options in "
          f"[{lo}, {hi}); {ordinal} ordinal decisions excluded (a scale is not a menu)", flush=True)
    if not contested:
        return 0

    from thelab.core.gpu import require_free_gpu
    require_free_gpu(3000)
    backbone = load(args.model, device=args.device)

    rows, t0 = [], time.perf_counter()
    for n, tid in enumerate(contested, 1):
        task, r = tasks[tid], first[tid]
        kind = task["question"]["type"]
        gold = str(task["expected"])
        top2 = [k for k, _ in sorted(r["probs"].items(), key=lambda kv: -kv[1])[:2]]
        by_id = {o.id: o for o in options_for(task)}
        # the harness's yes/no ids are the prompt's true/false
        key = {"yes": "true", "no": "false"} if kind == "noul" else {}
        chosen = [by_id[key.get(k, k)] for k in top2 if key.get(k, k) in by_id]
        if len(chosen) != 2:
            rows.append({"task_id": tid, "skipped": "option ids did not resolve"})
            continue
        readout = score(backbone.model, backbone.tokenizer, Decision(
            id=tid, evidence=task["state"], criterion=task["question"]["instructions"],
            options=tuple(chosen), ordinal=kind == "score"))
        probs = canonical(kind, readout.scores)
        pred = max(probs.items(), key=lambda kv: kv[1])[0]
        rows.append({
            "task_id": tid, "family": r.get("family"), "type": kind, "expected": gold,
            "n_options": len(r["probs"]), "top2": top2, "gold_in_top2": gold in top2,
            "first_pred": r.get("predicted"), "first_correct": bool(r.get("correct")),
            "runoff_pred": pred, "runoff_correct": pred == gold,
            "runoff_conf": max(probs.values()), "seconds": readout.total_seconds,
        })
        if n % 20 == 0:
            print(f"  {n}/{len(contested)}", flush=True)

    with open(args.out, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    scored = [r for r in rows if "runoff_correct" in r]
    ceiling = sum(r["gold_in_top2"] for r in scored) / len(scored)
    a = sum(r["first_correct"] for r in scored) / len(scored)
    b = sum(r["runoff_correct"] for r in scored) / len(scored)
    fixed = [r for r in scored if not r["first_correct"] and r["runoff_correct"]]
    broke = [r for r in scored if r["first_correct"] and not r["runoff_correct"]]
    print(f"\n{len(scored)} decisions re-asked in {time.perf_counter() - t0:.0f}s "
          f"({sum(r['seconds'] for r in scored) / len(scored) * 1000:.0f} ms each)")
    print(f"  first pass   {a:.3f}")
    print(f"  runoff       {b:.3f}   ({b - a:+.3f}; {len(fixed)} fixed, {len(broke)} broken)")
    print(f"  ceiling      {ceiling:.3f}   (how often the gold was in the top two at all)")
    lost = [r for r in scored if not r["gold_in_top2"]]
    print(f"  {len(lost)} decisions could not be recovered by any runoff: the gold was not offered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

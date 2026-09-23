# -*- coding: utf-8 -*-
"""Regression check, and the eager-vs-sdpa question, in one pass per implementation.

    python bench/attn_check.py eager  runs/attn-eager.json
    python bench/attn_check.py sdpa   runs/attn-sdpa.json

1. REGRESSION. Re-scores all 231 public JevBench tasks with today's code and
   environment, through the same option mapping the JevBench adapter uses, and
   compares each prediction with the recorded run of 2026-09-22 (which was
   eager). Identical predictions mean nothing in the benchmarked path moved.

2. SPEED. `model.load` defaults to eager attention, justified by a 130-token
   measurement. Qwen3.5 is hybrid, and its full-attention layers under eager
   materialise an n x n matrix per head - harmless at 130 tokens, suspected of
   the super-linear curve at browser sizes (3.2k tokens 2 s, 5.5k tokens 12 s).
   Times identical prompts at several lengths, after a warm-up, so Triton
   compilation is not counted.
"""

from __future__ import annotations

import json
import os
import statistics as st
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from jobe import Decision, Option, load, score        # noqa: E402
from jobe.server import options_for                    # noqa: E402  (pinned equal to the adapter's mapping)

PUBLIC = r"D:\Coding\jevbench\datasets\public"
RECORDED = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "runs", "2026-09-22-public231", "results.jsonl")
MODEL = r"D:\Coding\models\qwen35-4b"
#: Optional: a real page's evidence (a JSON string) to time as well - the
#: 2026-09-23 run used a 6,484-token DuckDuckGo results page.
DDG = os.environ.get("ATTN_PAGE_JSON", "")


def decision_for(task) -> Decision:
    q = task["question"]
    return Decision(id=task["id"], evidence=task["state"], criterion=q["instructions"],
                    options=tuple(options_for(q)), ordinal=q["type"] == "score")


def main() -> int:
    attn, out = sys.argv[1], sys.argv[2]
    tasks = []
    for name in ("easy.jsonl", "hard.jsonl", "original.jsonl"):
        with open(os.path.join(PUBLIC, name), encoding="utf-8") as fh:
            tasks += [json.loads(line) for line in fh]
    recorded = {}
    with open(RECORDED, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            recorded[r["task_id"]] = r

    t0 = time.perf_counter()
    bb = load(MODEL, device="auto", attn_implementation=attn)
    print("loaded %s in %.0fs" % (attn, time.perf_counter() - t0), flush=True)

    # ---- 1. every public task
    picks, lat = {}, []
    for task in tasks:
        r = score(bb.model, bb.tokenizer, decision_for(task), max_tokens=20000)
        choice = r.choice
        if task["question"]["type"] == "noul":            # the adapter reports yes/no
            choice = "yes" if choice == "true" else "no"
        picks[task["id"]] = {"choice": choice, "probs": r.scores}
        lat.append(r.total_seconds * 1000)
    same = sum(1 for t in tasks if picks[t["id"]]["choice"] == recorded[t["id"]]["predicted"])
    correct = sum(1 for t in tasks if picks[t["id"]]["choice"] == t["expected"])
    print("public 231: %d/231 correct | %d/231 identical to the recorded eager run | p50 %.0f ms"
          % (correct, same, st.median(lat)), flush=True)

    # ---- 2. browser-sized prompts, warmed
    opts = (Option("infra", "infra: outages and deploys"),
            Option("billing", "billing: payments and refunds"))
    word = "The checkout service emitted a structured log line about coupon validation. "
    evidences = [("ddg page", json.load(open(DDG, encoding="utf-8")))] if os.path.exists(DDG) else []
    evidences += [("~%dk tok" % (n // 1000), word * (n // 14)) for n in (1000, 3000, 5500)]
    timing = {}
    for label, ev in evidences:
        d = Decision(id="t", evidence=ev, criterion="Which team owns this?", options=opts)
        score(bb.model, bb.tokenizer, d, max_tokens=20000)           # warm: pays any compile
        ts = [score(bb.model, bb.tokenizer, d, max_tokens=20000) for _ in range(3)]
        timing[label] = {"tokens": ts[0].input_tokens,
                         "ms": st.median(x.total_seconds * 1000 for x in ts)}
        print("  %-9s %5d tokens  %7.0f ms" % (label, ts[0].input_tokens, timing[label]["ms"]),
              flush=True)

    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"attn": attn, "correct": correct, "identical_to_recorded": same,
                   "p50_ms": st.median(lat), "picks": picks, "timing": timing}, fh)
    print("wrote " + out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

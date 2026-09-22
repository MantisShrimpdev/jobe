# -*- coding: utf-8 -*-
"""Q1: does the readout's uncertainty find its own wrong answers?

Pure analysis over the official-harness results. For each tier: AUROC of three
uncertainty signals (top probability, top-1/top-2 margin, entropy) as detectors
of a WRONG answer, plus accuracy by confidence quartile. If the signal cannot
separate right from wrong on the hard tier, "reason when unsure" routes the
wrong decisions and the experiment is over before the GPU is touched.

  python bench/uncertainty_analysis.py [results.jsonl]
"""
import json
import math
import os

import sys

RES = sys.argv[1] if len(sys.argv) > 1 else "bench/runs/2026-09-22-public231/results.jsonl"


def tier_of(tid):
    return {"easy": "easy", "original": "standard", "hard": "hard"}[tid.split("-", 1)[0]]


def auroc(scores, labels):
    """AUROC via rank statistic. labels: 1 = wrong (the thing we want to detect).
    scores: higher = more uncertain. Ties handled by average rank."""
    pairs = sorted(zip(scores, labels))
    n = len(pairs); ranks = [0.0] * n; i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    pos = [r for r, (_, l) in zip(ranks, pairs) if l == 1]
    n_pos = len(pos); n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (sum(pos) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


rows = [json.loads(l) for l in open(RES, encoding="utf-8") if l.strip()]
rows = [r for r in rows if r.get("ok") and r.get("probs")]

print("Can uncertainty detect a WRONG answer?  (AUROC: 0.5 = coin flip, 1.0 = perfect)\n")
print(f"{'tier':9s} {'n':>4s} {'acc':>6s} | {'1-conf':>7s} {'1-margin':>9s} {'entropy':>8s} | "
      f"{'Q1 acc':>7s} {'Q2':>6s} {'Q3':>6s} {'Q4 acc':>7s}   (quartiles by confidence, low->high)")
print("-" * 104)
for tier in ("easy", "standard", "hard"):
    t = [r for r in rows if tier_of(r["task_id"]) == tier]
    if not t:
        continue
    conf, margin, ent, wrong = [], [], [], []
    for r in t:
        p = sorted(r["probs"].values(), reverse=True)
        conf.append(p[0]); margin.append(p[0] - (p[1] if len(p) > 1 else 0.0))
        ent.append(-sum(x * math.log(x) for x in p if x > 0))
        wrong.append(0 if r["correct"] else 1)
    a_conf = auroc([1 - c for c in conf], wrong)
    a_marg = auroc([1 - m for m in margin], wrong)
    a_ent = auroc(ent, wrong)
    order = sorted(range(len(t)), key=lambda i: conf[i])
    q = len(t) // 4
    quart = []
    for k in range(4):
        idx = order[k * q:(k + 1) * q] if k < 3 else order[3 * q:]
        quart.append(sum(1 - wrong[i] for i in idx) / len(idx))
    acc = 1 - sum(wrong) / len(wrong)
    print(f"{tier:9s} {len(t):4d} {acc:6.3f} | {a_conf:7.3f} {a_marg:9.3f} {a_ent:8.3f} | "
          f"{quart[0]:7.3f} {quart[1]:6.3f} {quart[2]:6.3f} {quart[3]:7.3f}")

# Routing simulation on hard: route the least-confident X% to a hypothetical oracle.
print("\nHard tier: if the least-confident X% were routed to a perfect reasoner, accuracy would be:")
t = [r for r in rows if tier_of(r["task_id"]) == "hard"]
conf = [max(r["probs"].values()) for r in t]; wrong = [0 if r["correct"] else 1 for r in t]
order = sorted(range(len(t)), key=lambda i: conf[i])
print(f"  {'route %':>8s} {'conf cutoff':>12s} {'wrong caught':>13s} {'acc if fixed':>13s}")
for pct in (10, 20, 30, 40, 50):
    k = round(len(t) * pct / 100)
    routed = set(order[:k])
    caught = sum(wrong[i] for i in routed)
    acc_fixed = (sum(1 - wrong[i] for i in range(len(t))) + caught) / len(t)
    print(f"  {pct:7d}% {conf[order[k - 1]]:12.3f} {caught:6d}/{sum(wrong):<6d} {acc_fixed:13.3f}")
print(f"  (ceiling if ALL routed: 1.000; readout alone: {1 - sum(wrong) / len(wrong):.3f}; "
      f"{sum(wrong)} wrong of {len(t)})")

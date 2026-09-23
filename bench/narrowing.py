# -*- coding: utf-8 -*-
"""Does narrowing cost accuracy? Measured against gold, not against intuition.

`jobe.wide` answers a choice too wide for the answer slots by grouping options
into a tree and multiplying probabilities down the path. That is a DIFFERENT
estimator from the flat readout, and `wide.py` has been saying so with nothing
behind it. This settles it.

The trick is that the question is testable on decisions that do not need
narrowing at all. Of JevBench's 231 public tasks, 142 declare 4 to 6 options -
comfortably inside the 16-slot protocol - so the flat readout can answer them
directly while a deliberately lowered `cap` forces the same decision through the
tree. Both are then scored against the same gold label.

Four arms, all on identical prompts:

  flat          the shipping readout, one pass
  cap3-full     tree of branching factor 3, every group opened
  cap2-full     tree of branching factor 2, every group opened - a deeper tree
  cap3-prune    branching factor 3, only the likeliest group opened

`*-full` isolates the tree-and-product estimator itself. `cap3-prune` adds the
expansion policy on top, which is the other half of what runs in production.

EVERY ARM SHARES ONE PRIMED PREFIX PER TASK, including the flat one. The states
here average ~3,700 tokens, so scoring flat through `score()` and narrowed
through the prefix cache would compare two things at once. Same mechanism, one
variable.

Reported as a PAIRED comparison - fixed, broken, and both-wrong - because two
accuracies that differ by 2 points can hide twenty disagreements, and only the
paired counts say whether narrowing moved anything real.

    python bench/narrowing.py --limit 40        # a quick pass
    python bench/narrowing.py                   # all 142
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics as st
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from jobe import Decision, Option, load                      # noqa: E402
from jobe.prefix import PrefixScorer                         # noqa: E402
from jobe.wide import score_wide                             # noqa: E402

PUBLIC = r"D:\Coding\jevbench\datasets\public"
MODEL = r"D:\Coding\models\qwen35-4b"
WEIGHTS_MIB = 8888          # measured: the two safetensors shards

ARMS = [
    ("flat", None),
    ("cap3-full", dict(cap=3, max_expand=9, expand_mass=1.1)),
    ("cap2-full", dict(cap=2, max_expand=9, expand_mass=1.1)),
    ("cap3-prune", dict(cap=3, max_expand=1, expand_mass=0.0)),
]


def free_vram_mb():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        used, total = (int(x) for x in out.replace(",", " ").split())
        return total - used
    except Exception:
        return None


def load_tasks(min_options: int):
    rows = []
    for name in ("easy.jsonl", "hard.jsonl", "original.jsonl"):
        with open(os.path.join(PUBLIC, name), encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if len(r["labels"]) >= min_options:
                    r["split"] = name.split(".")[0]
                    rows.append(r)
    return rows


def build_decision(task) -> Decision:
    """The same mapping `bench/jobe_direct.py` uses. Not a second opinion."""
    q = task["question"]
    crit = q.get("criteria")
    if q["type"] == "noul":
        c = crit or {}
        pairs = [(k, c.get(k) or f"The proposition is {k}.") for k in ("true", "false")]
    elif q["type"] == "choice":
        pairs = [(k, v or k) for k, v in (crit or {}).items()]
    else:
        pairs = [(str(i), lvl) for i, lvl in enumerate(crit or [])]
    return Decision(
        id=task["id"], evidence=task["state"], criterion=q["instructions"],
        options=tuple(Option(id=k, description=f"{k}: {v}") for k, v in pairs),
        ordinal=q["type"] == "score")


def total_variation(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-options", type=int, default=4)
    ap.add_argument("--out", default="runs/narrowing.json")
    args = ap.parse_args(argv)

    free = free_vram_mb()
    spilling = free is not None and free < WEIGHTS_MIB + 400
    print("free VRAM: %s MiB; weights need %d" % (free, WEIGHTS_MIB), flush=True)
    if spilling:
        print("NOTE: this will spill to host memory. ACCURACY IS UNAFFECTED - the\n"
              "      readout is deterministic - but every TIMING below is void.", flush=True)

    tasks = load_tasks(args.min_options)
    if args.limit:
        tasks = tasks[:args.limit]
    widths = collections.Counter(len(t["labels"]) for t in tasks)
    print("%d tasks, option counts %s" % (
        len(tasks), dict(sorted(widths.items()))), flush=True)

    bb = load(MODEL, device="auto")
    scorer = PrefixScorer(bb.model, bb.tokenizer)

    picks: dict[str, list] = {name: [] for name, _ in ARMS}
    dists: dict[str, list] = {name: [] for name, _ in ARMS}
    passes: dict[str, list] = {name: [] for name, _ in ARMS}
    gold, meta = [], []

    t0 = time.perf_counter()
    for i, task in enumerate(tasks, 1):
        decision = build_decision(task)
        scorer.prime(task["state"])          # once per task; every arm reuses it
        gold.append(task["expected"])
        meta.append({"id": task["id"], "split": task["split"],
                     "family": task["family"], "n": len(decision.options)})
        for name, cfg in ARMS:
            if cfg is None:
                r = scorer.score(decision)   # same prefix path as the narrowed arms
                picks[name].append(r.choice)
                dists[name].append(r.scores)
                passes[name].append(1)
            else:
                r = score_wide(bb.model, bb.tokenizer, decision, scorer=scorer, **cfg)
                picks[name].append(r.choice)
                dists[name].append(r.scores)
                passes[name].append(r.passes)
        if i % 10 == 0 or i == len(tasks):
            acc = {n: sum(p == g for p, g in zip(picks[n], gold)) / len(gold) for n, _ in ARMS}
            print("  %3d/%d  " % (i, len(tasks))
                  + "  ".join("%s %.3f" % (n, acc[n]) for n, _ in ARMS), flush=True)

    elapsed = time.perf_counter() - t0
    n = len(gold)
    print("\n%d decisions x %d arms in %.0fs%s"
          % (n, len(ARMS), elapsed, "  (VOID - spilled)" if spilling else ""))

    print("\n%-12s %8s %8s %10s %10s" % ("arm", "acc", "passes", "mean TV", "agree"))
    base = picks["flat"]
    rows = {}
    for name, _ in ARMS:
        correct = [p == g for p, g in zip(picks[name], gold)]
        acc = sum(correct) / n
        tv = st.mean(total_variation(a, b) for a, b in zip(dists["flat"], dists[name]))
        agree = sum(a == b for a, b in zip(base, picks[name])) / n
        rows[name] = {"accuracy": acc, "mean_passes": st.mean(passes[name]),
                      "mean_tv": tv, "agreement": agree}
        print("%-12s %8.3f %8.1f %10.3f %10.3f"
              % (name, acc, st.mean(passes[name]), tv, agree))

    print("\nPAIRED against flat (what actually moved):")
    for name, _ in ARMS[1:]:
        fixed = broken = both = 0
        for p, f, g in zip(picks[name], base, gold):
            if p == g and f != g:
                fixed += 1
            elif p != g and f == g:
                broken += 1
            elif p != g and f != g:
                both += 1
        rows[name].update(fixed=fixed, broken=broken, both_wrong=both)
        print("  %-12s fixed %2d   broken %2d   both wrong %2d   net %+d"
              % (name, fixed, broken, both, fixed - broken))

    print("\nBY OPTION COUNT (accuracy):")
    print("  %-6s %5s " % ("n", "tasks") + " ".join("%10s" % a for a, _ in ARMS))
    for w in sorted(widths):
        idx = [j for j, m in enumerate(meta) if m["n"] == w]
        if not idx:
            continue
        line = "  %-6d %5d " % (w, len(idx))
        for name, _ in ARMS:
            line += "%10.3f" % (sum(picks[name][j] == gold[j] for j in idx) / len(idx))
        print(line)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"n": n, "arms": rows, "spilled": spilling, "elapsed_s": elapsed,
                   "meta": meta, "gold": gold,
                   "picks": {k: v for k, v in picks.items()}}, fh)
    print("\nwrote " + args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Throughput of prefix-cache reuse: N questions about one long document.

Three paths over the same questions:
  uncached  every question re-encodes the whole document
  serial    document encoded once; each question a suffix off a cache copy
  batched   as serial, but several suffixes per forward pass (--batch sizes)

Reports ms/question, the speedups, peak GPU memory, and the largest slot-logit
disagreement against the uncached path - which must be rounding noise.

  python bench/prefix_bench.py --model D:\\Coding\\models\\qwen35-4b --questions 8 --batch 2 4 8
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from jobe import Decision, Option, PrefixScorer, load, score  # noqa: E402

PARA = ("Refunds require an original receipt and a purchase made within 30 days. Items marked "
        "final sale are excluded. A manager may approve an exception when the customer provides "
        "a bank statement showing the charge. Exchanges follow the same rule. ")
YESNO = (Option("yes", "yes: every required condition is established"),
         Option("no", "no: a condition is missing or a prohibition applies"))
CRITERIA = [
    "Is a refund permitted for a purchase 12 days ago with no receipt?",
    "Are final-sale items refundable?",
    "Can a manager approve a refund on a bank statement alone?",
    "Do exchanges follow the same rule as refunds?",
    "Is a purchase from 45 days ago refundable with a receipt?",
    "Is a receipt required for an exchange?",
    "Does the policy mention a time limit?",
    "Can a cashier approve an exception?",
]


def agreement(a, b):
    worst = max(max(abs(x - y) for x, y in zip(p.option_logits, q.option_logits)) for p, q in zip(a, b))
    flips = sum(p.choice != q.choice for p, q in zip(a, b))
    return worst, flips


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=40, help="paragraph repeats -> document length")
    ap.add_argument("--batch", type=int, nargs="*", default=[2, 4, 8])
    args = ap.parse_args()

    import torch

    bb = load(args.model, device="auto")
    state = PARA * args.repeat
    qs = [Decision(id=f"q{i}", evidence=state, criterion=CRITERIA[i % len(CRITERIA)], options=YESNO)
          for i in range(args.questions)]
    n = len(qs)
    score(bb.model, bb.tokenizer, qs[0])  # warm-up

    t0 = time.perf_counter()
    uncached = [score(bb.model, bb.tokenizer, d) for d in qs]
    t_un = time.perf_counter() - t0

    scorer = PrefixScorer(bb.model, bb.tokenizer)
    t0 = time.perf_counter()
    n_prefix = scorer.prime(state)
    t_prime = time.perf_counter() - t0
    t0 = time.perf_counter()
    serial = scorer.score_many(qs)
    t_se = time.perf_counter() - t0

    print(f"document {uncached[0].input_tokens} tokens (prefix {n_prefix}, suffix ~{serial[0].meta['suffix_tokens']}); "
          f"{n} questions; prime {t_prime * 1000:.0f} ms once")
    print(f"{'path':12s} {'ms/question':>12s} {'total s':>8s} {'vs uncached':>12s} {'vs serial':>10s} "
          f"{'peak GB':>8s} {'max|dlogit|':>12s} {'flips':>6s}")
    print("-" * 88)
    print(f"{'uncached':12s} {t_un / n * 1000:12.0f} {t_un:8.2f} {'1.0x':>12s} {'':>10s} {'':>8s} {'-':>12s} {'-':>6s}")
    w, f = agreement(serial, uncached)
    print(f"{'serial':12s} {t_se / n * 1000:12.0f} {t_se:8.2f} {t_un / t_se:11.1f}x {'1.0x':>10s} {'':>8s} {w:12.4f} {f:6d}")

    for b in args.batch:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            batched = scorer.score_batch(qs, max_batch=b)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{'batch ' + str(b):12s} {'OUT OF MEMORY':>12s}")
            continue
        t_b = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9
        w, f = agreement(batched, uncached)
        print(f"{'batch ' + str(b):12s} {t_b / n * 1000:12.0f} {t_b:8.2f} {t_un / t_b:11.1f}x "
              f"{t_se / t_b:9.1f}x {peak:8.2f} {w:12.4f} {f:6d}")
    print(f"\n(bf16 ulp at these logit magnitudes is 0.125; flips are argmax disagreements with the uncached path)")


if __name__ == "__main__":
    main()

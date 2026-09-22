"""Throughput of prefix-cache reuse: N questions about one long document.

Uncached, every question re-encodes the whole document. Cached, the document
is encoded once and each question is a short suffix. This reports both, plus
the largest slot-logit disagreement between the two paths - which must be
rounding noise, not a positional error.

  python bench/prefix_bench.py --model D:\\Coding\\models\\qwen35-4b --questions 8 --repeat 40
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=40, help="paragraph repeats -> document length")
    args = ap.parse_args()

    bb = load(args.model, device="auto")
    state = PARA * args.repeat
    qs = [Decision(id=f"q{i}", evidence=state, criterion=CRITERIA[i % len(CRITERIA)], options=YESNO)
          for i in range(args.questions)]

    # warm-up
    score(bb.model, bb.tokenizer, qs[0])

    t0 = time.perf_counter()
    uncached = [score(bb.model, bb.tokenizer, d) for d in qs]
    t_un = time.perf_counter() - t0

    scorer = PrefixScorer(bb.model, bb.tokenizer)
    t0 = time.perf_counter()
    n_prefix = scorer.prime(state)
    cached = scorer.score_many(qs)
    t_ca = time.perf_counter() - t0

    worst = max(max(abs(a - b) for a, b in zip(c.option_logits, u.option_logits))
                for c, u in zip(cached, uncached))
    flips = sum(c.choice != u.choice for c, u in zip(cached, uncached))

    print(f"document: {uncached[0].input_tokens} tokens (prefix {n_prefix}, suffix ~{cached[0].meta['suffix_tokens']})")
    print(f"questions: {len(qs)}")
    print(f"uncached : {t_un:6.2f} s total  {t_un / len(qs) * 1000:7.0f} ms/question  {len(qs) / t_un:5.2f} dec/s")
    print(f"cached   : {t_ca:6.2f} s total  {t_ca / len(qs) * 1000:7.0f} ms/question  {len(qs) / t_ca:5.2f} dec/s"
          f"   (prefix {scorer.prefix_seconds * 1000:.0f} ms once, then "
          f"{sum(c.forward_seconds for c in cached) / len(cached) * 1000:.0f} ms/question)")
    print(f"speedup  : {t_un / t_ca:.1f}x over {len(qs)} questions; "
          f"{uncached[0].forward_seconds / (sum(c.forward_seconds for c in cached) / len(cached)):.1f}x per question after the first")
    print(f"agreement: max |dlogit| {worst:.4f} (bf16 ulp at these magnitudes is 0.125); argmax flips {flips}/{len(qs)}")


if __name__ == "__main__":
    main()

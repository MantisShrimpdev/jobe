"""Calibration: one fitted temperature, and the metrics that say whether it
helped.

A restricted softmax produces numbers that sum to 1. That is not the same as
numbers that mean anything. Measured on a hosted model
(EveryAppKit's docs/SEMANTIC_DECISION_BENCHMARK.md), expected calibration error
ran 0.078 -> 0.245 -> 0.679 as decisions got harder: reliable on easy ones,
actively misleading on hard ones, with nothing in the score to say which you
were looking at.

Temperature scaling is the cheapest fix available and costs exactly one scalar.
It is fitted on a HELD-OUT split, never on the data you report, and it changes
no argmax - only how confident the distribution is.

Everything here is pure Python on stored logits, so once one pass has been
recorded the whole calibration loop runs instantly, with no model and no GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def softmax_with_temperature(logits: list[float], temperature: float) -> list[float]:
    """Softmax of ``logits / temperature``. T > 1 softens, T < 1 sharpens."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = [v / temperature for v in logits]
    top = max(scaled)
    weights = [math.exp(v - top) for v in scaled]
    total = sum(weights)
    return [w / total for w in weights]


def negative_log_likelihood(
    rows: list[list[float]], gold: list[int], temperature: float = 1.0
) -> float:
    """Mean NLL of the gold option under the temperature-scaled distribution.

    A strictly proper scoring rule: it is minimised only by reporting the true
    probabilities, which is exactly why it is the right thing to fit on.
    """
    if len(rows) != len(gold):
        raise ValueError("rows and gold must be the same length")
    if not rows:
        raise ValueError("no rows")
    total = 0.0
    for logits, index in zip(rows, gold):
        probs = softmax_with_temperature(logits, temperature)
        total -= math.log(max(probs[index], 1e-12))
    return total / len(rows)


def fit_temperature(
    rows: list[list[float]],
    gold: list[int],
    *,
    lo: float = 0.05,
    hi: float = 20.0,
    tolerance: float = 1e-4,
) -> float:
    """Fit one temperature by minimising NLL on a held-out split.

    Golden-section search over a one-dimensional, well-behaved objective — no
    optimiser, no gradients and no dependency. (SemIf's lineage uses L-BFGS for
    this; for a single scalar that is machinery without a purpose.)

    Args:
        rows: per-decision option logits, as recorded by the readout.
        gold: index of the correct option in each row.
        lo, hi: bracket to search. The default spans sharpening to heavy
            softening; a fit landing on either bound is reported as-is, and a
            bound-hugging result means the bracket was wrong for your data.
        tolerance: stop when the bracket is narrower than this.

    Returns:
        The temperature minimising held-out NLL. Divide logits by it.
    """
    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc = negative_log_likelihood(rows, gold, c)
    fd = negative_log_likelihood(rows, gold, d)
    while b - a > tolerance:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = negative_log_likelihood(rows, gold, c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = negative_log_likelihood(rows, gold, d)
    return (a + b) / 2.0


def brier_score(probs: list[float], gold: int) -> float:
    """Multi-class Brier: ``sum_k (p_k - y_k)^2``. Range [0, 2]. Proper."""
    return sum((p - (1.0 if i == gold else 0.0)) ** 2 for i, p in enumerate(probs))


@dataclass(frozen=True)
class CalibrationReport:
    """How much a set of scores can be trusted."""

    n: int
    accuracy: float
    mean_nll: float
    mean_brier: float
    ece: float
    mce: float
    #: (lo, hi, count, mean_confidence, accuracy) per occupied bin.
    bins: tuple[tuple[float, float, int, float, float], ...]
    temperature: float = 1.0


def calibration_report(
    rows: list[list[float]],
    gold: list[int],
    *,
    temperature: float = 1.0,
    n_bins: int = 10,
    min_bin_count: int = 1,
    predictions: list[int] | None = None,
) -> CalibrationReport:
    """Accuracy, NLL, Brier, ECE and MCE over stored logits.

    `predictions` overrides which option counts as the answer. It exists because
    **argmax is not the prediction for an ordinal question** — a `score` answer
    is the rounded probability-weighted value, which can differ from the
    highest-probability level. Passing ordinal rows without it silently scores
    them by argmax and skews accuracy and ECE together. Measured on 111 mixed
    tasks: argmax-only gave ECE 0.133 where the ordinal-aware answer was 0.106.

    ECE is the occupancy-weighted gap between confidence and accuracy across
    equal-width bins. MCE is the worst such gap, restricted to bins holding at
    least `min_bin_count` samples — otherwise a bin with one lucky sample
    dominates the number and it stops meaning anything.

    Invalid distributions are never repaired here: if a probability is missing
    it is absent, not imputed. A synthesised confidence is a lie about how much
    was measured.
    """
    if len(rows) != len(gold):
        raise ValueError("rows and gold must be the same length")
    if not rows:
        raise ValueError("no rows")

    if predictions is not None and len(predictions) != len(rows):
        raise ValueError("predictions must be the same length as rows")

    buckets = [{"n": 0, "conf": 0.0, "hit": 0} for _ in range(n_bins)]
    hits = 0
    nll = 0.0
    brier = 0.0
    for offset, (logits, index) in enumerate(zip(rows, gold)):
        probs = softmax_with_temperature(logits, temperature)
        predicted = (
            predictions[offset]
            if predictions is not None
            else max(range(len(probs)), key=lambda i: probs[i])
        )
        correct = predicted == index
        hits += correct
        nll -= math.log(max(probs[index], 1e-12))
        brier += brier_score(probs, index)
        # TOP-LABEL confidence, matching JevBench's `ece_top_label`. For a plain
        # classification this is the same as the probability of the prediction.
        # They diverge only for an ordinal question, where the answer is a
        # rounded expected value that need not be the highest-probability level
        # - and the convention has to be pinned, or two correct implementations
        # disagree. Measured: prob-of-prediction gave 0.126 where top-label
        # gave 0.106 on the same 111 rows.
        confidence = max(probs)
        slot = min(int(confidence * n_bins), n_bins - 1)
        buckets[slot]["n"] += 1
        buckets[slot]["conf"] += confidence
        buckets[slot]["hit"] += 1 if correct else 0

    n = len(rows)
    ece = 0.0
    mce = 0.0
    bins: list[tuple[float, float, int, float, float]] = []
    for i, bucket in enumerate(buckets):
        if not bucket["n"]:
            continue
        mean_conf = bucket["conf"] / bucket["n"]
        accuracy = bucket["hit"] / bucket["n"]
        gap = abs(mean_conf - accuracy)
        ece += (bucket["n"] / n) * gap
        if bucket["n"] >= min_bin_count:
            mce = max(mce, gap)
        bins.append((i / n_bins, (i + 1) / n_bins, bucket["n"], mean_conf, accuracy))

    return CalibrationReport(
        n=n,
        accuracy=hits / n,
        mean_nll=nll / n,
        mean_brier=brier / n,
        ece=ece,
        mce=mce,
        bins=tuple(bins),
        temperature=temperature,
    )

"""Order averaging: read the same decision under several option orders and
average the distributions.

Option letters are assigned by position, so the order you happen to declare
options in is a variable the caller never chose. Measured on a hosted model
(EveryAppKit's docs/SEMANTIC_DECISION_BENCHMARK.md): reversing the option list
flipped 4.2% of answers on easy decisions and **100%** on hard ones, where the
model had collapsed to always answering the first slot.

Averaging is not free - it costs one forward pass per order - and it **pays in
proportion to the fragility it fixes**. On the order-degenerate model it was
worth +8.2 accuracy points and cut ECE to 0.42x. On a model that was already
stable it *lost* 3 points and cut ECE only to 0.81x. So `flip_rate` is reported
alongside the averaged result: measure it before deciding to pay for this.
"""

from __future__ import annotations

from dataclasses import dataclass

from .prompt import Decision
from .readout import Readout


def order_variants(count: int, strategy: str = "reversed") -> list[list[int]]:
    """Build the option orders to score under.

    Args:
        count: number of options.
        strategy: ``"declared"`` (one pass, no averaging), ``"reversed"``
            (declared + reversed, the cheap default), or ``"rotations"``
            (every cyclic rotation - `count` passes, thorough and expensive).

    Returns:
        A list of index permutations, always starting with the declared order.
    """
    declared = list(range(count))
    if strategy == "declared":
        return [declared]
    if strategy == "reversed":
        return [declared, declared[::-1]]
    if strategy == "rotations":
        return [declared[i:] + declared[:i] for i in range(count)]
    raise ValueError(f"unknown order strategy {strategy!r}")


@dataclass(frozen=True)
class AveragedReadout:
    """A decision scored under several option orders."""

    decision_id: str
    #: Option ids in the order they were DECLARED (not as presented).
    option_ids: tuple[str, ...]
    #: Mean probability per option across all orders, summing to 1.
    probabilities: tuple[float, ...]
    #: Each order's readout, in the order they were run.
    per_order: tuple[Readout, ...]
    #: Share of orders whose argmax differs from the declared order's argmax.
    flip_rate: float
    #: Mean total-variation distance from the declared order's distribution.
    mean_total_variation: float

    @property
    def choice(self) -> str:
        best = max(range(len(self.option_ids)), key=lambda i: self.probabilities[i])
        return self.option_ids[best]

    @property
    def scores(self) -> dict[str, float]:
        return dict(zip(self.option_ids, self.probabilities))

    @property
    def total_seconds(self) -> float:
        return sum(r.total_seconds for r in self.per_order)

    def confidence(self) -> float:
        """Chance-corrected, as :meth:`Readout.confidence` — same caveat applies."""
        k = len(self.probabilities)
        return (max(self.probabilities) - 1.0 / k) / (1.0 - 1.0 / k)

    def disagreed(self) -> bool:
        """True when the orders did not all choose the same option.

        A useful routing signal: disagreement concentrates at low confidence,
        and these are the decisions worth escalating.
        """
        return self.flip_rate > 0.0


def total_variation(a: dict[str, float], b: dict[str, float]) -> float:
    """Total-variation distance between two distributions keyed by option id."""
    keys = set(a) | set(b)
    return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys) / 2.0


def average_readouts(
    decision: Decision, readouts: list[Readout]
) -> AveragedReadout:
    """Merge readouts taken under different orders. Pure — no model needed.

    Each readout reports probabilities in its own *presented* order, so they are
    re-keyed by option id before averaging. The result is aligned to the order
    the caller declared.
    """
    if not readouts:
        raise ValueError("no readouts to average")
    declared = tuple(o.id for o in decision.options)
    first = readouts[0].scores
    totals = {oid: 0.0 for oid in declared}
    flips = 0
    tv = 0.0
    for readout in readouts:
        scores = readout.scores
        if set(scores) != set(declared):
            raise ValueError(f"{decision.id}: readout options do not match the decision")
        for oid, p in scores.items():
            totals[oid] += p
        if readout.choice != readouts[0].choice:
            flips += 1
        tv += total_variation(first, scores)
    n = len(readouts)
    return AveragedReadout(
        decision_id=decision.id,
        option_ids=declared,
        probabilities=tuple(totals[oid] / n for oid in declared),
        per_order=tuple(readouts),
        flip_rate=flips / n,
        mean_total_variation=tv / n,
    )


def score_averaged(
    model,
    tokenizer,
    decision: Decision,
    *,
    strategy: str = "reversed",
    max_tokens: int = 4096,
) -> AveragedReadout:
    """Score a decision under several option orders and average the result.

    Costs one forward pass per order. Check `flip_rate` on a sample of your own
    decisions before adopting this: if it is near zero, averaging buys nothing.
    """
    from .readout import score

    orders = order_variants(len(decision.options), strategy)
    readouts = [
        score(model, tokenizer, decision.reordered(order), max_tokens=max_tokens)
        for order in orders
    ]
    return average_readouts(decision, readouts)

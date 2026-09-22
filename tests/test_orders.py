"""Order averaging. All pure — no model, no GPU."""

from __future__ import annotations

import math

import pytest

from jobe.orders import (
    average_readouts,
    order_variants,
    total_variation,
)
from jobe.prompt import Decision, Option
from jobe.readout import Readout

OPTS = (
    Option("billing", "Payments and refunds"),
    Option("infra", "Outages and deploys"),
    Option("sales", "Pricing and contracts"),
)
DECISION = Decision(
    id="d1", evidence="something happened", criterion="who owns it?", options=OPTS
)


def readout(ids: tuple[str, ...], probs: tuple[float, ...]) -> Readout:
    return Readout(
        decision_id="d1",
        option_ids=ids,
        probabilities=probs,
        option_logits=tuple(0.0 for _ in probs),
        input_tokens=10,
        forward_seconds=0.01,
        total_seconds=0.02,
        prompt_sha256="x",
    )


# ---------------------------------------------------------------------- orders


def test_order_variants_shapes():
    assert order_variants(3, "declared") == [[0, 1, 2]]
    assert order_variants(3, "reversed") == [[0, 1, 2], [2, 1, 0]]
    assert order_variants(3, "rotations") == [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    with pytest.raises(ValueError, match="unknown order strategy"):
        order_variants(3, "spiral")


def test_every_strategy_starts_from_the_declared_order():
    for strategy in ("declared", "reversed", "rotations"):
        assert order_variants(4, strategy)[0] == [0, 1, 2, 3]


def test_averaging_realigns_by_id_not_position():
    """The whole point: each order PRESENTS options differently, so averaging
    by position would add unrelated options together."""
    declared = readout(("billing", "infra", "sales"), (0.6, 0.3, 0.1))
    reversed_ = readout(("sales", "infra", "billing"), (0.2, 0.2, 0.6))
    avg = average_readouts(DECISION, [declared, reversed_])

    assert avg.option_ids == ("billing", "infra", "sales")
    assert math.isclose(avg.scores["billing"], 0.6, abs_tol=1e-12)
    assert math.isclose(avg.scores["infra"], 0.25, abs_tol=1e-12)
    assert math.isclose(avg.scores["sales"], 0.15, abs_tol=1e-12)
    assert math.isclose(sum(avg.probabilities), 1.0, abs_tol=1e-12)


def test_averaging_reports_agreement():
    same = [
        readout(("billing", "infra", "sales"), (0.6, 0.3, 0.1)),
        readout(("sales", "infra", "billing"), (0.1, 0.3, 0.6)),
    ]
    avg = average_readouts(DECISION, same)
    assert avg.flip_rate == 0.0
    assert avg.disagreed() is False
    assert avg.choice == "billing"


def test_averaging_reports_disagreement_and_distance():
    disagree = [
        readout(("billing", "infra", "sales"), (0.6, 0.3, 0.1)),
        readout(("sales", "infra", "billing"), (0.6, 0.3, 0.1)),  # picks sales
    ]
    avg = average_readouts(DECISION, disagree)
    assert avg.flip_rate == 0.5
    assert avg.disagreed() is True
    assert avg.mean_total_variation > 0.0


def test_averaging_rejects_readouts_from_a_different_decision():
    with pytest.raises(ValueError, match="do not match"):
        average_readouts(DECISION, [readout(("a", "b", "c"), (0.5, 0.3, 0.2))])
    with pytest.raises(ValueError, match="no readouts"):
        average_readouts(DECISION, [])


def test_total_variation_bounds():
    assert total_variation({"a": 1.0, "b": 0.0}, {"a": 1.0, "b": 0.0}) == 0.0
    assert math.isclose(
        total_variation({"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}), 1.0, abs_tol=1e-12
    )


def test_averaged_confidence_is_chance_corrected():
    flat = average_readouts(
        DECISION, [readout(("billing", "infra", "sales"), (1 / 3, 1 / 3, 1 / 3))]
    )
    assert math.isclose(flat.confidence(), 0.0, abs_tol=1e-12)

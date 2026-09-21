"""Order averaging and calibration. All pure — no model, no GPU."""

from __future__ import annotations

import math

import pytest

from jobe.calibrate import (
    brier_score,
    calibration_report,
    fit_temperature,
    negative_log_likelihood,
    softmax_with_temperature,
)
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


# ----------------------------------------------------------------- calibration


def test_temperature_one_is_the_identity():
    logits = [2.0, 1.0, 0.0]
    a = softmax_with_temperature(logits, 1.0)
    top = max(logits)
    weights = [math.exp(v - top) for v in logits]
    total = sum(weights)
    for x, y in zip(a, [w / total for w in weights]):
        assert math.isclose(x, y, abs_tol=1e-12)


def test_higher_temperature_softens_lower_sharpens():
    logits = [3.0, 0.0]
    assert max(softmax_with_temperature(logits, 5.0)) < max(
        softmax_with_temperature(logits, 1.0)
    )
    assert max(softmax_with_temperature(logits, 0.2)) > max(
        softmax_with_temperature(logits, 1.0)
    )
    with pytest.raises(ValueError, match="positive"):
        softmax_with_temperature(logits, 0.0)


def test_fit_recovers_a_known_temperature():
    """Build a set whose empirical label frequencies match softmax(logits / T),
    so the NLL minimum sits at T by construction, then check the fitter finds it."""
    true_t = 2.5
    base = [[3.0, 1.0, 0.0], [0.5, 2.0, 1.0], [4.0, 3.5, 3.0]]
    rows: list[list[float]] = []
    gold: list[int] = []
    for logits in base:
        probs = softmax_with_temperature(logits, true_t)
        for index, p in enumerate(probs):
            for _ in range(round(p * 2000)):
                rows.append(logits)
                gold.append(index)
    fitted = fit_temperature(rows, gold)
    assert abs(fitted - true_t) < 0.1, f"fitted {fitted}, expected ~{true_t}"


def test_fit_softens_an_overconfident_set():
    """Sharp logits whose labels are only sometimes right need T > 1."""
    rows = [[8.0, 0.0]] * 10
    gold = [0] * 6 + [1] * 4  # 60% accurate while claiming near-certainty
    assert fit_temperature(rows, gold) > 1.5


def test_nll_is_minimised_at_the_fitted_temperature():
    rows = [[5.0, 0.0]] * 10
    gold = [0] * 7 + [1] * 3
    t = fit_temperature(rows, gold)
    best = negative_log_likelihood(rows, gold, t)
    assert best <= negative_log_likelihood(rows, gold, t * 0.6) + 1e-9
    assert best <= negative_log_likelihood(rows, gold, t * 1.6) + 1e-9


def test_brier_rewards_the_truth():
    assert math.isclose(brier_score([1.0, 0.0], 0), 0.0, abs_tol=1e-12)
    assert math.isclose(brier_score([0.0, 1.0], 0), 2.0, abs_tol=1e-12)


def test_report_sees_perfect_and_broken_calibration():
    # Confident and right -> tiny ECE.
    good = calibration_report([[10.0, 0.0]] * 20, [0] * 20)
    assert good.accuracy == 1.0
    assert good.ece < 0.01

    # Equally confident, right only half the time -> ECE near 0.5.
    bad = calibration_report([[10.0, 0.0]] * 20, [0] * 10 + [1] * 10)
    assert bad.accuracy == 0.5
    assert bad.ece > 0.45
    assert bad.mce >= bad.ece


def test_temperature_improves_a_miscalibrated_report():
    rows = [[6.0, 0.0]] * 20
    gold = [0] * 13 + [1] * 7
    before = calibration_report(rows, gold)
    t = fit_temperature(rows, gold)
    after = calibration_report(rows, gold, temperature=t)
    assert after.ece < before.ece
    # Temperature scaling must never change which option wins.
    assert after.accuracy == before.accuracy


def test_report_rejects_mismatched_input():
    with pytest.raises(ValueError, match="same length"):
        calibration_report([[1.0, 0.0]], [0, 1])
    with pytest.raises(ValueError, match="no rows"):
        calibration_report([], [])

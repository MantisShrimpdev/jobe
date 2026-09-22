# -*- coding: utf-8 -*-
"""The long_policy worlds: long packs, exact outcomes, named mistakes, no JevBench text."""
import datetime as dt
import os
import random

import pytest

from jobe.worlds.base import contamination, evidence_chars, to_decision
from jobe.worlds.long_policy import LAWS, generate, working_days_after


@pytest.fixture(scope="module")
def worlds():
    return generate(6 * 30, seed=3)


def test_every_law_validates_is_long_and_is_deterministic(worlds):
    assert {w.law for w in worlds} == set(LAWS)
    for w in worlds:
        assert w.expected in dict(w.options)
        assert w.expected not in w.mistakes
        assert 7000 <= evidence_chars(w) <= 14000, (w.id, evidence_chars(w))   # the family's band, roughly 2-3.5k tokens
        assert w.family == "long_policy"
    again = generate(6 * 30, seed=3)
    assert [w.record() for w in again] == [w.record() for w in worlds]


def test_named_mistakes_distinguish_an_option_in_most_records(worlds):
    by_law = {}
    for w in worlds:
        by_law.setdefault(w.law, []).append(len(w.mistakes))
    for law, counts in by_law.items():
        assert sum(1 for c in counts if c) / len(counts) >= 0.5, f"{law}: too few records where a mistake changes the answer"


def test_every_option_is_the_gold_somewhere(worlds):
    """A law whose answer is always the same option teaches a reflex, not a law."""
    by_law = {}
    for w in worlds:
        by_law.setdefault(w.law, set()).add(w.expected)
    for law, golds in by_law.items():
        assert len(golds) >= 2, f"{law}: gold is always {golds}"


def test_records_become_valid_decisions(worlds):
    for w in worlds[:12]:
        d = to_decision(w.record())
        d.validate()


# --- working days with a closure period -----------------------------------------
def test_working_days_after_skips_weekends_holidays_and_closures():
    off = {dt.date(2026, 12, 25), dt.date(2026, 12, 26), dt.date(2027, 1, 1)} | {dt.date(2026, 12, 24) + dt.timedelta(days=i) for i in range(9)}
    # Mon 7 Dec 2026 -> 8-11 Dec (4), 14-18 (9), 21-23 (12), 4-6 Jan (15)
    assert working_days_after(dt.date(2026, 12, 7), 15, off) == dt.date(2027, 1, 6)
    assert working_days_after(dt.date(2026, 12, 7), 10, off) == dt.date(2026, 12, 21)
    assert working_days_after(dt.date(2026, 12, 4), 1, set()) == dt.date(2026, 12, 7)   # Friday -> Monday


# --- hand-checked outcomes on fixed seeds (seed 0, round-robin index) ------------------
def law_at(name: str, idx: int):
    return LAWS[name](random.Random(f"0:{name}:{idx}"), idx)


def test_approval_level_hand_check():
    w = law_at("approval_level", 0)
    p = w.params
    # 32,500 x 2 years + 15,000 + 2,400 = 82,400; the related purchase is 160 days old and the amendment does not apply
    assert (p["fee"], p["renewals"], p["impl"], p["training"], p["tcv"]) == (32500, 1, 15000, 2400, 82400)
    assert p["amount"] == 82400 and not p["amend_applies"]
    t1, t2, t3, t4 = p["thresholds"]
    assert t1 < 82400 <= t2
    assert w.expected == "department_head"
    assert w.mistakes.get("budget_holder") == "annual_fee_only"


def test_water_damage_hand_check():
    w = law_at("water_damage_claim", 1)
    # six weeks in plain view, family living there: repeated seepage, no concealed-water exception, no vacancy
    assert (w.params["weeks"], w.params["concealed"], w.params["occupancy"]) == (6, False, "lived_in")
    assert w.expected == "deny_repeated_seepage"
    assert w.mistakes == {"pay_full_estimate_less_deductible": "seepage_exclusion_missed"}


def test_alert_routing_hand_check():
    w = law_at("alert_routing", 2)
    # self-managed database replica at S1, no active window on it: Rule 4
    assert (w.params["source"], w.params["severity"], w.params["window_on_component"]) == ("self", "S1", False)
    assert w.expected == "database_oncall"
    assert w.mistakes["security_incident_response"] == "security_by_association"


def test_returns_decision_hand_check():
    w = law_at("returns_decision", 5)
    # delivered 28 Jun, complete request 29 Jul = day 31 of a 30-day window; X- code makes it configured; no fault found
    assert (w.params["delivery"], w.params["request"], w.params["window"]) == ("2026-06-28", "2026-07-29", 30)
    assert w.params["configured"] and not w.params["defect"]
    assert w.expected == "reject_return"


@pytest.mark.skipif(not os.environ.get("JEVBENCH_DIR"), reason="set JEVBENCH_DIR to a jevbench clone")
def test_no_ten_word_run_is_shared_with_the_public_tasks(worlds):
    assert contamination(worlds, os.environ["JEVBENCH_DIR"]) == []

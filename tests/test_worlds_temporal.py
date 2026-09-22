# -*- coding: utf-8 -*-
"""The temporal_numeric worlds: exact laws, hand-checked answers, honest options."""
import datetime as dt
import os
import random
from decimal import Decimal

import pytest

from jobe.worlds.temporal_numeric import (
    LAWS, ZONES, World, add_months, check, contamination, generate, money, to_decision,
)

NY, LONDON, TOKYO = ZONES["New York"], ZONES["London"], ZONES["Tokyo"]


# --- the clock-change law -----------------------------------------------------
def test_us_and_eu_transitions_2026():
    fwd, back = NY.transitions(2026)
    assert (fwd, back) == (dt.datetime(2026, 3, 8, 2), dt.datetime(2026, 11, 1, 2))
    fwd, back = LONDON.transitions(2026)
    assert (fwd, back) == (dt.datetime(2026, 3, 29, 1), dt.datetime(2026, 10, 25, 2))
    assert TOKYO.transitions(2026) is None


def test_offsets_and_round_trip():
    assert NY.offset(dt.datetime(2026, 1, 15, 12)) == dt.timedelta(hours=-5)
    assert NY.offset(dt.datetime(2026, 7, 15, 12)) == dt.timedelta(hours=-4)
    assert LONDON.offset(dt.datetime(2026, 3, 29, 0, 30)) == dt.timedelta(0)
    assert LONDON.offset(dt.datetime(2026, 3, 29, 3)) == dt.timedelta(hours=1)
    for z in ZONES.values():
        for w in (dt.datetime(2027, 2, 3, 9), dt.datetime(2027, 6, 3, 23), dt.datetime(2027, 12, 31, 5)):
            assert z.from_utc(z.to_utc(w)) == w


def test_change_instants_and_the_hour_before_a_spring_forward():
    assert NY.change_instants(2026) == (dt.datetime(2026, 3, 8, 7), dt.datetime(2026, 11, 1, 6))
    assert LONDON.change_instants(2026) == (dt.datetime(2026, 3, 29, 1), dt.datetime(2026, 10, 25, 1))
    # 06:30 UTC on the spring-forward day is still 01:30 standard time in New York
    assert NY.from_utc(dt.datetime(2026, 3, 8, 6, 30)) == dt.datetime(2026, 3, 8, 1, 30)
    assert NY.from_utc(dt.datetime(2026, 3, 8, 7, 0)) == dt.datetime(2026, 3, 8, 3, 0)
    # the hour after a fall-back reads 01:xx again, and is flagged as near the change
    assert NY.from_utc(dt.datetime(2026, 11, 1, 6, 30)) == dt.datetime(2026, 11, 1, 1, 30)
    assert NY.near_transition(dt.datetime(2026, 11, 1, 1, 30), 1)     # 05:30 UTC, half an hour before 06:00
    assert not NY.near_transition(dt.datetime(2026, 11, 1, 12, 0), 1)


def test_elapsed_across_a_spring_forward_is_one_hour_less_than_the_wall_clock():
    end = dt.datetime(2026, 3, 7, 22)          # Saturday night, before the jump
    start = dt.datetime(2026, 3, 8, 9)         # Sunday morning, after it
    assert (start - end) == dt.timedelta(hours=11)                          # what the wall clock says
    assert NY.to_utc(start) - NY.to_utc(end) == dt.timedelta(hours=10)      # what actually elapsed


# --- the month-end law ----------------------------------------------------------
def test_add_months_month_end_rule_and_leap_year():
    assert add_months(dt.date(2026, 8, 31), 18) == dt.date(2028, 2, 29)
    assert add_months(dt.date(2026, 8, 31), 18, month_end_rule=False) == dt.date(2028, 3, 2)
    assert add_months(dt.date(2027, 1, 31), 1) == dt.date(2027, 2, 28)
    assert add_months(dt.date(2027, 1, 15), 12) == dt.date(2028, 1, 15)


# --- every law, as a population ------------------------------------------------
@pytest.fixture(scope="module")
def worlds():
    return generate(11 * 40, seed=7)


def test_every_law_validates_and_is_deterministic(worlds):
    assert {w.law for w in worlds} == set(LAWS)
    for w in worlds:
        check(w)                                           # gold among options, mistakes among options, snake_case ids
        assert w.expected not in w.mistakes
        assert 2 <= len(w.options) <= 16
    again = generate(11 * 40, seed=7)
    assert [w.record() for w in again] == [w.record() for w in worlds]
    assert generate(11, seed=8)[0].record() != worlds[0].record()


def test_split_is_a_stable_hash_near_ten_percent(worlds):
    held = sum(w.split == "heldout" for w in worlds)
    assert 0.04 < held / len(worlds) < 0.18
    assert all(w.split == World(**{**w.__dict__}).split for w in worlds[:20])


def test_named_mistakes_exist_for_every_law(worlds):
    """A law whose mistakes never change the answer is teaching nothing."""
    by_law = {}
    for w in worlds:
        by_law.setdefault(w.law, []).append(len(w.mistakes))
    for law, counts in by_law.items():
        assert sum(counts) / len(counts) >= 0.25, f"{law}: mistakes rarely distinguish an option"


def test_records_become_valid_decisions(worlds):
    for w in worlds[:33]:
        d = to_decision(w.record())
        d.validate()
        assert [o.id for o in d.options] == [i for i, _ in w.options]


# --- hand-checked answers on fixed seeds (seed 0, the round-robin index) -----------
def law_at(name: str, idx: int) -> World:
    return LAWS[name](random.Random(f"0:{name}:{idx}"), idx)


def test_interest_period_hand_check():
    w = law_at("interest_period", 0)
    p = w.params
    assert (p["basis"], p["days"]) == ("ACT/365", 28)
    # 3,685,000 x 9.5 % x 28 / 365
    assert w.expected == "usd_26855_07"
    assert money(Decimal(p["principal"]) * Decimal(p["rate"]) * 28 / 365) == Decimal("26855.07")
    assert "inclusive_end_date" in w.mistakes.values() and "wrong_basis_365_vs_360" in w.mistakes.values()


def test_threshold_day_hand_check():
    w = law_at("threshold_day", 2)
    assert w.expected == "sep_17"                         # 90 % of 34,000 = 30,600 net, reached at the close of the 17th
    assert w.mistakes.get("sep_16") == "ignored_credits"  # gross charges get there a day earlier


def test_usage_allowance_hand_check():
    w = law_at("usage_allowance", 7)
    # 3.25 TiB = 3,573,412,790,272 B; 1018.89 + 808.08 - 70.27 + 632.41 + 1124.30 GB = 3,513.41 GB: 60 GB short
    assert w.expected == "under_100gb_below"
    assert w.mistakes.get("over_limit") == "tib_read_as_decimal_tb"


def test_credit_proration_hand_check():
    w = law_at("credit_proration", 10)
    # 1,460 credits over 365 days = 4/day; 11-20 Dec inclusive = 10 days, one suspended = 9 active
    assert w.expected == "36_credits"
    assert w.mistakes.get("40_credits") == "ignored_the_suspension"


def test_term_end_hand_check():
    w = law_at("term_end", 1)
    # 15 Aug 2026 + 36 months = 15 Aug 2029; ends 24:00 Helsinki (UTC+3) = 21:00 UTC;
    # claim 04:30 Singapore (UTC+8) on 16 Aug = 20:30 UTC on the 15th -> within, though the wall clock says otherwise
    assert w.expected == "yes"
    assert w.mistakes == {"no": "compared_wall_clocks_without_converting"}


# --- contamination, when a JevBench clone is at hand -------------------------------
@pytest.mark.skipif(not os.environ.get("JEVBENCH_DIR"), reason="set JEVBENCH_DIR to a jevbench clone")
def test_no_ten_word_run_is_shared_with_the_public_tasks(worlds):
    assert contamination(worlds, os.environ["JEVBENCH_DIR"]) == []

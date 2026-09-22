# -*- coding: utf-8 -*-
"""The multi_hop worlds: exact chains, decoys that each reach a named wrong option, no contradictions."""
import os
import random
import re

import pytest

from jobe.worlds.base import contamination, evidence_chars, to_decision
from jobe.worlds.multi_hop import LAWS, SKINS, generate


@pytest.fixture(scope="module")
def worlds():
    return generate(3 * 40, seed=5)


def test_every_law_validates_and_is_deterministic(worlds):
    assert {w.law for w in worlds} == set(LAWS)
    for w in worlds:
        assert w.expected in dict(w.options)
        assert w.expected not in w.mistakes
        assert w.family == "multi_hop"
        assert evidence_chars(w) >= 5000, (w.id, evidence_chars(w))
    assert [w.record() for w in generate(3 * 40, seed=5)] == [w.record() for w in worlds]


def test_chain_decoys_reach_every_wrong_option(worlds):
    """Six decoy paths, three wrong dispositions: each wrong option is reachable by a named path."""
    for w in worlds:
        if w.law == "identifier_chain":
            wrong = {i for i, _ in w.options} - {w.expected}
            assert set(w.mistakes) == wrong, (w.id, w.mistakes)


def test_chain_registers_never_map_one_identifier_two_ways(worlds):
    """A padding row that reuses a live code would make the file contradict itself."""
    for w in worlds:
        if w.law != "identifier_chain":
            continue
        seen = {}
        for line in w.evidence.split("\n"):
            m = re.match(r"^([A-Z]{2,4}-\d+|[A-Z]{3}) -> (.+?);", line)
            if m:
                code, target = m.group(1), m.group(2)
                assert seen.setdefault(code, target) == target, (w.id, code, seen[code], target)


def test_every_skin_and_every_disposition_occurs(worlds):
    chains = [w for w in worlds if w.law == "identifier_chain"]
    assert {w.params["skin"] for w in chains} == set(SKINS)
    by_skin = {}
    for w in chains:
        by_skin.setdefault(w.params["skin"], set()).add(w.expected)
    assert all(len(v) >= 2 for v in by_skin.values()), by_skin


def test_named_mistakes_distinguish_an_option_in_most_records(worlds):
    by_law = {}
    for w in worlds:
        by_law.setdefault(w.law, []).append(len(w.mistakes))
    for law, counts in by_law.items():
        assert sum(1 for c in counts if c) / len(counts) >= 0.6, f"{law}: too few records where a mistake changes the answer"


def test_records_become_valid_decisions(worlds):
    for w in worlds[:9]:
        to_decision(w.record()).validate()


# --- hand-checked outcomes on fixed seeds (seed 0, round-robin index) ------------------
def law_at(name: str, idx: int):
    return LAWS[name](random.Random(f"0:{name}:{idx}"), idx)


def test_identifier_chain_hand_check():
    w = law_at("identifier_chain", 0)
    # BOM-131 -> sensor module SM-3; MC-66 -> press line East -> IF-2; SM-3/IF-2 -> CP-39, whose in-force issue is on hold
    assert (w.params["skin"], w.params["q1"], w.params["q2"], w.params["cap"], w.params["frozen"]) == ("lot", 45, 25, 74, True)
    assert w.expected == "engineering_review"
    assert "CP-39 issue C (in force): the record is on hold" in w.evidence
    assert w.mistakes["normal_release"] == "used_superseded_revision"       # issue B's limit of 74 would release 70


def test_identifier_chain_band_hand_check():
    w = law_at("identifier_chain", 3)
    # 36 + 26 = 62 metres against a platform limit of 61 with a 20 % dispensation band: inside the band
    assert (w.params["skin"], w.params["q1"], w.params["q2"], w.params["cap"], w.params["tol"], w.params["frozen"]) == ("timetable", 36, 26, 61, 20, False)
    assert w.expected == "pass_fen_cross"


def test_invoice_tier_hand_check():
    w = law_at("invoice_tier", 1)
    # "Brenner Systems AG" -> V-1849, ELEVATED; payee NL is an approved country; GBP 8,500 / 0.837 = 10,155.32;
    # the same-vendor invoice is 52 days old, outside the 30-day window; ELEVATED Tier 2 runs 5,000-25,000
    assert (w.params["vendor"], w.params["risk"], w.params["override"], w.params["eur_now"], w.params["d_same"]) == ("V-1849", "ELEVATED", False, "10155.32", 52)
    assert w.expected == "tier_2_controller"


def test_oncall_page_hand_checks():
    w = law_at("oncall_page", 2)
    # ledgerrpt -> ledger-reports (tier 3): SEV3 lowered to SEV4, ticket only
    assert (w.params["service"], w.params["severity"]) == ("ledger-reports", 4)
    assert w.expected == "no_page_ticket_only"
    w = law_at("oncall_page", 5)
    # ledgerd -> ledger-core (tier 1): SEV3 raised to SEV2; 07:50 UTC is 09:50 Berlin, inside Chen's 06:00-10:00 override
    assert (w.params["service"], w.params["severity"], w.params["local_time"], w.params["override"]) == ("ledger-core", 2, "09:50", True)
    assert w.expected == "chen"
    assert w.mistakes["no_page_ticket_only"] == "severity_not_adjusted"


@pytest.mark.skipif(not os.environ.get("JEVBENCH_DIR"), reason="set JEVBENCH_DIR to a jevbench clone")
def test_no_ten_word_run_is_shared_with_the_public_tasks(worlds):
    assert contamination(worlds, os.environ["JEVBENCH_DIR"]) == []

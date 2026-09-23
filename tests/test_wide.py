# -*- coding: utf-8 -*-
"""Choices wider than the slot protocol: narrowing, and what it must preserve.

No GPU. A fake scorer stands in for the forward pass, built to favour whichever
option mentions a marker — at group level that is the group whose summary
contains it, at leaf level the option itself. So "does narrowing actually find
the right option among 2,525" is a real question here, not a stubbed one.

What is NOT tested here is whether a narrowed distribution agrees with the flat
one on the same decision. That needs the real backbone, and `wide.py` says so.
"""

from __future__ import annotations

import pytest

from jobe import Decision, DecisionError, Option
from jobe.readout import Readout
from jobe import wide


def readout(decision, probs):
    return Readout(
        decision_id=decision.id,
        option_ids=tuple(o.id for o in decision.options),
        probabilities=tuple(probs),
        option_logits=tuple(0.0 for _ in probs),
        input_tokens=7, forward_seconds=0.0, total_seconds=0.0,
        prompt_sha256="0" * 64)


class FakeScorer:
    """Puts its mass wherever the marker is. Everything else is uniform."""

    def __init__(self, marker="TARGET", hit=0.8):
        self.marker, self.hit = marker, hit
        self.calls = 0
        self.primed = None
        self.seen_widths = []

    def prime(self, evidence):
        self.primed = evidence
        return 10

    def score(self, decision):
        self.calls += 1
        n = len(decision.options)
        self.seen_widths.append(n)
        hits = [i for i, o in enumerate(decision.options) if self.marker in o.description]
        if not hits or len(hits) == n:
            return readout(decision, [1.0 / n] * n)
        rest = (1.0 - self.hit) / (n - len(hits))
        return readout(decision, [self.hit / len(hits) if i in hits else rest
                                  for i in range(n)])

    def score_batch(self, decisions, *, max_batch=2):
        return [self.score(d) for d in decisions]


def options(n, marker_at=None, marker="TARGET"):
    return tuple(
        Option(f"el-{i}", f"element {i}" + (f" {marker}" if i == marker_at else ""))
        for i in range(n))


def decide(n, marker_at=None):
    return Decision(id="pick", evidence={"page": "a page"},
                    criterion="Which element should the next action act on?",
                    options=options(n, marker_at))


# ------------------------------------------------------------------ the tree


@pytest.mark.parametrize("n", [17, 30, 114, 230, 240, 275, 2525])
def test_tree_respects_the_cap_and_keeps_every_option_in_order(n):
    """Order matters: a DOM's neighbours are related, which is why the caller
    groups consecutive elements too. Losing one option would be silent."""
    nodes = wide.build_tree(options(n), cap=16)
    assert len(nodes) <= 16
    flat = [o.id for node in nodes for o in node.options]
    assert flat == [f"el-{i}" for i in range(n)]


def test_tree_of_a_narrow_list_is_all_leaves():
    nodes = wide.build_tree(options(9), cap=16)
    assert len(nodes) == 9 and all(n.leaf is not None for n in nodes)


def test_summary_respects_its_budget():
    node = wide.build_tree(options(2525), cap=16)[0]
    assert len(wide.summarise(node, limit=220)) <= 220
    assert wide.summarise(node, limit=220).startswith("element 0")


# ------------------------------------------------- narrow decisions untouched


def test_a_decision_within_the_cap_is_the_flat_readout(monkeypatch):
    """No tree, no extra pass, no change in behaviour. This is the common case:
    the median page in the caller's bench has 4 elements."""
    d = decide(9, marker_at=3)
    monkeypatch.setattr(wide, "score",
                        lambda m, t, dec, **kw: readout(dec, [0.2] + [0.1] * 8))
    r = wide.score_wide(None, None, d, cap=16)
    assert r.flat is True and r.passes == 1 and r.depth == 1
    assert r.option_ids == tuple(o.id for o in d.options)
    assert r.probabilities[0] == pytest.approx(0.2)
    assert r.estimated == frozenset()


def test_exactly_at_the_cap_still_does_not_narrow(monkeypatch):
    monkeypatch.setattr(wide, "score", lambda m, t, dec, **kw: readout(dec, [1 / 16] * 16))
    assert wide.score_wide(None, None, decide(16), cap=16).flat is True


def test_one_over_the_cap_narrows():
    """`webform` in the caller's bench misses by exactly one option: 17."""
    r = wide.score_wide(None, None, decide(17, marker_at=5), cap=16, scorer=FakeScorer())
    assert r.flat is False


# ---------------------------------------------------- what the caller needs


@pytest.mark.parametrize("n", [17, 114, 230, 275, 2525])
def test_every_declared_option_gets_a_probability_and_they_sum_to_one(n):
    """jev-browser's resolve() sorts target.probabilities and walks down it for
    an element that fits the chosen tool, with a >= 0.1 floor. A winner alone
    would break that, and a distribution that does not sum to 1 would make the
    floor mean something different at every page size."""
    r = wide.score_wide(None, None, decide(n, marker_at=n // 3), cap=16,
                        scorer=FakeScorer())
    assert len(r.option_ids) == n
    assert r.option_ids == tuple(f"el-{i}" for i in range(n))
    assert sum(r.probabilities) == pytest.approx(1.0)
    assert all(p >= 0.0 for p in r.probabilities)


@pytest.mark.parametrize("n", [17, 30, 114, 230, 256])
@pytest.mark.parametrize("where", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_narrowing_finds_the_marked_option_at_depth_two(n, where):
    """The whole point. Up to 256 options the tree is two levels deep, every
    group summary covers all of its members, and recovery is complete.

    This is the range that matters in practice: `jev-browser` caps a choice at
    240 options (`MAX_SINGLE`), doing its own grouping above that, so a caller
    on that harness never leaves depth 2.
    """
    at = min(n - 1, int(where * (n - 1)))
    r = wide.score_wide(None, None, decide(n, at), cap=16, scorer=FakeScorer())
    assert r.depth == 2
    assert r.choice == f"el-{at}"


@pytest.mark.parametrize("n,at", [(2525, 1234), (2525, 2000)])
def test_the_widest_trees_need_a_bigger_summary_budget(n, at):
    """A known limit, pinned rather than hidden.

    Past 256 options the tree grows a third level. At 2,525 options one group
    holds 256 members, and 700 characters cannot describe them: recovery on this
    probe is 2/5, against 5/5 at 230, 256 and 275. Raising the budget restores
    it, which locates the fault - the narrowing is not what fails, the summary
    is. Head-truncation scores the same 2/5, so this is a budget problem and not
    a choice of summarisation.
    """
    assert wide.score_wide(None, None, decide(n, at), cap=16,
                           scorer=FakeScorer()).depth == 3
    generous = wide.score_wide(None, None, decide(n, at), cap=16,
                               scorer=FakeScorer(), summary_limit=20_000)
    assert generous.choice == f"el-{at}"


def test_a_group_summary_shows_its_span_not_just_its_head():
    """Cutting the joined text would describe only the first 5% of a 256-member
    group, so the level deciding whether to open it could not see most of it."""
    top = wide.build_tree(options(2525), cap=16)[7]
    text = wide.summarise(top)
    assert text.startswith("element 1792")          # first member kept
    assert "…" in text                               # and says members were skipped
    members = [o.id for o in top.options]
    assert members[0] == "el-1792" and members[-1] == "el-2047"


# ------------------------------------------------------------ cost and honesty


def test_cost_is_bounded_by_the_expansion_policy():
    """Opening every group would cost 159 passes on a 2,525-element page. The
    policy the caller already uses - top groups to 0.9 mass, at most 4 - is what
    keeps it to a handful."""
    scorer = FakeScorer()
    r = wide.score_wide(None, None, decide(2525, marker_at=900), cap=16,
                        scorer=scorer, max_expand=4)
    assert r.depth == 3                       # ceil(log16(2525))
    assert r.passes <= 1 + 4 + 16
    assert scorer.calls == r.passes


def test_unopened_groups_keep_their_mass_and_are_flagged():
    """Zeroing them would sum to less than 1 and would claim those options were
    ruled out. They were never looked at, which is a different statement."""
    r = wide.score_wide(None, None, decide(230, marker_at=5), cap=16,
                        scorer=FakeScorer(), max_expand=1)
    assert r.estimated                                   # some were never opened
    assert sum(r.probabilities) == pytest.approx(1.0)
    for oid in r.estimated:
        assert r.scores[oid] > 0.0
    assert r.meta["examined"] == 230 - len(r.estimated)


def test_confidence_is_scaled_against_the_full_option_set():
    """The caller asked a 230-way question. Rescaling against the 16 the winning
    readout happened to see would read far more confident than the decision was."""
    r = wide.score_wide(None, None, decide(230, marker_at=5), cap=16, scorer=FakeScorer())
    k = 230
    assert r.confidence() == pytest.approx((max(r.probabilities) - 1 / k) / (1 - 1 / k))


def test_a_uniform_model_stays_uniform():
    """No marker anywhere: nothing should invent a preference."""
    r = wide.score_wide(None, None, decide(64), cap=16, scorer=FakeScorer())
    assert max(r.probabilities) == pytest.approx(min(r.probabilities))


def test_cap_below_two_is_refused():
    with pytest.raises(DecisionError, match="at least 2"):
        wide.score_wide(None, None, decide(30), cap=1, scorer=FakeScorer())


def test_the_evidence_is_primed_once():
    scorer = FakeScorer()
    d = decide(230, marker_at=5)
    wide.score_wide(None, None, d, cap=16, scorer=scorer)
    assert scorer.primed == d.evidence

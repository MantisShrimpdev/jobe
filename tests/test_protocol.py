"""Protocol tests. Everything here runs without loading a model; the two
tokenizer tests use a small cached one and skip if it is unavailable.
"""

from __future__ import annotations

import json
import math

import pytest

from jobe.prompt import (
    SYSTEM_PROMPT,
    Decision,
    DecisionError,
    Option,
    build_messages,
    prompt_digest,
    render_prompt,
)
from jobe.readout import Readout, restricted_softmax
from jobe.slots import LETTERS, MAX_OPTIONS, SlotError, resolve_slots

OPTS = (
    Option("billing", "Payments, invoices and refunds"),
    Option("infra", "Outages, latency and deploys"),
)


def decision(**over) -> Decision:
    base = dict(
        id="d1",
        evidence="The checkout page returns a 500 when applying a coupon.",
        criterion="Which team should own this ticket?",
        options=OPTS,
    )
    base.update(over)
    return Decision(**base)


# ----------------------------------------------------------------- validation


def test_accepts_a_well_formed_decision():
    decision().validate()


def test_rejects_fewer_than_two_options():
    with pytest.raises(DecisionError, match="at least 2 options"):
        decision(options=(OPTS[0],)).validate()


def test_rejects_more_options_than_letters():
    many = tuple(Option(f"o{i}", f"option {i}") for i in range(MAX_OPTIONS + 1))
    with pytest.raises(DecisionError, match=f"at most {MAX_OPTIONS}"):
        decision(options=many).validate()


def test_rejects_duplicate_option_ids():
    dupes = (Option("same", "one"), Option("same", "two"))
    with pytest.raises(DecisionError, match="duplicate option id"):
        decision(options=dupes).validate()


@pytest.mark.parametrize("bad", ["", "   ", None])
def test_rejects_empty_evidence(bad):
    with pytest.raises(DecisionError, match="evidence"):
        decision(evidence=bad).validate()


def test_rejects_blank_criterion_and_blank_description():
    with pytest.raises(DecisionError, match="criterion"):
        decision(criterion="  ").validate()
    with pytest.raises(DecisionError, match="needs a description"):
        decision(options=(Option("a", " "), OPTS[1])).validate()


def test_accepts_structured_evidence():
    decision(evidence={"status": 500, "path": "/checkout"}).validate()


# --------------------------------------------------------------------- prompt


def test_letters_are_assigned_positionally():
    _, user = build_messages(decision())
    payload = json.loads(user["content"])
    assert [o["letter"] for o in payload["options"]] == ["A", "B"]


def test_option_ids_are_never_shown_to_the_model():
    # An id can encode the answer ("billing"); only descriptions may be seen.
    _, user = build_messages(decision())
    assert "billing" not in user["content"]
    assert "Payments, invoices and refunds" in user["content"]


def test_evidence_precedes_criterion_so_it_is_a_reusable_prefix():
    _, user = build_messages(decision())
    body = user["content"]
    assert body.index("evidence") < body.index("criterion")


def test_system_prompt_demands_a_bare_letter():
    system, _ = build_messages(decision())
    assert system["content"] == SYSTEM_PROMPT
    assert "only its uppercase letter" in system["content"]


def test_structured_evidence_keeps_its_shape():
    _, user = build_messages(decision(evidence={"status": 500}))
    assert json.loads(user["content"])["evidence"] == {"status": 500}


def test_build_messages_validates_first():
    with pytest.raises(DecisionError):
        build_messages(decision(options=()))


def test_reordering_preserves_ids_and_rejects_non_permutations():
    d = decision()
    flipped = d.reordered([1, 0])
    assert [o.id for o in flipped.options] == ["infra", "billing"]
    with pytest.raises(DecisionError, match="not a permutation"):
        d.reordered([0, 0])


def test_prompt_digest_is_stable_and_sensitive():
    assert prompt_digest("abc") == prompt_digest("abc")
    assert prompt_digest("abc") != prompt_digest("abd")


# -------------------------------------------------------------------- readout


def test_restricted_softmax_is_a_distribution_in_input_order():
    p = restricted_softmax([2.0, 1.0, 0.0])
    assert math.isclose(sum(p), 1.0, abs_tol=1e-12)
    assert p[0] > p[1] > p[2]


def test_restricted_softmax_is_stable_at_large_magnitudes():
    p = restricted_softmax([1000.0, 999.0])
    assert all(math.isfinite(v) for v in p)
    assert math.isclose(p[0], math.e / (math.e + 1), abs_tol=1e-12)


def test_restricted_softmax_rejects_degenerate_input():
    with pytest.raises(ValueError):
        restricted_softmax([1.0])
    with pytest.raises(ValueError):
        restricted_softmax([1.0, float("nan")])
    with pytest.raises(ValueError):
        restricted_softmax([1.0, float("inf")])


def _readout(probs, ids=("a", "b")) -> Readout:
    return Readout(
        decision_id="d",
        option_ids=ids,
        probabilities=probs,
        option_logits=tuple(0.0 for _ in probs),
        input_tokens=1,
        forward_seconds=0.0,
        total_seconds=0.0,
        prompt_sha256="x",
    )


def test_choice_and_scores_track_the_distribution():
    r = _readout((0.25, 0.75))
    assert r.choice == "b"
    assert r.scores == {"a": 0.25, "b": 0.75}


def test_confidence_is_chance_corrected():
    # Two options: p_max 0.5 is pure chance -> 0.0; certainty -> 1.0.
    assert math.isclose(_readout((0.5, 0.5)).confidence(), 0.0, abs_tol=1e-12)
    assert math.isclose(_readout((1.0, 0.0)).confidence(), 1.0, abs_tol=1e-12)
    # Four options: chance is 0.25, so 0.25 must also map to 0.0.
    four = _readout((0.25, 0.25, 0.25, 0.25), ids=("a", "b", "c", "d"))
    assert math.isclose(four.confidence(), 0.0, abs_tol=1e-12)


def test_expected_value_weights_ordinal_levels():
    r = _readout((0.5, 0.0, 0.5), ids=("0", "1", "2"))
    assert math.isclose(r.expected_value(), 1.0, abs_tol=1e-12)


def test_probability_status_says_uncalibrated():
    assert "uncalibrated" in _readout((0.5, 0.5)).probability_status


# ------------------------------------------------------------------ tokenizer

TOKENIZER = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    except Exception as exc:  # noqa: BLE001 - not cached, nothing to test against
        pytest.skip(f"{TOKENIZER} unavailable: {type(exc).__name__}")


def test_slots_resolve_in_context_not_standalone(tokenizer):
    """SentencePiece encodes a bare letter with a word-boundary marker, so the
    standalone id differs from the one that actually follows a prompt. Resolving
    in context is what makes this tokenizer family usable at all."""
    d = decision()
    prompt = render_prompt(tokenizer, d)
    ids = tokenizer.encode(prompt, add_special_tokens=False)

    slots = resolve_slots(tokenizer, prompt, ids, 2)
    assert [tokenizer.decode([s]) for s in slots] == ["A", "B"]
    assert len(set(slots)) == 2

    standalone = tokenizer.encode("A", add_special_tokens=False)
    assert standalone != [slots[0]], (
        "expected this tokenizer to differ standalone vs in context; "
        "if it stops differing the regression guard is worthless"
    )


def test_resolve_slots_rejects_an_impossible_option_count(tokenizer):
    d = decision()
    prompt = render_prompt(tokenizer, d)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    with pytest.raises(SlotError, match="need 2"):
        resolve_slots(tokenizer, prompt, ids, 1)
    with pytest.raises(SlotError, match="need 2"):
        resolve_slots(tokenizer, prompt, ids, MAX_OPTIONS + 1)


def test_resolve_slots_detects_a_moved_boundary(tokenizer):
    """If the caller passes ids that are not this prompt's, the guard must fire
    rather than silently reading the wrong position."""
    d = decision()
    prompt = render_prompt(tokenizer, d)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    with pytest.raises(SlotError, match="re-tokenizes|costs"):
        resolve_slots(tokenizer, prompt, ids[:-1], 2)


def test_every_letter_is_one_token_in_context(tokenizer):
    d = decision(options=tuple(Option(f"o{i}", f"option number {i}") for i in range(MAX_OPTIONS)))
    prompt = render_prompt(tokenizer, d)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    slots = resolve_slots(tokenizer, prompt, ids, MAX_OPTIONS)
    assert len(slots) == MAX_OPTIONS
    assert [tokenizer.decode([s]) for s in slots] == list(LETTERS)

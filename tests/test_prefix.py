"""Prefix-cache reuse.

The tokenizer tests use a small cached model and skip if it is unavailable. The
guard tests use a stub model, so they need no weights at all. The equality
test against a real backbone is opt-in via JOBE_GPU_MODEL, because it loads
several gigabytes and needs a GPU to be meaningful.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from jobe.prefix import (
    PrefixError,
    PrefixScorer,
    evidence_key,
    prefix_ids_for,
    score_with_prefix,
)
from jobe.prompt import Decision, Option, render_prompt

EVIDENCE = (
    "Refunds require an original receipt and a purchase within 30 days. Items marked "
    "final sale are excluded. A manager may approve an exception with a bank statement."
)
YESNO = (Option("yes", "yes: every required condition is established"),
         Option("no", "no: a condition is missing or a prohibition applies"))


def decision(criterion="Is a refund permitted?", options=YESNO, evidence=EVIDENCE, id="d"):
    return Decision(id=id, evidence=evidence, criterion=criterion, options=options)


# ---------------------------------------------------------------- tokenizer

TOKENIZER = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{TOKENIZER} unavailable: {type(exc).__name__}")


def test_prefix_is_a_token_prefix_of_every_full_prompt(tokenizer):
    """The whole point. Varied criteria and option counts, same evidence."""
    prefix = prefix_ids_for(tokenizer, EVIDENCE)
    assert len(prefix) > 10
    many = tuple(Option(f"o{i}", f"option number {i}") for i in range(16))
    cases = [
        decision(),
        decision(criterion="Which team should own this ticket?",
                 options=(Option("billing", "billing"), Option("infra", "infra"), Option("sales", "sales"))),
        decision(criterion="Rate the severity from the listed levels.", options=many),
        decision(criterion="x"),
    ]
    for d in cases:
        ids = tokenizer.encode(render_prompt(tokenizer, d), add_special_tokens=False)
        assert ids[: len(prefix)] == prefix, d.criterion


def test_prefix_contains_the_evidence_and_never_the_placeholder(tokenizer):
    text = tokenizer.decode(prefix_ids_for(tokenizer, EVIDENCE))
    assert "original receipt" in text
    assert "placeholder" not in text
    assert "prefix boundary" not in text


def test_prefix_depends_on_the_evidence(tokenizer):
    assert prefix_ids_for(tokenizer, EVIDENCE) != prefix_ids_for(tokenizer, EVIDENCE + " More.")


def test_structured_evidence_is_supported(tokenizer):
    ev = {"policy": "receipt required", "days": 30}
    prefix = prefix_ids_for(tokenizer, ev)
    ids = tokenizer.encode(render_prompt(tokenizer, decision(evidence=ev)), add_special_tokens=False)
    assert ids[: len(prefix)] == prefix


def test_evidence_key_is_content_based():
    assert evidence_key({"a": 1, "b": 2}) == evidence_key({"b": 2, "a": 1})
    assert evidence_key("x") != evidence_key("y")


# ------------------------------------------------------------------- guards


class _StubModel:
    """Enough of a causal LM for the guards to run: no weights, no GPU."""

    def __init__(self):
        import torch
        self._p = torch.nn.Parameter(torch.zeros(1))
        self.calls = 0

    def parameters(self):
        yield self._p

    def forward(self, input_ids, attention_mask, past_key_values=None, use_cache=False,
                return_dict=True, position_ids=None, logits_to_keep=0):
        pass

    def __call__(self, **kw):
        import torch
        self.calls += 1
        return SimpleNamespace(past_key_values=object(), logits=torch.zeros(1, 1, 64))


def test_score_before_prime_is_refused(tokenizer):
    scorer = PrefixScorer(_StubModel(), tokenizer)
    assert scorer.is_primed is False
    with pytest.raises(PrefixError, match="call prime"):
        scorer.score(decision())


def test_mismatched_evidence_is_refused_not_silently_scored(tokenizer):
    model = _StubModel()
    scorer = PrefixScorer(model, tokenizer)
    scorer.prime(EVIDENCE)
    assert scorer.is_primed and scorer.prefix_tokens > 0
    with pytest.raises(PrefixError, match="differs from the primed evidence"):
        scorer.score(decision(evidence="a completely different document"))
    # Only the prime ran a forward pass; the refused decision never reached the model.
    assert model.calls == 1


def test_priming_the_same_evidence_twice_is_a_no_op(tokenizer):
    model = _StubModel()
    scorer = PrefixScorer(model, tokenizer)
    n1 = scorer.prime(EVIDENCE)
    n2 = scorer.prime(EVIDENCE)
    assert n1 == n2 and model.calls == 1
    scorer.prime(EVIDENCE + " changed")
    assert model.calls == 2


# -------------------------------------------------------------- real model

GPU_MODEL = os.environ.get("JOBE_GPU_MODEL")


@pytest.mark.skipif(not GPU_MODEL, reason="set JOBE_GPU_MODEL=<path-or-id> to run on a real backbone")
def test_cached_suffix_matches_uncached_full_prompt():
    """Slot logits from prefix+suffix must equal the uncached forward, and the
    argmax must be identical. bf16 rounding paths differ, so allow two ulps."""
    torch = pytest.importorskip("torch")
    from jobe import load, score

    bb = load(GPU_MODEL, device="auto")
    state = ("Refunds require an original receipt and a purchase made within 30 days. "
             "Items marked final sale are excluded. A manager may approve an exception "
             "when the customer provides a bank statement. ") * 30
    questions = [
        decision(evidence=state, id="a"),
        decision(criterion="Are final-sale items refundable?", options=(Option("yes", "yes"), Option("no", "no")),
                 evidence=state, id="b"),
        decision(criterion="Which team owns a duplicate-charge ticket?",
                 options=(Option("billing", "billing"), Option("infra", "infra"), Option("sales", "sales")),
                 evidence=state, id="c"),
    ]
    cached = score_with_prefix(bb.model, bb.tokenizer, state, questions)
    for d, c in zip(questions, cached):
        u = score(bb.model, bb.tokenizer, d)
        assert c.choice == u.choice, d.id
        assert c.prompt_sha256 == u.prompt_sha256, "same prompt must hash the same"
        assert c.input_tokens == u.input_tokens
        assert max(abs(a - b) for a, b in zip(c.option_logits, u.option_logits)) <= 0.25, d.id
        assert c.meta["prefix_tokens"] > 0 and c.meta["suffix_tokens"] < 200

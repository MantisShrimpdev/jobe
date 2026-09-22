"""Batched suffix scoring off one prefix cache.

Layout tests are pure. Guard tests use a stub model. The equality test against
a real backbone is opt-in via JOBE_GPU_MODEL.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from jobe.prefix import PrefixError, PrefixScorer, suffix_layout
from jobe.prompt import Decision, Option

EVIDENCE = ("Refunds require an original receipt and a purchase within 30 days. Items marked "
            "final sale are excluded. A manager may approve an exception with a bank statement.")
YESNO = (Option("yes", "yes: every required condition is established"),
         Option("no", "no: a condition is missing or a prohibition applies"))


def decision(criterion="Is a refund permitted?", options=YESNO, evidence=EVIDENCE, id="d"):
    return Decision(id=id, evidence=evidence, criterion=criterion, options=options)


# ------------------------------------------------------------------- layout


def test_layout_right_pads_and_marks_each_row_end():
    lay = suffix_layout([[7, 8, 9], [5], [1, 2]], prefix_len=100, pad_id=0)
    assert lay.width == 3
    assert lay.input_ids == [[7, 8, 9], [5, 0, 0], [1, 2, 0]]
    # mask spans the cached prefix plus the real suffix, zero over pads
    assert lay.attention_mask == [[1] * 103, [1] * 101 + [0, 0], [1] * 102 + [0]]
    # positions continue from the prefix; pads get 0 (masked, so irrelevant)
    assert lay.position_ids == [[100, 101, 102], [100, 0, 0], [100, 101, 0]]
    # the readout lives at each row's last REAL token
    assert lay.ends == [2, 0, 1]


def test_layout_refuses_empty_suffixes():
    with pytest.raises(PrefixError, match="non-empty suffix"):
        suffix_layout([[1, 2], []], prefix_len=10, pad_id=0)
    with pytest.raises(PrefixError):
        suffix_layout([], prefix_len=10, pad_id=0)


def test_layout_is_a_no_op_on_equal_lengths():
    lay = suffix_layout([[1, 2], [3, 4]], prefix_len=5, pad_id=9)
    assert lay.input_ids == [[1, 2], [3, 4]] and lay.ends == [1, 1]
    assert all(0 not in m for m in lay.attention_mask)


# ------------------------------------------------------------------- guards

TOKENIZER = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{TOKENIZER} unavailable: {type(exc).__name__}")


class _StubModel:
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


def test_batch_refuses_before_any_forward_when_one_decision_is_foreign(tokenizer):
    model = _StubModel()
    scorer = PrefixScorer(model, tokenizer)
    scorer.prime(EVIDENCE)
    good = decision(id="ok")
    bad = decision(evidence="a different document", id="bad")
    with pytest.raises(PrefixError, match="bad: evidence differs"):
        scorer.score_batch([good, bad])
    assert model.calls == 1, "only the prime ran; the batch was refused before the model was touched"


def test_batch_requires_prime_and_a_sane_batch_size(tokenizer):
    scorer = PrefixScorer(_StubModel(), tokenizer)
    with pytest.raises(PrefixError, match="call prime"):
        scorer.score_batch([decision()])
    scorer.prime(EVIDENCE)
    with pytest.raises(ValueError, match="max_batch"):
        scorer.score_batch([decision()], max_batch=0)
    assert scorer.score_batch([]) == []


# --------------------------------------------------------------- real model

GPU_MODEL = os.environ.get("JOBE_GPU_MODEL")


@pytest.mark.skipif(not GPU_MODEL, reason="set JOBE_GPU_MODEL=<path-or-id> to run on a real backbone")
def test_batched_rows_match_uncached_and_serial():
    """Every batched row must reproduce the uncached forward's slot logits to
    rounding noise and the same argmax. The tolerance (0.5) is set between what
    right-padding measured (<= 0.25) and what left-padding measured (>= 0.625),
    so a regression to the wrong padding side fails this test."""
    from jobe import load, score

    bb = load(GPU_MODEL, device="auto")
    state = ("Refunds require an original receipt and a purchase made within 30 days. Items "
             "marked final sale are excluded. A manager may approve an exception when the "
             "customer provides a bank statement. ") * 30
    qs = [
        decision(evidence=state, id="a"),
        decision(criterion="Are final-sale items refundable?",
                 options=(Option("yes", "yes"), Option("no", "no")), evidence=state, id="b"),
        decision(criterion="Which team owns a duplicate-charge ticket?",
                 options=(Option("billing", "billing"), Option("infra", "infra"), Option("sales", "sales")),
                 evidence=state, id="c"),
        decision(criterion="Can a manager approve on a bank statement alone?", evidence=state, id="d"),
        decision(criterion="Is a receipt required for an exchange?", evidence=state, id="e"),
    ]
    scorer = PrefixScorer(bb.model, bb.tokenizer)
    scorer.prime(state)
    batched = scorer.score_batch(qs, max_batch=3)  # forces two chunks, ragged widths
    serial = scorer.score_many(qs)
    assert [r.decision_id for r in batched] == [d.id for d in qs]
    for d, b, s in zip(qs, batched, serial):
        u = score(bb.model, bb.tokenizer, d)
        assert b.choice == u.choice == s.choice, d.id
        assert b.prompt_sha256 == u.prompt_sha256
        assert max(abs(x - y) for x, y in zip(b.option_logits, u.option_logits)) <= 0.5, d.id
        assert b.meta["padding"] == "right" and b.meta["batch"] in (2, 3)

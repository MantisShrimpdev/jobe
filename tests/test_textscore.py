"""The text readout: option ids scored as continuations, with no option ceiling.

Guards and the prompt contract are tested with a cached tokenizer. The load-
bearing test — that scoring options as batched rows off one shared cache equals
scoring each option on its own — is opt-in against a real backbone via
JOBE_GPU_MODEL, because that equality is the whole claim and a stub cannot
establish it.
"""

from __future__ import annotations

import math
import os

import pytest

from jobe.prompt import (
    TEXT_PROMPT_VERSION,
    Decision,
    DecisionError,
    Option,
    build_messages,
    build_text_messages,
    render_text_prompt,
)
from jobe.textscore import TextScoreError, option_continuations, score_text

TOKENIZER = "D:/Coding/models/qwen35-4b"
EVIDENCE = (
    "Refunds require an original receipt and a purchase within 30 days. Items marked "
    "final sale are excluded. A manager may approve an exception with a bank statement."
)


def wide(n: int) -> Decision:
    """A decision with `n` options — beyond the letter protocol's sixteen."""
    return Decision(
        id="wide",
        evidence=EVIDENCE,
        criterion="Which queue should this go to?",
        options=tuple(Option(f"queue_{i:03d}", f"the {i}th queue") for i in range(n)),
    )


# ------------------------------------------------------------------ the prompt


def test_the_letter_path_keeps_its_ceiling_and_says_where_to_go():
    with pytest.raises(DecisionError, match="at most 16 options"):
        build_messages(wide(40))
    with pytest.raises(DecisionError, match="score_text"):
        build_messages(wide(40))


def test_the_text_path_has_no_ceiling_and_shows_ids():
    messages = build_text_messages(wide(120))
    assert len(messages) == 2
    assert "queue_119" in messages[1]["content"]
    assert "letter" not in messages[1]["content"]
    assert "id" in messages[0]["content"]


def test_the_text_path_still_refuses_a_malformed_record():
    with pytest.raises(DecisionError, match="at least 2 options"):
        build_text_messages(
            Decision(id="d", evidence="e", criterion="c", options=(Option("only", "one"),))
        )
    with pytest.raises(DecisionError, match="duplicate option id"):
        build_text_messages(
            Decision(
                id="d",
                evidence="e",
                criterion="c",
                options=(Option("a", "one"), Option("a", "two")),
            )
        )


# ------------------------------------------------------------------ tokenizer


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    except Exception as exc:  # pragma: no cover - depends on the local cache
        pytest.skip(f"tokenizer unavailable: {exc}")


def test_continuations_are_tokenized_in_context(tokenizer):
    decision = wide(3)
    prompt = render_text_prompt(tokenizer, decision)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    conts = option_continuations(tokenizer, prompt, prompt_ids, decision)

    assert len(conts) == 3
    for option, cont in zip(decision.options, conts):
        assert len(cont) >= 1
        # the continuation decodes back to the id, and re-encoding the pair is
        # exactly the prompt followed by it
        assert tokenizer.decode(cont) == option.id
        assert tokenizer.encode(prompt + option.id, add_special_tokens=False) == (
            prompt_ids + cont
        )


def test_continuations_refuse_an_id_that_adds_nothing(tokenizer):
    decision = Decision(
        id="d",
        evidence=EVIDENCE,
        criterion="c",
        options=(Option("", "empty"), Option("b", "bee")),
    )
    prompt = render_text_prompt(tokenizer, Decision(**{**decision.__dict__, "options": (
        Option("a", "aye"), Option("b", "bee"))}))
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    with pytest.raises(TextScoreError, match="adds no tokens"):
        option_continuations(tokenizer, prompt, prompt_ids, decision)


def test_continuations_refuse_a_prompt_that_re_tokenizes(tokenizer):
    """A prompt whose tail merges with the id would score a different sequence."""
    decision = wide(2)
    prompt = "the quick brown fox jump"  # an id starting with 's' merges into "jumps"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    merging = Decision(
        id="d",
        evidence=EVIDENCE,
        criterion="c",
        options=(Option("s", "merges into the tail"), decision.options[1]),
    )
    merged = tokenizer.encode(prompt + "s", add_special_tokens=False)
    if merged[: len(prompt_ids)] == prompt_ids:
        pytest.skip("this tokenizer does not merge across the chosen boundary")
    with pytest.raises(TextScoreError, match="re-tokenizes"):
        option_continuations(tokenizer, prompt, prompt_ids, merging)


def test_a_long_prompt_is_refused_rather_than_truncated(tokenizer):
    decision = Decision(
        id="long",
        evidence="word " * 5000,
        criterion="c",
        options=(Option("a", "aye"), Option("b", "bee")),
    )
    with pytest.raises(ValueError, match="no truncation"):
        score_text(_NoModel(), tokenizer, decision, max_tokens=256)


class _NoModel:
    """Asserts the prompt guards run before anything touches a model."""

    def parameters(self):  # pragma: no cover - only reached on a guard failure
        raise AssertionError("the model must not be touched when a guard fails")


# ------------------------------------------------- against a real backbone


GPU_MODEL = os.environ.get("JOBE_GPU_MODEL")
gpu_only = pytest.mark.skipif(not GPU_MODEL, reason="set JOBE_GPU_MODEL to run")


@pytest.fixture(scope="module")
def backbone():
    from jobe.model import load

    return load(GPU_MODEL, device="auto", attn_implementation="sdpa")


def naive_scores(model, tokenizer, decision, length_norm=True):
    """Score every option in its own forward pass, with no cache and no batching."""
    import torch

    prompt = render_text_prompt(tokenizer, decision)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    conts = option_continuations(tokenizer, prompt, prompt_ids, decision)
    device = next(model.parameters()).device
    out = []
    for cont in conts:
        ids = prompt_ids + cont
        with torch.inference_mode():
            logits = model(
                input_ids=torch.tensor([ids], dtype=torch.long, device=device),
                attention_mask=torch.ones((1, len(ids)), dtype=torch.long, device=device),
                use_cache=False,
                return_dict=True,
            ).logits[0].float()
        total = 0.0
        for k, token in enumerate(cont):
            at = len(prompt_ids) - 1 + k  # the position whose logits predict `token`
            row = logits[at]
            total += (row[token] - row.logsumexp(-1)).item()
        out.append(total / len(cont) if length_norm else total)
    return out


@gpu_only
def test_batched_continuations_equal_scoring_each_option_alone(backbone):
    """The claim this module rests on: the cache and the batching change nothing."""
    decision = Decision(
        id="equal",
        evidence=EVIDENCE,
        criterion="Is a refund permitted, and by whom?",
        options=(
            Option("yes", "a refund is permitted"),
            Option("no", "a refund is not permitted"),
            Option("manager_exception", "only a manager may approve it"),
            Option("needs_receipt", "the receipt is missing"),
            Option("final_sale_excluded", "the item is final sale"),
        ),
    )
    got = score_text(backbone.model, backbone.tokenizer, decision, max_batch=2)
    want = naive_scores(backbone.model, backbone.tokenizer, decision)
    for option, a, b in zip(decision.options, got.option_logits, want):
        assert a == pytest.approx(b, abs=2e-3), f"{option.id}: batched {a} vs naive {b}"
    assert got.choice == decision.options[
        max(range(len(want)), key=lambda i: want[i])
    ].id


@gpu_only
def test_a_wide_menu_is_answered_where_the_letter_readout_refuses(backbone):
    from jobe.readout import score

    decision = wide(40)
    with pytest.raises(DecisionError):
        score(backbone.model, backbone.tokenizer, decision)

    readout = score_text(backbone.model, backbone.tokenizer, decision, max_batch=4)
    assert len(readout.probabilities) == 40
    assert readout.choice in {o.id for o in decision.options}
    assert math.isclose(sum(readout.probabilities), 1.0, abs_tol=1e-6)
    assert readout.prompt_version == TEXT_PROMPT_VERSION
    assert readout.meta["options"] == 40


@gpu_only
def test_length_normalisation_changes_the_score_but_not_the_mechanism(backbone):
    decision = Decision(
        id="lengths",
        evidence=EVIDENCE,
        criterion="Which applies?",
        options=(
            Option("no", "short id"),
            Option("escalate_to_a_manager_for_approval", "a much longer id"),
        ),
    )
    normed = score_text(backbone.model, backbone.tokenizer, decision, length_norm=True)
    summed = score_text(backbone.model, backbone.tokenizer, decision, length_norm=False)
    # the summed form penalises the long id purely for being long
    long_index = 1
    assert summed.option_logits[long_index] < normed.option_logits[long_index]
    assert normed.meta["length_norm"] == "mean per token"
    assert summed.meta["length_norm"] == "summed"

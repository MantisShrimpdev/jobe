"""The text readout: score each option's own tokens, so the menu has no ceiling.

`readout.score` presents every option as one answer letter and reads that
letter's logit. That is one forward pass and a clean measurement, and it stops
at sixteen options because there are sixteen letters. This module answers the
other case: score the log-probability of each option's *id* as a continuation
of the prompt, and take the softmax over those scores. Nothing is generated,
the answer is still structurally one of the declared ids, and the number of
options is bounded only by what fits the context.

WHY NOT THE BLOCK-DIAGONAL TRICK. The usual way to do this in one pass is to
concatenate every option after the prompt under a 4D mask, so each option
attends to the prompt and to itself but not to its neighbours. That is correct
on a standard transformer and wrong on this one: Qwen3.5 is hybrid, with
linear-attention layers between the full-attention ones (`layer_types` in its
config), and a recurrent layer has no attention mask to restrict - option two
would see option one through the recurrent state with nothing to prevent it.
So options are scored as separate rows off one shared KV cache instead, which
is exactly `prefix.score_batch`'s mechanism: right-padded rows, the cache
expanded with `reorder_cache`, one forward per chunk. Each row is an
independent continuation, so there is no cross-contamination to mask away.

COST. The prompt is encoded once. An option that is a single token in context
costs nothing further - so a menu of single-token ids is one forward pass, the
same as the letter readout, with no cap. Longer ids cost one batched pass over
the option tokens, chunked by `max_batch`.

LENGTH NORMALISATION IS A CHOICE, NOT A MEASUREMENT. A summed log-probability
systematically favours short ids: `billing` (one token) would outscore
`escalate_to_engineering` (five) before the evidence is even considered. This
module therefore defaults to the MEAN log-probability per token, which removes
that bias, and `length_norm=False` gives the summed form that the Decision
Index's reference engine uses. Neither is measured to be better here yet; the
default is the one whose failure mode is not systematic.

WHAT DOES NOT CHANGE. `score` and its letter slots are untouched and remain the
default path for two to sixteen options, and that is now measured rather than
assumed: across the 231 public JevBench tasks the letter readout scores 0.801
against this module's 0.792, wins the standard tier by 4.2 points, runs about a
quarter faster, and is the more decisive of the two (mean top probability 0.838
against 0.768). Prefer letters wherever they apply. Prefer a hierarchy over a
wide menu where a taxonomy exists at all - menu size is this family's measured
weakness, and answering a 128-way question is not the same as answering it
well. Reach for this module when the menu genuinely will not fit in sixteen.
"""

from __future__ import annotations

import time

from .prompt import (
    TEXT_PROMPT_VERSION,
    Decision,
    prompt_digest,
    render_text_prompt,
)
from .readout import Readout, restricted_softmax


class TextScoreError(ValueError):
    """Raised when a decision cannot be scored by its option text."""


def option_continuations(
    tokenizer, prompt: str, prompt_ids: list[int], decision: Decision
) -> list[list[int]]:
    """Tokenize every option id as a continuation of the prompt, in context.

    In context for the same reason `slots.resolve_slots` is: tokenizers merge
    greedily across a boundary, so an id tokenized on its own is not necessarily
    the sequence that would follow this prompt. Each id is encoded as
    ``prompt + id`` and the prompt's own ids must survive unchanged, or the
    continuation being scored is not the one the model would produce.

    Raises:
        TextScoreError: an id that contributes no tokens, or a prompt tail that
            re-tokenizes when the id is appended.
    """
    out: list[list[int]] = []
    for option in decision.options:
        merged = tokenizer.encode(prompt + option.id, add_special_tokens=False)
        # The boundary is checked FIRST: a merge is the likelier failure and the
        # more specific diagnosis. A merge can also shorten the sequence, which
        # would otherwise be reported as the blander "adds no tokens".
        if merged[: len(prompt_ids)] != prompt_ids:
            raise TextScoreError(
                f"appending option id {option.id!r} re-tokenizes the prompt boundary; "
                "its continuation would not be the one the model scores"
            )
        if len(merged) == len(prompt_ids):
            raise TextScoreError(f"option id {option.id!r} adds no tokens to the prompt")
        out.append(merged[len(prompt_ids) :])
    return out


def score_text(
    model,
    tokenizer,
    decision: Decision,
    *,
    max_tokens: int = 4096,
    length_norm: bool = True,
    max_batch: int = 4,
) -> Readout:
    """Score a decision by the likelihood of each option's id. No option ceiling.

    Args:
        model: a loaded causal LM.
        tokenizer: its tokenizer.
        decision: the decision to score; any number of options.
        max_tokens: refuse a prompt longer than this rather than truncating.
        length_norm: score each option by its MEAN token log-probability rather
            than the sum, so a long id is not penalised for its length.
        max_batch: how many multi-token options to score per forward pass. The
            logits of a batch are ``(batch, width, vocab)``, and this vocabulary
            is 248k, so raising it costs memory quickly.

    Returns:
        A :class:`Readout` whose ``option_logits`` are the per-option scores and
        whose ``probabilities`` are the softmax over them.

    Raises:
        TextScoreError: an option id cannot be scored as a continuation.
        DecisionError: the decision record is malformed.
        ValueError: the prompt exceeds `max_tokens`.
    """
    import torch

    started = time.perf_counter()
    decision.validate(max_options=None)

    prompt = render_text_prompt(tokenizer, decision)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids:
        raise ValueError(f"{decision.id}: empty prompt")
    if len(prompt_ids) > max_tokens:
        raise ValueError(
            f"{decision.id}: {len(prompt_ids)} input tokens exceed the limit "
            f"{max_tokens}; no truncation is performed"
        )
    continuations = option_continuations(tokenizer, prompt, prompt_ids, decision)

    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise TextScoreError("tokenizer has neither a pad nor an eos token to pad with")

    forward_started = time.perf_counter()
    # One pass over the prompt: its last position scores every option's FIRST
    # token, and its cache carries the rest.
    with torch.inference_mode():
        out = model(
            input_ids=torch.tensor([prompt_ids], dtype=torch.long, device=device),
            attention_mask=torch.ones((1, len(prompt_ids)), dtype=torch.long, device=device),
            use_cache=True,
            return_dict=True,
        )
    first_logits = out.logits[0, -1].float()
    first_logprobs = first_logits - first_logits.logsumexp(-1)
    totals = [first_logprobs[cont[0]].item() for cont in continuations]

    # Options of two or more tokens need their remaining tokens scored. Feeding
    # t0..t(n-2) yields the predictions for t1..t(n-1).
    rest = [i for i, cont in enumerate(continuations) if len(cont) > 1]
    if rest:
        prefix_cache = out.past_key_values
        from .prefix import suffix_layout

        for start in range(0, len(rest), max_batch):
            chunk = rest[start : start + max_batch]
            suffixes = [continuations[i][:-1] for i in chunk]
            layout = suffix_layout(suffixes, len(prompt_ids), pad_id)
            n = len(chunk)
            import copy as _copy

            cache = _copy.deepcopy(prefix_cache)
            cache.reorder_cache(torch.zeros(n, dtype=torch.long, device=device))
            with torch.inference_mode():
                logits = model(
                    input_ids=torch.tensor(layout.input_ids, dtype=torch.long, device=device),
                    attention_mask=torch.tensor(
                        layout.attention_mask, dtype=torch.long, device=device
                    ),
                    position_ids=torch.tensor(
                        layout.position_ids, dtype=torch.long, device=device
                    ),
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                    logits_to_keep=layout.width,
                ).logits.float()
            # logits[r, j] predicts the token after suffix position j, i.e. t(j+1).
            targets = torch.tensor(
                [
                    continuations[i][1:] + [pad_id] * (layout.width - (len(continuations[i]) - 1))
                    for i in chunk
                ],
                dtype=torch.long,
                device=device,
            )
            chosen = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            logprobs = chosen - logits.logsumexp(-1)
            for row, i in enumerate(chunk):
                take = len(continuations[i]) - 1
                totals[i] += logprobs[row, :take].sum().item()
            del cache
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - forward_started

    scores = [
        total / len(cont) if length_norm else total
        for total, cont in zip(totals, continuations)
    ]
    return Readout(
        decision_id=decision.id,
        option_ids=tuple(option.id for option in decision.options),
        probabilities=tuple(restricted_softmax(scores)),
        option_logits=tuple(scores),
        input_tokens=len(prompt_ids),
        forward_seconds=forward_seconds,
        total_seconds=time.perf_counter() - started,
        prompt_sha256=prompt_digest(prompt),
        prompt_version=TEXT_PROMPT_VERSION,
        meta={
            "readout": (
                "option-text likelihood: log-probability of each option id as a "
                "continuation, scored as separate rows off one shared prefix cache"
            ),
            "device": str(device),
            "options": len(decision.options),
            "length_norm": "mean per token" if length_norm else "summed",
            "multi_token_options": len(rest),
            "prompt_tokens": len(prompt_ids),
        },
    )

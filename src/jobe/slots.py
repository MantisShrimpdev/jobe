"""Answer slots: the letters an option can be answered with, and the proof that
reading their logits actually measures what we think it measures.

Protocol adapted from TheoLeeCJ/SemIf (MIT), `src/semif_phase1/direct.py`.

Two guards live here, and both are load-bearing. A decision is read from the
next-token distribution at one position, so:

1. Every answer letter must be **exactly one token** that round-trips through
   the tokenizer. If "A" encodes to two tokens, or decodes back to something
   else, its "logit" is not the probability of answering A.

2. Appending the letter to the prompt must not **re-tokenize the prompt's
   tail**. Tokenizers merge greedily across boundaries; if `encode(prompt + "A")`
   is not `encode(prompt) + [id("A")]`, then the position whose logits we read
   is not the position the answer would occupy, and the numbers are confident
   nonsense.

Neither guard is expensive, and skipping them produces a working-looking system
that is silently wrong. They fail loudly instead.
"""

from __future__ import annotations

LETTERS = "ABCDEFGHIJKLMNOP"
"""Answer slots in order; option i is presented as LETTERS[i]."""

MAX_OPTIONS = len(LETTERS)
"""Hard cap on options per decision - one single-token letter each."""


class SlotError(ValueError):
    """Raised when a tokenizer cannot support the answer-slot contract."""


def slot_token_ids(tokenizer, count: int) -> list[int]:
    """Return the single token id for each of the first `count` answer letters.

    Args:
        tokenizer: a HuggingFace tokenizer.
        count: how many options this decision declares, 2..MAX_OPTIONS.

    Raises:
        SlotError: if a letter is not one exact round-trip token, or two
            letters collide on the same id.
    """
    if not 2 <= count <= MAX_OPTIONS:
        raise SlotError(f"need 2..{MAX_OPTIONS} options, got {count}")
    ids: list[int] = []
    for letter in LETTERS[:count]:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1:
            raise SlotError(f"answer slot {letter!r} encodes to {len(encoded)} tokens, not 1")
        if tokenizer.decode(encoded) != letter:
            raise SlotError(f"answer slot {letter!r} does not round-trip through the tokenizer")
        ids.append(encoded[0])
    if len(set(ids)) != len(ids):
        raise SlotError("answer-slot tokens collide - two letters share one id")
    return ids


def verify_boundary(tokenizer, prompt: str, prompt_ids: list[int], slot_ids: list[int]) -> None:
    """Prove that appending an answer letter does not re-tokenize the prompt tail.

    Args:
        tokenizer: a HuggingFace tokenizer.
        prompt: the rendered prompt text, ending where the answer begins.
        prompt_ids: `tokenizer.encode(prompt, add_special_tokens=False)`.
        slot_ids: aligned to LETTERS.

    Raises:
        SlotError: if any letter changes the tokenization of the prompt.
    """
    for letter, slot in zip(LETTERS, slot_ids):
        merged = tokenizer.encode(prompt + letter, add_special_tokens=False)
        if merged != [*prompt_ids, slot]:
            raise SlotError(
                f"appending answer slot {letter!r} re-tokenizes the prompt boundary; "
                "the last-position logits would not correspond to the answer"
            )


def resolve_slots(tokenizer, prompt: str, prompt_ids: list[int], count: int) -> list[int]:
    """Derive answer-slot ids **in context**, which is the only correct way.

    Encoding a letter on its own is not reliable. SentencePiece tokenizers
    (Llama-2, Mistral, TinyLlama) prepend a word-boundary marker to standalone
    text, so ``encode("A")`` yields ``▁A`` while the token that actually follows
    a prompt is a bare ``A`` — a different id. Reading the logit of ``▁A`` would
    measure the wrong thing, confidently.

    So each slot is taken from ``encode(prompt + letter)[-1]``, and the guard
    becomes the part that genuinely matters: everything *before* that last token
    must still be exactly ``prompt_ids``. If the prompt's tail re-tokenizes, the
    final position is not the answer position and no id can fix it.

    Args:
        tokenizer: a HuggingFace tokenizer.
        prompt: the rendered prompt, ending where the answer begins.
        prompt_ids: ``tokenizer.encode(prompt, add_special_tokens=False)``.
        count: how many options this decision declares.

    Returns:
        One token id per answer letter, aligned to ``LETTERS``.

    Raises:
        SlotError: a letter costs more than one token in context, the prompt
            tail re-tokenizes, or two letters resolve to the same id.
    """
    if not 2 <= count <= MAX_OPTIONS:
        raise SlotError(f"need 2..{MAX_OPTIONS} options, got {count}")
    ids: list[int] = []
    for letter in LETTERS[:count]:
        merged = tokenizer.encode(prompt + letter, add_special_tokens=False)
        if len(merged) != len(prompt_ids) + 1:
            raise SlotError(
                f"answer slot {letter!r} costs {len(merged) - len(prompt_ids)} tokens in "
                "context, not 1"
            )
        if merged[:-1] != prompt_ids:
            raise SlotError(
                f"appending answer slot {letter!r} re-tokenizes the prompt boundary; "
                "the last-position logits would not correspond to the answer"
            )
        ids.append(merged[-1])
    if len(set(ids)) != len(ids):
        raise SlotError("answer-slot tokens collide in context - two letters share one id")
    return ids

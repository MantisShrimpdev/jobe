"""The decision record, its validation, and the prompt it renders to.

Protocol adapted from TheoLeeCJ/SemIf (MIT), `src/semif_phase1/core.py`.

Evidence is placed FIRST and the criterion last. That ordering is deliberate:
it makes one body of evidence a reusable KV-cache prefix when many questions are
asked about the same state, which is where the serial/batched speedups come
from. Option ids are never shown to the model - only letters and descriptions -
so the model cannot key on an id that leaks the answer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .slots import LETTERS, MAX_OPTIONS

SYSTEM_PROMPT = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)

PROMPT_VERSION = "jobe-options-v1"


class DecisionError(ValueError):
    """Raised when a decision record cannot be represented by the protocol."""


@dataclass(frozen=True)
class Option:
    """One choice the model may return."""

    id: str
    description: str


@dataclass(frozen=True)
class Decision:
    """One decision: evidence, the criterion to apply, and the options."""

    id: str
    evidence: object
    criterion: str
    options: tuple[Option, ...]
    #: Ordinal questions are scored by probability-weighted value, not argmax.
    ordinal: bool = False

    def validate(self, *, max_options: int | None = MAX_OPTIONS) -> None:
        """Raise :class:`DecisionError` on anything the protocol cannot express.

        `max_options` is the letter protocol's ceiling by default. The text
        readout (:mod:`jobe.textscore`) has no such ceiling - it scores an
        option's own tokens rather than a letter slot - so it passes ``None``.
        """
        if self.evidence is None or (
            isinstance(self.evidence, str) and not self.evidence.strip()
        ):
            raise DecisionError(f"{self.id}: evidence must be a non-empty value")
        if not isinstance(self.criterion, str) or not self.criterion.strip():
            raise DecisionError(f"{self.id}: criterion must be a non-empty string")
        n = len(self.options)
        if n < 2:
            raise DecisionError(f"{self.id}: need at least 2 options, got {n}")
        if max_options is not None and n > max_options:
            raise DecisionError(
                f"{self.id}: need at most {max_options} options, got {n}; "
                "score_text() reads option text and has no such ceiling"
            )
        seen: set[str] = set()
        for option in self.options:
            if not option.id:
                raise DecisionError(f"{self.id}: every option needs a non-empty id")
            if not option.description or not option.description.strip():
                raise DecisionError(f"{self.id}: option {option.id} needs a description")
            if option.id in seen:
                raise DecisionError(f"{self.id}: duplicate option id {option.id}")
            seen.add(option.id)

    def reordered(self, order: list[int]) -> "Decision":
        """Return the same decision with options permuted (ids preserved)."""
        if sorted(order) != list(range(len(self.options))):
            raise DecisionError(f"{self.id}: order is not a permutation of the options")
        return Decision(
            id=self.id,
            evidence=self.evidence,
            criterion=self.criterion,
            options=tuple(self.options[i] for i in order),
            ordinal=self.ordinal,
        )


def build_messages(decision: Decision) -> list[dict[str, str]]:
    """Render a decision to the two-message chat form. Pure."""
    decision.validate()
    payload = {
        "evidence": decision.evidence,
        "criterion": decision.criterion,
        "options": [
            {"letter": LETTERS[i], "description": option.description}
            for i, option in enumerate(decision.options)
        ],
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


TEXT_SYSTEM_PROMPT = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only that option's id, copied exactly, with no explanation or reasoning."
)

TEXT_PROMPT_VERSION = "jobe-options-text-v1"


def build_text_messages(decision: Decision) -> list[dict[str, str]]:
    """Render a decision for the TEXT readout: options carry their ids, not letters.

    The letter form cannot express more than sixteen options because there are
    sixteen answer letters. Here the answer is the option's own id, so the
    payload shows ids and the model is asked to copy one - and there is no cap.
    """
    decision.validate(max_options=None)
    payload = {
        "evidence": decision.evidence,
        "criterion": decision.criterion,
        "options": [
            {"id": option.id, "description": option.description}
            for option in decision.options
        ],
    }
    return [
        {"role": "system", "content": TEXT_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _apply_template(tokenizer, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def render_text_prompt(tokenizer, decision: Decision) -> str:
    """The text readout's prompt, ending where the option id begins."""
    return _apply_template(tokenizer, build_text_messages(decision))


def render_prompt(tokenizer, decision: Decision) -> str:
    """Apply the tokenizer's chat template, with thinking disabled.

    `enable_thinking=False` is what keeps this usable on reasoning models: their
    first generated token is otherwise a preamble ("Okay", "First", "<think>"),
    never the answer, and the readout measures nothing. Tokenizers that do not
    accept the flag simply ignore it.
    """
    return _apply_template(tokenizer, build_messages(decision))


def prompt_digest(text: str) -> str:
    """SHA-256 of the exact prompt, so a score can be traced to what produced it."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

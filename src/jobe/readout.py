"""The readout: one forward pass, last-position logits, restricted to the
declared answer slots.

Adapted from TheoLeeCJ/SemIf (MIT), `src/semif_phase1/direct.py`.

Nothing is generated. The model is run once over the prompt, the logits at the
final position are taken, and only the declared answer letters are kept and
renormalised. A decision therefore costs zero output tokens no matter how many
options it has, and the answer is structurally one of the ids the caller
declared - it cannot emit prose, drift off-format, or need repair.

What this buys over reading `logprobs` from a hosted API, measured in
EveryAppKit's docs/SEMANTIC_DECISION_BENCHMARK.md:

  * the FULL vocabulary is visible, so no option is ever unmeasured (an API
    returns only a top-N window and loses the correct answer ~0.9% of the time);
  * `enable_thinking=False` is reachable, so reasoning models work (through an
    API their first token is a preamble and the readout fails outright);
  * no provider needs to expose logprobs at all - Anthropic never does, and GLM
    accepts the parameter while returning null.
"""

from __future__ import annotations

import inspect
import math
import time
from dataclasses import dataclass, field

from .prompt import PROMPT_VERSION, Decision, prompt_digest, render_prompt
from .slots import resolve_slots


@dataclass(frozen=True)
class Readout:
    """One decision's measured distribution."""

    decision_id: str
    #: Option ids in the order they were PRESENTED to the model.
    option_ids: tuple[str, ...]
    #: Probabilities aligned to `option_ids`, summing to 1.
    probabilities: tuple[float, ...]
    #: Raw selected logits, before the restricted softmax.
    option_logits: tuple[float, ...]
    input_tokens: int
    forward_seconds: float
    total_seconds: float
    prompt_sha256: str
    prompt_version: str = PROMPT_VERSION
    #: Deliberately not called "confidence" - see `confidence()`.
    probability_status: str = (
        "conditional option score; uncalibrated as decision confidence"
    )
    meta: dict = field(default_factory=dict)

    @property
    def choice(self) -> str:
        """The highest-scoring option id."""
        best = max(range(len(self.option_ids)), key=lambda i: self.probabilities[i])
        return self.option_ids[best]

    @property
    def scores(self) -> dict[str, float]:
        """Probabilities keyed by option id."""
        return dict(zip(self.option_ids, self.probabilities))

    def expected_value(self) -> float:
        """Probability-weighted value for ordinal questions (ids must be ints)."""
        return sum(int(oid) * p for oid, p in zip(self.option_ids, self.probabilities))

    def confidence(self) -> float:
        """Chance-corrected confidence: ``(p_max - 1/K) / (1 - 1/K)``.

        Rescales the top probability against what guessing would give, so a
        2-option and a 9-option decision are comparable. **It does not detect
        "none of these fit"** - the softmax is normalised over the declared
        options, so it sums to 1 even when every option is wrong. Validate the
        evidence before the call; this number cannot do it for you.
        """
        k = len(self.probabilities)
        p_max = max(self.probabilities)
        return (p_max - 1.0 / k) / (1.0 - 1.0 / k)


def restricted_softmax(values: list[float]) -> list[float]:
    """Softmax over selected logits only. Max-subtracted for stability."""
    if len(values) < 2 or any(not math.isfinite(v) for v in values):
        raise ValueError("restricted_softmax needs at least two finite scores")
    top = max(values)
    weights = [math.exp(v - top) for v in values]
    total = sum(weights)
    return [w / total for w in weights]


def _forward_last_logits(model, inputs):
    """Run the model, returning logits at the final position only.

    `logits_to_keep=1` skips the LM head over every earlier position, which is
    the bulk of the head's cost on a long prompt. Models that do not accept it
    fall back to slicing.
    """
    import torch

    params = inspect.signature(model.forward).parameters
    kwargs = dict(inputs, use_cache=False, return_dict=True)
    if "logits_to_keep" in params:
        kwargs["logits_to_keep"] = 1
    with torch.inference_mode():
        out = model(**kwargs)
    return out.logits[:, -1, :]


def score(model, tokenizer, decision: Decision, *, max_tokens: int = 4096) -> Readout:
    """Measure one decision in a single forward pass.

    Args:
        model: a loaded causal LM.
        tokenizer: its tokenizer.
        decision: the decision to score.
        max_tokens: refuse prompts longer than this rather than truncating -
            silently dropping evidence would change the answer.

    Raises:
        SlotError: the tokenizer cannot support the answer-slot contract.
        DecisionError: the decision record is malformed.
        ValueError: the prompt exceeds `max_tokens`.
    """
    import torch

    started = time.perf_counter()
    decision.validate()

    prompt = render_prompt(tokenizer, decision)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids:
        raise ValueError(f"{decision.id}: empty prompt")
    if len(prompt_ids) > max_tokens:
        raise ValueError(
            f"{decision.id}: {len(prompt_ids)} input tokens exceed the limit "
            f"{max_tokens}; no truncation is performed"
        )

    slots = resolve_slots(tokenizer, prompt, prompt_ids, len(decision.options))

    device = next(model.parameters()).device
    inputs = {
        "input_ids": torch.tensor([prompt_ids], dtype=torch.long, device=device),
        "attention_mask": torch.ones((1, len(prompt_ids)), dtype=torch.long, device=device),
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_started = time.perf_counter()
    vocabulary = _forward_last_logits(model, inputs)[0].float()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - forward_started

    selected = vocabulary[slots].tolist()
    return Readout(
        decision_id=decision.id,
        option_ids=tuple(o.id for o in decision.options),
        probabilities=tuple(restricted_softmax(selected)),
        option_logits=tuple(selected),
        input_tokens=len(prompt_ids),
        forward_seconds=forward_seconds,
        total_seconds=time.perf_counter() - started,
        prompt_sha256=prompt_digest(prompt),
        meta={
            "readout": "full-vocabulary last-position logits restricted to declared answer slots",
            "device": str(device),
        },
    )

"""Jobe's adapter for TheLab's training loop — the part that touches tokens.

TheLab's `thelab.decisions.train` needs, per record, the prompt ids, the token
id of each answer letter at the answer position, and the gold index. Those are
exactly what the readout computes — `render_prompt` then `resolve_slots` in
context — so training optimises the very logits `score()` reads. Nothing else
about the readout changes: a trained adapter is loaded onto the same backbone
and scored by the same path.

    python -m thelab.decisions.train --adapter jobe.train_adapter:JobeAdapter \
        --backbone D:/Coding/models/qwen35-4b --records temporal_numeric.jsonl --out runs/lora-tn
"""
from __future__ import annotations

from thelab.decisions.train import Encoded

from .model import load
from .prompt import render_prompt
from .records import to_decision
from .slots import resolve_slots


class JobeAdapter:
    # Qwen3.5's attention projections; the linear-attention layers use other names and are left frozen
    default_target_modules = ("q_proj", "k_proj", "v_proj", "o_proj")

    def __init__(self, model, tokenizer, *, max_tokens: int = 4096):
        self.model, self.tokenizer, self.max_tokens = model, tokenizer, max_tokens
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    @classmethod
    def from_backbone(cls, name: str, *, revision: str | None = None, device: str = "auto",
                      attn_implementation: str = "sdpa", max_tokens: int = 4096) -> "JobeAdapter":
        bb = load(name, revision=revision, device=device, attn_implementation=attn_implementation)
        return cls(bb.model, bb.tokenizer, max_tokens=max_tokens)

    def encode(self, record) -> Encoded:
        decision = to_decision(record)
        decision.validate()
        prompt = render_prompt(self.tokenizer, decision)
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > self.max_tokens:
            raise ValueError(f"{decision.id}: {len(ids)} tokens exceed {self.max_tokens}; no truncation")
        slots = resolve_slots(self.tokenizer, prompt, ids, len(decision.options))
        expected = record["expected"] if isinstance(record, dict) else record.expected
        option_ids = [o.id for o in decision.options]
        return Encoded(record_id=decision.id, input_ids=ids, slot_ids=slots, gold=option_ids.index(expected),
                       option_ids=option_ids)

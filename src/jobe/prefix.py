"""Prefix-cache reuse: encode the evidence once, answer many questions as cheap
suffixes off a copy of that cache.

Adapted from TheoLeeCJ/SemIf (MIT), `src/semif_phase1/serial.py`.

Jobe's prompt puts evidence FIRST and the criterion + options last precisely so
that one body of evidence is a reusable prefix. This module cashes that in.
The state is run through the model once with a KV cache kept; each question is
then only its own suffix — `, "criterion": ..., "options": [...]}` plus the
template's assistant turn — continued from a copy of that cache.

Measured on Qwen3.5-4B, bf16, RTX 3080, a 1,921-token state: prefix encoded
once in 1.23 s; each question then 0.10–0.16 s against ~1.3 s for the same
prompt uncached. About 10× per question after the first. Slot logits agree with
the uncached forward to within one bf16 ulp and the argmax is identical.

Two guards make this safe, and both fail loudly rather than degrade:

  * the cached prefix must be a token-for-token prefix of every full prompt it
    is reused for. Tokenizers merge greedily across boundaries, so the prefix
    drops its final token (JSON punctuation can fuse with what follows) and the
    full prompt's ids are checked against it before any forward pass;
  * a decision whose evidence differs from the primed evidence is refused. It
    is never silently scored against another state's cache.

Two variants live here. `score` is serial: one question at a time, one cache
copy each. `score_batch` runs several suffixes per forward pass off a batch-
expanded copy of the cache (SemIf's `shared.py`), right-padded, each row read at
its last real token; on a 10 GB card it measured fastest at a batch of 2 and
slower than serial at 8. Both are the right shape for "many questions about one
document"; neither helps a workload with one question per document.
"""

from __future__ import annotations

import copy
import inspect
import json
import time

from .prompt import PROMPT_VERSION, Decision, Option, build_messages, prompt_digest, render_prompt
from .readout import Readout, restricted_softmax
from .slots import resolve_slots


class PrefixError(ValueError):
    """Raised when a prefix cannot be established or must not be reused."""


def evidence_key(evidence: object) -> str:
    """Canonical identity of an evidence value, so dicts and strings compare by content."""
    return json.dumps(evidence, ensure_ascii=False, sort_keys=True)


def prefix_ids_for(tokenizer, evidence: object) -> list[int]:
    """Token ids of the rendered prompt up to the end of the evidence value.

    Renders a placeholder decision to obtain the chat template's framing, then
    cuts the text right after the serialised evidence — before the criterion,
    before the options. The final token is dropped: JSON punctuation after the
    evidence can merge with the last token of the value, and a prefix that ends
    on a token which would have merged is not a prefix at all.

    Raises:
        PrefixError: the template altered the payload (so its position cannot be
            located), or the evidence is not where the protocol puts it.
    """
    placeholder = Decision(
        id="__prefix__",
        evidence=evidence,
        criterion="prefix boundary placeholder",
        options=(Option("a", "a"), Option("b", "b")),
    )
    prompt = render_prompt(tokenizer, placeholder)
    payload = build_messages(placeholder)[1]["content"]
    if prompt.count(payload) != 1:
        raise PrefixError("cannot locate the unmodified payload in the chat template")
    head = json.dumps({"evidence": evidence}, ensure_ascii=False)[:-1]  # drop closing brace
    if not payload.startswith(head):
        raise PrefixError("evidence serialisation is not the payload prefix")
    text = prompt[: prompt.index(payload)] + head
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) < 2:
        raise PrefixError("prefix is too short to reuse")
    return ids[:-1]


class PrefixScorer:
    """Score many decisions about one piece of evidence off a single prefix pass.

        scorer = PrefixScorer(backbone.model, backbone.tokenizer)
        scorer.prime(document)                 # one forward pass, cache kept
        for q in questions:
            r = scorer.score(Decision(evidence=document, ...))   # suffix only

    `score` refuses a decision whose evidence is not the primed evidence.
    """

    def __init__(self, model, tokenizer, *, max_tokens: int = 4096):
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self._device = next(model.parameters()).device
        params = inspect.signature(model.forward).parameters
        self._logits_to_keep = "logits_to_keep" in params
        self._cache_position = "cache_position" in params
        self._cache = None
        self._prefix_ids: list[int] = []
        self._key: str | None = None
        self.prefix_seconds: float = 0.0

    # --------------------------------------------------------------- priming

    @property
    def is_primed(self) -> bool:
        return self._cache is not None

    @property
    def prefix_tokens(self) -> int:
        return len(self._prefix_ids)

    def prime(self, evidence: object) -> int:
        """Encode `evidence` once and keep its cache. Returns the prefix length.

        A no-op when the same evidence is already primed. Priming a different
        evidence replaces the cache.
        """
        import torch

        key = evidence_key(evidence)
        if self._cache is not None and key == self._key:
            return len(self._prefix_ids)
        ids = prefix_ids_for(self.tokenizer, evidence)
        if len(ids) > self.max_tokens:
            raise PrefixError(f"prefix of {len(ids)} tokens exceeds the limit {self.max_tokens}")

        self._cache = None  # release the old cache before allocating a new one
        started = time.perf_counter()
        kwargs = dict(
            input_ids=torch.tensor([ids], dtype=torch.long, device=self._device),
            attention_mask=torch.ones((1, len(ids)), dtype=torch.long, device=self._device),
            use_cache=True,
            return_dict=True,
        )
        # Only the cache is wanted from this pass. Without this the model also
        # materialises logits for every prefix position - 1,921 tokens x a 248k
        # vocabulary in bf16 is ~950 MB of nothing, which on a 10 GB card is the
        # difference between fitting and not.
        if self._logits_to_keep:
            kwargs["logits_to_keep"] = 1
        with torch.inference_mode():
            out = self.model(**kwargs)
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        self.prefix_seconds = time.perf_counter() - started

        self._cache = out.past_key_values
        self._prefix_ids = ids
        self._key = key
        return len(ids)

    # --------------------------------------------------------------- scoring

    def score(self, decision: Decision) -> Readout:
        """Score one decision as a suffix off the primed cache.

        Raises:
            PrefixError: nothing is primed, the decision's evidence is not the
                primed evidence, or the full prompt does not begin with the
                cached prefix (a tokenizer-boundary failure).
        """
        import torch

        started = time.perf_counter()
        if self._cache is None:
            raise PrefixError("no evidence is primed; call prime() first")
        decision.validate()
        if evidence_key(decision.evidence) != self._key:
            raise PrefixError(
                f"{decision.id}: evidence differs from the primed evidence; "
                "refusing to score it against another state's cache"
            )

        prompt = render_prompt(self.tokenizer, decision)
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > self.max_tokens:
            raise ValueError(
                f"{decision.id}: {len(ids)} input tokens exceed the limit {self.max_tokens}; "
                "no truncation is performed"
            )
        n = len(self._prefix_ids)
        if ids[:n] != self._prefix_ids:
            raise PrefixError(
                f"{decision.id}: full prompt does not begin with the cached prefix; "
                "the tokenizer re-tokenised the boundary and the cache cannot be reused"
            )
        suffix = ids[n:]
        slots = resolve_slots(self.tokenizer, prompt, ids, len(decision.options))

        forward_started = time.perf_counter()
        cache = copy.deepcopy(self._cache)
        positions = torch.arange(n, len(ids), device=self._device)
        kwargs = dict(
            input_ids=torch.tensor([suffix], dtype=torch.long, device=self._device),
            attention_mask=torch.ones((1, len(ids)), dtype=torch.long, device=self._device),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
            position_ids=positions.unsqueeze(0),
        )
        if self._logits_to_keep:
            kwargs["logits_to_keep"] = 1
        if self._cache_position:
            kwargs["cache_position"] = positions
        with torch.inference_mode():
            vocabulary = self.model(**kwargs).logits[:, -1, :][0].float()
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        forward_seconds = time.perf_counter() - forward_started

        selected = vocabulary[slots].tolist()
        return Readout(
            decision_id=decision.id,
            option_ids=tuple(o.id for o in decision.options),
            probabilities=tuple(restricted_softmax(selected)),
            option_logits=tuple(selected),
            input_tokens=len(ids),
            forward_seconds=forward_seconds,
            total_seconds=time.perf_counter() - started,
            prompt_sha256=prompt_digest(prompt),
            prompt_version=PROMPT_VERSION,
            meta={
                "readout": (
                    "prefix-cache suffix; full-vocabulary last-position logits "
                    "restricted to declared answer slots"
                ),
                "device": str(self._device),
                "prefix_tokens": n,
                "suffix_tokens": len(suffix),
            },
        )

    def score_many(self, decisions: list[Decision]) -> list[Readout]:
        """Score decisions that all share the primed evidence, in order."""
        return [self.score(d) for d in decisions]

    # --------------------------------------------------------------- batched

    def _prepare(self, decision: Decision) -> tuple[list[int], list[int], list[int], str]:
        """All the guards, none of the forward pass. Returns (ids, suffix, slots, prompt)."""
        decision.validate()
        if evidence_key(decision.evidence) != self._key:
            raise PrefixError(
                f"{decision.id}: evidence differs from the primed evidence; "
                "refusing to score it against another state's cache"
            )
        prompt = render_prompt(self.tokenizer, decision)
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > self.max_tokens:
            raise ValueError(
                f"{decision.id}: {len(ids)} input tokens exceed the limit {self.max_tokens}; "
                "no truncation is performed"
            )
        n = len(self._prefix_ids)
        if ids[:n] != self._prefix_ids:
            raise PrefixError(
                f"{decision.id}: full prompt does not begin with the cached prefix; "
                "the tokenizer re-tokenised the boundary and the cache cannot be reused"
            )
        if len(ids) == n:
            raise PrefixError(f"{decision.id}: empty suffix")
        slots = resolve_slots(self.tokenizer, prompt, ids, len(decision.options))
        return ids, ids[n:], slots, prompt

    def score_batch(self, decisions: list[Decision], *, max_batch: int = 2) -> list[Readout]:
        """Score decisions sharing the primed evidence, several per forward pass.

        Adapted from SemIf's `shared.py`. Suffixes are RIGHT-padded to a common
        width, the prefix cache is expanded to the batch with `reorder_cache`
        (the beam-search path, which every cache layer type implements — the
        linear-attention layers in a hybrid model lack `batch_repeat_interleave`),
        and one forward pass scores the whole chunk. Each row's readout is taken
        at its last REAL token, gathered before the LM head so the head runs on
        one vector per row rather than every padded position.

        Right-padding is a measured choice, not a habit: on Qwen3.5-4B, right-
        padded rows matched the uncached forward to within 1–2 bf16 ulps while
        left-padded rows drifted to 5–7. Pads placed between the prefix and the
        suffix perturb the recurrent/conv layers even when masked.

        `max_batch` defaults to 2, which measured fastest on a 10 GB card with a
        ~2k-token prefix: 59 ms/question at batch 2, 74 at batch 4, and batch 8
        peaked at 10.06 GB, spilled to host memory, and ran SLOWER than serial
        (189 ms). The ceiling is the hardware, not the code - run
        bench/prefix_bench.py --batch on yours before raising it. Every guard runs for every decision before any
        forward pass, so a bad decision refuses the whole call up front rather
        than after half of it has been paid for.
        """
        import torch

        if not decisions:
            return []
        if self._cache is None:
            raise PrefixError("no evidence is primed; call prime() first")
        if max_batch < 1:
            raise ValueError("max_batch must be at least 1")
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise PrefixError("tokenizer has neither a pad nor an eos token to pad with")

        started = time.perf_counter()
        prepared = [self._prepare(d) for d in decisions]  # guards first, all of them
        base = getattr(self.model, "model", None)
        head = getattr(self.model, "lm_head", None)
        gather_then_head = base is not None and head is not None

        results: list[Readout] = []
        n_prefix = len(self._prefix_ids)
        for start in range(0, len(decisions), max_batch):
            chunk = list(zip(decisions[start:start + max_batch], prepared[start:start + max_batch]))
            suffixes = [p[1] for _, p in chunk]
            layout = suffix_layout(suffixes, n_prefix, pad_id)
            n = len(chunk)

            forward_started = time.perf_counter()
            cache = copy.deepcopy(self._cache)
            cache.reorder_cache(torch.zeros(n, dtype=torch.long, device=self._device))
            kwargs = dict(
                input_ids=torch.tensor(layout.input_ids, dtype=torch.long, device=self._device),
                attention_mask=torch.tensor(layout.attention_mask, dtype=torch.long, device=self._device),
                position_ids=torch.tensor(layout.position_ids, dtype=torch.long, device=self._device),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            with torch.inference_mode():
                if gather_then_head:
                    hidden = base(**kwargs).last_hidden_state  # (n, width, hidden)
                    rows = torch.arange(n, device=self._device)
                    ends = torch.tensor(layout.ends, dtype=torch.long, device=self._device)
                    vocab = head(hidden[rows, ends]).float()  # (n, vocab)
                else:
                    kwargs["logits_to_keep"] = layout.width
                    logits = self.model(**kwargs).logits.float()
                    vocab = logits[torch.arange(n), torch.tensor(layout.ends)]
            if self._device.type == "cuda":
                torch.cuda.synchronize(self._device)
            forward_seconds = time.perf_counter() - forward_started
            del cache

            for row, (decision, (ids, suffix, slots, prompt)) in enumerate(chunk):
                selected = vocab[row][slots].tolist()
                results.append(Readout(
                    decision_id=decision.id,
                    option_ids=tuple(o.id for o in decision.options),
                    probabilities=tuple(restricted_softmax(selected)),
                    option_logits=tuple(selected),
                    input_tokens=len(ids),
                    forward_seconds=forward_seconds / n,
                    total_seconds=0.0,  # filled below once the whole call is timed
                    prompt_sha256=prompt_digest(prompt),
                    prompt_version=PROMPT_VERSION,
                    meta={
                        "readout": (
                            "prefix-cache batched suffix; full-vocabulary last-real-token "
                            "logits restricted to declared answer slots"
                        ),
                        "device": str(self._device),
                        "prefix_tokens": n_prefix,
                        "suffix_tokens": len(suffix),
                        "batch": n,
                        "padding": "right",
                    },
                ))
        total = time.perf_counter() - started
        return [
            Readout(**{**r.__dict__, "total_seconds": total / len(results)}) for r in results
        ]


class SuffixLayout:
    """A right-padded batch of suffixes, ready for one forward pass."""

    __slots__ = ("input_ids", "attention_mask", "position_ids", "ends", "width")

    def __init__(self, input_ids, attention_mask, position_ids, ends, width):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.position_ids = position_ids
        self.ends = ends
        self.width = width


def suffix_layout(suffixes: list[list[int]], prefix_len: int, pad_id: int) -> SuffixLayout:
    """Right-pad suffixes to a common width. Pure.

    For each row: ids are the suffix followed by pads; the attention mask covers
    the cached prefix and the real suffix tokens and is zero over the pads;
    position ids continue from `prefix_len` over the real tokens; `ends[i]` is
    the index of the row's last real token, which is where its readout lives.
    """
    if not suffixes or any(len(s) == 0 for s in suffixes):
        raise PrefixError("every decision needs a non-empty suffix")
    width = max(len(s) for s in suffixes)
    ids, masks, positions, ends = [], [], [], []
    for s in suffixes:
        pad = width - len(s)
        ids.append(list(s) + [pad_id] * pad)
        masks.append([1] * (prefix_len + len(s)) + [0] * pad)
        positions.append(list(range(prefix_len, prefix_len + len(s))) + [0] * pad)
        ends.append(len(s) - 1)
    return SuffixLayout(ids, masks, positions, ends, width)


def score_with_prefix(model, tokenizer, evidence: object, decisions: list[Decision],
                      *, max_tokens: int = 4096) -> list[Readout]:
    """Prime `evidence` once and score every decision against it."""
    scorer = PrefixScorer(model, tokenizer, max_tokens=max_tokens)
    scorer.prime(evidence)
    return scorer.score_many(decisions)

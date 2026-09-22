# Jobe

A local, frozen-backbone decision readout. Give it evidence, a criterion and a
set of options; it returns a probability for each option, read from the model's
next-token distribution in **one forward pass**. Nothing is generated, so there
is no text to parse, nothing to repair, and no way for the answer to be
something other than one of the ids you declared.

**Status: conformance-clean through JevBench's own harness, scoring in
SemIf's band.** 231/231 public tasks driven by their `Runner`, every result
`strict_valid`, zero failures; 0.805 accuracy, hard tier 0.604. Scored with
their `composite_v12` under their partial-run rule and their self-hosted
pricing convention for a 4B: **74.9**, against SemIf's published 74.7 and Jev's
75.4, with the judge tier unmeasured and held-out unseen (`bench/RESULTS.md`).
No training has happened — v1 freezes the backbone entirely, which is the
design, not a shortcut. Three of the top five open systems are frozen
backbones.

```python
from jobe import Decision, Option, load, score

backbone = load("unsloth/llama-3.2-3b-instruct")
r = score(backbone.model, backbone.tokenizer, Decision(
    id="ticket-1",
    evidence="The API has returned 503 for every request since the deploy.",
    criterion="Which team should own this ticket?",
    options=(
        Option("billing", "Payments, invoices, refunds and charges"),
        Option("infra", "Outages, latency, deploys and system errors"),
    ),
))

r.choice        # "infra"
r.scores        # {"billing": 0.003, "infra": 0.997}
r.confidence()  # 0.994  — chance-corrected; read the caveat below
```

## Why local, rather than an API

The same protocol run over a hosted API (`logprobs` on an OpenAI-compatible
endpoint) has three hard limits, each measured rather than assumed — see
EveryAppKit's `docs/SEMANTIC_DECISION_BENCHMARK.md`. Reading logits in-process
removes all three:

| Through an API | Here |
|---|---|
| Anthropic exposes no logprobs; GLM accepts the parameter and returns null | Reads logits directly — no provider involved |
| Reasoning models unusable: the first token is `Okay`/`<think>`, never the answer | `enable_thinking=False` in the chat template |
| Only a top-N window is returned; loses the correct option in ~0.9% of decisions | The **full** vocabulary is visible; nothing is ever unmeasured |

## One improvement over the protocol it adapts

SemIf verifies answer slots by encoding each letter **standalone**. That rejects
every SentencePiece tokenizer — Llama-2, Mistral, TinyLlama — because they
prepend a word-boundary marker to standalone text: `encode("A")` gives `▁A`
(id 319 on TinyLlama) while the token that actually follows a prompt is a bare
`A` (id 29909). Reading the logit of `▁A` would measure the wrong thing, with
full confidence.

`resolve_slots` derives each slot **in context**, from `encode(prompt + letter)[-1]`,
and keeps the guard that genuinely matters: everything before that final token
must still be exactly the prompt's own ids. If the prompt's tail re-tokenizes,
the last position is not the answer position and no id can repair it. This makes
SentencePiece backbones usable instead of rejected, and is correct by
construction on BPE ones.

## Measured latency

`llama-3.2-3b-instruct`, bf16, RTX 3080, 130-token prompt, batch 1, after clock
warm-up (n=48):

| | p50 | p95 |
|---|---:|---:|
| forward pass | 35.9 ms | 36.5 ms |
| **end to end** | **36.9 ms** | **37.5 ms** |

Pipeline overhead — tokenising, resolving slots in context, and the restricted
softmax — is **1.0 ms** of that. The same decision on CPU costs ~3,400 ms, so
the GPU is worth **92×** here.

For context rather than comparison: Laya is published at 32.8 ms on a T4 and
TypeSafe Jev has been independently measured at 236–276 ms.

Two findings behind those numbers, both of which cost real time to run down:

- **`eager` attention beats SDPA at this scale** — 39.1 ms against 50.4 ms on a
  130-token prompt at batch 1. A decision prompt is short and un-batched, so
  SDPA's kernel overhead is never repaid. `load()` therefore defaults to eager;
  pass `attn_implementation="sdpa"` for long evidence, where that should invert.
- **Check what else holds VRAM before trusting any timing.** The first GPU run
  measured 457 ms p50 with a 2,458 ms p95 and wildly inconsistent
  attention-implementation results. Ollama was holding 3.3 GB of a 10 GB card,
  leaving the model thrashing, and the GPU was sitting at 210 MHz of 2,130.
  After `ollama stop` and a proper warm-up the same benchmark was 12× faster and
  the distribution tightened to a 0.6 ms spread. A one-off timing on a busy GPU
  is not a measurement.

## Prefix cache — many questions about one document

The prompt puts evidence first so that it can be a reusable prefix, and
`jobe.prefix` cashes that in. `PrefixScorer.prime(document)` runs the model
once with its KV cache kept; each `score(decision)` then continues from a copy
of that cache with only the question's suffix — `, "criterion": …, "options":
[…]}` plus the assistant turn.

```python
from jobe import PrefixScorer

scorer = PrefixScorer(backbone.model, backbone.tokenizer)
scorer.prime(document)
for d in decisions:            # every one with evidence=document
    r = scorer.score(d)        # suffix only
```

Measured with `bench/prefix_bench.py` — Qwen3.5-4B, bf16, RTX 3080, a
1,995-token document (prefix 1,921 tokens, suffix ~74), eight questions:

| | per question | eight questions |
|---|---:|---:|
| uncached | 1,302 ms | 10.41 s |
| **cached** | **117 ms** after a one-time 1,772 ms prime | **2.80 s** |

**11× per question after the first; 3.7× over eight.** Break-even is the second
question. Slot logits agree with the uncached path to within two bf16 ulps and
the argmax never flipped (0/8). Qwen3.5's hybrid recurrent-state layers survive
the cache copy correctly — that was the part least certain in advance, and the
equality test is what settles it.

Two guards, both fail-loud rather than degrade:

- **The cached prefix must be a token-for-token prefix of every prompt it is
  reused for.** Tokenizers merge greedily across boundaries, so the prefix drops
  its final token (JSON punctuation can fuse with the end of the evidence) and
  the full prompt's ids are checked against it before any forward pass.
- **A decision whose evidence differs from the primed one is refused** with
  `PrefixError`. It is never silently scored against another document's cache.

This is the serial variant — one question at a time, one cache copy each
(5–10 ms). Batching suffixes in parallel, SemIf's `shared.py`, is the remaining
optimisation on top.

## Known limits

- **16 options maximum** — one single-token letter each. Above roughly that,
  retrieve-then-decide beats decide-over-everything anyway; every system in this
  family degrades sharply with menu size.
- **`confidence()` cannot detect "none of these fit".** The softmax is
  normalised over the *declared* options, so it sums to 1 even when every option
  is wrong. Measured: a model answered `billing_question` at p = 0.998 for the
  evidence *"Text Anna that I'll be ten minutes late."* Validate the evidence
  before the call; this number cannot do it for you.
- **A weak backbone produces a confident reflex, not a decision.** TinyLlama-1.1B
  answered the same letter on every case at p ≈ 0.98, scoring exactly the base
  rate. Run the control battery before believing any accuracy number.

## Verifying a backbone

EveryAppKit ships `tools/decisionGate.ts`, which runs the three controls that
catch this class of failure — shuffled-context (E1), option-order (E2), and a
positional-bias check — reported **per tier**, because an aggregate hides
exactly the failure it is meant to catch. Point it at any endpoint; it exits
non-zero on failure.

## Layout

```
src/jobe/
  slots.py     answer-letter slots + the in-context resolution and boundary guard
  prompt.py    the decision record, its validation, and the rendered prompt
  readout.py   one forward pass, last-position logits, restricted softmax
  model.py     frozen backbone loading; device and attention resolved once
  orders.py    score under several option orders, average, report the flip rate
  calibrate.py temperature fitting + ECE/MCE/Brier/NLL over stored logits
  prefix.py    encode the evidence once, score many questions as suffixes off the cache
tests/         53 tests; a few need a tokenizer, one is opt-in on a real GPU
```

## Next

1. ~~CUDA torch, then re-measure latency.~~ Done — 36.9 ms p50.
2. ~~Order averaging across option orders.~~ Built (`jobe.orders`) — reports
   `flip_rate` alongside the averaged result, because it pays only in proportion
   to the fragility it fixes. Not yet validated at volume on a real task set.
3. ~~One fitted temperature.~~ Built (`jobe.calibrate`) — golden-section on
   held-out NLL, no optimiser and no dependency. Also not yet fitted on real data.
4. ~~Measure on real tasks.~~ Done — `bench/RESULTS.md`. The backbone was the
   lever: Qwen3.5-4B beats llama-3.2-3b by **+25.6 points overall** and +44.4 on
   the standard tier, on identical prompts. And a **single global temperature
   does not transfer across tiers** — it helps one and hurts the other, in
   opposite directions for the two backbones.
5. ~~Control battery.~~ Done — `bench/gate.py`, **GATE PASSED on all three
   tiers**. E1 destroys all above-chance skill *including on hard*
   (0.613 → 0.306, where gemma4-8B had not moved at all); E2 flip rate is 2.1%
   easy / 8.3% standard / 23.4% hard.
6. ~~Average the hard tier and see whether it pays.~~ Done — **it does not.**
   Accuracy 0.613 → 0.613 for double the forward passes; ECE moves 0.005. Of
   the 11 answers averaging changed, it fixed 3 and broke 3. The 22.5% order
   instability is symmetric noise, not a correctable bias — so a flip rate does
   **not** predict whether averaging helps, which corrects what I claimed
   earlier.
7. ~~Run through the official harness.~~ Done — 231/231, strict 1.000,
   composite 74.9 under SemIf-class pricing. See `bench/README-submission.md`.
8. **Submit**: weights pinned to `851bf6e8…` plus `bench/jobe_direct.py`. No
   server is needed — Benchmark Heaven runs in-process adapters on their own
   infrastructure, and the spec forbids a home endpoint.
9. ~~Prefix cache.~~ Done — `jobe.prefix`, **11× per question after the first**
   on a 2k-token document, logits within two bf16 ulps, 0/8 flips. Serial only;
   batched suffixes are the remaining SemIf optimisation.

Only after those plateau is training worth considering.

## License and attribution

**MIT** — see `LICENSE`. The decision protocol is adapted from
[TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf), also MIT; see `NOTICE`
for what was taken and what was changed.

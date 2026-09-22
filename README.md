# Jobe

A local, frozen-backbone decision readout. Give it evidence, a criterion and a
set of options; it returns a probability for each option, read from the model's
next-token distribution in **one forward pass**. Nothing is generated, so there
is no text to parse, nothing to repair, and no way for the answer to be
something other than one of the ids you declared.

**Status: the readout runtime works, and reaches SemIf parity on the public
JevBench tasks** — 0.801 overall with Qwen3.5-4B, hard tier 0.613 against SemIf's
published 0.595 (`bench/RESULTS.md`). 45 tests pass; `llama-3.2-3b-instruct`
answers 6/6 on a triage smoke set with sensible confidence gradation (0.41 on a
genuinely ambiguous case, 0.99 on clear ones). No training has happened — v1
freezes the backbone entirely, which is the design, not a shortcut. Three of the
top five open systems on JevBench are frozen backbones, and the best of them
sits 0.7 points behind a closed commercial model.

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
tests/         45 tests; only two need a tokenizer, none need a model or a GPU
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
6. Average **the hard tier only** and see whether it pays — the flip rates say
   there is nothing to repair on easy or standard.
7. A `/v1/systemone`-compatible server.

Only after those plateau is training worth considering.

## Attribution

The decision protocol is adapted from [TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf)
(MIT). See `NOTICE`.

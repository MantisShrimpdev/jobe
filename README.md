# Jobe

A local, frozen-backbone decision readout. Give it evidence, a criterion and a
set of options; it returns a probability for each option, read from the model's
next-token distribution in **one forward pass**. Nothing is generated, so there
is no text to parse, nothing to repair, and no way for the answer to be
something other than one of the ids you declared.

**Status: the readout runtime works.** 28 protocol tests pass; `llama-3.2-3b-instruct`
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

## Known limits

- **CPU-only torch is installed** (`2.10.0+cpu`) on a machine with an RTX 3080.
  A decision costs **~3.4 s** on CPU for a 3B model, against roughly 0.2 s needed
  to be competitive on a latency axis. A CUDA build is the single biggest
  outstanding dependency; wheels for Python 3.14 may not exist yet.
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
  model.py     frozen backbone loading; device resolved once
tests/         28 protocol tests, no model required
```

## Next

1. CUDA torch, then re-measure latency — the whole speed argument rests on it.
2. Order averaging across 2+ option orders (pays in proportion to the
   order-fragility it fixes; measure the flip rate first, it is not free).
3. One fitted temperature on a held-out split.
4. A `/v1/systemone`-compatible server.

Only after those plateau is training worth considering.

## Attribution

The decision protocol is adapted from [TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf)
(MIT). See `NOTICE`.

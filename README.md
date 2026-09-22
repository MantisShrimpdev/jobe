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
backbones. Tag **`v0.1.0`** pins this exact state — weights revision, prompt
version, adapter and run — so every later change is measured against it.

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

### Batched suffixes

`score_batch(decisions, max_batch=2)` runs several suffixes per forward pass off
a batch-expanded copy of the cache — SemIf's `shared.py` step. Same document,
eight questions:

| path | ms/question | vs uncached | vs serial | peak GB |
|---|---:|---:|---:|---:|
| uncached | 906 | 1.0× | | |
| serial | 113 | 8.1× | 1.0× | |
| **batch 2** | **59** | **15.3×** | **1.9×** | 9.12 |
| batch 4 | 74 | 12.2× | 1.5× | 9.58 |
| batch 8 | 189 | 4.8× | **0.6×** | 10.06 |

0 argmax flips at every size; logits within two bf16 ulps of the uncached path.
SemIf reports 1.9× for its own serial→batched step, which is a reassuring
cross-check.

Three things the measurement decided rather than the design:

- **Right-padding, not left.** Right-padded rows sit within 1–2 ulps of ground
  truth; left-padded rows drift to 5–7. Pads placed between the prefix and the
  suffix perturb the hybrid recurrent/conv layers even when masked out.
- **Expansion via `reorder_cache`** — the beam-search path — because
  `batch_repeat_interleave` is not implemented for Qwen3.5's linear-attention
  cache layers.
- **The default batch is 2, and 8 is slower than serial.** The model takes
  8.5 GB of a 10.2 GB card; batch 8 peaks at 10.06 GB and the driver spills to
  host RAM. The ceiling is the hardware, not the code — `bench/prefix_bench.py
  --batch` finds it on yours.

Each row's readout is gathered at its last real token *before* the LM head, so
the head runs on one vector per row rather than every padded position. Every
guard runs for every decision before any forward pass: one foreign decision
refuses the whole call up front, not after half of it has been paid for.

## Worlds — training data from exact laws

The hard-tier gap is knowledge, not thinking depth (see `bench/RESULTS.md`), so
the next lever is training — and training data for exact-law families cannot
be labelled by a model that gets them wrong. `jobe.worlds` generates decisions
whose answers are **computed by code** from rules stated in full in the
evidence. `worlds/temporal_numeric.py` covers the family where the readout most
often reproduces the same wrong number: day-count interest, month-end roll with
leap years and time zones, threshold-crossing days on net cost, cumulative card
limits, rest periods across a clock change, business-day windows, unit
conversion with one rounding, binary-vs-decimal allowances, FX reimbursement,
refund and credit proration — eleven laws. `worlds/long_policy.py` builds
8–13k-character policy packs the way that family builds them: definitions with
a carve-out, numbered clauses, an amendment with its own applicability date, a
superseded text left in the file, the case records and a desk note applying
the naive rule — six laws: approval routing by aggregated contract value, a
water-damage claim through exclusion, exception and endorsement, alert routing
through ordered runbook rules, payment-relief eligibility on a defined income,
appeal admissibility on working days with closures, and a commercial returns
decision.

The wrong options are not random. Each is the answer a **named mistake**
produces — wall-clock time across a clock change, an inclusive end date, a
365-day basis, tax counted in net cost — and the record says which
(`mistakes: {option_id: mistake}`), so a trained readout is graded on the law
and not the surface. Clock changes follow the US and EU rules from first
principles in stdlib, and every evidence text states the change it relies on.

```bash
PYTHONPATH=src python -m jobe.worlds.temporal_numeric --n 2000 --seed 0 --out temporal_numeric.jsonl
PYTHONPATH=src python -m jobe.worlds.temporal_numeric --n 500 --check-contamination <jevbench clone>
```

Records are reproducible from the seed, carry a stable `train`/`heldout` split
(≈10 % held out by id hash) and the rationale, and convert to a `Decision` with
`to_decision`. Nothing in them is JevBench text: the contamination check fails
on any 10-word run shared with the public task files, and it caught my own
first draft, which had paraphrased two of their rule sentences too closely.

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
- **What `confidence()` *is* good for: handing off.** On good input it does find
  the readout's own wrong answers — hard tier AUROC 0.79, the least-confident
  quartile 15% accurate against 83% for the most. Use a threshold (≈0.57 on
  JevBench hard) to route to a person or a larger model. Not to a thinking pass
  of the same backbone: measured, that fixes 8 and breaks 3 of 33, and 15 of
  the 21 shared misses are the identical wrong answer (`bench/RESULTS.md`).

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
  worlds/      exact-law generators: synthetic decisions whose answers are computed by code
tests/         85 tests; a few need a tokenizer, two are opt-in on a real GPU, two need a JevBench clone
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
   on a 2k-token document, logits within two bf16 ulps, 0/8 flips. Batched
   suffixes too: **1.9× over serial, ~15× over uncached at batch 2**; batch 8
   is slower than serial on a 10 GB card.
10. ~~Reason only when unsure.~~ Done — **it does not pay.** Routing the 30%
    least-confident hard decisions to a budget-forced 512-token thinking pass
    lifts hard 0.604 → 0.649 (8 fixed, 3 broken, p = 0.23) for ~14 s per hard
    decision, and under the composite the thinking tokens cost more than that
    earns: 74.85 → 73.62. Free thinking never terminates on this backbone at any
    cap. The finding underneath: when both modes miss, 71% of the time it is the
    *same* wrong answer — a knowledge gap, not a thinking gap.

The frozen levers have plateaued; **`v0.1.0`** freezes this state as the
baseline. What follows it, in order: **submit** (item 8 — the held-out number
is the one that counts), then **training** on the families where the model
reproduces the same wrong answer — temporal_numeric, long_policy, multi_hop —
from exact-law generators (`jobe.worlds`; temporal_numeric and long_policy are
built, multi_hop is not), LoRA on the frozen backbone, CE on the option slots plus a Brier
term, gated by E1 and `bench/gate.py` and compared to `v0.1.0` through the same
harness.

## License and attribution

**MIT** — see `LICENSE`. The decision protocol is adapted from
[TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf), also MIT; see `NOTICE`
for what was taken and what was changed.

# Jobe on JevBench public tasks

Run 2026-09-22. All 231 public tasks (easy 48, original 72, hard 111), single
declared option order, no calibration applied, RTX 3080, bf16. 231/231 returned
a usable distribution for both backbones — zero protocol failures, zero parse
failures.

**Not a JevBench score.** These are the public tasks scored locally; the
leaderboard's tiers are larger (534 decisions) and its `standard` tier is not the
same set as the `original` file used here. No rank is claimed. The comparison
that matters below is backbone-vs-backbone on identical inputs.

## The backbone was the lever

| backbone | easy | original | hard | ALL | chance |
|---|---:|---:|---:|---:|---:|
| llama-3.2-3b-instruct | 0.979 | 0.528 | 0.369 | 0.545 | 0.318 |
| **Qwen3.5-4B** | 0.979 | **0.972** | **0.613** | **0.801** | 0.318 |
| *delta* | *0.000* | ***+0.444*** | ***+0.244*** | ***+0.256*** | |

Identical protocol, identical prompts, identical scoring — only the weights
changed. **+25.6 points overall, +44.4 on the standard tier.**

The two are indistinguishable on easy (0.979 each), which is exactly why easy
tiers are useless for choosing a backbone. The separation appears where the
decision is actually hard.

By question type:

| backbone | tier | choice | noul | score |
|---|---|---:|---:|---:|
| llama-3.2-3b | original | 0.472 | 0.583 | 0.583 |
| Qwen3.5-4B | original | **1.000** | 0.917 | **1.000** |
| llama-3.2-3b | hard | 0.343 | 0.421 | 0.333 |
| Qwen3.5-4B | hard | **0.582** | **0.711** | 0.333 |

`score` (ordinal) is the one place Qwen does not pull ahead on the hard tier —
both sit at 0.333 on n=6, which is too small to read anything into.

## Against SemIf, which runs the same backbone

SemIf is #2 on JevBench (74.7) using Qwen3.5-4B frozen. Its published per-tier
accuracy against Jobe's, with the tier-mismatch caveat above:

| | easy | standard / original | hard |
|---|---:|---:|---:|
| SemIf (published) | 1.000 | 0.979 | 0.595 |
| **Jobe + Qwen3.5-4B** | 0.979 | 0.972 | **0.613** |

Close enough to call it protocol parity on the public set, with Jobe slightly
ahead on the hard tier and slightly behind on easy. That is the expected result —
it is the same protocol on the same weights — and it is the evidence that the
readout is implemented correctly.

## A single global temperature does not transfer

Fitted on `original`, reported held-out on the other two tiers:

| backbone | fitted T | tier | ECE before | ECE after | NLL before | NLL after |
|---|---:|---|---:|---:|---:|---:|
| llama-3.2-3b | 3.952 | easy | 0.081 | **0.361** | 0.116 | 0.502 |
| llama-3.2-3b | 3.952 | hard | 0.402 | **0.172** | 2.196 | 1.221 |
| Qwen3.5-4B | 0.348 | easy | 0.016 | **0.010** | 0.020 | 0.014 |
| Qwen3.5-4B | 0.348 | hard | 0.133 | **0.281** | 0.950 | 1.738 |

**Each temperature helps one tier and hurts the other, and the two backbones need
opposite corrections.**

The mechanism is visible in the raw numbers. Qwen on `original` is 97.2% accurate
at 0.940 confidence — *under*-confident, so the fit sharpens (T = 0.348 < 1). On
hard it is 61.3% accurate at 0.708 confidence — *over*-confident, which needs
T > 1. One scalar cannot move in both directions, so fitting on the middle tier
sharpens a tier that was already too sharp. llama shows the mirror image: fitted
T = 3.95 rescues hard (ECE 0.402 → 0.172) by wrecking easy (0.081 → 0.361).

This matters for any plan that specifies "one fitted temperature". It is the
right instrument, but a **single global** scalar is the wrong shape whenever
accuracy varies sharply with difficulty — which is the defining property of a
tiered benchmark. Options: fit per tier, fit against a difficulty estimate, or
fit on a mixture that matches production traffic rather than on one tier.

Note Qwen's raw ECE is already excellent where it is competent — **0.016 on easy,
0.133 on hard, untouched**. Calibration may be a smaller problem than expected
for this backbone; the honest move is to measure before correcting.

## Latency

| backbone | easy | original | hard | ALL p50 |
|---|---:|---:|---:|---:|
| llama-3.2-3b | 40.1 ms | 39.9 ms | 165.9 ms | 46.0 ms |
| Qwen3.5-4B | 101.7 ms | 103.6 ms | 213.2 ms | 111.1 ms |

Qwen is ~2.4× slower: 4.66B against 3.2B parameters, and a 248k vocabulary
against 128k, which the LM head pays for. Hard-tier prompts are much longer,
which is where both rise.

Applying the leaderboard's published speed formula to Jobe's 111.1 ms p50,
*including* its ×2 + 0.15 s self-hosted penalty, gives **S ≈ 88.6** — against
SemIf's published 83.7 on the same backbone. That is a computed figure from our
measurement, not a measured leaderboard entry, and the hardware differs.

## Control battery — Qwen3.5-4B passes all three, on every tier

`bench/gate.py`, a port of EveryAppKit's `tools/decisionGate.ts`. That tool
drives an OpenAI-compatible HTTP endpoint; routing a local model through one
would reintroduce the top-N logprob window Jobe exists to remove, so the
controls are reimplemented here against the full-vocabulary readout. Criteria
are identical, deliberately.

| tier | acc | E1 acc | chance | skill kept | conf | E1 conf | Δ | flip | TV | pos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| easy | 0.979 | 0.292 | 0.284 | **1%** | 0.984 | 0.767 | −0.217 | **2.1%** | 0.027 | 29% |
| original | 0.972 | 0.278 | 0.311 | **−5%** | 0.940 | 0.847 | −0.093 | 8.3% | 0.116 | 33% |
| hard | 0.613 | 0.306 | 0.336 | **−11%** | 0.708 | 0.688 | −0.020 | 23.4% | 0.168 | 37% |

**GATE PASSED.**

**E1 is emphatic, including on hard.** Shuffled evidence destroys essentially all
above-chance skill on every tier — and on `original` and `hard` it lands *below*
chance, which is the correct behaviour for a model that genuinely reads the
evidence: given evidence that points elsewhere, it is actively misled rather
than falling back on priors.

The hard tier is the one that matters. Measured earlier through the API path,
gemma4-8B's hard accuracy did not move *at all* under E1 (0.297 → 0.297) because
it was emitting the letter "A" every time. **Qwen3.5-4B drops 0.613 → 0.306.**
That is the difference between a decision and a reflex, and only E1 can see it.

**E2 shows real order stability**, which is not a given:

| system | hard-tier flip rate | mean TV |
|---|---:|---:|
| gemma4-8B (via API) | 100.0% | 0.967 |
| llama-3.2-3b (via API) | 61.3% | 0.515 |
| **Qwen3.5-4B (here)** | **23.4%** | **0.168** |
| Verdict-open-jev, fine-tuned ModernBERT (published) | 3.0% | 0.034 |

On the easy tier Qwen's 2.1% flip / 0.027 TV actually **beats the purpose-trained
decision model's published figure**. Stability degrades with difficulty, and
23.4% on hard is close enough to the 25% threshold to watch.

**Positional bias is mild and passing** — 29 / 33 / 37% concentration against
~25 / 26 / 29% expected, versus gemma4's 100%.

### This answers the order-averaging question

Averaging costs one forward pass per extra order and pays in proportion to the
fragility it fixes. The flip rates say where that is:

- **easy (2.1%) and original (8.3%): not worth it.** There is almost nothing to
  repair, and on a stable model averaging measured a *loss* elsewhere.
- **hard (23.4%): worth testing**, and only here.

So the plan is not "average everything" but "average the hard tier", which is
also where the latency budget can least afford it — a real trade to make
deliberately rather than by default.

### One honest caveat on confidence

Confidence does fall on irrelevant evidence, so unlike gemma4 (whose confidence
*rose*) the score carries some signal. But **the drop shrinks exactly where it
would be most useful**: −0.217 on easy, −0.093 on original, **−0.020 on hard**.
On the hard tier confidence is essentially flat between real and nonsense
evidence. The structural caveat stands — validate evidence before the call.

## What this does not cover

- **Order averaging is still unrun as a scoring mode.** E2 measured the flip
  rate, but no result above uses an averaged distribution — the accuracy table's
  `flip` column is 0.0% by construction because that run used a single order.
- **The control battery covers Qwen3.5-4B only.** llama-3.2-3b has not been
  gated here; its API-path numbers are from a different run and a quantised
  build, so they are context rather than a like-for-like comparison.
- **Public tasks only** (231 of 534), and the held-out half is where a
  leaderboard result would actually be decided.

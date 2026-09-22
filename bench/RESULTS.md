# Jobe on JevBench public tasks

Run 2026-09-22. All 231 public tasks (easy 48, original 72, hard 111), single
declared option order, no calibration applied, RTX 3080, bf16. 231/231 returned
a usable distribution for both backbones — zero protocol failures, zero parse
failures.

**Not a JevBench score.** These are the public tasks scored locally; the
leaderboard's tiers are larger (534 decisions) and its `standard` tier is not the
same set as the `original` file used here. No rank is claimed. The comparison
that matters below is backbone-vs-backbone on identical inputs.

## Through the official harness — the numbers that count

Everything above this section was measured with Jobe's own runner. This section
is the same 231 public tasks driven by **JevBench's `Runner`** via
`bench/jobe_direct.py`, scored by **JevBench's `summarize()` and
`composite_v12`** unmodified. Artefacts: `bench/runs/2026-09-22-public231/`.

**Conformance: 231/231 attempted, operational success 1.000, schema validity
1.000, strict 1.000, 0 renormalised.** Every distribution Jobe returned was
inside the pre-registered tolerance, not merely the headline one. This is the
spec's Step 1 gate — *"all 231 items through the official MIT harness, zero
parse failures"* — passed at its strictest reading.

| tier | n | accuracy | ECE | Brier | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|
| easy | 48 | **1.000** | 0.016 | 0.011 | 0.108 s | 0.144 s |
| standard (`original`) | 72 | **0.986** | 0.046 | 0.036 | 0.107 s | 0.125 s |
| hard | 111 | **0.604** | 0.133 | 0.500 | 0.224 s | **3.253 s** |
| judge | — | *not public* | | | | |
| **all** | 231 | **0.805** | 0.049 | 0.254 | | |

Also from their summary: **paraphrase consistency 0.972** over 36 pairs (a
metric never computed here before), macro accuracy 0.804, ordinal MAE 0.256.

**Run-to-run jitter, stated.** Each tier differs from the earlier own-runner
pass by exactly one task (48 vs 47, 71 vs 70, 67 vs 68). Same weights, same
prompts, same device; bf16 CUDA kernels are not bitwise deterministic and
near-tie decisions flip. Treat any single-run tier accuracy as ±1 task on top of
its sampling interval. The two runners agreeing to within that is also the
evidence that the own-runner numbers above were sound.

### Axes, computed by `composite_v12`

| axis | score | how |
|---|---:|---|
| Intelligence | **82.9** | partial run: judge absent, their renormalisation → easy 0.19 / standard 0.39 / hard 0.42 |
| Calibration | **71.7** | mean of ECE term 73.3 (hard ECE 0.133) and fidelity term 70.1 (mean TVD 0.299 over 10 gold-distribution items) |
| Speed | **88.4** | standard-tier p50 0.107 s / p95 0.125 s, endpoint `gpu` → ×2 + 0.15 s |
| Cost | *assumed* | not measurable locally; see below |

Two readings of those, both honest:

- **The Intelligence figure is likely pessimistic.** Leaving judge out
  renormalises hard from 30% to 42% of the axis, and hard is the weakest tier.
  If judge behaved like standard for Jobe (SemIf scores 95.2% there),
  Intelligence would be ≈ 86. It is not measured, so 82.9 stands — but the
  direction of the bias is known.
- **The fidelity half of Calibration costs about 3 points, not 30.** The feared
  drag from mean TVD 0.299 lands the axis at 71.7 against an ECE-only 73.3 —
  level with SemIf's published 72.6.

### The composite

Their `cost()` refuses a missing price and the geometric mean floors a missing
axis at 1, so cost cannot be omitted. Benchmark Heaven prices self-hosted
entrants at the hosted-provider reference rate for the weight class; SemIf, the
same 4B class, is listed at ~$0.022 per 1,000 decisions. That is the defensible
assumption, shown with its sensitivity:

| assumed $/1k decisions | Cost axis | **JevBench Score** |
|---:|---:|---:|
| 0.010 | 70.0 | 77.9 |
| **0.022 (SemIf's class)** | **59.7** | **74.9** |
| 0.040 | 51.9 | 72.3 |

For reference, published: **Jev 1.13.0 75.4 · SemIf 74.7** (I 85.9, C 72.6,
S 83.7, K 59.5) · **#10 68.9**.

Under the same-class price, Jobe scores **74.9 — within 0.2 of SemIf and 0.5 of
Jev.** Across a 4× price range it stays between 72.3 and 77.9, all of which is
above the #10 line. That is the strongest statement the evidence supports, and
these are its conditions: a partial run with no judge tier, standard
approximated by the 72-task cohort, speed on standard alone, cost assumed, and
the held-out half unseen. Public-set placement is a floor.

**One tail risk worth naming:** hard-tier p95 latency is **3.25 s** — the long
policy documents. Speed is scored on standard+judge, not hard, so it does not
enter the axis here; but if judge tasks are long documents, that tail would.
Eager attention is O(n²) in prompt length; SDPA is the lever if it bites.

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

**These are not distinguishable, and the comparison is weaker than it looks.**
Jobe's hard figure is over the **111 public** hard tasks; SemIf's published
figure is over the **full 220**. A two-proportion test on 0.613 vs 0.595 gives
z = 0.32, p ≈ 0.75, and Jobe's own 95% interval is [0.520, 0.698] — ±8.9 points.
"Slightly ahead on hard" is not a supportable claim and an earlier version of
this document made it.

What the numbers *do* support: Jobe is **in SemIf's band**, which is the expected
result for the same protocol on the same weights, and is evidence the readout is
implemented correctly rather than evidence of an improvement. Note also that
`original` (72 tasks, the published Jev cohort) is **not** the leaderboard's
`standard` tier (96 tasks), so that column is not a like-for-like comparison
either.

## A single global temperature does not transfer

Fitted on `original`, reported held-out on the other two tiers:

| backbone | fitted T | tier | ECE before | ECE after | NLL before | NLL after |
|---|---:|---|---:|---:|---:|---:|
| llama-3.2-3b | 3.952 | easy | 0.081 | **0.361** | 0.116 | 0.502 |
| llama-3.2-3b | 3.952 | hard | 0.407 | **0.163** | 2.196 | 1.221 |
| Qwen3.5-4B | 0.348 | easy | 0.016 | **0.010** | 0.020 | 0.014 |
| Qwen3.5-4B | 0.348 | hard | 0.106 | **0.272** | 0.950 | 1.738 |

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
0.106 on hard, untouched**. Calibration may be a smaller problem than expected
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

### Order averaging on the hard tier: measured, and it buys nothing

The hard tier was the one place the flip rate suggested averaging might pay.
It does not.

| strategy | n | acc | Δacc | ECE | Δece | Brier | conf |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 order (declared) | 111 | 0.613 | — | 0.106 | — | 0.500 | 0.708 |
| 2 orders (declared + reversed) | 111 | **0.613** | **+0.000** | 0.101 | −0.005 | 0.495 | 0.684 |

**Zero accuracy change, and ECE improves by 0.005** — for double the forward
passes. (The wall-clock multiple this run reported is not quotable: it executed
while another process held the GPU, so its latency is polluted. The *nominal*
cost is 2× passes.)

The mechanism is worth stating exactly, because it is not what the flip rate
implied:

- 22.5% of tasks flip between the two orders — which agrees with E2's
  independently measured 23.4%, a useful cross-check on both.
- Averaging changed the final answer on **11 of 111** tasks.
- Of those 11: it **fixed 3, broke 3**, and left 5 neutral.

**Averaging is a coin flip here.** The instability is real but *symmetric* — it
is noise, not a bias pointing consistently at the wrong answer. Averaging can
only help when the instability is skewed toward errors, and it is not.

### The correction this forces

I claimed earlier — in the E2 write-up and in `decisionGate`'s own
documentation — that measuring the flip rate tells you whether order averaging
is worth paying for. **That is wrong.** The flip rate measures *instability*; it
says nothing about *correctability*. A 23% flip rate is equally consistent with
"averaging recovers half of those" and "averaging fixes as many as it breaks",
and only a fixed-versus-broke count distinguishes them.

The flip rate is still a valid control — a model whose answer moves under a
no-op perturbation is worth knowing about, and at 100% it exposed a degenerate
reflex. It is just not a proxy for the value of averaging.

The earlier +8.2-point gain from averaging on gemma4-8B is consistent with this:
there the instability *was* biased, because a position-0 collapse puts a
different arbitrary option first in each order, so averaging recovered signal
from a model that had none. That is repairing a broken readout, not improving a
working one — and Qwen3.5-4B is a working one.

### What the flip rate said before this measurement

Averaging costs one forward pass per extra order and pays in proportion to the
fragility it fixes. The flip rates say where that is:

- **easy (2.1%) and original (8.3%): not worth it.** There is almost nothing to
  repair, and on a stable model averaging measured a *loss* elsewhere.
- **hard (23.4%): worth testing**, and only here.

That reasoning pointed at the hard tier, and the measurement above then showed
even the hard tier does not pay. The conclusion stands but for a different
reason than the flip rates suggested: **order averaging is not worth it for
Qwen3.5-4B on any tier.**

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

## Methodology, and its limits

What holds up:

- JevBench's own task files, mapped exactly as `jevbench/adapters/semif_direct.py`
  maps them, so the prompts are the reference adapter's.
- Every result **per tier**, never pooled — the aggregate is what hides a
  degenerate tier.
- **My metrics now provably reproduce JevBench's own**: ECE and accuracy agree to
  0.0000 on all three tiers against `jevbench/metrics.py` unmodified. Getting
  there took two fixes, both found by checking rather than assuming — see below.
- GPU contention caught twice and the polluted timings discarded rather than
  published.
- One independent cross-check: `orders.flip_rate` (22.5%) agreed with E2's
  separately computed 23.4%.
- The composite formula verified by reproducing SemIf's published 74.7 from its
  own published axes.

Two bugs the cross-check against their code exposed, after these numbers were
first written up:

1. **Ordinal questions were scored by argmax.** A `score` answer is the rounded
   probability-weighted value, which need not be the highest-probability level.
   `calibration_report` now takes an explicit `predictions` list. This moved
   hard-tier accuracy by one task and ECE by 0.027.
2. **"Confidence" was ambiguous.** I used the probability *of the prediction*;
   JevBench's `ece_top_label` uses the **top-label** probability. Identical for
   classification, divergent for ordinal. Pinned to their convention.

What still limits every number here:

- **Public tasks only** — 231 of 534. Worse for comparisons: the leaderboard's
  published per-tier figures are over the **full** tiers (hard = 220), so even
  the hard-tier comparison is 111 tasks against 220.
- **`original` is not `standard`.** 72 tasks (the published Jev cohort) versus
  their 96. Do not read that column across.
- **No judge tier at all**, which is 28% of the leaderboard's Intelligence axis.
  A composite score therefore cannot be computed, only guessed at.
- **Single run, single seed.** E1's derangement used seed 42 once; nothing is
  repeated. At n=111 a hard-tier accuracy carries ±8.9 points at 95%, so
  differences smaller than about 10 points are not measurements.
- **Calibration is half-measured** — the ECE half only. The probability-fidelity
  component (total variation against exact gold distributions on 20 items) is
  not computed.

The backbone result survives all of this comfortably: 0.801 vs 0.545 on the same
231 tasks is z = 5.9, p < 1e-10. The fine-grained comparisons against published
entrants do not.

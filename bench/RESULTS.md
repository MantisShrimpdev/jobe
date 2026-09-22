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

> **Rule change, 22 Sep 2026 — JevBench v1.3.0** scores Intelligence as chance-corrected accuracy per tier
> (`(acc − chance)/(1 − chance)`, tier chances from their published option-count histograms: easy 0.284,
> standard 0.317, hard 0.336, judge 0.292) and multiplies the score by `(I/50)²` below I = 50. Everything
> below was computed under v1.2. Re-scored with their `composite_v13`: I 74.3 (hard tier 40.3 chance-corrected),
> **JevBench Score 72.8** at the same $0.022 assumption (73.5 at our measured tokens), against SemIf 73.1,
> Jev 74.4, djev 73.0, #10 66.6. Same band, same story: the hard tier is the whole gap.

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

## Reason only when unsure — measured, and it does not pay

The OODA-shaped idea: readout first, and route only the decisions the readout
is unsure about to a thinking pass. Three questions in order, each able to end
the experiment early. Scripts: `bench/uncertainty_analysis.py`,
`bench/reason_route.py`, `bench/route_composite.py`; artifacts in
`bench/runs/2026-09-22-reason-route/`.

### Q1 — can the readout's uncertainty find its own wrong answers? Yes.

Pure analysis over the official-harness results. On the hard tier, 1 − top
probability detects a wrong answer with **AUROC 0.787** (margin 0.750, entropy
0.779); accuracy by confidence quartile, low to high, is **0.148 / 0.667 /
0.741 / 0.833**. Routing the least-confident 30% (confidence ≤ 0.573) would
catch 26 of the 44 hard-tier misses; on the standard tier the AUROC is 0.972.
The caveat above — confidence barely moves under nonsense evidence on hard —
does not bite here: that is about *bad input*, this is about *wrong answers on
good input*, and the two are different questions.

### Q2 — does a thinking pass beat the readout on those decisions? Barely, and not significantly.

Same backbone, same system prompt and options, `enable_thinking=True`, the
model card's thinking recipe (temperature 1.0, top-p 0.95, top-k 20, presence
penalty 1.5 as a logits processor). **Free thinking never terminates:** greedy
and the recipe both ran to a 2,048-token cap without emitting `</think>`,
looping, on both diagnostic tasks — one greedy trace had already reached the
right conclusion ("Therefore, the $15,000 sublimit applies") and could not
stop. So the pass is **budget-forced**: 512 thinking tokens, then `</think>` is
injected and the answer letter is read greedily (`reason_diag.out`).

On the 33 least-confident hard tasks (the 30% cell): readout **7/33 = 0.212**,
thinking **12/33 = 0.364**. **8 fixed, 3 broken, 4 kept — and 15 of the 21
shared misses are the identical wrong answer.** Sign test on the 11 discordant
pairs: two-sided p = 0.23. Mean 45.3 s per routed task (36–77 s) for 512
tokens — 3.9–14 tok/s on this card, prompt-length dependent; 0/33 closed the
think on its own.

| family | n | readout right | routed right |
|---|---:|---:|---:|
| temporal_numeric | 10 | 1 | 3 |
| long_policy | 8 | 2 | 4 |
| multi_hop | 6 | 0 | 1 |
| probability | 3 | 2 | 2 |
| tradeoff | 3 | 1 | 0 |
| adversarial | 1 | 0 | 1 |
| ambiguous | 1 | 0 | 1 |
| judge_hard | 1 | 1 | 0 |

### Q3 — the routing curve

Hard tier (n = 111), readout for the confident rest:

| routed | tasks | hard acc | Δ | mean s per hard decision |
|---:|---:|---:|---:|---:|
| 0% | 0 | 0.604 | | 0.9 |
| 10% | 11 | 0.631 | +0.027 | 5.5 |
| 20% | 22 | 0.640 | +0.036 | 9.9 |
| 30% | 33 | 0.649 | +0.045 | 14.4 |

### What the composite does with it

Two facts from the harness (`composite_v12.py`, `results/v1.2`). **Speed is
the standard+judge run only** — hard-tier latency never enters it, and at this
cutoff the router touches 2 of 72 standard tasks, both already right, so p95
is untouched: the latency above is free in the score. **Cost is token-priced**
at the 4B reference tariff, $0.03/M in and **$0.15/M out**, with hard 220/534
of the average — thinking tokens land on the axis the geometric mean punishes
hardest. With the measured +0.045 (`route_composite.py`, harness functions):

| think tokens | usd / 1,000 | K | JevBench Score | 60:20:20 accuracy view |
|---:|---:|---:|---:|---:|
| 0 (readout) | 0.0220 | 59.7 | 74.85 | 78.66 |
| 256 (not run; assumes the gain survives) | 0.0272 | 56.9 | 74.38 (−0.47) | 78.97 (+0.31) |
| 512 (measured) | 0.0325 | 54.7 | 73.62 (−1.23) | 78.32 (−0.34) |

Break-even on the headline score needed hard +0.185 at 512 tokens or +0.097 at
256; measured +0.045. This is GPT-5.6 Luna's placement in miniature — I 96.8,
K 28.5, #20 — the score is built so accuracy cannot buy back cost.

### Reading

The identical-miss rate is the finding. When both modes are wrong, **71% of
the time it is the same wrong answer**, and every temporal_numeric miss
reproduced the readout's exact number. For this backbone the hard-tier gap is
mostly not a reasoning-depth problem; it is what the model knows or can
compute at all — the case for training, not for a longer think. Routing stays
a product option (+4.5 points on hard for ~14 s per hard decision, when
accuracy matters more than the bill), not a leaderboard move, and it is not
wired into `jobe_direct.py`.

## Training, first full run — it does not ship (2026-09-22)

TheLab's loop (`thelab.decisions.train`): LoRA r=16 on the attention projections
of the frozen Qwen3.5-4B (3.1 M trainable of 4.2 B), loss = CE on the option
slots + 0.5 × Brier, one epoch over the three exact-law families at a
2,048-token cap — 3,646 records (all of temporal_numeric, 35 % of long_policy,
67 % of multi_hop; 1,347 dropped over the cap, never truncated), 407 held out,
456 optimiser steps, 1 h 47 min on the RTX 3080 with flash-linear-attention
kernels. Best checkpoint by held-out NLL: step 300. Artifacts:
`bench/runs/2026-09-22-lora-3fam-v1/` (training summary and manifest, the
harness results and score, the gate report); the adapter weights themselves
stay out of git (`runs/lora-3fam-v1/best/`, 12 MB).

**On its own held-out worlds it learned a great deal:**

| step | acc | NLL | Brier | ECE |
|---:|---:|---:|---:|---:|
| 0 (v0.1.0) | 0.295 | 1.567 | 0.824 | 0.228 |
| 100 | 0.565 | 1.007 | 0.533 | 0.074 |
| **300 (kept)** | **0.619** | **0.816** | **0.461** | **0.056** |
| 456 | 0.582 | 0.841 | 0.479 | 0.099 |

**On JevBench it did not transfer, and it cost the untrained families** — the
official harness on the same 231 public tasks, step-300 adapter merged into
the weights (`JOBE_LORA_DIR`), gate passed on every tier:

| tier | n | v0.1.0 | adapter | Δ | ECE v0.1.0 → adapter |
|---|---:|---:|---:|---:|---:|
| easy | 48 | 1.000 | 1.000 | 0 | 0.016 → 0.022 |
| standard | 72 | 0.986 | 0.958 | **−2** | 0.046 → 0.062 |
| hard | 111 | 0.604 | 0.586 | **−2** | 0.133 → 0.128 |
| all | 231 | 0.805 | 0.788 | −4 | 0.049 → 0.049 |

By hard-tier family the shape is exactly what one epoch on three families
should produce: the trained families moved up — temporal_numeric 3/15 → 5/15,
multi_hop +1, probability +1, tradeoff +1 — and the untrained ones paid for
it: **judge_hard 13/17 → 9/17 (−4)**, ambiguous −2, long_policy −1; on the
standard tier adequacy −2 and intent −1. 14 hard items fixed, 16 broken.
Calibration improved (axis 71.7 → 74.9; hard ECE 0.133 → 0.128), which is the
Brier term working as designed. Under v1.3 the composite reads 73.3 against
72.8 — a tick up that comes entirely from calibration and a run-to-run speed
difference, while Intelligence fell 74.3 → 71.6.

**Decision: v0.1.0 stays the shipped readout.** The rule was "ship only if
hard rises without easy or standard falling"; hard did not rise and standard
fell. The adapter is kept as a measurement, not a release.

**What it says.** The worlds teach the habits they contain, and the model
learned them — but the JevBench hard tier is mostly *other* families
(judge_hard, ambiguous, trap, adversarial) that a LoRA on eight attention
layers forgets while it learns ours. The fix is not more epochs; it is
anchoring: a KL-to-the-frozen-base term on decisions the base already answers
well, so learning the three families cannot move the rest. That is the next
change to the loop, and the next run is judged the same way.

## The uncertain band is where everything happens (2026-09-23)

Splitting the v0.1.0 public run by the readout's own top probability, and then
asking where the lora-3fam-v1 adapter actually changed an answer:

| baseline confidence | n | share | accuracy | answer moved | fixed | broken |
|---|---:|---:|---:|---:|---:|---:|
| 0.00 – 0.45 | 15 | 6% | 0.067 | 9 (60%) | +6 | −0 |
| 0.45 – 0.85 | 70 | 30% | 0.643 | 29 (41%) | +9 | **−17** |
| 0.85 – 1.00 | 146 | 63% | 0.959 | 2 (1%) | +0 | −2 |

"Answer moved" counts the decisions whose *prediction* changed, not just the
ones whose correctness did: a wrong answer replaced by a different wrong answer
leaves both accuracy columns untouched and is still the model being unstable.
By that measure the adapter reached 60% of the bottom band and 41% of the
contested one and **1% of the confident one**.

Three populations, not one. Two tasks in three come back above 0.85 and are
right 96% of the time; a LoRA on eight attention layers moved 1% of them and
a thinking pass would be spending its budget on settled questions. One task in
six-and-a-half comes back below 0.45, where the model is right once in fifteen
against the 3.33 that uniform guessing over those menus would give. That gap
does not clear significance on fifteen items — P(X ≤ 1) under uniform guessing
is 0.12 — so it is a direction, not a finding; what *is* solid is that the band
is near or below chance and nothing in it is being answered. Everything
contested lives in between: 30% of the benchmark, 64% accurate.

**The training run's whole −4 is the middle band.** Below 0.45 it fixed six and
broke none, because there was nothing there to break. Above 0.85 it was a
non-event. Between them it fixed nine and broke seventeen. "It did not
transfer" was the right verdict and the wrong description: it transferred
precisely where the base was already lost, and it churned where the base was
marginally right.

This is the same 30% the routed-reasoning experiment spent its budget on, and
it is the band any future intervention should be judged on. `bench/confidence_bands.py`
produces the table.

### Where the correct answer actually sits

Joining the run to the task file gives the rank the correct option held in each
decision's distribution — not just whether it won:

| baseline confidence | n | correct is #1 | inside the top two | gold not in the top two |
|---|---:|---:|---:|---:|
| below 0.45 | 15 | 0.067 | 0.333 | 10 |
| 0.45 – 0.85 | 70 | 0.614 | **0.871** | 9 |
| above 0.85 | 146 | 0.959 | 0.993 | 1 |

**In the contested band the answer is already in hand 87% of the time.** What
is failing is not retrieval, it is the ordering of two candidates — and the gap
between 0.614 and 0.871 is 18 decisions, larger than anything training or
thinking has moved.

One tempting reading is wrong and worth recording so it is not retried: below
0.45 the correct option is not systematically the *second* choice. Its ranks
there are #1 once, #2 four times, #3 six times, #4 three times, #5 once — taking
the second choice instead would score 4 of 15 against the 3.3 uniform guessing
expects. There is no free inversion at the bottom.

The top-two result is a different matter, and `bench/runoff.py` tests it: re-ask
each contested decision with only its two candidates. Renormalising the existing
distribution over those two changes nothing, because the ordering is the
ordering — the runoff is a fresh prompt, which is a different computation, and
there is a measured reason to expect it to help. Menu size is this family's
documented weakness, so a five-way question asked as a two-way question is being
asked in the shape the model is best at.

Each pair is scored in **both orders** and averaged. The two candidates arrive
sorted by the first pass, so prompting them that way would hand the previous
winner slot A every time, and this backbone has a measured position bias
(hard-tier concentration 0.396 against the 0.294 its menus imply). A runoff
built that way would mostly confirm itself and look like a result. Two short
forward passes on the third of decisions that are contested.

**Registered before it ran.** On the 39 contested choice decisions the first
pass scores 0.667 and the ceiling — how often the gold is in the top two at all
— is 0.846, so the headroom is 7 decisions. On the 14 in the bottom band it is
0.071 against a 0.357 ceiling, or 4 decisions. I expect **a small gain, +0.00
to +0.08 on the contested band and near zero at the bottom**, and the two
orders to disagree on 15–30% of pairs. The reason to expect little: the model
ranked these two the way it did by reading the evidence, and a shorter menu
does not change the reading. A 512-token thinking pass is a far heavier
perturbation than a re-ask and it left 15 of 21 shared misses on the *identical*
wrong answer. If thinking barely moved this model, a re-ask should move it
less. The order-disagreement rate is the number I am least sure of and the one
most worth having.

### The top-two runoff — measured, and it measures position, not judgement

| band | n | first pass | runoff | Δ | fixed | broken | ceiling |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.45 – 0.85 | 39 | 0.667 | 0.667 | **+0.000** | 2 | 2 | 0.846 |
| 0.00 – 0.45 | 14 | 0.071 | 0.214 | **+0.143** | 2 | 0 | 0.357 |

Seven recoverable decisions in the contested band and it recovered none of them
net. Four in the bottom band and it took two, breaking nothing.

**What the two orders reveal is worth more than the headline.** Scoring each
pair both ways was a guard against a biased design; it turned into the result:

| condition | re-elects the first pass | accuracy |
|---|---:|---:|
| previous winner shown **first** | 71.8% | 0.590 |
| previous winner shown **second** | 92.3% | 0.667 |

The model prefers whatever sits in the second slot by about twenty points, in a
prompt with **two** options and nothing else to distract it. The
winner-shown-second column is simply the first pass re-elected, which is why it
inherits its 0.667; the winner-shown-first column is the position bias
overriding a judgement that was right two times in three. The orders disagree
on 25.6% of contested pairs and 35.7% of bottom-band ones.

So a narrower menu did not make the model read the evidence again. It made the
slot matter more. Menu size is a real weakness, but shrinking the menu is not
the lever — the decision was already made before the runoff was asked.

**Three interventions, one law.** A LoRA adapter, a 512-token thinking pass and
a two-option re-ask share no mechanism, and they behave identically across the
bands:

| intervention | below 0.45 | 0.45 – 0.85 |
|---|---|---|
| LoRA, three exact-law families | +6 / −0 | +9 / **−17** |
| 512-token thinking pass | +4 / −0 | +4 / −3 |
| top-two runoff | +2 / −0 | +2 / −2 |

Everything helps where the model is lost and nothing helps where it is
undecided. The bottom band is 6% of the benchmark and cheap to recover. The
contested 30% has now resisted three different attacks, and the one that tried
hardest did the most damage.

**Scoring the prediction.** Registered before the run: +0.00 to +0.08 on the
contested band (got +0.000, at the boundary), near zero at the bottom (**wrong**
— got +0.143, two of the four available), 15–30% order disagreement (got 25.6%
and 35.7%). The reasoning I gave for the fix was also wrong in its direction: I
argued that always placing the previous winner first would make the runoff
"mostly confirm itself", and in fact that placement *disagrees* most. The fix
was still necessary; a single-order runoff would have reported −0.077 or
+0.000 depending purely on which slot the incumbent got.

### The same band re-decides the routed-reasoning result

The routed pass was judged as one rule — think on the bottom 30% of hard tasks
by confidence — and it did not pay. Split by the same bands, it has the same
signature as the training run:

| baseline confidence | n | readout | after thinking | fixed | broken |
|---|---:|---:|---:|---:|---:|
| 0.00 – 0.45 | 15 | 0.067 | 0.333 | +4 | **−0** |
| 0.45 – 0.85 | 18 | 0.333 | 0.389 | +4 | −3 |

Thinking is free of risk below 0.45 and a coin toss above it — the same split a
LoRA produced by a completely different mechanism. The rule that was tested
routes both halves. An absolute threshold at 0.45 routes 15 of 111 hard tasks
instead of 33, and **zero easy and zero standard tasks**, which matters because
Speed is scored on the standard and judge tiers only: the tested rule spilled
onto 2 standard decisions and this one spills onto none.

Repricing both through the harness's own axis functions, at the budget that was
actually measured (512 tokens):

| rule | routed | hard Δ | headline | vs baseline | 60:20:20 | vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| baseline, no thinking | — | — | 74.85 | 0 | 78.66 | 0 |
| bottom 30% quantile | 33 hard + 2 std | +0.045 | 73.64 | −1.22 | 78.33 | −0.33 |
| **confidence < 0.45** | **15 hard** | **+0.036** | **74.45** | **−0.40** | **78.89** | **+0.23** |

Same measurement, same budget, a different threshold: the headline loss shrinks
from −1.22 to −0.40 and the accuracy-weighted view turns from −0.33 to +0.23.
It gives up one of the five net fixes and 55% of the bill. **"Reason when
unsure" did not fail because thinking does not help. It failed because the rule
was routing twice as many decisions as it should have**, and half of those were
ones the readout was already getting right.

At a 256-token budget the same arithmetic gives −0.05 headline and +0.53 on the
accuracy view, which would make it free — but that row assumes the gain
survives halving the budget, and that is not yet measured. `bench/route_composite.py`
takes the rule as arguments now, so any threshold can be priced.

### A prediction, written before the anchored run was scored

The anchored run (below) was launched before this analysis existed, so the
following is a real forecast rather than a description. A KL term costs the
most where the base distribution is sharp and the least where it is flat —
moving a 0.99 is expensive, moving a 0.65 is cheap. The damage above is
concentrated at 0.45–0.85. **The anchor is therefore structurally weakest
exactly where the problem is**, and the rehearsal share (30% of the gradient
now coming from general multiple-choice rather than worlds) is likely to do
more of the work than the KL term.

So: the bottom band's +6 should mostly survive; the middle band should improve
from −8 net but not reach zero (predict −5 to 0); standard should recover most
of its −2; hard should land between −1 and +2; and held-out world accuracy
should come in *below* lora-3fam-v1's 0.619, because 30% of the same step
budget was spent on rehearsal. Under that forecast the ship rule — hard rises
with easy and standard flat — is more likely to be missed than met.

## Letter readout vs option-text readout — letters win where both apply (2026-09-23)

`jobe.textscore.score_text` scores each option's own id as a continuation of the
prompt instead of reading one answer letter, which removes the sixteen-option
ceiling the letter protocol inherits from having sixteen letters. Whether it
should therefore become the default is a separate question, and this is the
measurement: same backbone, same 231 tasks, same option mapping, both readouts
in one process off one model load, length normalisation on.

| tier | n | letter | text | Δ | letter ms | text ms |
|---|---:|---:|---:|---:|---:|---:|
| easy | 48 | 0.979 | 0.979 | +0.000 | 117 | 232 |
| standard | 72 | 0.972 | 0.931 | **−0.042** | 109 | 166 |
| hard | 111 | 0.613 | 0.622 | +0.009 | 1046 | 1271 |
| all | 231 | 0.801 | 0.792 | −0.009 | 561 | 710 |

They agree on 204 of 231 (88.3%); text fixes 11 and breaks 13, and no task
failed to score. The standard-tier losses are all `choice` items and all in
routing and intent, where the option ids are words of unequal token length and
the length normalisation is doing real work. Text is also the less decisive
readout — mean top probability 0.768 against 0.838 — which matters because
calibration is a scored axis.

**Letters stay the default; text stays the fallback above sixteen options.**
Note what this does and does not establish: every task here has between two and
six options, so the comparison is entirely inside the range where the letter
readout applies. It says letters are better *where both work*. It says nothing
about the regime text exists for, which no public JevBench task reaches.

## Training, second full run — every training metric improved and every benchmark metric got worse (2026-09-23)

The first full run's write-up named the fix: "a KL-to-the-frozen-base term on
decisions it already answers well, so learning the families cannot move the
rest." This is that run. TheLab's loop gained `--anchor-weight` (forward KL from
the frozen base over the restricted options, reference precomputed before LoRA
is applied), `--anchor-families` and `--anchor-correct-only`. The mixture was
70% worlds and 30% MMLU rehearsal, **no JevBench items** — training on the
public split is allowed and declared by others, but every number here was
measured on it — matched to lora-3fam-v1 on *encoded* rows so both runs take the
same 455 optimiser steps at the same 2,048-token cap. 787 of the 1,105
rehearsal records carried an anchor, being the ones the base already answered
correctly. 1 h 53 min, best checkpoint step 400.

**On the training side it beat the first run on every axis:**

| | anchored (step 400) | lora-3fam-v1 (step 300) |
|---|---:|---:|
| held-out worlds | **0.650** | 0.619 |
| gain from its own baseline | **+0.368** | +0.324 |
| held-out NLL | **0.782** | 0.816 |
| held-out ECE | **0.022** | 0.056 |
| MMLU rehearsal | **0.685 → 0.730** | not in the mixture |
| order flip rate (gate) | **0.095** | 0.113 |
| position concentration (gate) | **0.351** | 0.368 |

It learned the worlds faster with 30% fewer world records, ended better
calibrated, raised general ability rather than merely holding it, and became
more stable under option reversal. The anchor did exactly what it was built to
do. The gate passed.

**On JevBench every one of those gains reversed:**

| | v0.1.0 | 3fam | anchored |
|---|---:|---:|---:|
| easy (48) | 1.000 | 1.000 | 1.000 |
| standard (72) | 0.986 | 0.958 | 0.958 |
| hard (111) | 0.604 | 0.586 | **0.550** |
| all (231) | 0.805 | 0.788 | **0.771** |
| Brier | 0.254 | 0.282 | **0.297** |
| ECE | 0.049 | 0.049 | **0.063** |

Scored by JevBench's own `scoring.score_task` and `summarize`, via
`bench/to_official_records.py`, which reproduces both archived official runs
exactly (186/231 and 182/231, zero prediction mismatches).

**The single most damaging number is temporal_numeric.** On its own held-out
worlds that family went 0.243 → 0.615, a 37-point gain over 247 items. On
JevBench's family of the same name it went **3/15 → 2/15**, where the
*less*-trained 3fam adapter had gone 3/15 → 5/15. Learning the generator
harder made the benchmark family worse. long_policy is the same shape: 0.286 →
0.857 held out, 10/19 → 7/19 on the benchmark against 3fam's 10/19 → 9/19.

Across three runs the relationship is monotone in the wrong direction:

| | held-out worlds | JevBench hard |
|---|---:|---:|
| v0.1.0 | 0.282 | **0.604** |
| 3fam | 0.619 | 0.586 |
| anchored | 0.650 | **0.550** |

Three points is not a law. It is enough to stop treating held-out world
accuracy as a proxy for anything.

**By confidence band the shape is the familiar one**, and the anchor did not
protect the contested middle:

| baseline confidence | n | answer moved | fixed | broken |
|---|---:|---:|---:|---:|
| 0.00 – 0.45 | 15 | 11 (73%) | +3 | −1 |
| 0.45 – 0.85 | 70 | 30 (43%) | +10 | **−17** |
| 0.85 – 1.00 | 146 | 3 (2%) | +0 | −3 |

The middle band is −7 against 3fam's −8, so the anchor bought essentially
nothing there — as predicted, for the predicted reason: a KL term costs least
where the base distribution is flat, and that is exactly where the damage is.
It also broke one decision in the bottom band, where 3fam broke none.

### The anchor did its job, measured directly

Both adapters scored on the *same* held-out set — the worlds they were trained
on and the MMLU rehearsal items that stand in for "general decisions the base
could already make" — so this is the one comparison that isolates the recipe
from the data split:

| arm | worlds | MMLU rehearsal | rehearsal NLL | rehearsal ECE |
|---|---:|---:|---:|---:|
| base (`v0.1.0`) | 0.282 | 0.685 | 0.799 | 0.068 |
| lora-3fam-v1, worlds only | 0.592 | 0.676 (**−0.009**) | 0.798 | 0.080 |
| lora-anchor-v1 | **0.644** | **0.721 (+0.036)** | 0.678 | 0.055 |

The worlds-only run lost general ability and got *less* calibrated on it. The
anchored run gained general ability, lowered its NLL on it by a sixth, and
improved its calibration — while beating the same run on the worlds it was
trained for. Whatever else is true, the anchor plus rehearsal demonstrably did
the thing it was built to do.

**Which makes the JevBench result worse, not better.** Two independent proxies
for "did this damage the model" — held-out worlds and held-out MMLU — both say
no, emphatically, and the benchmark says yes. There is no version of "it
forgot" that survives this table. It did not forget; it learned something that
is wrong for the benchmark's own decisions.

### Scoring the prediction

The forecast was committed before the run was scored. It was wrong five ways,
and wrong in *both* directions:

| prediction | outcome |
|---|---|
| held-out worlds below 0.619 | 0.650 — too pessimistic |
| bottom band's +6 mostly survives | +3/−1 — too optimistic |
| middle band improves to between −5 and 0 | −7 — too optimistic |
| standard recovers most of its −2 | stayed at −2 |
| hard lands between −1 and +2 | −6 |

The only part that held was the framing: "the ship rule is more likely to be
missed than met." It was missed by more than I thought, and the mechanism I
distrusted (the anchor) worked, while the outcome I expected it to buy did not
arrive. Registering the forecast is what makes that legible instead of
retrofittable.

**Decision: `v0.1.0` stays the shipped readout.** Two adapters have now been
trained and neither ships. The rule was "ship only if hard rises without easy
or standard falling"; hard fell further than last time.

**What it actually says.** The first run's conclusion was "the worlds teach the
habits they contain, and the JevBench hard tier is mostly other families." That
was too kind. This run learned the habits *better* and transferred *worse*, on
the very families the generators are named after. The exact-law generators are
not a harder version of JevBench's temporal_numeric and long_policy; they are a
different distribution wearing the same names, and the gradient that fits one
moves away from the other. The next change is not another regulariser on the
same data. It is either training data drawn from the benchmark's own
distribution, or abandoning the training track for the inference-time
interventions in this file, which are so far the only things that have moved a
number in the right direction.

## What this does not cover

- **Order averaging is still unrun as a scoring mode.** E2 measured the flip
  rate, but no result above uses an averaged distribution — the accuracy table's
  `flip` column is 0.0% by construction because that run used a single order.
- **The control battery covers Qwen3.5-4B only.** llama-3.2-3b has not been
  gated here; its API-path numbers are from a different run and a quantised
  build, so they are context rather than a like-for-like comparison.
- **Public tasks only** (231 of 534), and the held-out half is where a
  leaderboard result would actually be decided.
- **The routed pass ran one budget (512) on one cell (30%).** A 256-token
  budget is the only one the composite could tolerate and it was not run; the
  table assumes the gain survives halving the budget, which is untested.
- **No critique pass.** A LOOP-shaped second pass that attacks the readout's
  answer was not run; with 3 of 7 routed-right answers broken by a single
  think, it would have to be conservative to help.

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

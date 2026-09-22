# Submission package — Jobe v0.1.0 to JevBench

Everything a maintainer needs to run Jobe on their own machine and list it, in
the shape their results file uses. `README-submission.md` holds the *decisions*
behind the unit (no temperature, no order averaging, the failure policy, the
stated weak spot); this file is the *package*: card, unit, numbers, pricing,
reproduction, and the issue text to paste.

How systems get onto the board (from their v1.2 additions notes): a GitHub
issue on `fstandhartinger/jevbench` naming the repo and commit, the weights and
revision, and a runnable adapter or server. They run it on **their** RunPod GPU
so the held-out hard items can be sent; a submitter-operated endpoint only earns
an unranked partial row. Jobe is an in-process adapter: no endpoint, nothing to
keep online.

## System card (paste-ready)

| field | value |
|---|---|
| key | `jobe-qwen3.5-4b` |
| display | Jobe (Qwen3.5-4B, MantisShrimpdev) |
| class | jev-rebuild |
| open | yes |
| author | MantisShrimpdev |
| repo | https://github.com/MantisShrimpdev/jobe — tag `v0.1.0` (commit `732f01e`); package + adapter unchanged at `6ebb77d` |
| licence | MIT (code); Qwen3.5 weights Apache-2.0 |
| underlying | `Qwen/Qwen3.5-4B`, frozen, bf16, hub revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| has_distribution | true |
| probability_source | native — full-vocabulary last-position logits, restricted softmax over the declared answer slots |
| endpoint_kind | gpu (in-process adapter, run on the maintainers' machine) |
| temperature / order averaging | none / none (measured, see README-submission.md) |
| output tokens | 0 per decision — nothing is generated |

## The unit

- **Adapter:** `bench/jobe_direct.py` → copy to `jevbench/adapters/jobe_direct.py`. Registers as `--adapter jobe_direct`. Returns HTTP-style **422** for anything outside the protocol (more than 16 options, a prompt over `max_tokens`, a tokenizer that cannot give single-token answer slots) so the runner records it and continues; infrastructure faults propagate.
- **Harness registration:** `bench/jevbench-registration.patch` — the five edits to `jevbench/cli.py` and `jevbench/adapters/__init__.py`, taken against harness `v1.2.11` (`bb83e5d`), the revision the conformance run used.
- **Windows only:** put `bench/winshim` first on `PYTHONPATH` (a no-op `fcntl` for their `budget.py`). Linux needs nothing.
- **Load:** `JEVBENCH_WARM_LOAD=1` loads the backbone at adapter construction so the first timed decision is not a cold start. VRAM ≈ 8.6 GB at bf16; any 10 GB card.

```bash
git clone https://github.com/MantisShrimpdev/jobe && cd jobe && git checkout v0.1.0
pip install -e .                                   # torch, transformers
cp bench/jobe_direct.py <jevbench>/jevbench/adapters/
( cd <jevbench> && git apply <jobe>/bench/jevbench-registration.patch )
cd <jevbench> && JEVBENCH_WARM_LOAD=1 python -m jevbench.cli run --adapter jobe_direct \
    --model D:/path/or/hub-id --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a ...
```

## Public-set numbers (their harness, `bench/runs/2026-09-22-public231/`)

231 of 231 attempted, 231 strict-valid, 0 failures.

| tier | n | accuracy | ECE | p50 | p95 |
|---|---:|---:|---:|---:|---:|
| easy | 48 | 1.000 | 0.016 | 0.108 s | 0.144 s |
| standard | 72 | 0.986 | 0.046 | 0.107 s | 0.125 s |
| hard | 111 | 0.604 | 0.133 | 0.224 s | 3.253 s |
| all | 231 | 0.805 | 0.049 | | |

Latencies are raw, on an RTX 3080 in-process, batch 1, `eager` attention;
their ×2 + 0.15 s self-hosted adjustment applies. Axes under `composite_v12`
with the judge tier absent (weights renormalised): I 82.9 · C 71.7 · S 88.4
(standard tier) · K 61.9 at the pricing below → **JevBench Score 74.9–75.4**
depending on the cost line; 72.3–77.9 across a 4× price band. A public-set
estimate, not a placement.

## Pricing basis

Same weights as SemIf, so the same reference tariff applies — deepinfra
`Qwen/Qwen3.5-4B`, **$0.03 per M input, $0.15 per M output** — with Jobe's own
measured token counts and **zero output tokens**:

| tier | mean input tokens / decision | $ per 1,000 decisions |
|---|---:|---:|
| easy | 164 | 0.0049 |
| standard | 168 | 0.0050 |
| hard | 1,274 | 0.0382 |

Under their rule `(v1.1 × 314 + hard × 220) / 534`, with the public
easy+standard mean as the proxy for the v1.1 tiers: **≈ $0.0187 per 1,000
decisions** (cost axis 61.9). The judge tier will move the v1.1 mean; they
price from the tokens their run measures.

## What a local run cannot tell you

The judge tier (68 tasks, none public, 28 % of Intelligence), the held-out
hard half (where placement is decided), speed on their hardware, and the exact
weighting of probability fidelity inside Calibration — on the 10 public items
with a gold distribution, Jobe's mean total variation is 0.299 (stated in full
in README-submission.md).

## Pre-flight

- [x] History scanned for secrets and private addresses across all 22 commits: none.
- [x] `v0.1.0` tag pushed; weights revision pinned and shard hashes verified against the hub.
- [x] Adapter conformance-clean through their `Runner` (231/231 strict-valid).
- [x] Repo made public — 2026-09-22.
- [x] Issue opened — https://github.com/fstandhartinger/jevbench/issues/28 (2026-09-22).

## Issue text (paste into a new issue on fstandhartinger/jevbench)

**Title:** Submission: Jobe — frozen Qwen3.5-4B decision readout, in-process adapter

Repo: https://github.com/MantisShrimpdev/jobe, tag `v0.1.0` (commit `732f01e`). MIT.

What it is: a local, frozen-backbone decision readout over `Qwen/Qwen3.5-4B`
(bf16, hub revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`). One forward
pass per decision; the distribution is the restricted softmax over the
declared options' answer-letter logits at the last position — `native`
probabilities, zero generated tokens. No temperature, no order averaging
(both measured and rejected; see `bench/README-submission.md`). The
protocol is adapted from SemIf (MIT, attributed in `NOTICE`); the answer
slots are resolved in context rather than standalone, which is the one
divergence.

How to run: in-process adapter, no endpoint — `bench/jobe_direct.py` drops
into `jevbench/adapters/`, and `bench/jevbench-registration.patch` applies
the `cli.py` registration against v1.2.11. Full steps in `bench/SUBMISSION.md`.
Needs one 10 GB GPU. On Linux nothing else; on Windows `bench/winshim` on the path.

Public-set run through your harness (`bench/runs/2026-09-22-public231/`):
231/231 strict-valid, 0 failures; easy 1.000, standard 0.986, hard 0.604
(ECE 0.016 / 0.046 / 0.133); standard-tier p50 0.107 s, p95 0.125 s raw on
an RTX 3080.

Pricing: same weights as SemIf — deepinfra Qwen3.5-4B $0.03/M in, $0.15/M
out — with measured input tokens (164 easy / 168 standard / 1,274 hard per
decision) and 0 output tokens; ≈ $0.019 per 1,000 decisions under your v1.2.3 rule.

Known limits: 16 options maximum (returns 422 beyond it); prompts are never
truncated (422 over `max_tokens`); probability fidelity on exact-distribution
items is the weak spot (mean TV 0.299 on the 10 public ones).

Happy to adjust anything about the adapter to fit your run.

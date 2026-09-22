# Submitting Jobe to JevBench

Benchmark Heaven runs entrants on its own infrastructure; you submit **weights
plus a harness-compatible adapter**, never a home endpoint. This is the
submission unit and the decisions behind it, each with the evidence that made
it.

## The unit

| | |
|---|---|
| Backbone | `Qwen/Qwen3.5-4B`, **revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`** |
| Weights identity | shard 1 `26a93f066e1916ad…`, shard 2 `cb544bd9bfae93dc…` (sha256, byte-identical to the hub revision — verified, not assumed) |
| Precision | bf16 |
| Attention | `eager` (measured faster than SDPA at this prompt length; see main README) |
| Readout | one forward pass, full-vocabulary last-position logits, restricted softmax over declared answer slots |
| Order averaging | **none** |
| Temperature | **1.0 — none applied** |
| Adapter | `bench/jobe_direct.py` → `jevbench/adapters/jobe_direct.py` |
| `probs_source` | `native` |

## Decisions, and why

### No temperature

Every fitted temperature helped one tier and hurt another, and the direction
flipped between backbones (`RESULTS.md`, "A single global temperature does not
transfer"). Raw hard-tier ECE is **0.106**, which under the leaderboard's own
formula is C ≈ 78.8 on the ECE component — already above SemIf's published 72.6
overall. Applying a scalar that we know moves the wrong way on at least one tier
would be trading a measured number for a hoped-for one. T = 1.0 is a decision
with evidence, not a default.

### No order averaging

Measured on the hard tier: accuracy 0.613 → 0.613, ECE 0.106 → 0.101, for
double the forward passes. Of 11 answers it changed, it fixed 3 and broke 3.
Symmetric instability is noise, and averaging noise costs latency for nothing.

### Failure policy: 422 for out-of-contract, abort-eligible for infrastructure

The runner aborts after three consecutive failures but exempts
`status_code == 422`. The adapter therefore returns **422** for anything the
protocol cannot represent — more than 16 options, a tokenizer that cannot
support single-token answer slots for a prompt, a prompt over `max_tokens`
(never truncated: dropping evidence changes answers). Those are recorded as
unprocessable and do not kill the run. A CUDA OOM or driver fault returns
without a status and *does* count, because three of those in a row is a real
infrastructure problem and stopping is correct.

**Known ceiling: 16 options.** Public tasks peak at 6. Held-out is unknown. A
17-option task fails soft (422) rather than killing the run.

## A weak spot, stated

**Probability fidelity is poor.** On the 10 public items that carry an exact
gold distribution (`provenance.gold_probs` — the `probability` family), Jobe's
mean total variation from gold is **0.299**, max **0.797**, and argmax accuracy
on those items is **40%** on a binary question. This is half of the
leaderboard's Calibration axis, and its exact weighting is not in the public
scoring code. The frozen backbone is not good at exact-probability reasoning.
This is the single most likely thing to pull the composite below the ECE-only
estimate — and it is the family the exact-law-worlds training thesis targets,
so it is also the strongest argument for eventually training.

n = 10. Read the magnitude, not the decimals.

## Reproducing the conformance run

The official harness, not Jobe's own runner. On Linux nothing extra is needed.
On Windows, `jevbench/budget.py` imports `fcntl` for an advisory lock on its
cost ledger; `bench/winshim/fcntl.py` stubs it as a no-op, which is safe for a
single-process local run and changes nothing about scoring. Put it FIRST on the
path, for the run only.

```bash
# 1. register the adapter
cp bench/jobe_direct.py <jevbench>/jevbench/adapters/jobe_direct.py
#    then in <jevbench>/jevbench/adapters/__init__.py:
#      from .jobe_direct import JobeDirectAdapter
#    and in <jevbench>/jevbench/cli.py:
#      - import JobeDirectAdapter alongside SemIfDirectAdapter
#      - add  "jobe_direct": JobeDirectAdapter  to `kinds`
#      - add  "jobe_direct"  to the no-endpoint-required tuple
#      - add  "jobe_direct"  to the revision-kwarg tuple
#      - add  "jobe_direct"  to the `choices=` list on the --adapter argument
#        (argparse validates against this separately; missing it rejects the
#        name before anything loads - five places, not four)

# 2. run (Windows: prefix PYTHONPATH with bench/winshim)
cd <jevbench>
PYTHONPATH=<jobe>/src JEVBENCH_WARM_LOAD=1 \
python -m jevbench.cli run \
  --tasks datasets/public/easy.jsonl,datasets/public/original.jsonl,datasets/public/hard.jsonl \
  --adapter jobe_direct \
  --endpoint Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --model Qwen/Qwen3.5-4B \
  --results results.jsonl --ledger ledger.json --raw-dir /tmp/raw --cap-usd 10
```

`JEVBENCH_WARM_LOAD=1` loads the weights before the clock starts, as a server
that is already up would be — the leaderboard's own convention for in-process
entrants.

## What a local run cannot tell you

- **The judge tier** — 68 tasks in the manifest, none public, 28% of the
  Intelligence axis.
- **Held-out** — where placement is actually decided.
- **Speed on their hardware** — ours is an RTX 3080; theirs is RunPod, with a
  ×2 + 0.15 s self-hosted adjustment applied.
- **Cost** — priced by their reference methodology for the weight class.

So a composite score is not computable locally, only bounded. Public-set
placement is the floor.

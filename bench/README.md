# `bench/` — what each script is for

Thirteen scripts in one directory is a directory that needs a map. The findings
they produced live in [`RESULTS.md`](RESULTS.md); this is only the index.

Every script that touches the GPU refuses a busy card rather than contending
with whatever else is on it.

## Running the benchmark

| script | what it does |
|---|---|
| `jobe_direct.py` | the in-process JevBench adapter — the submission unit, no HTTP and no top-N logprob window |
| `run_jevbench.py` | runs the public tasks through Jobe and writes `results.jsonl` plus a manifest. `--orders reversed` re-runs with the options flipped |
| `score_official.py` | scores a run with **JevBench's own code** and computes the partial-run composite. The number to quote |
| `score_jevbench.py` | our own per-tier scoring, and fits a temperature on the stored logits |
| `gate.py` | the control battery (E1 evidence swap, E2 order reversal, position concentration), per tier, non-zero exit on failure |

## Reading a run

| script | what it does |
|---|---|
| `compare_runs.py` | two runs side by side, **by tier and by family**. A tier delta of −2 and a tier delta of −2 that is really +4/−6 are different results |
| `confidence_bands.py` | splits a run by the readout's own top probability. The public 231 are three populations and only the middle one is contested |
| `uncertainty_analysis.py` | does the readout's confidence find its own wrong answers? (hard-tier AUROC 0.79) |

## Interventions, and what they cost

| script | what it does |
|---|---|
| `runoff.py` | re-asks a contested decision with only its top two options. In that band the answer is in the top two 87% of the time |
| `reason_route.py` | a thinking pass over the least-confident decisions. `--max-conf` states the rule as a threshold; `--frac` as a quantile |
| `route_composite.py` | prices a routing rule through the harness's own axis functions. Takes `--route-hard` and `--route-std`, because the rule is what decides whether it pays |
| `compare_orders.py` | whether averaging over option orders pays, on one tier |
| `prefix_bench.py` | throughput of prefix-cache reuse: many questions about one long document |

## Artifacts

`runs/<date>-<name>/` holds what a claim rests on — `results.jsonl`,
`official_score.json`, `gate.json`, and for a trained adapter the training
`summary.json` and `train_manifest.json`. Adapter weights stay out of git.

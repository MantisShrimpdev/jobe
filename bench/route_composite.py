# -*- coding: utf-8 -*-
"""What a routed thinking pass does to the JevBench Score, using the harness's own axis functions.
Facts from results/v1.2: Speed = standard+judge run only (hard latency never enters);
Cost for a 4B self-hosted row = deepinfra tariff $0.03/M in, $0.15/M out; usd_per_1000 = (v11*314 + hard*220)/534.

  python bench/route_composite.py --jevbench <clone>
"""
import argparse
import json
import math
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--jevbench", required=True, help="path to the jevbench clone")
ap.add_argument("--official-score", default="bench/runs/2026-09-22-public231/official_score.json")
ap.add_argument("--usd", type=float, default=0.022, help="cost assumption behind the baseline composite")
ap.add_argument("--measured-hard-delta", type=float, default=None,
                help="hard-tier accuracy gain measured at 30%% routing (0.045 on 2026-09-22): print the composite under it")
args = ap.parse_args()
sys.path.insert(0, args.jevbench)
from jevbench.composite_v12 import cost  # noqa: E402

_axes = json.load(open(args.official_score, encoding="utf-8"))["axes"]
I0, C0, S0 = _axes["intelligence"], _axes["calibration"], _axes["speed"]  # partial-run axes
USD0 = args.usd
HARD_SHARE, STD_SHARE = 220 / 534, 242 / 534        # decision shares in the cost average
OUT_USD_PER_TOKEN = 0.15 / 1e6
W_HARD_PARTIAL = 0.30 / 0.72                        # renormalised (no judge tier): hard = 41.7 % of I
W_HARD_FULL = 0.30

def score_with(I, C, S, K, w=(0.25, 0.25, 0.25, 0.25)):
    axes = (I, C, S, K)
    return math.exp(sum(wi * math.log(max(a, 1)) for wi, a in zip(w, axes)))

K0 = cost(USD0)
base = score_with(I0, C0, S0, K0)
print(f"baseline: I {I0:.1f} C {C0:.1f} S {S0:.1f} K {K0:.1f} -> JevBench Score {base:.2f}\n")
print(f"{'route':>6s} {'std spill':>9s} {'think tok':>9s} {'usd/1000':>9s} {'K':>6s} {'dK':>6s} | "
      f"{'break-even d(hard) 25:25:25:25':>30s} | {'60:20:20 accuracy view':>22s}")
for f, f_std in ((0.3, 2 / 72), (0.5, 4 / 72)):
    for T in (256, 512, 1024, 2048):
        usd = USD0 + 1000 * OUT_USD_PER_TOKEN * T * (HARD_SHARE * f + STD_SHARE * f_std)
        K = cost(usd)
        # break-even dI for the headline score: 0.25 ln I' + 0.25 ln K' = 0.25 ln I + 0.25 ln K  ->  I' = I * K/K'
        dI_head = I0 * K0 / K - I0
        # accuracy view 60:20:20 (no calibration): 0.6 ln I' + 0.2 ln K' = ...  -> I' = I * (K/K')**(1/3)
        dI_acc = I0 * (K0 / K) ** (0.2 / 0.6) - I0
        print(f"{f:6.0%} {f_std:9.1%} {T:9d} {usd:9.4f} {K:6.1f} {K - K0:+6.1f} | "
              f"hard +{dI_head / W_HARD_PARTIAL / 100:.3f} (partial) / +{dI_head / W_HARD_FULL / 100:.3f} (full run) | "
              f"hard +{dI_acc / W_HARD_PARTIAL / 100:.3f} / +{dI_acc / W_HARD_FULL / 100:.3f}")
print("\n(hard is 0.604 in the official run; a break-even above +0.396 is unreachable)")

if args.measured_hard_delta is not None:
    d = args.measured_hard_delta
    acc_w = (0.6, 0.0, 0.2, 0.2)
    b_head, b_acc = score_with(I0, C0, S0, K0), score_with(I0, C0, S0, K0, acc_w)
    print(f"\nmeasured hard delta {d:+.3f} at 30% routing (hard weight in this partial run {W_HARD_PARTIAL:.3f}):")
    print(f"{'think tok':>9s} {'usd/1000':>9s} {'K':>6s} {'I':>6s} | {'headline':>9s} {'delta':>6s} | {'60:20:20':>9s} {'delta':>6s}")
    for T in (0, 256, 512):
        usd = USD0 + 1000 * OUT_USD_PER_TOKEN * T * (HARD_SHARE * 0.3 + STD_SHARE * 2 / 72)
        K = cost(usd)
        I = I0 + (100 * W_HARD_PARTIAL * d if T else 0.0)
        s, sa = score_with(I, C0, S0, K), score_with(I, C0, S0, K, acc_w)
        print(f"{T:9d} {usd:9.4f} {K:6.1f} {I:6.1f} | {s:9.2f} {s - b_head:+6.2f} | {sa:9.2f} {sa - b_acc:+6.2f}")

# what Luna's trade looks like in the same units
print(f"\nLuna: I 96.8 C 89.8 S 77.5 K 28.5 -> {score_with(96.821, 89.792, 77.546, 28.490):.1f}; "
      f"SemIf: I 85.9 C 72.6 S 83.7 K 59.5 -> {score_with(85.938, 72.601, 83.704, 59.467):.1f}")

"""Score a JevBench run with JevBench's OWN code, and compute the partial-run
composite with the leaderboard's own axis functions.

Nothing here reimplements a metric. `summarize`, `intelligence`, `calibration`,
`speed`, `cost`, `tvd` and `jevbench_score` are imported from the harness and
called as-is. What this script adds is only the plumbing: splitting a run by
tier, feeding each axis what it expects, and saying out loud where the inputs
fall short of a full leaderboard run.

Those shortfalls, stated once here and again in the output:

  * JUDGE tier absent (not public; 28% of Intelligence). The harness's own
    partial-run rule applies: the tier is left out and the remaining tier
    weights are renormalised. Still a partial run.
  * STANDARD is approximated by the public `original` file (72 tasks, the
    published Jev cohort); the leaderboard's standard tier is 96.
  * SPEED convention is p50/p95 over the standard+judge run. With no judge,
    standard-only latencies are used. Endpoint kind is "gpu" (our own server:
    x2 + 0.15 s), exactly as the leaderboard adjusts self-hosted entrants.
  * COST cannot be measured locally and `cost()` refuses a missing price. The
    composite is therefore reported under explicit price assumptions with a
    sensitivity row - never as one number.

  python bench/score_official.py --jevbench <clone> --tasks all231.jsonl \
      --results results.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def tier_of(task_id: str) -> str:
    head = task_id.split("-", 1)[0]
    return {"easy": "easy", "original": "standard", "hard": "hard"}.get(head, head)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jevbench", required=True, help="path to the jevbench clone")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--endpoint-kind", default="gpu", choices=["gpu", "cpu", "demo", "api"])
    ap.add_argument("--cost-usd-per-1000", type=float, nargs="*",
                    default=[0.010, 0.022, 0.040],
                    help="assumed prices to show the composite under (0.022 ~ SemIf's estimate)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    sys.path.insert(0, args.jevbench)
    from jevbench.composite_v12 import (  # noqa: E402
        TIER_WEIGHTS, calibration, cost, intelligence, jevbench_score, speed, tvd,
    )
    from jevbench.metrics import latency_summary  # noqa: E402
    from jevbench.summarize import summarize  # noqa: E402
    from jevbench.tasks import load_jsonl  # noqa: E402

    tasks = load_jsonl(args.tasks)
    records = [json.loads(l) for l in open(args.results, encoding="utf-8") if l.strip()]
    by_task = {t.id: t for t in tasks}

    # ------------------------------------------------------------ headline
    overall = summarize(tasks, records)
    print("OFFICIAL HARNESS SUMMARY (jevbench.summarize, unmodified)")
    print(f"  attempted {overall['n_attempted']}/{overall['n_planned']}   "
          f"operational_success {overall['operational_success']:.3f}   "
          f"schema_validity {overall['schema_validity']:.3f}   "
          f"strict {overall['schema_validity_strict']:.3f}   "
          f"renormalized {overall['n_renormalized']}")
    print(f"  accuracy {overall['accuracy']:.3f}   macro_accuracy {overall['macro_accuracy']:.3f}   "
          f"brier {overall['brier_mean']:.3f}   ece {overall['ece']['ece']:.3f}   "
          f"ordinal_mae {overall['ordinal_mae']}")
    pc = overall["paraphrase_consistency"]
    print(f"  paraphrase pairs {pc['pairs']}  both_valid {pc['both_valid']}  "
          f"agreement {pc['agreement']}")
    print(f"  probability_sources {overall['probability_sources']}   "
          f"cost_basis {overall['cost_basis']}")

    # ------------------------------------------------------------ per tier
    tiers: dict[str, dict] = {}
    print("\nPER TIER (their summarize on each tier's tasks)")
    print(f"  {'tier':9s} {'n':>4s} {'acc':>7s} {'ece':>7s} {'brier':>7s} {'p50 s':>7s} {'p95 s':>7s} {'strict':>7s}")
    for tier in ["easy", "standard", "hard"]:
        tt = [t for t in tasks if tier_of(t.id) == tier]
        tr = [r for r in records if r["task_id"] in {t.id for t in tt}]
        if not tt:
            continue
        s = summarize(tt, tr)
        tiers[tier] = s
        print(f"  {tier:9s} {s['n_attempted']:4d} {s['accuracy']:7.3f} {s['ece']['ece']:7.3f} "
              f"{s['brier_mean']:7.3f} {s['latency']['p50_s']:7.3f} {s['latency']['p95_s']:7.3f} "
              f"{s['schema_validity_strict']:7.3f}")
    print("  judge     (not public - absent)")

    # ------------------------------------------------------------ axes
    acc = {t: tiers[t]["accuracy"] for t in tiers}
    acc["judge"] = None
    I = intelligence(acc)
    used = {t: w for t, w in TIER_WEIGHTS.items() if acc.get(t) is not None}
    renorm = {t: w / sum(used.values()) for t, w in used.items()}

    hard_tasks = [t for t in tasks if tier_of(t.id) == "hard"]
    hard_rec = {r["task_id"]: r for r in records}
    tvds = []
    for t in hard_tasks:
        gp = (t.provenance or {}).get("gold_probs")
        r = hard_rec.get(t.id)
        if gp and r and r.get("probs"):
            tvds.append(tvd(r["probs"], gp, t.labels))
    mean_tvd = sum(tvds) / len(tvds) if tvds else None
    hard_ece = tiers["hard"]["ece"]["ece"]
    C_full = calibration(hard_ece, mean_tvd)
    C_ece_only = calibration(hard_ece)

    std_lat = [r["latency_s"] for r in records
               if tier_of(r["task_id"]) == "standard" and r.get("latency_s") is not None]
    ls = latency_summary(std_lat)
    S = speed(ls["p50_s"], ls["p95_s"], args.endpoint_kind)
    all_lat = latency_summary([r["latency_s"] for r in records if r.get("latency_s") is not None])
    S_all = speed(all_lat["p50_s"], all_lat["p95_s"], args.endpoint_kind)

    print("\nAXES (their composite_v12 functions)")
    print(f"  Intelligence  {I:6.1f}   partial run: judge absent, weights renormalised to "
          + ", ".join(f"{t} {w:.2f}" for t, w in renorm.items()))
    print(f"  Calibration   {C_full:6.1f}   = mean of ECE term {C_ece_only:.1f} and fidelity term "
          f"{100 * (1 - mean_tvd):.1f}  (hard ECE {hard_ece:.3f}, mean TVD {mean_tvd:.3f} over "
          f"{len(tvds)} gold-distribution items)")
    print(f"  Speed         {S:6.1f}   standard-tier p50 {ls['p50_s']:.3f}s / p95 {ls['p95_s']:.3f}s, "
          f"endpoint '{args.endpoint_kind}' (x2 +0.15s)   [all tiers would give {S_all:.1f}]")
    print(f"  Cost            n/a   not measurable locally; composite shown under assumptions below")

    # ------------------------------------------------------------ composite
    print("\nJEVBENCH SCORE under explicit cost assumptions (their jevbench_score)")
    print(f"  {'$ / 1k decisions':>17s} {'K':>6s} {'score':>7s}   context")
    for usd in args.cost_usd_per_1000:
        K = cost(usd)
        score = jevbench_score({"intelligence": I, "calibration": C_full, "speed": S, "cost": K})
        note = "~SemIf's published estimate" if abs(usd - 0.022) < 1e-9 else ""
        print(f"  {usd:17.4f} {K:6.1f} {score:7.1f}   {note}")
    print("  for reference, published: Jev 1.13.0 75.4 | SemIf 74.7 (I 85.9 C 72.6 S 83.7 K 59.5) | #10 68.9")
    print("\nThis is a PARTIAL-RUN figure: no judge tier, standard approximated by the 72-task")
    print("original file, speed on standard only, cost assumed. Public-set placement is a floor.")

    if args.out:
        out = {"overall": overall, "tiers": tiers,
               "axes": {"intelligence": I, "calibration": C_full, "calibration_ece_only": C_ece_only,
                        "hard_ece": hard_ece, "mean_tvd": mean_tvd, "tvd_n": len(tvds),
                        "speed": S, "speed_all_tiers": S_all, "renormalised_weights": renorm},
               "composite_under_cost_assumptions": {
                   str(u): jevbench_score({"intelligence": I, "calibration": C_full,
                                           "speed": S, "cost": cost(u)})
                   for u in args.cost_usd_per_1000}}
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, default=str)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

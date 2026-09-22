"""Score a Jobe JevBench run, per tier, and fit a temperature on stored logits.

Per tier because an aggregate hides the failure it is meant to catch: in an
earlier run elsewhere the pooled shuffled-context drop read a healthy +0.221
while the hard tier moved +0.000.

The temperature is fitted on ONE tier and applied to the others, so the number
reported is held-out rather than fitted-and-reported on the same data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from jobe.calibrate import calibration_report, fit_temperature  # noqa: E402


def load(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def predict(probs: dict[str, float], kind: str) -> str:
    if kind == "score":
        return str(round(sum(int(k) * v for k, v in probs.items())))
    return max(probs.items(), key=lambda kv: kv[1])[0]


def rows_and_gold(rows: list[dict]) -> tuple[list[list[float]], list[int], list[int]]:
    """Logit rows, the gold index, and the PREDICTED index for calibration.

    The prediction is returned explicitly because argmax is not the answer for
    an ordinal question - a `score` answer is the rounded probability-weighted
    value. Scoring those by argmax disagreed with JevBench's own metrics
    (ECE 0.133 vs 0.106 on 111 mixed tasks), which is what caught it.
    """
    out_rows: list[list[float]] = []
    out_gold: list[int] = []
    out_pred: list[int] = []
    for r in rows:
        if not r.get("ok") or "logits" not in r:
            continue
        ids = r["logit_ids"]
        gold = str(r["expected"])
        if r["type"] == "noul":
            gold = "true" if gold == "yes" else "false"
        if gold not in ids:
            continue
        out_rows.append(r["logits"])
        out_gold.append(ids.index(gold))
        pred = predict(r["probs"], r["type"])
        if r["type"] == "noul":
            pred = "true" if pred == "yes" else "false"
        out_pred.append(ids.index(pred) if pred in ids else out_gold[-1] - 1)
    return out_rows, out_gold, out_pred


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("ok")]
    if not ok:
        return {}
    correct = sum(1 for r in ok if predict(r["probs"], r["type"]) == str(r["expected"]))
    chance = sum(1.0 / len(r["labels"]) for r in ok) / len(ok)
    per_type: dict[str, dict] = {}
    for r in ok:
        bucket = per_type.setdefault(r["type"], {"n": 0, "hit": 0})
        bucket["n"] += 1
        bucket["hit"] += predict(r["probs"], r["type"]) == str(r["expected"])
    secs = sorted(r["seconds"] for r in ok)
    return {
        "n": len(rows),
        "ok": len(ok),
        "accuracy": correct / len(ok),
        "chance": chance,
        "mean_conf": sum(max(r["probs"].values()) for r in ok) / len(ok),
        "flip_rate": sum(r.get("flip_rate", 0.0) for r in ok) / len(ok),
        "p50_s": secs[len(secs) // 2],
        "per_type": {k: {"n": v["n"], "acc": v["hit"] / v["n"]} for k, v in sorted(per_type.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="label=path.jsonl")
    ap.add_argument("--fit-tier", default="original", help="tier to fit the temperature on")
    args = ap.parse_args()

    print(f"{'model':26s} {'tier':10s} {'n':>4s} {'acc':>7s} {'chance':>7s} "
          f"{'conf':>6s} {'flip':>6s} {'p50 ms':>7s}  per-type")
    print("-" * 112)
    for spec in args.runs:
        label, path = spec.split("=", 1)
        rows = load(path)
        tiers: dict[str, list[dict]] = {}
        for r in rows:
            tiers.setdefault(r["tier"], []).append(r)
        for tier in ["easy", "original", "hard"]:
            if tier not in tiers:
                continue
            s = summarize(tiers[tier])
            if not s:
                continue
            types = " ".join(f"{k}={v['acc']:.3f}({v['n']})" for k, v in s["per_type"].items())
            print(f"{label:26s} {tier:10s} {s['ok']:4d} {s['accuracy']:7.3f} {s['chance']:7.3f} "
                  f"{s['mean_conf']:6.3f} {s['flip_rate']:6.1%} {s['p50_s']*1000:7.1f}  {types}")
        allrows = [r for rs in tiers.values() for r in rs]
        s = summarize(allrows)
        if s:
            print(f"{label:26s} {'ALL':10s} {s['ok']:4d} {s['accuracy']:7.3f} {s['chance']:7.3f} "
                  f"{s['mean_conf']:6.3f} {s['flip_rate']:6.1%} {s['p50_s']*1000:7.1f}")
        print("-" * 112)

    print("\nCALIBRATION — temperature fitted on one tier, reported on the others (held out)")
    print(f"{'model':26s} {'T':>6s} {'tier':10s} {'ECE before':>11s} {'ECE after':>10s} {'NLL before':>11s} {'NLL after':>10s}")
    print("-" * 92)
    for spec in args.runs:
        label, path = spec.split("=", 1)
        rows = load(path)
        fit_rows, fit_gold, _ = rows_and_gold([r for r in rows if r.get("tier") == args.fit_tier])
        if len(fit_rows) < 10:
            print(f"{label:26s} (not enough rows on tier {args.fit_tier} to fit)")
            continue
        t = fit_temperature(fit_rows, fit_gold)
        for tier in ["easy", "hard"]:
            tier_rows, tier_gold, tier_pred = rows_and_gold([r for r in rows if r.get("tier") == tier])
            if len(tier_rows) < 5:
                continue
            before = calibration_report(tier_rows, tier_gold, predictions=tier_pred)
            after = calibration_report(tier_rows, tier_gold, temperature=t, predictions=tier_pred)
            assert after.accuracy == before.accuracy, "temperature must not change argmax"
            print(f"{label:26s} {t:6.3f} {tier:10s} {before.ece:11.3f} {after.ece:10.3f} "
                  f"{before.mean_nll:11.3f} {after.mean_nll:10.3f}")
        print("-" * 92)


if __name__ == "__main__":
    main()

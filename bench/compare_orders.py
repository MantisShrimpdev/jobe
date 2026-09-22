"""Does order averaging pay? One tier, several order strategies.

Averaging costs one forward pass per extra order. It repairs instability, so it
pays in proportion to the instability there is to repair — which is an empirical
question about your model and your tier, not a default.

Metrics come straight from the recorded probabilities; no logits needed, since
an averaged distribution is not the softmax of any single logit row.
"""

from __future__ import annotations

import argparse
import json


def load(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def predict(probs: dict[str, float], kind: str) -> str:
    if kind == "score":
        return str(round(sum(int(k) * v for k, v in probs.items())))
    return max(probs.items(), key=lambda kv: kv[1])[0]


def ece(pairs: list[tuple[float, bool]], n_bins: int = 10) -> float:
    buckets = [[0, 0.0, 0] for _ in range(n_bins)]
    for conf, hit in pairs:
        i = min(int(min(max(conf, 0.0), 1.0) * n_bins), n_bins - 1)
        buckets[i][0] += 1
        buckets[i][1] += conf
        buckets[i][2] += 1 if hit else 0
    total = len(pairs)
    out = 0.0
    for n, conf_sum, hits in buckets:
        if n:
            out += (n / total) * abs(conf_sum / n - hits / n)
    return out


def brier(probs: dict[str, float], gold: str) -> float:
    return sum((p - (1.0 if k == gold else 0.0)) ** 2 for k, p in probs.items())


def summarize(rows: list[dict], only_tier: str | None) -> dict:
    ok = [r for r in rows if r.get("ok") and (only_tier is None or r["tier"] == only_tier)]
    pairs, hits, briers = [], 0, 0.0
    for r in ok:
        gold = str(r["expected"])
        hit = predict(r["probs"], r["type"]) == gold
        hits += hit
        pairs.append((max(r["probs"].values()), hit))
        briers += brier(r["probs"], gold)
    secs = sorted(r["seconds"] for r in ok)
    return {
        "n": len(ok),
        "acc": hits / len(ok),
        "conf": sum(c for c, _ in pairs) / len(pairs),
        "ece": ece(pairs),
        "brier": briers / len(ok),
        "flip": sum(r.get("flip_rate", 0.0) for r in ok) / len(ok),
        "p50": secs[len(secs) // 2],
        "total_s": sum(r["seconds"] for r in ok),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="label=path.jsonl")
    ap.add_argument("--tier", default=None, help="restrict to one tier")
    args = ap.parse_args()

    print(f"{'strategy':14s} {'n':>4s} {'acc':>7s} {'Δacc':>7s} {'ECE':>7s} {'Δece':>7s} "
          f"{'Brier':>7s} {'conf':>6s} {'flip':>6s} {'p50 ms':>8s} {'cost':>6s}")
    print("-" * 96)
    base = None
    for spec in args.runs:
        label, path = spec.split("=", 1)
        s = summarize(load(path), args.tier)
        if base is None:
            base = s
            dacc = dece = 0.0
            cost = 1.0
        else:
            dacc = s["acc"] - base["acc"]
            dece = s["ece"] - base["ece"]
            cost = s["total_s"] / base["total_s"]
        print(f"{label:14s} {s['n']:4d} {s['acc']:7.3f} {dacc:+7.3f} {s['ece']:7.3f} "
              f"{dece:+7.3f} {s['brier']:7.3f} {s['conf']:6.3f} {s['flip']:6.1%} "
              f"{s['p50']*1000:8.1f} {cost:5.2f}x")
    print("-" * 96)
    print("cost = wall-clock relative to the first row. Averaging is worth it only if")
    print("Δacc or Δece justifies that multiple.")


if __name__ == "__main__":
    main()

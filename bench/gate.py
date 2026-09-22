"""The control battery, run against a LOCAL Jobe backbone.

A port of EveryAppKit's tools/decisionGate.ts. That tool drives an
OpenAI-compatible HTTP endpoint; routing a local model through one would
reintroduce the top-N logprob window Jobe exists to remove, so the controls are
reimplemented here against the full-vocabulary readout. The criteria are
identical, deliberately, so results are comparable.

  E1  shuffled-context  give each decision another decision's evidence. Real
                        skill should mostly vanish. If it does not, the model is
                        reading the option menu, not the evidence.
  E2  option-order      present the options reversed. Ids are preserved, so
                        nothing that matters changed. A high flip rate means the
                        answer is unstable under noise.
  POS positional bias   which slot does the chosen option sit in? A model
                        concentrating on one index is not deciding.

Reported PER TIER. An aggregate hides the failure it is meant to catch: measured
elsewhere, a pooled E1 drop read a healthy +0.221 while the hard tier moved
+0.000 because the model was answering "A" every time.

Exits non-zero if any tier fails.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from jobe import Decision, Option, load, score  # noqa: E402

SEED = 42


def options_for(task: dict) -> list[Option]:
    q = task["question"]
    crit = q.get("criteria")
    if q["type"] == "noul":
        c = crit or {}
        pairs = [(k, c.get(k) or f"The proposition is {k}.") for k in ("true", "false")]
    elif q["type"] == "choice":
        pairs = [(k, v or k) for k, v in (crit or {}).items()]
    else:
        pairs = [(str(i), lvl) for i, lvl in enumerate(crit or [])]
    return [Option(id=k, description=f"{k}: {v}") for k, v in pairs]


def canonical(kind: str, scores: dict[str, float]) -> dict[str, float]:
    if kind != "noul":
        return scores
    return {"yes": scores.get("true", 0.0), "no": scores.get("false", 0.0)}


def predict(probs: dict[str, float], kind: str) -> str:
    if kind == "score":
        return str(round(sum(int(k) * v for k, v in probs.items())))
    return max(probs.items(), key=lambda kv: kv[1])[0]


def derange(n: int, seed: int = SEED) -> list[int]:
    """Sattolo - a single cycle, so no element keeps its own index."""
    idx = list(range(n))
    rng = random.Random(seed)
    for i in range(n - 1, 0, -1):
        j = rng.randint(0, i - 1)
        idx[i], idx[j] = idx[j], idx[i]
    return idx


def total_variation(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys) / 2.0


def run_one(backbone, task, options, evidence):
    kind = task["question"]["type"]
    decision = Decision(
        id=task["id"],
        evidence=evidence,
        criterion=task["question"]["instructions"],
        options=tuple(options),
        ordinal=kind == "score",
    )
    readout = score(backbone.model, backbone.tokenizer, decision)
    probs = canonical(kind, readout.scores)
    pred = predict(probs, kind)
    return {
        "pred": pred,
        "correct": pred == str(task["expected"]),
        "conf": max(probs.values()),
        "probs": probs,
        "position": [o.id for o in options].index(
            readout.choice if kind != "noul" else readout.choice
        ),
        "k": len(options),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-flip", type=float, default=0.25)
    ap.add_argument("--max-position", type=float, default=0.60)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    files = (
        [os.path.join(args.tasks, f) for f in sorted(os.listdir(args.tasks)) if f.endswith(".jsonl")]
        if os.path.isdir(args.tasks)
        else [f.strip() for f in args.tasks.split(",") if f.strip()]
    )

    backbone = load(args.model, device=args.device)
    print(f"decisionGate: {backbone.name} on {backbone.device}/{backbone.dtype}", file=sys.stderr)

    report: dict = {"model": args.model, "tiers": {}}
    any_fail = False
    warmed = False

    for path in files:
        tier = os.path.splitext(os.path.basename(path))[0]
        tasks = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        opts = [options_for(t) for t in tasks]

        if not warmed:
            for _ in range(12):
                run_one(backbone, tasks[0], opts[0], tasks[0]["state"])
            warmed = True

        print(f"  {tier}: baseline…", file=sys.stderr)
        base = [run_one(backbone, t, o, t["state"]) for t, o in zip(tasks, opts)]
        acc = sum(r["correct"] for r in base) / len(base)
        chance = sum(1.0 / len(t["labels"]) for t in tasks) / len(tasks)
        mean_conf = sum(r["conf"] for r in base) / len(base)
        tier_report: dict = {"n": len(base), "accuracy": acc, "chance": chance,
                             "mean_confidence": mean_conf}
        verdicts: list[str] = []

        # POSITION
        counts: dict[int, int] = {}
        for r in base:
            counts[r["position"]] = counts.get(r["position"], 0) + 1
        top = max(counts.values()) / len(base)
        expected = len(tasks) / sum(len(o) for o in opts)
        bad = top > args.max_position
        any_fail |= bad
        tier_report["position_concentration"] = top
        verdicts.append(
            f"POSITION {'FAIL' if bad else 'pass'} — {top:.0%} of answers in one slot "
            f"(~{expected:.0%} expected)"
        )

        # E1
        print(f"  {tier}: E1 shuffled-context…", file=sys.stderr)
        idx = derange(len(tasks))
        e1 = [run_one(backbone, t, o, tasks[idx[i]]["state"])
              for i, (t, o) in enumerate(zip(tasks, opts))]
        e1acc = sum(r["correct"] for r in e1) / len(e1)
        e1conf = sum(r["conf"] for r in e1) / len(e1)
        skill = acc - chance
        kept = e1acc - chance
        at_chance = skill <= 0.05
        bad = (not at_chance) and kept > skill * 0.5
        any_fail |= bad
        tier_report.update({"e1_accuracy": e1acc, "e1_drop": acc - e1acc,
                            "e1_mean_confidence": e1conf})
        if at_chance:
            verdicts.append(
                f"E1       n/a  — baseline {acc:.0%} is already at chance {chance:.0%}; "
                "nothing to destroy"
            )
        else:
            verdicts.append(
                f"E1       {'FAIL' if bad else 'pass'} — shuffled evidence kept "
                f"{kept / skill:.0%} of above-chance skill ({acc:.0%} → {e1acc:.0%})"
            )
        if e1conf >= mean_conf:
            verdicts.append(
                "         note — confidence did not fall on irrelevant evidence; "
                "the score cannot be used to detect bad input"
            )

        # E2
        print(f"  {tier}: E2 reversed order…", file=sys.stderr)
        e2 = [run_one(backbone, t, list(reversed(o)), t["state"])
              for t, o in zip(tasks, opts)]
        flips = sum(1 for a, b in zip(base, e2) if a["pred"] != b["pred"]) / len(base)
        tv = sum(total_variation(a["probs"], b["probs"]) for a, b in zip(base, e2)) / len(base)
        bad = flips > args.max_flip
        any_fail |= bad
        tier_report.update({"e2_flip_rate": flips, "e2_mean_tv": tv,
                            "e2_accuracy": sum(r["correct"] for r in e2) / len(e2)})
        verdicts.append(
            f"E2       {'FAIL' if bad else 'pass'} — {flips:.0%} of answers flip on "
            f"reversal (mean TV {tv:.3f})"
        )

        report["tiers"][tier] = tier_report
        print(f"\n── {tier} ──  n={len(base)}  accuracy {acc:.1%}  (chance {chance:.1%})")
        for v in verdicts:
            print(f"   {v}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
        print(f"\nwrote {args.json}")
    print(f"\n{'GATE FAILED' if any_fail else 'GATE PASSED'}")
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()

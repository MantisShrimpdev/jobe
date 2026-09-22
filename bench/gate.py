"""The control battery, run against a LOCAL Jobe backbone.

The driver: scores every task three ways — as is, with another task's evidence
(E1), with the options reversed (E2) — and hands the rows to TheLab's verdicts
(`thelab.decisions.gate`), which are the same criteria as EveryAppKit's
tools/decisionGate.ts so results stay comparable. Routing a local model through
an HTTP endpoint would reintroduce the top-N logprob window Jobe exists to
remove, which is why the driver is here and not there.

Reported PER TIER. Exits non-zero if any tier fails.

  python bench/gate.py --tasks <dir or files> --model D:/Coding/models/qwen35-4b
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from thelab.decisions.gate import GateThresholds, derange, e1_verdict, e2_verdict, position_verdict  # noqa: E402

from jobe import Decision, Option, load, score  # noqa: E402


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


def run_one(backbone, task, options, evidence):
    kind = task["question"]["type"]
    decision = Decision(id=task["id"], evidence=evidence, criterion=task["question"]["instructions"],
                        options=tuple(options), ordinal=kind == "score")
    readout = score(backbone.model, backbone.tokenizer, decision)
    probs = canonical(kind, readout.scores)
    pred = predict(probs, kind)
    return {"pred": pred, "correct": pred == str(task["expected"]), "conf": max(probs.values()), "probs": probs,
            "position": [o.id for o in options].index(readout.choice), "k": len(options)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-flip", type=float, default=GateThresholds.max_flip)
    ap.add_argument("--max-position", type=float, default=GateThresholds.max_position)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    thresholds = GateThresholds(max_flip=args.max_flip, max_position=args.max_position)

    files = ([os.path.join(args.tasks, f) for f in sorted(os.listdir(args.tasks)) if f.endswith(".jsonl")]
             if os.path.isdir(args.tasks) else [f.strip() for f in args.tasks.split(",") if f.strip()])

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
        chance = sum(1.0 / len(t["labels"]) for t in tasks) / len(tasks)
        print(f"  {tier}: E1 shuffled-context…", file=sys.stderr)
        idx = derange(len(tasks))
        e1 = [run_one(backbone, t, o, tasks[idx[i]]["state"]) for i, (t, o) in enumerate(zip(tasks, opts))]
        print(f"  {tier}: E2 reversed order…", file=sys.stderr)
        e2 = [run_one(backbone, t, list(reversed(o)), t["state"]) for t, o in zip(tasks, opts)]

        tier_report: dict = {"n": len(base)}
        verdicts: list[str] = []
        fail, rep, line = position_verdict([r["position"] for r in base], [r["k"] for r in base], thresholds)
        any_fail |= fail; tier_report.update(rep); verdicts.append(line)
        fail, rep, lines = e1_verdict([r["correct"] for r in base], [r["correct"] for r in e1], chance,
                                      [r["conf"] for r in base], [r["conf"] for r in e1], thresholds)
        any_fail |= fail; tier_report.update(rep); verdicts.extend(lines)
        fail, rep, line = e2_verdict([r["pred"] for r in base], [r["pred"] for r in e2], [r["probs"] for r in base],
                                     [r["probs"] for r in e2], [r["correct"] for r in e2], thresholds)
        any_fail |= fail; tier_report.update(rep); verdicts.append(line)

        report["tiers"][tier] = tier_report
        print(f"\n── {tier} ──  n={len(base)}  accuracy {tier_report['accuracy']:.1%}  (chance {chance:.1%})")
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

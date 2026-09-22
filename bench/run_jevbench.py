"""Run JevBench public tasks through Jobe and record everything needed to score.

Task mapping mirrors jevbench/adapters/semif_direct.py, so numbers are
comparable in shape to how SemIf itself was mapped:

  noul   -> options "true"/"false", remapped to the canonical "yes"/"no"
  choice -> one option per criteria key
  score  -> level indices "0".."k-1"
  every description prefixed "<id>: "

Raw option logits are recorded per decision, so a temperature can be fitted
afterwards with no further inference — calibration then iterates in
milliseconds instead of minutes.

  python bench/run_jevbench.py --tasks <dir> --model <hf-id> --out results.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from jobe import Decision, Option, load, score  # noqa: E402
from jobe.orders import score_averaged  # noqa: E402


def options_for(task: dict) -> list[Option]:
    question = task["question"]
    kind = question["type"]
    criteria = question.get("criteria")
    if kind == "noul":
        crit = criteria or {}
        pairs = [(k, crit.get(k) or f"The proposition is {k}.") for k in ("true", "false")]
    elif kind == "choice":
        pairs = [(k, v or k) for k, v in (criteria or {}).items()]
    else:
        pairs = [(str(i), level) for i, level in enumerate(criteria or [])]
    return [Option(id=k, description=f"{k}: {v}") for k, v in pairs]


def canonical(kind: str, scores: dict[str, float]) -> dict[str, float]:
    """noul is scored against JevBench's ["no", "yes"] labels."""
    if kind != "noul":
        return scores
    return {"yes": scores.get("true", 0.0), "no": scores.get("false", 0.0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="directory of .jsonl files, or comma-separated files")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--orders", default="declared", choices=["declared", "reversed", "rotations"])
    ap.add_argument("--limit", type=int, default=0, help="first N tasks per tier (0 = all)")
    args = ap.parse_args()

    if os.path.isdir(args.tasks):
        files = [os.path.join(args.tasks, f) for f in sorted(os.listdir(args.tasks)) if f.endswith(".jsonl")]
    else:
        files = [f.strip() for f in args.tasks.split(",") if f.strip()]

    backbone = load(args.model, device=args.device)
    print(f"loaded {backbone.name} on {backbone.device}/{backbone.dtype} "
          f"attn={backbone.attn_implementation}", file=sys.stderr)

    rows: list[dict] = []
    warmed = False
    for path in files:
        tier = os.path.splitext(os.path.basename(path))[0]
        tasks = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        if args.limit:
            tasks = tasks[: args.limit]
        started = time.perf_counter()
        for n, task in enumerate(tasks, 1):
            options = options_for(task)
            decision = Decision(
                id=task["id"],
                evidence=task["state"],
                criterion=task["question"]["instructions"],
                options=tuple(options),
                ordinal=task["question"]["type"] == "score",
            )
            if not warmed:  # spin GPU clocks up before anything is timed
                for _ in range(12):
                    score(backbone.model, backbone.tokenizer, decision)
                warmed = True
            try:
                if args.orders == "declared":
                    readout = score(backbone.model, backbone.tokenizer, decision)
                    flip, tv = 0.0, 0.0
                    logits = list(readout.option_logits)
                    ids = list(readout.option_ids)
                    seconds = readout.total_seconds
                else:
                    averaged = score_averaged(
                        backbone.model, backbone.tokenizer, decision, strategy=args.orders
                    )
                    readout = averaged
                    flip, tv = averaged.flip_rate, averaged.mean_total_variation
                    # Logits are per-order; keep the declared order's for calibration.
                    logits = list(averaged.per_order[0].option_logits)
                    ids = list(averaged.per_order[0].option_ids)
                    seconds = averaged.total_seconds
                kind = task["question"]["type"]
                rows.append({
                    "id": task["id"],
                    "tier": tier,
                    "family": task.get("family"),
                    "type": kind,
                    "labels": task["labels"],
                    "expected": task["expected"],
                    "ok": True,
                    "probs": canonical(kind, readout.scores),
                    "logit_ids": ids,
                    "logits": logits,
                    "flip_rate": flip,
                    "mean_tv": tv,
                    "seconds": seconds,
                })
            except Exception as exc:  # noqa: BLE001 - a failure is a result
                rows.append({
                    "id": task["id"], "tier": tier, "type": task["question"]["type"],
                    "labels": task["labels"], "expected": task["expected"],
                    "ok": False, "error": f"{type(exc).__name__}: {exc}",
                })
            if n % 25 == 0:
                print(f"  {tier}: {n}/{len(tasks)}", file=sys.stderr)
        print(f"  {tier}: {len(tasks)} tasks in {time.perf_counter()-started:.1f}s", file=sys.stderr)

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(json.dumps(r) for r in rows) + "\n")
    ok = sum(1 for r in rows if r.get("ok"))
    print(f"wrote {len(rows)} rows ({ok} ok) -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

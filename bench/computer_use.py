# -*- coding: utf-8 -*-
"""Run the computer-use decision set through Jobe and report per decision type.

Binary decisions are scored in BOTH option orders, because a two-option prompt
on this backbone has a measured ~20-point preference for the second slot and an
agent loop is mostly binaries.

Element menus over sixteen options are refused by the letter readout by design;
those fall through to `score_text`, and both facts are reported rather than
hidden, because a real page routinely has more than sixteen controls.
"""
import json
import os
import statistics as st
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jobe import Decision, Option, load, score  # noqa: E402
from jobe.prompt import DecisionError  # noqa: E402
from jobe.textscore import score_text  # noqa: E402
from thelab.core.gpu import require_free_gpu  # noqa: E402

import computer_use_tasks as T  # noqa: E402

MODEL = os.environ.get("JOBE_MODEL", r"D:\Coding\models\qwen35-4b")
require_free_gpu(3000)
bb = load(MODEL, device="auto")
print("backbone %s on %s\n" % (bb.name, bb.device), flush=True)

rows = []


def ask(tid, kind, evidence, criterion, options, gold, note=""):
    """Score once on the declared order; record the outcome."""
    d = Decision(id=tid, evidence=evidence, criterion=criterion, options=tuple(options))
    t0 = time.perf_counter()
    path = "letter"
    try:
        r = score(bb.model, bb.tokenizer, d)
    except DecisionError:
        path = "text"
        r = score_text(bb.model, bb.tokenizer, d)
    rows.append({
        "id": tid, "kind": kind, "path": path, "k": len(options),
        "pred": r.choice, "gold": gold, "correct": r.choice == gold,
        "conf": max(r.probabilities), "ms": (time.perf_counter() - t0) * 1000,
        "note": note,
    })
    return rows[-1]


def ask_both_orders(tid, kind, evidence, criterion, a, b, gold):
    """A two-option decision, scored a-then-b and b-then-a."""
    out = {}
    for tag, opts in (("ab", (a, b)), ("ba", (b, a))):
        d = Decision(id=tid + "-" + tag, evidence=evidence, criterion=criterion, options=opts)
        t0 = time.perf_counter()
        r = score(bb.model, bb.tokenizer, d)
        out[tag] = {"pred": r.choice, "p": dict(zip(r.option_ids, r.probabilities)),
                    "ms": (time.perf_counter() - t0) * 1000}
    avg = {o.id: (out["ab"]["p"][o.id] + out["ba"]["p"][o.id]) / 2 for o in (a, b)}
    pred_avg = max(avg.items(), key=lambda kv: kv[1])[0]
    rows.append({
        "id": tid, "kind": kind, "path": "letter", "k": 2,
        "pred": pred_avg, "gold": gold, "correct": pred_avg == gold,
        "pred_ab": out["ab"]["pred"], "pred_ba": out["ba"]["pred"],
        "correct_ab": out["ab"]["pred"] == gold, "correct_ba": out["ba"]["pred"] == gold,
        "order_disagrees": out["ab"]["pred"] != out["ba"]["pred"],
        "conf": max(avg.values()), "ms": out["ab"]["ms"] + out["ba"]["ms"], "note": "",
    })
    return rows[-1]


# ---------------------------------------------------------------- 1. elements
print("ELEMENT SELECTION  (Browser Use's one repeated question)", flush=True)
for tid, goal, rowsrc, crit, gold in T.ELEMENT:
    opts = [Option(str(i), text) for i, text in enumerate(rowsrc)]
    ev = goal + "\n\nInteractive elements on the page:\n" + T.table(rowsrc)
    r = ask(tid, "element", ev, crit, opts, gold, note="k=%d" % len(opts))
    flag = "" if r["correct"] else "   <- wrong, said [%s]" % r["pred"]
    print("  %-10s k=%2d  %-6s -> [%s] gold [%s] p=%.2f %6.0f ms%s"
          % (tid, r["k"], r["path"], r["pred"], gold, r["conf"], r["ms"], flag), flush=True)

# --------------------------------------------------------------- 2. operation
print("\nOPERATION  (CLICK / TYPE_TEXT / SELECT / SCROLL_UP / SCROLL_DOWN / WAIT / DONE / BLOCKED)", flush=True)
op_opts = [Option(o, T.OP_DESC[o]) for o in T.OPS]
for tid, ev, crit, gold in T.OPERATION:
    r = ask(tid, "operation", ev, crit, op_opts, gold)
    flag = "" if r["correct"] else "   <- wrong, said %s" % r["pred"]
    print("  %-10s -> %-11s gold %-11s p=%.2f %6.0f ms%s"
          % (tid, r["pred"], gold, r["conf"], r["ms"], flag), flush=True)

# ------------------------------------------------------------- 3. step status
print("\nSTEP STATUS  (done / continue / error / irreversible / blocked)", flush=True)
st_opts = [Option(k, v) for k, v in T.STATUS.items()]
for tid, ev, gold in T.STEP_STATUS:
    r = ask(tid, "status", ev, "What is the status of this step?", st_opts, gold)
    flag = "" if r["correct"] else "   <- wrong, said %s" % r["pred"]
    print("  %-10s -> %-13s gold %-13s p=%.2f %6.0f ms%s"
          % (tid, r["pred"], gold, r["conf"], r["ms"], flag), flush=True)

# ------------------------------------------------ 4. binary gates, both orders
print("\nBINARY GATES  (tool-risk and urgency; scored in both orders)", flush=True)
YES, NO = Option("yes", "yes"), Option("no", "no")
for tid, ev, crit, gold in T.GATES + T.MODERATION:
    kind = "gate" if tid.startswith("cu-ga") else "moderation"
    r = ask_both_orders(tid, kind, ev, crit, YES, NO, gold)
    flag = "" if r["correct"] else "   <- wrong"
    dis = " ORDER-FLIP" if r["order_disagrees"] else ""
    print("  %-10s -> %-3s gold %-3s  (yes-first %s / no-first %s)%s%s"
          % (tid, r["pred"], gold, r["pred_ab"], r["pred_ba"], dis, flag), flush=True)

# --------------------------------------------------------------- 5. routing
print("\nMODEL ROUTING  (fast / powerful; scored in both orders)", flush=True)
F, P = Option(*T.FAST), Option(*T.POWERFUL)
for tid, ev, gold in T.ROUTING:
    r = ask_both_orders(tid, "routing", ev, "Which model should handle this task?", F, P, gold)
    flag = "" if r["correct"] else "   <- wrong"
    dis = " ORDER-FLIP" if r["order_disagrees"] else ""
    print("  %-10s -> %-9s gold %-9s  (fast-first %s / powerful-first %s)%s%s"
          % (tid, r["pred"], gold, r["pred_ab"], r["pred_ba"], dis, flag), flush=True)

# ------------------------------------------------------------------- summary
print("\n" + "=" * 78)
print("%-12s %4s %8s %9s %10s %9s" % ("decision", "n", "correct", "accuracy", "median ms", "path"))
order = ["element", "operation", "status", "gate", "moderation", "routing"]
for kind in order:
    g = [r for r in rows if r["kind"] == kind]
    if not g:
        continue
    paths = sorted({r["path"] for r in g})
    print("%-12s %4d %8d %9.3f %10.0f %9s"
          % (kind, len(g), sum(r["correct"] for r in g), sum(r["correct"] for r in g) / len(g),
             st.median(r["ms"] for r in g), "+".join(paths)))
print("%-12s %4d %8d %9.3f %10.0f" % ("ALL", len(rows), sum(r["correct"] for r in rows),
      sum(r["correct"] for r in rows) / len(rows), st.median(r["ms"] for r in rows)))

bin_rows = [r for r in rows if "pred_ab" in r]
if bin_rows:
    ab = sum(r["correct_ab"] for r in bin_rows) / len(bin_rows)
    ba = sum(r["correct_ba"] for r in bin_rows) / len(bin_rows)
    flips = sum(r["order_disagrees"] for r in bin_rows)
    print("\nbinary decisions (%d): yes/fast first %.3f, no/powerful first %.3f, "
          "averaged %.3f; the orders disagree on %d"
          % (len(bin_rows), ab, ba, sum(r["correct"] for r in bin_rows) / len(bin_rows), flips))

over = [r for r in rows if r["k"] > 16]
if over:
    print("\n%d decisions had more than 16 options (k=%s) and fell through to the text "
          "readout: %d correct, median %.0f ms"
          % (len(over), sorted({r["k"] for r in over}), sum(r["correct"] for r in over),
             st.median(r["ms"] for r in over)))

out = r"D:\Coding\jobe\runs\computer-use.json"
with open(out, "w", encoding="utf-8") as fh:
    json.dump(rows, fh, indent=1)
print("\nwrote " + out)

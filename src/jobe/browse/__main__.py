# -*- coding: utf-8 -*-
"""Drive a browser from the command line - one line per goal, every decision printed.

    PYTHONPATH=src python -m jobe.browse "search nike" "open the top one"
    PYTHONPATH=src python -m jobe.browse --url tests/fixtures/stays.html "find a stay in Lisbon"

Each argument is one turn of a conversation against the SAME browser, so the
second can refer to what the first left on screen.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

MODEL = os.environ.get("JOBE_MODEL", r"D:\Coding\models\qwen35-4b")


def printer(log_path=None):
    fh = open(log_path, "a", encoding="utf-8") if log_path else None

    def emit(kind, data):
        data = {k: v for k, v in data.items() if k not in ("screenshot", "trace")}
        if fh:
            fh.write(json.dumps({"t": time.time(), "kind": kind, **data}, default=str) + "\n")
            fh.flush()
        if kind == "decision":
            ops = sorted(data["op_probs"].items(), key=lambda kv: -kv[1])[:3]
            t = data["timings"]
            line = "  %2d  %-10s %s" % (data["step"], data["operation"],
                                        " ".join("%s=%.2f" % kv for kv in ops))
            if data.get("done_check") is not None:
                line += "\n        done-check p(achieved)=%.2f%s" % (
                    data["done_check"], "  -> DONE overruled" if data.get("overruled") else "")
            if data.get("target_line"):
                line += "\n        -> %s  p=%.2f" % (data["target_line"][:70], data["target_p"])
            line += "\n        %d tok, prime %.0f ms, op %.0f ms%s, total %.0f ms" % (
                data["evidence_tokens"], t.get("prime_ms", 0), t.get("operation_ms", 0),
                (", target %.0f ms" % t["target_ms"]) if "target_ms" in t else "", t.get("total_ms", 0))
            print(line, flush=True)
        elif kind == "intent":
            print("  intent: %s (%.2f, %.0f ms)" % (data["intent"], data["confidence"], data["ms"]), flush=True)
        elif kind in ("typing", "acted", "note", "status", "confirm"):
            print("        %s: %s" % (kind, data.get("summary") or data.get("text") or data.get("what")),
                  flush=True)
        elif kind == "done":
            print("  => %s  (%s steps, %.1fs)%s" % (data["status"].upper(), data.get("steps", "-"),
                                                   data.get("ms", 0) / 1000,
                                                   ("  " + data["url"]) if data.get("url") else ""), flush=True)
        elif kind == "error":
            print("  !! " + data["text"], flush=True)
    return emit


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("turns", nargs="+", help="one conversation turn each")
    ap.add_argument("--url", help="start here instead of the home page")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--log", help="append every event as JSONL")
    ap.add_argument("--max-steps", type=int, default=18)
    args = ap.parse_args(argv)

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from jobe import load
    from jobe.browse.agent import Session
    from jobe.browse.policy import Policy

    t0 = time.perf_counter()
    bb = load(MODEL, device="auto")
    print("model loaded in %.0fs" % (time.perf_counter() - t0), flush=True)
    session = Session(Policy(bb), printer(args.log), headless=args.headless,
                      max_steps=args.max_steps, screenshots=False)
    try:
        if args.url:
            url = args.url
            if os.path.exists(url):
                from pathlib import Path
                url = Path(url).resolve().as_uri()
            session._ensure_browser().goto(url)
        for turn in args.turns:
            print("\n> " + turn, flush=True)
            session.say(turn)
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

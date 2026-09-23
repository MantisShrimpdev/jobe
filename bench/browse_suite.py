# -*- coding: utf-8 -*-
"""Browser tasks with independent success checks - the model's DONE is not evidence.

Each task starts a fresh page, gives the agent one goal (or a short
conversation), and then asks the PAGE whether it succeeded: a JavaScript
predicate on the fixture's real state, or the URL on a live site. The agent's
own status is recorded beside it, which is how a false DONE - the agent
claiming success on a page that disagrees - gets counted rather than trusted.

    PYTHONPATH=src python bench/browse_suite.py                 # everything
    PYTHONPATH=src python bench/browse_suite.py --only fixture  # offline only

Fixture tasks run on `tests/fixtures/stays.html` (jev-ultrafast's fixture,
MIT): deterministic, local, no network. Live tasks touch real sites and are
read-only - searching and opening results, never submitting anything.

Live tasks run HEADED by default, because that is how the chat app runs and
because headless Chromium is walled: on 2026-09-23 one headless results page
per engine got a challenge from DuckDuckGo, Google, Brave and Ecosia and a
refusal from Mojeek and Startpage. Round 4's DuckDuckGo "passes" were checked
by URL alone - `q=nike` is in the URL of the challenge page too - so a live
pass now also needs result links on the page and no wall. `stretch` tasks are
reported but not counted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

FIXTURE = (Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "stays.html").as_uri()
MODEL = os.environ.get("JOBE_MODEL", r"D:\Coding\models\qwen35-4b")

CARDS = "[...document.querySelectorAll('#results .card h2')].map(e => e.textContent)"

#: At least five visible links that leave the search engine: a results page, not a wall.
RESULTS = ("() => [...document.querySelectorAll('a[href^=\"http\"]')].filter(a => a.offsetParent "
           "&& !/(^|\\.)(duckduckgo|bing|google|microsoft)\\./.test(new URL(a.href).hostname)).length >= 5")

TASKS = [
    # ---------------------------------------------------------------- fixture
    {"id": "fx-search", "set": "fixture", "url": FIXTURE,
     "turns": ["find a stay in Lisbon"],
     "check": "() => searched && query.trim().toLowerCase() === 'lisbon'"},
    {"id": "fx-category", "set": "fixture", "url": FIXTURE,
     "turns": ["show only nature stays"],
     "check": "() => category === 'Nature'"},
    {"id": "fx-free", "set": "fixture", "url": FIXTURE,
     "turns": ["show stays with free cancellation"],
     "check": "() => free === true"},
    {"id": "fx-open", "set": "fixture", "url": FIXTURE,
     "turns": ["open Serra Lodge"],
     "check": "() => location.hash === '#serra-lodge'"},
    {"id": "fx-search-open", "set": "fixture", "url": FIXTURE,
     "turns": ["find a stay in Copenhagen and open it"],
     "check": "() => location.hash === '#the-glasshouse'"},
    {"id": "fx-two-turns", "set": "fixture", "url": FIXTURE,
     "turns": ["search for Lisbon", "open the cheapest one"],
     "check": "() => location.hash === '#serra-lodge'"},
    {"id": "fx-combined", "set": "fixture", "url": FIXTURE,
     "turns": ["find a Lisbon stay with free cancellation"],
     "check": "() => searched && query.toLowerCase().includes('lisbon') && free === true"},
    {"id": "fx-reading", "set": "fixture", "url": FIXTURE,
     "turns": ["open the reading room"],
     "check": "() => /reading/i.test(document.body.innerText.slice(0, 400)) && !document.getElementById('destination')"},
    # ------------------------------------------------------------------- live
    {"id": "ddg-search", "set": "live", "url": "https://duckduckgo.com",
     "turns": ["search nike"],
     "check_url": lambda u: "q=nike" in u.lower(), "check": RESULTS},
    {"id": "ddg-search-open", "set": "live", "url": "https://duckduckgo.com",
     "turns": ["search for the python programming language", "open the top result"],
     "check_url": lambda u: "duckduckgo.com" not in u and u.startswith("http")},
    {"id": "wiki-search", "set": "live", "url": "https://en.wikipedia.org/wiki/Main_Page",
     "turns": ["search wikipedia for the eiffel tower"],
     "check_url": lambda u: "eiffel_tower" in u.lower()},
    {"id": "ddg-news", "set": "live", "url": "https://duckduckgo.com",
     "turns": ["search for the latest news on github"],
     "check_url": lambda u: "q=" in u and "github" in u.lower(), "check": RESULTS},
    # the person's own script, word for word
    {"id": "ddg-news-open", "set": "live", "url": "https://duckduckgo.com",
     "turns": ["search for the latest news on github", "open the top one"],
     "check_url": lambda u: "duckduckgo.com" not in u and u.startswith("http")},
    {"id": "bing-search-open", "set": "live", "url": "https://www.bing.com",
     "turns": ["search for the python programming language", "open the top result"],
     "check_url": lambda u: "bing.com" not in u and u.startswith("http")},
    # ---- the first real session, 2026-09-24, word for word; each failed then
    {"id": "bing-click", "set": "live", "url": "https://www.bing.com",
     "turns": ["click image creator"],
     "check_url": lambda u: "/images/create" in u, "expect_status": "done", "expect_typed": []},
    {"id": "run-on", "set": "live", "url": "https://www.bing.com",
     "turns": ["search for the latest news on github open the top one"],
     "check_url": lambda u: "bing.com" not in u and u.startswith("http"),
     "expect_typed": ["latest news on github"]},
    {"id": "search-again", "set": "live", "url": "https://www.bing.com/search?q=latest+news+on+github",
     "turns": ["search github"],
     "check_url": lambda u: "q=github" in u.lower(), "expect_typed": ["github"]},
    {"id": "go-to-site", "set": "live", "url": "https://www.bing.com",
     "turns": ["got to github"],
     "check_url": lambda u: (urlparse(u).hostname or "").endswith("github.com")},
    {"id": "open-by-name", "set": "live", "url": "https://github.com/login",
     "turns": ["open blender"],
     "check_url": lambda u: "blender.org" in u, "expect_typed": []},
    {"id": "browser-again", "set": "live", "url": "https://www.bing.com",
     "turns": ["open browser", "open browser"],
     "check_url": lambda u: "bing.com" in u, "expect_status": "ready"},
    # ------------------------------------------------------------ stretch
    {"id": "flight-search", "set": "stretch", "url": "https://duckduckgo.com",
     "turns": ["find the best flight from melbourne to perth in october"],
     "check_url": lambda u: "perth" in u.lower(), "check": RESULTS},
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["fixture", "live", "stretch"])
    ap.add_argument("--ids", help="comma-separated task ids")
    ap.add_argument("--headless", action="store_true", help="live tasks headless too (they get walled)")
    ap.add_argument("--headed", action="store_true", help="fixture tasks headed too")
    ap.add_argument("--out", default="runs/browse-suite.json")
    args = ap.parse_args(argv)

    from jobe import load
    from jobe.browse.agent import Session
    from jobe.browse.policy import Policy

    tasks = [t for t in TASKS if (not args.only or t["set"] == args.only)
             and (not args.ids or t["id"] in args.ids.split(","))]
    bb = load(MODEL, device="auto")
    policy = Policy(bb)
    results = []

    for t in tasks:
        events: list = []
        headless = args.headless if t["set"] != "fixture" else not args.headed
        session = Session(policy, lambda k, d: events.append((k, d)), headless=headless,
                          screenshots=False, max_steps=14)
        t0 = time.perf_counter()
        status, url, ok, err, wall = "?", "", False, None, None
        try:
            session._ensure_browser().goto(t["url"])
            for turn in t["turns"]:
                session.say(turn)
            done = [d for k, d in events if k == "done"]
            status = done[-1]["status"] if done else "no-done"
            errs = [d["text"] for k, d in events if k == "error"]
            err = errs[-1] if errs else None
            page = session.browser.page
            url = page.url
            wall = session.browser.challenge(session.browser.observe())
            # every check the task has must hold, and a wall is never a pass: a
            # challenge page has q=nike in its URL too
            ok = not wall
            if "check_url" in t:
                ok = ok and bool(t["check_url"](url))
            if "check" in t:
                ok = ok and bool(page.evaluate(t["check"]))
            if "expect_status" in t:
                ok = ok and status == t["expect_status"]
            if "expect_typed" in t:
                ok = ok and [d["text"] for k, d in events if k == "typing"] == t["expect_typed"]
            if wall:
                err = "WALL: " + wall
        except Exception as exc:  # noqa: BLE001
            err = "%s: %s" % (type(exc).__name__, str(exc)[:160])
        finally:
            session.close()
        decisions = [d for k, d in events if k == "decision"]
        claimed = status == "done"
        row = {"id": t["id"], "set": t["set"], "ok": ok, "status": status, "wall": wall,
               "headless": headless,
               "false_done": claimed and not ok, "steps": len(decisions),
               "seconds": time.perf_counter() - t0, "url": url, "error": err,
               "trace": ["%s%s%s" % (d["operation"],
                                     (" -> " + (d["target_line"] or "")[:48]) if d.get("target_line") else "",
                                     " [DONE overruled]" if d.get("overruled") else "")
                         for d in decisions],
               "typed": [d["text"] for k, d in events if k == "typing"],
               "op_probs": [{k: round(v, 3) for k, v in sorted(d["op_probs"].items(), key=lambda kv: -kv[1])[:4]}
                            for d in decisions],
               "decision_ms": [d["timings"].get("total_ms", 0) for d in decisions],
               "evidence_tokens": [d["evidence_tokens"] for d in decisions]}
        results.append(row)
        print("%-16s %-4s %-12s %2d steps %5.1fs  %s" % (
            t["id"], "PASS" if ok else "FAIL", status.upper(), row["steps"], row["seconds"],
            ("typed " + repr(row["typed"])) if row["typed"] else ""), flush=True)
        if not ok:
            for line in row["trace"]:
                print("      . " + line, flush=True)
            if err:
                print("      ! " + err, flush=True)

    counted = [r for r in results if r["set"] != "stretch"]
    n = len(counted)
    passed = sum(r["ok"] for r in counted)
    false_done = sum(r["false_done"] for r in counted)
    all_ms = [ms for r in results for ms in r["decision_ms"]]
    all_tok = [x for r in results for x in r["evidence_tokens"]]
    print("\n%d/%d passed | %d false DONE | %d walls | %d decisions, median %.0f ms, median %d evidence tokens"
          % (passed, n, false_done, sum(bool(r["wall"]) for r in results), len(all_ms),
             sorted(all_ms)[len(all_ms) // 2] if all_ms else 0,
             sorted(all_tok)[len(all_tok) // 2] if all_tok else 0))
    for r in results:
        if r["set"] == "stretch":
            print("stretch %-16s %s %s" % (r["id"], "PASS" if r["ok"] else "FAIL", r["status"]))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"passed": passed, "n": n, "false_done": false_done, "results": results}, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

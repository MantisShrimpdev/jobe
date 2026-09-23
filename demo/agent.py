# -*- coding: utf-8 -*-
"""A browser agent whose every decision is one forward pass of a frozen 4B.

The split is the one people ship with Jev: a plan says what outcome is wanted,
in words, and the model decides. Nothing here is generated. Each loop takes an
atomic DOM snapshot, renders it as a numbered table of interactive controls, and
asks the readout three questions - which element, which operation, what is the
status of this step - then acts with Playwright.

WHY THE NARROWING STEP EXISTS. Browser Use passes the whole snapshot and lets
the model pick from it. This readout cannot: it answers with single-token
letters and so stops at sixteen options, and scoring option text instead is
about eighteen times slower (measured, `bench/RESULTS.md`). So a page with more
than sixteen controls is split into groups, the group is chosen, then the
element within it. Two 80 ms decisions instead of one, which is still an order
of magnitude under a generated answer.

WHAT IT CANNOT DO. It cannot read a value off the page and type it. A dropdown
is enumerable so choosing a value is a decision, but free text has to come from
the plan. That boundary is real and the demo does not hide it.

    python demo/agent.py                       # the bundled site, headed
    python demo/agent.py --url http://... --goal "..."
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from jobe import Decision, Option, load, score  # noqa: E402

MAX_OPTIONS = 16
GROUP = 6
OP_DESC = {
    "CLICK": "click the element",
    "SELECT": "choose a value from this dropdown",
    "TYPE_TEXT": "type text into this field",
}
#: What each kind of control can actually take. Asking beyond this invites a
#: confident-sounding answer to a question the DOM has already settled.
LEGAL_OPS = {
    "dropdown": ("SELECT",),
    "text": ("TYPE_TEXT", "CLICK"),
    "search": ("TYPE_TEXT", "CLICK"),
    "textarea": ("TYPE_TEXT", "CLICK"),
    "checkbox": ("CLICK",),
    "button": ("CLICK",),
    "a": ("CLICK",),
}
# DONE is deliberately NOT an operation. An operation is what you do TO an
# element; whether the task is finished is a different question with a different
# answer set, and offering it in both places let a 0.62-confidence DONE stop the
# run one click short of the goal.

SNAPSHOT_JS = r"""
() => {
  const sel = 'a[href],button,input,select,textarea,[role=button],[role=link]';
  const out = [];
  for (const el of document.querySelectorAll(sel)) {
    const r = el.getBoundingClientRect();
    const st = getComputedStyle(el);
    if (!r.width || !r.height || st.visibility === 'hidden' || st.display === 'none') continue;
    if (el.disabled) continue;
    if (r.bottom < 0 || r.top > innerHeight * 3) continue;
    const tag = el.tagName.toLowerCase();
    let label = (el.getAttribute('aria-label') || el.labels?.[0]?.textContent ||
                 el.innerText || el.placeholder || el.name || '').trim().replace(/\s+/g, ' ');
    let kind = tag === 'input' ? (el.type || 'text') : tag;
    let value = ('value' in el && el.value !== undefined) ? String(el.value) : '';
    if (tag === 'select') { value = (el.options[el.selectedIndex] || {}).textContent || '(none)'; value = value.trim(); kind = 'dropdown'; }
    if (el.type === 'checkbox') value = el.checked ? 'checked' : 'unchecked';
    const row = el.closest('tr');
    const ctx = row ? Array.from(row.children).map(c => c.innerText.trim())
                          .filter(Boolean).join(', ') : '';
    out.push({
      i: out.length, kind, label: label.slice(0, 60), value: value.slice(0, 40),
      context: ctx.slice(0, 120),
      id: el.id || '', x: Math.round(r.x + r.width / 2), y: Math.round(r.y + r.height / 2),
    });
  }
  return out;
}
"""

HUD_JS = r"""
(lines) => {
  let hud = document.getElementById('__jobe_hud');
  if (!hud) {
    hud = document.createElement('div');
    hud.id = '__jobe_hud';
    hud.style.cssText = 'position:fixed;right:16px;top:16px;width:430px;z-index:2147483647;' +
      'background:#0d1117ee;color:#e6edf3;font:12px/1.55 ui-monospace,Menlo,Consolas,monospace;' +
      'padding:14px 16px;border-radius:12px;box-shadow:0 8px 30px rgba(0,0,0,.35);' +
      'backdrop-filter:blur(3px);white-space:pre-wrap;pointer-events:none';
    document.body.appendChild(hud);
  }
  hud.textContent = lines;
  // Reserve the space rather than cover the page: a recording of an agent is
  // useless if the panel narrating it sits on top of what it changed.
  document.body.style.paddingRight = '470px';
}
"""

PAGE_TEXT_JS = r"""
() => {
  const main = document.querySelector('main') || document.body;
  return (main.innerText || '').trim().replace(/\n{2,}/g, '\n').slice(0, 700);
}
"""

HILITE_JS = r"""
(box) => {
  let h = document.getElementById('__jobe_hilite');
  if (!h) {
    h = document.createElement('div');
    h.id = '__jobe_hilite';
    h.style.cssText = 'position:fixed;z-index:2147483646;border:3px solid #f0883e;' +
      'border-radius:8px;box-shadow:0 0 0 4px #f0883e33;pointer-events:none;' +
      'transition:all .18s ease';
    document.body.appendChild(h);
  }
  if (!box) { h.style.opacity = '0'; return; }
  h.style.opacity = '1';
  h.style.left = box.x + 'px'; h.style.top = box.y + 'px';
  h.style.width = box.w + 'px'; h.style.height = box.h + 'px';
}
"""


def signature(els: list) -> str:
    """What the page looks like, cheaply. Two identical signatures across an
    action mean the action did nothing, and repeating it is a loop."""
    return json.dumps([[e["kind"], e["label"], e["value"]] for e in els])


def describe(e: dict) -> str:
    bits = [e["kind"]]
    if e["label"]:
        bits.append(e["label"])
    if e["value"]:
        bits.append("= " + e["value"])
    if e["context"] and e["context"] != e["label"]:
        bits.append("(" + e["context"] + ")")
    return " ".join(bits)


class Agent:
    def __init__(self, backbone, page, log):
        self.bb, self.page, self.log = backbone, page, log
        self.decisions = 0
        self.ms = 0.0
        self.tokens = 0

    def decide(self, what, evidence, criterion, options, ordinal=False):
        """One decision. Returns (choice, probability, milliseconds)."""
        t0 = time.perf_counter()
        r = score(self.bb.model, self.bb.tokenizer, Decision(
            id="%s-%d" % (what, self.decisions), evidence=evidence,
            criterion=criterion, options=tuple(options), ordinal=ordinal))
        ms = (time.perf_counter() - t0) * 1000
        self.decisions += 1
        self.ms += ms
        self.tokens += r.input_tokens
        return r.choice, max(r.probabilities), ms, r.input_tokens

    def pick_element(self, goal, els):
        """Which element to act on. Grouped when the menu exceeds the letter ceiling."""
        # Option ids are the element's index in the FULL snapshot, which is not its
        # position in this list once already-tried ones have been filtered out.
        by_i = {str(e["i"]): e for e in els}
        table = "\n".join("[%d] %s" % (e["i"], describe(e)) for e in els)
        ev = goal + "\n\nInteractive elements on the page:\n" + table
        crit = "Which element should I act on next to make progress toward the goal?"
        if len(els) <= MAX_OPTIONS:
            opts = [Option(str(e["i"]), describe(e)) for e in els]
            choice, p, ms, tk = self.decide("element", ev, crit, opts)
            self.log("  decide  which element   -> [%s] %-34s %5.0f ms  p=%.2f  %4d tok"
                     % (choice, describe(by_i[choice])[:34], ms, p, tk))
            return by_i[choice], ms

        groups = [els[i:i + GROUP] for i in range(0, len(els), GROUP)]
        gopts = [Option("g%d" % gi, "elements [%d]-[%d]: %s" % (
            g[0]["i"], g[-1]["i"], "; ".join(describe(e)[:28] for e in g)))
            for gi, g in enumerate(groups)]
        gchoice, gp, gms, gtk = self.decide("group", ev, crit, gopts)
        g = groups[int(gchoice[1:])]
        self.log("  narrow  %d controls -> group %s of %d  [%d]-[%d]   %5.0f ms  p=%.2f  %4d tok"
                 % (len(els), gchoice[1:], len(groups), g[0]["i"], g[-1]["i"], gms, gp, gtk))
        opts = [Option(str(e["i"]), describe(e)) for e in g]
        choice, p, ms, tk = self.decide("element", ev, crit, opts)
        self.log("  decide  which element   -> [%s] %-34s %5.0f ms  p=%.2f  %4d tok"
                 % (choice, describe(by_i[choice])[:34], ms, p, tk))
        return by_i[choice], gms + ms

    def pick_operation(self, goal, el):
        """Only what the element can take. A question the DOM has already
        settled is not a decision, and asking it anyway is how you get a
        0.41-confidence TYPE_TEXT on a dropdown."""
        legal = LEGAL_OPS.get(el["kind"], ("CLICK",))
        if len(legal) == 1:
            self.log("  forced  which action    -> %-38s        (a %s takes only this)"
                     % (legal[0], el["kind"]))
            return legal[0], 0.0
        ev = goal + chr(10) + chr(10) + "The chosen element is: " + describe(el)
        opts = [Option(o, OP_DESC[o]) for o in legal]
        choice, p, ms, tk = self.decide(
            "operation", ev, "What operation do I perform on the chosen element?", opts)
        self.log("  decide  which action    -> %-38s %5.0f ms  p=%.2f  %4d tok" % (choice, ms, p, tk))
        return choice, ms

    def pick_value(self, goal, el, values):
        """A dropdown's options are enumerable, so the value is a decision too."""
        ev = goal + "\n\nThe dropdown is: " + describe(el)
        opts = [Option(v, v) for v in values if v.strip()][:MAX_OPTIONS]
        choice, p, ms, tk = self.decide(
            "value", ev, "Which value should this dropdown be set to?", opts)
        self.log("  decide  which value     -> %-38s %5.0f ms  p=%.2f  %4d tok" % (choice, ms, p, tk))
        return choice, ms

    def pick_status(self, goal, els, text=""):
        """Whether the task is finished needs the page's WORDS, not its controls.

        Measured the hard way: given only the control list, this decision missed a
        completed booking, because a confirmation page's controls are just the nav
        bar. The sentence "Booking held - Ryanair" is the whole signal and it is
        not a control.
        """
        ev = goal
        if text:
            ev += "\n\nThe page now reads:\n" + text
        ev += "\n\nAnd offers these controls:\n" + "\n".join(
            "[%d] %s" % (e["i"], describe(e)) for e in els[:24])
        opts = [
            Option("continue", "the goal is not achieved yet and more steps are needed"),
            Option("done", "the page confirms the goal has been achieved, stop"),
            Option("blocked", "the page cannot be progressed at all"),
            Option("error", "the page shows an error"),
        ]
        choice, p, ms, tk = self.decide("status", ev, "What is the status of this task?", opts)
        self.log("  decide  step status     -> %-38s %5.0f ms  p=%.2f  %4d tok" % (choice, ms, p, tk))
        return choice, ms, p


def keep_warm(stop):
    """A tiny periodic GPU op, because an idle card downclocks.

    Measured here: between decisions the process waits on the browser, the 3080
    falls from about 2,130 MHz to 285, and a decision that costs 80 ms under
    load costs two to three seconds cold. Production traffic never idles, so
    80 ms is the honest steady-state figure; an interactive loop has to hold the
    clock up itself or it ends up reporting the power state rather than the
    model. Run with --no-keep-warm to watch it happen.
    """
    import torch
    if not torch.cuda.is_available():
        return
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    while not stop.is_set():
        for _ in range(8):
            a = a @ a.T / 2048.0
        torch.cuda.synchronize()
        time.sleep(0.01)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://localhost:8824")
    # A plan that never says what finished LOOKS like cannot expect a decision
    # model to infer it. The last sentence is the whole difference between this
    # loop stopping at the confirmation page and wandering into the nav bar.
    ap.add_argument("--goal", default="Goal: search for a flight from London to Rome, "
                                      "direct flights only, and select the cheapest direct "
                                      "fare. The task is complete once the page confirms a "
                                      "booking is held.")
    ap.add_argument("--model", default=os.environ.get("JOBE_MODEL", r"D:\Coding\models\qwen35-4b"))
    ap.add_argument("--typed", default='{"Return": "", "Depart": "12 Oct"}',
                    help="free text the PLAN supplies; the readout never invents it")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--slow", type=int, default=450, help="ms of dwell between steps, for recording")
    ap.add_argument("--out", default="")
    ap.add_argument("--shot", default="", help="save a final screenshot here")
    ap.add_argument("--gate", type=float, default=0.80,
                    help="hand over when the status decision is less sure than this")
    ap.add_argument("--no-keep-warm", action="store_true",
                    help="let the GPU downclock between decisions; see keep_warm()")
    args = ap.parse_args(argv)

    from playwright.sync_api import sync_playwright

    typed = json.loads(args.typed)
    lines: list[str] = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    print("loading the backbone ...", flush=True)
    t0 = time.perf_counter()
    bb = load(args.model, device="auto")
    print("ready in %.0f s  -  %s" % (time.perf_counter() - t0, bb.name), flush=True)
    import threading
    stop = threading.Event()
    if not args.no_keep_warm:
        threading.Thread(target=keep_warm, args=(stop,), daemon=True).start()
        print("holding the GPU at clock between decisions (--no-keep-warm to see the cost)")
    print("\n" + args.goal + "\n", flush=True)

    trace = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=args.headless, args=["--window-size=1560,980"])
        page = browser.new_page(viewport={"width": 1540, "height": 940})
        page.goto(args.url)
        page.wait_for_load_state("networkidle")
        agent = Agent(bb, page, log)
        tried: set[tuple] = set()

        for step in range(1, args.max_steps + 1):
            els = page.evaluate(SNAPSHOT_JS)
            before = signature(els)
            # "Already tried" is keyed to the PAGE STATE, not to the element. An
            # element that did nothing here may be exactly right two steps later
            # (Search before a destination is chosen), and an element that flips a
            # checkbox back and forth changes the page every time, so a plain
            # no-change test never catches the oscillation. Pairing the two does.
            live = [e for e in els if (before, e["id"] or describe(e)) not in tried]
            if len(live) < len(els):
                log("step %d   %d controls, %d already tried in this exact state"
                    % (step, len(els), len(els) - len(live)))
            if not live:
                log("step %d   every control has been tried in this state; stopping" % step)
                break
            els = live
            if not els:
                log("step %d: no interactive elements; stopping" % step)
                break
            log("step %d   snapshot: %d interactive controls" % (step, len(els)))

            el, ms_el = agent.pick_element(args.goal, els)
            box = page.evaluate(
                "(id) => { const e = id ? document.getElementById(id) : null;"
                " if (!e) return null; const r = e.getBoundingClientRect();"
                " return {x:r.x, y:r.y, w:r.width, h:r.height}; }", el["id"])
            page.evaluate(HILITE_JS, box)
            page.wait_for_timeout(args.slow)

            op, ms_op = agent.pick_operation(args.goal, el)
            ms_val = 0.0
            handle = page.locator("#" + el["id"]) if el["id"] else None

            if op == "DONE":
                log("  act     DONE\n")
                trace.append({"step": step, "op": "DONE"})
                break
            if handle is None:
                log("  act     skipped, the element has no id to target\n")
                continue
            try:
                if op == "SELECT":
                    values = page.evaluate(
                        "(id) => Array.from(document.getElementById(id).options)"
                        ".map(o => o.textContent.trim())", el["id"])
                    value, ms_val = agent.pick_value(args.goal, el, values)
                    handle.select_option(label=value)
                    log("  act     SELECT %r" % value)
                elif op == "TYPE_TEXT":
                    value = typed.get(el["label"], typed.get(el["id"], ""))
                    handle.fill(value)
                    log("  act     TYPE_TEXT %r  (supplied by the plan, not the readout)" % value)
                elif op == "SCROLL_DOWN":
                    page.mouse.wheel(0, 500)
                    log("  act     SCROLL_DOWN")
                elif op == "WAIT":
                    page.wait_for_timeout(600)
                    log("  act     WAIT")
                else:
                    handle.click()
                    log("  act     CLICK")
            except Exception as exc:  # noqa: BLE001 - a demo must say what broke, not die
                log("  act     failed: %s" % str(exc).splitlines()[0][:90])
            page.wait_for_timeout(args.slow)
            page.evaluate(HILITE_JS, None)

            after = page.evaluate(SNAPSHOT_JS)
            tried.add((before, el["id"] or describe(el)))
            if signature(after) == before and op not in ("WAIT", "DONE"):
                log("  note    the page did not change")
            status, ms_st, status_p = agent.pick_status(
                args.goal, after, page.evaluate(PAGE_TEXT_JS))
            trace.append({"step": step, "element": describe(el), "op": op,
                          "status": status, "ms": ms_el + ms_op + ms_val + ms_st})
            page.evaluate(HUD_JS, "\n".join(lines[-14:]))
            log("")
            # The gate measured in bench/computer_use.py: a status below 0.80 is
            # where every one of that set's errors lived, and "is this finished"
            # is the decision this readout is weakest at. So an unsure status is
            # handed over rather than acted on.
            # Status is the decision this readout is measurably weakest at
            # (4 of 6 in bench/computer_use.py), and an agent that does not know
            # where it is should stop rather than guess. So ANY status under the
            # gate hands over, whichever way it points.
            if status_p < args.gate:
                log("stopping: status %r at only p=%.2f, under the %.2f gate - "
                    "handing to a human" % (status, status_p, args.gate))
                break
            if status in ("done", "blocked", "error"):
                log("stopping: status is %s at p=%.2f" % (status, status_p))
                break

        page.evaluate(HUD_JS, "\n".join(lines[-14:]))
        log("%d decisions, %.0f ms of model time, %.0f ms and %d prompt tokens each"
            % (agent.decisions, agent.ms, agent.ms / max(1, agent.decisions),
               agent.tokens / max(1, agent.decisions)))
        if args.shot:
            page.screenshot(path=args.shot, full_page=False)
            print("wrote " + args.shot)
        page.wait_for_timeout(2500)
        browser.close()
    stop.set()

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"goal": args.goal, "url": args.url, "trace": trace,
                       "decisions": agent.decisions, "model_ms": agent.ms}, fh, indent=1)
        print("wrote " + args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

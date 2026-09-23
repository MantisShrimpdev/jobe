# -*- coding: utf-8 -*-
"""The loop: observe, decide, guard, act - one goal at a time, one browser session.

A conversation is a sequence of goals against ONE browser that persists
between them, which is what makes "open the top one" mean something: the
results page from the previous line is still there.

Everything the loop does is emitted as an event, so the chat window can show
each decision as it happens - operation, target, probabilities, timing and a
screenshot with the target outlined.

What the loop tells the model after every action is WHAT CHANGED - "Click
'View Serra Lodge' -> URL now ...#serra-lodge; new text: Serra Lodge, A
hillside retreat". Round 1 without it: on a detail page the model could not
tell the thing it was asked to open was already open, and kept clicking for
fourteen steps.

Bounded on purpose: a step budget per goal, a stop when the same action (by
MEANING - operation, label, text) repeats three times or two actions alternate,
and a stop after three actions that changed nothing. "Changed" is semantic:
single-page apps re-render with fresh node identities, so a raw fingerprint
changes on every click even when nothing a person would notice did. And one
boundary that is this project's: an action the readout judges hard to undo
pauses for the person to allow it.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import traceback

from .browser import Browser, StalePage
from .policy import Choice, Policy
from .text import CLAUSES as _CLAUSES, OPEN as _OPEN, unset_controls, unused_words

HOME = "https://duckduckgo.com"
IRREVERSIBLE_AT = 0.6

#: What a line from the person can be. Round 1 read "show only nature stays" as
#: a request to describe the page, so `read` is narrow and `act` spells out that
#: filtering and showing things are actions.
INTENTS = {
    "act": "Do something in the browser: search, open, click, type, filter, sort, show only certain "
           "results, select an option, go to a site.",
    "open": "Only open or start the browser, or go back to a start page. No task beyond that.",
    "read": "Only answer a question ABOUT the page in words, like 'what is on the page?' or 'read it "
            "to me'. No clicking or typing.",
    "close": "Close or quit the browser.",
}

#: A click on one of these sets a value; it does not open anything. Round 6: for
#: "open Serra Lodge" a ticked checkbox satisfied the open-goal rule and DONE
#: was called on the search page.
_NOT_OPENING = {"checkbox", "radio", "switch", "combobox", "textbox", "searchbox", "spinbutton",
                "menuitemcheckbox", "menuitemradio"}

#: After Jobe stops at a human-verification page, the person solves it and says so.
_CONTINUE = re.compile(r"(continue|go on|carry on|resume|keep going|try again|done|ok(?:ay)?|"
                       r"i did it|i've done it|solved(?: it)?|all good|go ahead)[.!]*", re.I)

_CANCEL = re.compile(r"(no|n|nope|cancel|stop|don't|do not|never ?mind|leave it)[.!]*", re.I)

_URL = re.compile(r"\b((?:https?://)?(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s]*)?)", re.I)


class Stop(Exception):
    """The person pressed Stop."""


def meaning(page: dict) -> str:
    """A fingerprint of what a person would notice: URL, title, text, controls - no node ids."""
    controls = sorted((a.get("kind"), a.get("role"), a.get("label"), str(a.get("value", "")),
                       str(a.get("checked")), str(a.get("expanded")))
                      for a in page.get("actions") or [] if a.get("kind") in ("click", "fill", "select"))
    blob = json.dumps([page.get("url"), page.get("title"), page.get("text"), controls], default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def what_changed(before: dict, after: dict) -> str:
    """One short clause: the part of the page the last action changed."""
    parts = []
    if before.get("url") != after.get("url"):
        parts.append("URL now " + _trim(after.get("url", ""), 90))
    elif before.get("title") != after.get("title"):
        parts.append("title now " + _trim(after.get("title", ""), 60))
    seen = set((before.get("text") or "").split("\n"))
    new = [ln.strip() for ln in (after.get("text") or "").split("\n") if ln.strip() and ln not in seen]
    if new:
        parts.append("new text: " + _trim(" · ".join(new), 150))
    if not parts:
        fields_b = {a.get("label"): a.get("value") for a in before.get("actions") or [] if a.get("kind") == "fill"}
        fields_a = {a.get("label"): a.get("value") for a in after.get("actions") or [] if a.get("kind") == "fill"}
        diff = [k for k in fields_a if fields_a[k] != fields_b.get(k)]
        if diff:
            parts.append("%s now %s" % (_q(diff[0]), _q(fields_a[diff[0]])))
    return "; ".join(parts) if parts else "nothing visible changed"


class Session:
    """One browser, one conversation, all decisions through one Policy."""

    def __init__(self, policy: Policy, emit, *, headless: bool = False, home: str = HOME,
                 max_steps: int = 14, screenshots: bool = True):
        self.policy = policy
        self.emit = emit
        self.headless = headless
        self.home = home
        self.max_steps = max_steps
        self.screenshots = screenshots
        self.browser: Browser | None = None
        self.history: list[dict] = []
        self.pending_text: dict | None = None
        self.stop_requested = False
        self.pending: dict | None = None      # an irreversible action awaiting the person
        self.blocked_goal: str | None = None  # a goal stopped at a human-verification page
        self._current_goal = ""
        self._last_goal = ""

    # ------------------------------------------------------------ session

    def _ensure_browser(self) -> Browser:
        if self.browser is None or not self.browser.alive:
            if self.browser is not None:
                self.browser.close()
            self.browser = Browser(headless=self.headless)
            self.history = []
            self.pending_text = None
        return self.browser

    def close(self):
        if self.browser is not None:
            self.browser.close()
            self.browser = None
        self.history = []
        self.pending_text = None

    def _check_stop(self):
        if self.stop_requested:
            self.stop_requested = False
            raise Stop()

    # ------------------------------------------------------------- intent

    def intent(self, text: str) -> tuple[str, dict, float]:
        """Which kind of request this is - a choice over four, never a parse."""
        from ..prompt import Option
        state = "UTTERANCE: %s\nBROWSER OPEN: %s\nCURRENT PAGE: %s" % (
            text, "yes" if self.browser and self.browser.alive else "no",
            (self.browser.page.url if self.browser and self.browser.alive else "none"))
        r = self.policy._ask("intent", state, "What does the utterance ask for?",
                             [Option(k, "%s: %s" % (k, v)) for k, v in INTENTS.items()])
        return r.choice, r.scores, r.confidence()

    # --------------------------------------------------------------- turns

    def say(self, text: str) -> None:
        """One line from the person. Emits events; never raises to the caller."""
        started = time.perf_counter()
        text = " ".join(text.split())
        try:
            if self.pending and re.fullmatch(r"(yes|y|allow|ok|okay|go|go ahead|do it|confirm)[.!]*",
                                             text, re.I):
                self._resume_pending()
                return
            if self.pending:
                self.emit("note", {"text": "Cancelled the paused action."})
                self.pending = None
                # A bare "no" (what the window's Cancel button sends) only cancels;
                # anything longer is the person's next request, and runs.
                if _CANCEL.fullmatch(text):
                    self.emit("done", {"status": "stopped", "ms": _ms(started)})
                    return
            if self.blocked_goal and _CONTINUE.fullmatch(text):
                goal, self.blocked_goal = self.blocked_goal, None
                self.emit("note", {"text": "Picking up where I stopped: " + goal})
                self.run_goal(goal, started, resume=True)
                return
            self.blocked_goal = None

            t0 = time.perf_counter()
            intent, probs, conf = self.intent(text)
            self.emit("intent", {"intent": intent, "probabilities": probs, "confidence": conf,
                                 "ms": (time.perf_counter() - t0) * 1000})

            if intent == "close":
                self.close()
                self.emit("done", {"status": "closed", "ms": _ms(started)})
                return

            url = _URL.search(text)
            fresh = self.browser is None or not self.browser.alive
            b = self._ensure_browser()
            if intent == "open" or fresh:
                target = url.group(1) if (url and intent != "read") else self.home
                self.emit("status", {"text": "Opening " + target})
                b.goto(target)
                self._snapshot_event("opened")
                if intent == "open" or (url and _only_navigation(text)):
                    self.emit("done", {"status": "ready", "url": b.page.url, "ms": _ms(started)})
                    return
            elif url and _only_navigation(text):
                self.emit("status", {"text": "Going to " + url.group(1)})
                b.goto(url.group(1))
                self._snapshot_event("opened")
                self.emit("done", {"status": "ready", "url": b.page.url, "ms": _ms(started)})
                return

            if intent == "read":
                page = b.observe(screenshot=self.screenshots)
                self.emit("page", {"title": page.get("title"), "url": page.get("url"),
                                   "text": " ".join((page.get("text") or "").split())[:1200],
                                   "screenshot": page.get("screenshot")})
                self.emit("done", {"status": "read", "url": page.get("url"), "ms": _ms(started)})
                return

            self.run_goal(text, started)
        except Stop:
            self.emit("done", {"status": "stopped", "ms": _ms(started)})
        except Exception as exc:  # noqa: BLE001 - the chat must always get an answer
            self.emit("error", {"text": "%s: %s" % (type(exc).__name__, str(exc)[:300]),
                                "trace": traceback.format_exc()[-1500:]})

    def run_goal(self, goal: str, started: float, *, resume: bool = False) -> None:
        """Work on one goal until DONE, BLOCKED, a stop rule, or the step budget.

        `resume` continues the SAME goal - after the person allowed a paused
        action or got past a verification page - so what was already done still
        counts toward it, instead of opening a fresh goal that must start over.
        """
        self._current_goal = goal
        b = self._ensure_browser()
        if not resume:
            if self.history:
                # Earlier goals stay visible ("open the top one" needs the results), but
                # marked, so their actions are not read as progress on this one.
                self.history.append({"goal_marker": True, "summary": self._last_goal or "earlier"})
            self._last_goal = goal
            self.pending_text = None
        keys: list[str] = []
        steps, unchanged = 0, 0
        while steps < self.max_steps:
            self._check_stop()
            try:
                page = b.observe(screenshot=self.screenshots)
            except StalePage:
                time.sleep(0.3)
                continue
            mine = self._goal_history()
            skip = {h["label"] for h in mine if h.get("operation") == "CLICK" and not h.get("changed")
                    and h.get("label")}
            typed = {h["text"].lower() for h in mine if h.get("operation") == "TYPE_TEXT" and h.get("text")
                     and h.get("submitted")}
            wall = b.challenge(page)
            if wall:
                # Never attempted, never clicked through: a person solves it.
                self.blocked_goal = goal
                self.emit("done", {"status": "blocked", "steps": steps, "url": page.get("url"),
                                   "ms": _ms(started), "captcha": True,
                                   "text": "I stopped at %s. I don't get past those myself - if it is "
                                           "something you can do, do it in the browser window, then "
                                           "say “continue”." % wall})
                return
            allowed = self._done_allowed(goal)
            texts = [h.get("text") for h in mine if h.get("operation") == "TYPE_TEXT"]
            unset = unset_controls(goal, page.get("actions") or [], texts)
            # The unused-word rule is a DONE guard, not advice: shown on the first
            # step, "NOT YET USED: serra, lodge" sent "open Serra Lodge" (round 6)
            # to typing the name into the search box instead of clicking it.
            unused = unused_words(goal, page.get("actions") or [], typed=texts,
                                  clicked=[h.get("label") for h in mine if h.get("operation") == "CLICK"],
                                  title=page.get("title") or "",
                                  url=page.get("url") or "") if allowed and not unset else []
            ch = self.policy.choose(goal, page, self.history, self.pending_text,
                                    allow_done=allowed and not unset and not unused,
                                    skip_clicks=skip, typed=typed, unset=unset, unused=unused)
            steps += 1
            self._decision_event(steps, ch, page)
            self._check_stop()

            if ch.operation in ("DONE", "BLOCKED"):
                self.emit("done", {"status": ch.operation.lower(), "steps": steps,
                                   "url": page.get("url"), "ms": _ms(started)})
                return

            action, text = self._action_for(ch, page.get("url") or "")
            if action is None:
                self.emit("done", {"status": "blocked", "steps": steps,
                                   "text": "nothing to act on for " + ch.operation, "ms": _ms(started)})
                return

            key = "%s|%s|%s" % (ch.operation, _short(ch.target_line), text or "")
            keys.append(key)
            if keys.count(key) >= 3 or (len(keys) >= 4 and keys[-1] == keys[-3] and keys[-2] == keys[-4]
                                        and keys[-1] != keys[-2]):
                self.emit("done", {"status": "stuck", "steps": steps, "ms": _ms(started),
                                   "text": "repeating the same actions without getting further"})
                return

            if ch.operation in ("CLICK", "SUBMIT"):
                p_irrev = self.policy.irreversible(ch)
                if p_irrev >= IRREVERSIBLE_AT:
                    self.pending = {"goal": goal, "choice": ch, "action": action, "text": text,
                                    "page": page, "steps": steps}
                    self.emit("confirm", {"what": _summary(ch, text), "p": p_irrev,
                                          "text": "This looks hard to undo. Reply “yes” to let me do it, "
                                                  "or say anything else to cancel."})
                    return

            if ch.operation == "TYPE_TEXT":
                self.emit("typing", {"text": text, "field": ch.target_line})

            changed = self._execute(ch, action, text, page)
            unchanged = 0 if changed or ch.operation == "WAIT" else unchanged + 1
            if unchanged >= 3:
                self.emit("done", {"status": "stuck", "steps": steps,
                                   "text": "three actions in a row changed nothing", "ms": _ms(started)})
                return
        self.emit("done", {"status": "step_budget", "steps": steps, "ms": _ms(started)})

    def _goal_history(self) -> list[dict]:
        mine = []
        for h in reversed(self.history):
            if h.get("goal_marker"):
                break
            mine.append(h)
        return list(reversed(mine))

    def _done_allowed(self, goal: str) -> bool:
        """Structural preconditions for DONE, each from a measured false DONE.

        * At least one action in THIS goal. An imperative goal is almost never
          already satisfied by the page it starts on; round 2 declared "show only
          nature stays" DONE on step 1 with nothing filtered. (jev-browser demands
          0.9 confidence on round 0 for the same reason.)
        * If the goal's last clause asks to open something, the last action must be
          a click that changed the page.
        """
        mine = self._goal_history()
        if not mine:
            return False
        last_clause = _CLAUSES.split(goal)[-1]
        if _OPEN.search(last_clause):
            since = []
            for h in reversed(mine):
                if h.get("operation") in ("TYPE_TEXT", "SUBMIT"):
                    break
                since.append(h)
            return any(h.get("operation") == "CLICK" and h.get("changed")
                       and h.get("role") not in _NOT_OPENING for h in since)
        return True

    # ------------------------------------------------------------- acting

    def _action_for(self, ch: Choice, url: str = ""):
        if ch.operation == "SUBMIT":
            if not self.pending_text:
                return None, None
            return {"kind": "submit", "node": self.pending_text["node"]}, None
        if ch.action is None:
            return None, None
        text = None
        if ch.operation == "TYPE_TEXT":
            text, _ = self.policy.choose_text(self._current_goal, ch, url)
            if not text:
                return None, None
        return ch.action, text

    def _execute(self, ch: Choice, action: dict, text, page) -> bool:
        b = self.browser
        label = _summary(ch, text)
        b.highlight(action, label)
        try:
            b.act(action, text=text)
        except StalePage as exc:
            self.emit("note", {"text": "Page moved before I could act (%s) - looking again." % exc})
            return True
        try:
            after = b.observe(screenshot=False)
        except StalePage:
            after = page
        changed = meaning(after) != meaning(page)
        change = what_changed(page, after) if changed else "nothing visible changed"
        self.history.append({"step": sum(1 for h in self.history if not h.get("goal_marker")) + 1,
                             "summary": "%s -> %s" % (label, change), "changed": changed,
                             "operation": ch.operation, "node": action.get("node"),
                             "label": (ch.element or {}).get("label"), "text": text,
                             "role": (ch.element or {}).get("role"),
                             "url": after.get("url")})
        submitted = ch.operation == "SUBMIT" or after.get("url") != page.get("url") or \
            (ch.operation == "CLICK" and (ch.element or {}).get("role") in ("button", "link"))
        if submitted:
            for h in self._goal_history():
                if h.get("operation") == "TYPE_TEXT":
                    h["submitted"] = True
        # typed text stays "pending" until it is submitted or the page moves on
        if ch.operation == "TYPE_TEXT":
            self.pending_text = {"node": action.get("node"), "text": text,
                                 "label": (ch.element or {}).get("label", "the field")}
        elif ch.operation == "SUBMIT" or after.get("url") != page.get("url") or \
                (ch.operation == "CLICK" and (ch.element or {}).get("role") in ("button", "link")):
            self.pending_text = None
        self.emit("acted", {"summary": label, "change": change, "changed": changed,
                            "url": after.get("url")})
        return changed

    def _resume_pending(self):
        p, self.pending = self.pending, None
        started = time.perf_counter()
        self._current_goal = p["goal"]
        self.emit("note", {"text": "Allowed: " + _summary(p["choice"], p["text"])})
        self._execute(p["choice"], p["action"], p["text"], p["page"])
        self.run_goal(p["goal"], started, resume=True)

    # -------------------------------------------------------------- events

    def _decision_event(self, step: int, ch: Choice, page: dict):
        self.emit("decision", {
            "step": step, "operation": ch.operation, "op_probs": ch.op_probs,
            "op_confidence": ch.op_confidence, "target": ch.target, "target_line": ch.target_line,
            "target_p": ch.target_p, "done_check": ch.done_check, "overruled": ch.overruled,
            "top_targets": sorted(ch.target_probs.items(), key=lambda kv: -kv[1])[:5],
            "rect": (ch.element or {}).get("rect"), "viewport": [page.get("w"), page.get("h")],
            "timings": ch.timings, "evidence_tokens": ch.evidence_tokens, "passes": ch.passes,
            "url": page.get("url"), "title": page.get("title"),
            "screenshot": page.get("screenshot")})

    def _snapshot_event(self, why: str):
        if not self.screenshots or self.browser is None:
            return
        try:
            page = self.browser.observe(screenshot=True)
            self.emit("page", {"title": page.get("title"), "url": page.get("url"),
                               "screenshot": page.get("screenshot"), "why": why})
        except Exception:
            pass


def _only_navigation(text: str) -> bool:
    """'go to github.com' is navigation; 'go to github.com and open the first repo' is not."""
    return len(text.split()) <= 5 and not re.search(r"\b(and|then|search|find|open the|click)\b",
                                                   text, re.I)


def _summary(ch: Choice, text=None) -> str:
    if ch.operation == "TYPE_TEXT":
        return "Type “%s” into %s" % (text or "", _short(ch.target_line))
    if ch.operation == "SUBMIT":
        return "Press Enter"
    if ch.operation == "SELECT":
        return "Select " + _short(ch.target_line)
    if ch.operation == "CLICK":
        return "Click " + _short(ch.target_line)
    return ch.operation.replace("_", " ").title()


def _short(line: str | None) -> str:
    if not line:
        return "?"
    m = re.match(r"\[\d+\]\s+\S+\s+(\".*?\")", line)
    if not m:
        return line[:60]
    tail = line.split(" → choose ", 1)
    return m.group(1) + ((" → " + tail[1]) if len(tail) == 2 else "")


def _trim(s: str, n: int) -> str:
    s = " ".join(str(s or "").split())
    return s[:n] + ("…" if len(s) > n else "")


def _q(s) -> str:
    return "“%s”" % _trim(s, 40)


def _ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000

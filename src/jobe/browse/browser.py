# -*- coding: utf-8 -*-
"""An isolated browser, observed atomically and acted on with pre-input checks.

The executor logic is adapted from browser-use/jev-ultrafast (MIT; see NOTICE),
moved from their CDP daemon onto Playwright. The reason for moving is not
taste: jev-ultrafast attaches to the user's own Chrome through remote
debugging and opens tabs inside their logged-in profile. An agent acting on
someone's behalf should not be driving the browser that holds their sessions,
so this launches its own Chromium with a fresh, throwaway profile.

What is kept from their executor, because each line of it was paid for:

  * code-owned node identities from `snapshot.js` - the model names an index,
    never a selector or coordinate;
  * immediately before input, the node must still be connected, visible,
    enabled, and actually the element under its own centre point (not covered
    by a modal) - otherwise the decision is stale and the page is re-observed;
  * typing is select-all then insert, so a field's old contents are replaced;
  * after typing into a combobox, wait (at most 200 ms) for suggestions to be
    visible before observing, so the next decision can see them.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

SNAPSHOT_JS = (Path(__file__).with_name("snapshot.js")).read_text(encoding="utf-8")

#: Resolve an observed node and prove it can take input RIGHT NOW. Mirrors
#: jev-ultrafast's pre-input check; returns the centre point, or null.
RESOLVE_JS = """(action => {
  const e = window.__jevFast?.nodes.get(action.node);
  if (!e?.isConnected || e.matches(':disabled') || e.closest('[aria-disabled="true"],[inert]') ||
      !e.checkVisibility({checkOpacity:true, checkVisibilityCSS:true})) return null;
  if (action.kind === 'fill' && (e.readOnly || e.getAttribute('aria-readonly') === 'true')) return null;
  e.scrollIntoView({block:'nearest', inline:'nearest'});
  const r = e.getBoundingClientRect(), x = r.x + r.width/2, y = r.y + r.height/2;
  if (!r.width || !r.height || x < 0 || y < 0 || x >= innerWidth || y >= innerHeight) return null;
  const hit = document.elementFromPoint(x, y);
  if (!(e === hit || e.contains(hit) || hit?.contains(e) && hit.tagName === 'LABEL')) return null;
  if (action.kind === 'select') {
    if (e.tagName !== 'SELECT' || ![...e.options].some(o => o.value === action.value && !o.disabled))
      return null;
    e.value = action.value;
    e.dispatchEvent(new Event('input', {bubbles:true}));
    e.dispatchEvent(new Event('change', {bubbles:true}));
  }
  return {x, y, w: r.width, h: r.height, left: r.x, top: r.y};
})"""

#: After input: let the page react. Two animation frames or 50 ms normally;
#: an editable combobox waits (max 200 ms) for its suggestions to appear.
SETTLE_JS = """(action => new Promise(resolve => {
  const field = window.__jevFast?.nodes.get(action.node);
  const autocomplete = action.kind === 'fill' && field?.getAttribute('role') === 'combobox';
  let frames = 0, stopped = false;
  const finish = () => { stopped = true; resolve(); };
  setTimeout(finish, autocomplete ? 200 : 50);
  const ready = () => {
    if (stopped) return;
    const opts = [...document.querySelectorAll('[role="option"]')];
    if (++frames >= 2 && (!autocomplete || opts.some(o => {
      const r = o.getBoundingClientRect();
      return r.width && r.height && r.bottom > 0 && r.top < innerHeight;
    }))) finish(); else requestAnimationFrame(ready);
  };
  requestAnimationFrame(ready);
}))"""

#: How much page there is: text length, control count, URL. After a click or
#: Enter it is sampled until it holds still. Measured 2026-09-23 on DuckDuckGo
#: after Enter: `load` fires with an empty body (1 control at +0.1 s), a
#: skeleton at +0.6 s, the results at +1.6 s. Round 5 decided on the empty
#: first frame - DONE for a search, and the Search button for "open the top
#: result" - in five of its six failures.
QUIET_JS = """() => [document.body ? document.body.innerText.length : 0,
  document.querySelectorAll('a,button,input,select,textarea,[role]').length, location.href]"""

#: Which observed nodes sit inside an advert. Search engines mark their ads in
#: the markup (DuckDuckGo `data-layout="ad"`, Google `#tads` / `data-text-ad`,
#: Bing `.b_ad`); elsewhere an ad card carries a label line of its own, "Ad" or
#: "Sponsored". Measured 2026-09-23: the first ten result links DuckDuckGo showed
#: for "python programming language" were two adverts and their sitelinks, and
#: "open the top result" is not a request to open an advert.
AD_JS = """(nodes => nodes.filter(n => {
  const e = window.__jevFast?.nodes.get(n);
  if (!e) return false;
  if (e.closest('[data-layout="ad"],[data-testid="ad"],[data-text-ad],#tads,#bottomads,.b_ad,' +
                '[aria-label="Ads"],[aria-label="Sponsored"]')) return true;
  const card = e.closest('article,li,[role="article"]');
  return !!card && card.innerText.length < 1500 &&
    card.innerText.split('\\n').some(l => /^(ad|ads|sponsored|promoted|sponsored result)$/i.test(l.trim()));
}))"""

#: The "preview" in the live browser: outline the element about to be acted on,
#: with its operation, for a moment before acting - so a person watching the
#: window sees what was decided, not just what happened.
HIGHLIGHT_JS = """(({node, label}) => {
  document.querySelectorAll('[data-jobe-overlay]').forEach(n => n.remove());
  const e = window.__jevFast?.nodes.get(node);
  if (!e) return;
  const r = e.getBoundingClientRect();
  const box = document.createElement('div');
  box.setAttribute('data-jobe-overlay', '');
  Object.assign(box.style, {position:'fixed', left:(r.left-4)+'px', top:(r.top-4)+'px',
    width:(r.width+8)+'px', height:(r.height+8)+'px', border:'2.5px solid #6d5dfc',
    borderRadius:'8px', boxShadow:'0 0 0 4px rgba(109,93,252,.18)', zIndex:2147483647,
    pointerEvents:'none', transition:'opacity .25s'});
  const tag = document.createElement('div');
  tag.textContent = label;
  Object.assign(tag.style, {position:'absolute', left:'-2px', top: r.top > 34 ? '-30px' : (r.height+10)+'px',
    background:'#6d5dfc', color:'#fff', font:'600 12px/1 system-ui,Segoe UI,sans-serif',
    padding:'6px 9px', borderRadius:'6px', whiteSpace:'nowrap', letterSpacing:'.01em'});
  box.appendChild(tag);
  document.documentElement.appendChild(box);
  setTimeout(() => { box.style.opacity = '0'; setTimeout(() => box.remove(), 300); }, 1100);
})"""


#: A human-verification challenge. The agent never attempts one - it stops and
#: hands the page to the person. Found the hard way: after a day of automated
#: test searches DuckDuckGo answered "Unfortunately, bots use DuckDuckGo too.
#: Select all squares containing a duck", and the agent clicked its Submit.
CHALLENGE_TEXT = re.compile(
    r"not a robot|are you a robot|bots use|verify (?:that )?you(?:'| a)re (?:a )?human|"
    r"made by a human|unusual traffic|human verification|press (?:and|&) hold|"
    r"checking (?:if|that) the site connection is secure|security check to continue", re.I)
#: A page that refuses outright. Measured on 2026-09-23, one results page per
#: engine: HEADLESS, Mojeek answered "403 - Forbidden" and Startpage "Access
#: Denied" while DuckDuckGo, Google, Brave and Ecosia challenged - only Bing
#: served results. HEADED, all of those served results except Mojeek.
REFUSED_TITLE = re.compile(r"^\s*(?:401|403|429)\b|forbidden|access denied|too many requests|"
                           r"attention required|just a moment", re.I)
CHALLENGE_FRAME = re.compile(r"recaptcha|hcaptcha|challenges\.cloudflare|turnstile|arkoselabs|funcaptcha",
                             re.I)


#: Origins the agent's browser must never load: the chat window's own server.
#: A page could otherwise lead the agent to its own control panel, where
#: clicking and typing would be sending itself commands. DeepSeek Harness's
#: desktop browser filters its host endpoint and loopback aliases the same way.
PROTECTED: set[str] = set()

#: Environment variables that never reach the launched Chromium. It does not
#: need them, and DeepSeek Harness learned to scrub a launched browser's
#: environment after a launcher inherited one.
_SECRET_ENV = re.compile(r"KEY|TOKEN|SECRET|PASSW|CREDENTIAL|COOKIE|AUTH", re.I)


def protect(port: int) -> None:
    """Keep the agent's browser off 127.0.0.1:<port> and its aliases."""
    for host in ("127.0.0.1", "localhost", "[::1]", "0.0.0.0"):
        PROTECTED.add("%s:%d" % (host, port))


class StalePage(RuntimeError):
    """The decision no longer refers to what is on the page. Observe again."""


def fingerprint(state: dict) -> str:
    content = {k: state.get(k) for k in ("url", "text", "actions", "scroll")}
    for a in content.get("actions") or []:
        a.pop("rect", None)
    return hashlib.sha256(json.dumps(content, sort_keys=True, default=str).encode()).hexdigest()


class Browser:
    """One Chromium, one throwaway profile, one active page."""

    def __init__(self, *, headless: bool = False, width: int = 1180, height: int = 820,
                 x: int = 520, y: int = 40, highlight: bool = True):
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        args = ["--window-position=%d,%d" % (x, y), "--window-size=%d,%d" % (width, height + 90),
                "--disable-blink-features=AutomationControlled", "--no-first-run",
                "--no-default-browser-check"]
        env = {k: v for k, v in os.environ.items() if not _SECRET_ENV.search(k)}
        self._browser = self._pw.chromium.launch(headless=headless, args=args, env=env)
        # No downloads: an agent that follows links on someone's behalf should
        # never leave files behind on their disk.
        self._context = self._browser.new_context(viewport={"width": width, "height": height},
                                                  locale="en-AU", accept_downloads=False)
        for origin in PROTECTED:
            for scheme in ("http", "https", "ws"):
                self._context.route("%s://%s/**" % (scheme, origin), lambda route: route.abort())
        self.page = self._context.new_page()
        # A link that opens a new tab moves the agent to that tab: "open the top
        # one" on a results page that uses target=_blank must follow the result.
        self._context.on("page", self._adopt)
        self.highlight_enabled = highlight and not headless
        self.headless = headless
        self._last_input = None

    # ---------------------------------------------------------------- pages

    def _adopt(self, new_page):
        self.page = new_page
        try:
            new_page.wait_for_load_state("domcontentloaded", timeout=8000)
            new_page.bring_to_front()
        except Exception:
            pass

    @property
    def alive(self) -> bool:
        try:
            return self._browser.is_connected() and not self.page.is_closed()
        except Exception:
            return False

    def goto(self, url: str) -> None:
        if not url.startswith(("http://", "https://", "file:", "about:")):
            url = "https://" + url
        self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
        self._settle_load()
        self._wait_quiet()

    def _settle_load(self, timeout_ms: int = 4000):
        try:
            self.page.wait_for_load_state("load", timeout=timeout_ms)
        except Exception:
            pass

    def _wait_quiet(self, network_ms: int = 2500, max_s: float = 2.0, every: float = 0.25) -> None:
        """Let a page that is still filling in finish, within a bound.

        Network idle first (no request for 500 ms, at most `network_ms`): it is
        what separates DuckDuckGo's skeleton, which is waiting on its results
        request, from a page that is done. Then the DOM must hold still for two
        samples in a row, for pages that re-render without a navigation. A busy
        site that never goes idle costs the bound, never more.
        """
        try:
            self.page.wait_for_load_state("networkidle", timeout=network_ms)
        except Exception:
            pass
        end, last, same = time.perf_counter() + max_s, None, 0
        while time.perf_counter() < end:
            try:
                sig = self.page.evaluate(QUIET_JS)
            except Exception:          # mid-navigation
                sig = None
            same = same + 1 if (sig is not None and sig == last and sig[0] > 0) else 0
            if same >= 2:
                return
            last = sig
            time.sleep(every)

    def challenge(self, page: dict | None = None) -> str | None:
        """A short description if the current page is a human-verification wall, else None.

        Text first: the phrases challenge pages use, read from the title and the
        top of the page - and the bare word "captcha" only on a short page, so an
        article ABOUT captchas is not a wall. Then frames: a VISIBLE reCAPTCHA,
        hCaptcha or Turnstile widget, but not the invisible reCAPTCHA that many
        ordinary forms embed.
        """
        title = (page or {}).get("title") or ""
        text = (page or {}).get("text") or ""
        head = "%s %s" % (title, text[:1500])
        m = CHALLENGE_TEXT.search(head) or (len(text) < 2000 and re.search(r"captcha", head, re.I))
        if m:
            return "a human-verification check (the page says “%s”)" % m.group(0)
        if len(text) < 2000 and REFUSED_TITLE.search(title):
            return "a page refusing access (“%s”)" % title.strip()[:60]
        for frame in getattr(self.page, "frames", []):
            url = getattr(frame, "url", "") or ""
            if not CHALLENGE_FRAME.search(url) or "size=invisible" in url:
                continue
            try:
                box = frame.frame_element().bounding_box()
            except Exception:
                continue
            if box and box["width"] >= 60 and box["height"] >= 40:
                return "a human-verification check (a verification widget is showing)"
        return None

    # -------------------------------------------------------------- observe

    def observe(self, *, screenshot: bool = False) -> dict:
        if self._last_input is not None:
            action, self._last_input = self._last_input, None
            try:
                self.page.evaluate(SETTLE_JS, action)
            except Exception:
                pass
        last_error = None
        for _ in range(12):
            try:
                state = self.page.evaluate(SNAPSHOT_JS)
            except Exception as exc:          # navigating mid-read
                last_error = exc
                self._settle_load(1500)
                continue
            if state is None:                 # no document.body yet
                time.sleep(0.1)
                continue
            self._mark_ads(state)
            state["fingerprint"] = fingerprint(json.loads(json.dumps(state)))
            if screenshot:
                shot = self.page.screenshot(type="jpeg", quality=62, scale="css")
                state["screenshot"] = base64.b64encode(shot).decode()
            return state
        raise StalePage("page never settled enough to read (%s)" % last_error)

    def _mark_ads(self, state: dict) -> None:
        actions = state.get("actions") or []
        nodes = sorted({a["node"] for a in actions if type(a.get("node")) is int})
        try:
            ads = set(self.page.evaluate(AD_JS, nodes)) if nodes else set()
        except Exception:
            return
        for a in actions:
            if a.get("node") in ads:
                a["ad"] = True

    # ------------------------------------------------------------------ act

    def highlight(self, action: dict, label: str, hold_s: float = 0.45) -> None:
        if not self.highlight_enabled or action.get("node") is None:
            return
        try:
            self.page.evaluate(HIGHLIGHT_JS, {"node": action["node"], "label": label})
            time.sleep(hold_s)
        except Exception:
            pass

    def act(self, action: dict, text: str | None = None) -> None:
        kind = action.get("kind")
        if kind == "wait":
            time.sleep(0.6)
            return
        if kind == "scroll":
            self.page.mouse.wheel(0, action.get("delta", 560))
            time.sleep(0.25)
            return
        if kind == "submit":
            # Focus goes back to the field that was typed into first: after the
            # text, the agent may have ticked a checkbox or opened a menu, and
            # Enter pressed there would do something else entirely.
            if type(action.get("node")) is int:
                self.page.evaluate("n => window.__jevFast?.nodes.get(n)?.focus()", action["node"])
            self.page.keyboard.press("Enter")
            self._after_navigation_maybe()
            return
        if type(action.get("node")) is not int:
            raise StalePage("no observed node to act on")
        hit = self.page.evaluate(RESOLVE_JS, action)
        if hit is None:
            raise StalePage("the element changed, moved or is covered")
        if kind == "select":
            self._last_input = action
            return
        self.page.mouse.click(hit["x"], hit["y"])
        if kind == "fill":
            mod = "Meta" if sys.platform == "darwin" else "Control"
            self.page.keyboard.press(mod + "+A")
            self.page.keyboard.insert_text(text or "")
        self._last_input = action
        if kind == "click":
            self._after_navigation_maybe()

    def _after_navigation_maybe(self):
        # A click or Enter may start a navigation. Give it a moment to begin,
        # then wait for the new document and for it to finish filling in - but
        # never for more than a few seconds.
        time.sleep(0.15)
        self._settle_load(5000)
        self._wait_quiet()

    def close(self) -> None:
        for fn in (lambda: self._context.close(), lambda: self._browser.close(),
                   lambda: self._pw.stop()):
            try:
                fn()
            except Exception:
                pass

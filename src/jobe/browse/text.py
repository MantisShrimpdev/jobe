# -*- coding: utf-8 -*-
"""What to type, chosen among the user's own words - never generated.

jev-ultrafast hands TYPE_TEXT to a small text-generating LLM. This backbone
never generates, and it does not need to: the value to type is almost always
already IN the goal. "search nike" types "nike"; "flights from Zurich to
London" types "Zurich" into one field and "London" into the other. So the
candidates are contiguous spans of the goal, and the readout picks which span
belongs in which field - a choice over declared options, the one thing it is
built for.

The cost of that honesty: a value the user never said cannot be typed. That is
the right failure for an agent acting on someone's behalf - it will not invent
a name, an address or a date.
"""

from __future__ import annotations

import base64
import re
from urllib.parse import parse_qs, urljoin, urlparse

#: Words that never start or end a useful value on their own.
STOP = {
    "a", "an", "the", "for", "to", "from", "of", "in", "on", "at", "by", "with", "and", "or",
    "into", "onto", "about", "me", "my", "i", "please", "it", "this", "that", "then", "is", "are",
}
#: Command verbs: they describe the action, not the value.
COMMAND = {
    "search", "find", "look", "lookup", "up", "type", "enter", "google", "go", "open", "visit",
    "show", "get", "navigate", "browse", "check", "fill", "put", "write", "hi", "hey", "jobe",
    "click", "select", "choose", "book", "search:",
}

#: Words that point AT something on the page rather than being a value to type:
#: "open the top result", "click the cheapest one". A goal made only of these
#: has nothing to type - and offering TYPE_TEXT anyway is how round 1 typed
#: "top result" into DuckDuckGo fourteen times.
REFERENCE = {
    "top", "first", "second", "third", "last", "next", "previous", "cheapest", "best", "newest",
    "latest", "result", "results", "one", "ones", "link", "links", "page", "button", "item",
    "option", "them", "that", "those", "above", "below", "here", "there", "same", "other",
}

_WORD = re.compile(r"[\w'’.@:/+\-]+", re.UNICODE)


def _words(goal: str) -> list[str]:
    return [w.strip(".,;:!?\"'“”") for w in _WORD.findall(goal) if w.strip(".,;:!?\"'“”")]


#: Goals whose LAST clause asks to open something. Round 2: "find a stay in
#: Copenhagen and open it" searched, saw the stay, and declared DONE without
#: opening it; "open the top result" stopped before clicking anything.
OPEN = re.compile(r"\b(open|click|view|visit|follow|go into|go to the)\b", re.I)
#: Where one request becomes the next. A bare "and" splits only before a command
#: verb, so "search for rock and roll" stays one search.
CLAUSES = re.compile(
    r",\s*(?:and\s+|then\s+)?|\s+(?:and\s+then|then)\s+|\s+and\s+(?=(?:open|click|view|visit|follow|go|"
    r"search|find|look|select|choose|type|enter|show|sort|filter|scroll|book|press|submit)\b)", re.I)

#: A clause that points at something to click rather than naming something to type.
_POINTING = re.compile(r"\s*(?:and\s+|then\s+)?(?:click|open|view|visit|follow|select|choose|press|tap)\b",
                       re.I)

#: A request that ends by pointing at a result with nothing in between - "search
#: for the latest news on github open the top one". 2026-09-24: with no "and" to
#: split on, all of it was typed into Bing.
_TRAILING_POINT = re.compile(
    r"\s+(?:(?:and\s+then|and|then)\s+)?((?:open|click|visit|view)\s+(?:on\s+)?(?:(?:it|that|this)|"
    r"(?:the\s+)?(?:(?:top|first|second|third|last|cheapest|best|newest|latest|next)\s+)?"
    r"(?:one|result|link|item|page|video|article|story|site)s?))\s*[.!]*$", re.I)


def last_clause(goal: str) -> str:
    return CLAUSES.split(goal)[-1]


def clauses(goal: str) -> list[str]:
    return [c.strip() for c in CLAUSES.split(goal) if c and c.strip()]


def pointing(clause: str) -> bool:
    """Does this clause point at something to click, rather than name a value to type?"""
    return bool(_POINTING.match(clause))


def normalise(goal: str) -> str:
    """Give a run-on request its missing clause break: "... github open the top one"."""
    m = _TRAILING_POINT.search(goal)
    if not m:
        return goal
    head = goal[:m.start()].rstrip(" ,")
    if not has_value(head):
        return goal                       # nothing before it: "open the top one" alone
    return "%s, then %s" % (head, m.group(1))


def typeable(goal: str) -> bool:
    """Is there anything to TYPE - a value in a clause that is not just pointing?

    2026-09-24: "click image creator" was offered TYPE_TEXT because "image
    creator" looks like a value, and after the click it typed it into Bing's
    image prompt. In a pointing clause those words are a label to click.
    """
    return any(has_value(c) for c in clauses(goal) if not pointing(c))


_SEARCH = re.compile(r"\s*(?:(?:hi|hey|ok|okay)\s+)?(?:jobe[,!.]?\s+)?(?:please\s+)?"
                     r"(?:search|find|look\s+up|look\s+for|lookup|google)\b", re.I)


def search_first(goal: str) -> bool:
    """A request that starts by searching for something: type it before clicking anything.

    2026-09-24: "search github" on a results page clicked a GitHub link instead
    of searching - the word matched a link, and the readout went with the lure.
    """
    first = clauses(goal)[0] if clauses(goal) else goal
    return bool(_SEARCH.match(first)) and has_value(first)


#: Going to a site by name: "go to github", "open blender".
_SITE = re.compile(
    r"\s*(?:(?:hi|hey|ok|okay)\s+)?(?:jobe[,!.]?\s+)?(?:please\s+)?(?:can\s+you\s+)?"
    r"(?P<verb>go\s+to|goto|got\s+to|go\s+on\s+to|visit|navigate\s+to|take\s+me\s+to|load|open\s+up|open)\s+"
    r"(?:the\s+)?(?P<name>.+?)(?:\s+(?:website|web\s+site|site|homepage|home\s+page))?(?:\s+please)?"
    r"\s*[.!?]*$", re.I)
_NOT_A_SITE = re.compile(r"^(?:a\s+)?(?:new\s+)?(?:web\s+)?(?:browser|chrome|tab|window)$", re.I)


def site_request(clause: str) -> tuple[str, str] | None:
    """("go"|"open", name) when a clause asks for a site by name, else None.

    2026-09-24: "got to github" was typed into Bing as a search, and "open
    blender" on GitHub's login page typed "blender" into the username box.
    Neither name was on the page; both meant "take me to that site".
    """
    m = _SITE.fullmatch(clause)
    if not m:
        return None
    name = m.group("name").strip()
    if _NOT_A_SITE.match(name) or not has_value(name):
        return None                        # "open the top one", "open it", "open a new tab"
    verb = m.group("verb").lower()
    return ("open" if verb.startswith("open") else "go", name)


def wants_a_result(goal: str) -> bool:
    """'open the top one', 'click the first result': open something named only by position."""
    clause = last_clause(goal)
    return bool(OPEN.search(clause)) and not has_value(clause)


#: Click-tracking paths engines send result clicks through (Bing /ck/a, Google /url).
_REDIRECT = re.compile(r"^/(?:ck/a|url|l/|link|r/|aclk)", re.I)


def _site(host: str | None) -> str:
    parts = (host or "").lower().split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else (host or "")


def unwrap(url: str) -> str | None:
    """Where an engine's click-tracking redirect really goes, when the URL says.

    Round 7: Bing sends EVERY link through /ck/a - its own Images tab and its
    related searches as well as results - so the redirect alone does not mean
    "leaves Bing". The destination rides in `u=` as "a1" + base64url.
    Google's /url carries it in `q=` or `url=`.
    """
    try:
        p = urlparse(url)
    except ValueError:
        return None
    q = parse_qs(p.query)
    if p.path.startswith("/ck/a") and q.get("u") and q["u"][0].startswith("a1"):
        raw = q["u"][0][2:]
        try:
            return base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            return None
    if p.path == "/url":
        for key in ("q", "url"):
            if q.get(key) and q[key][0].startswith(("http://", "https://")):
                return q[key][0]
    return None


def leaves_site(href: str | None, page_url: str) -> bool:
    """Does this link go to another site? A click-tracking redirect is judged by where it goes."""
    if not href:
        return False
    try:
        full = urljoin(page_url, href)
        dest, here = urlparse(full), urlparse(page_url)
    except ValueError:
        return False
    if dest.scheme not in ("http", "https"):
        return False
    if _site(dest.hostname) != _site(here.hostname):
        return True
    if not _REDIRECT.match(dest.path or ""):
        return False
    real = unwrap(full)
    if real is None:
        return True                  # a redirect that does not say where: assume it leaves
    return _site(urlparse(urljoin(page_url, real)).hostname) != _site(here.hostname)


def destination_host(href: str | None, page_url: str) -> str:
    """The host a link really leads to, looking through an engine's click-tracker."""
    if not href:
        return ""
    try:
        full = urljoin(page_url, href)
        dest = urlparse(full)
        if _REDIRECT.match(dest.path or ""):
            real = unwrap(full)
            if real:
                dest = urlparse(urljoin(page_url, real))
        return (dest.hostname or "").lower()
    except ValueError:
        return ""


def names_host(name: str, host: str) -> bool:
    """Does a host carry the name? "blender" -> www.blender.org, "new york times" -> nytimes.com."""
    squashed = re.sub(r"[^a-z0-9]", "", host.lower())
    return any(w in squashed for w in (re.sub(r"[^a-z0-9]", "", w) for w in named_words(name)) if len(w) >= 3)


#: A page's own furniture, never a result - even when it links off-site.
#: 2026-09-24: Bing's "Accessibility Help" leads to microsoft.com, so it passed
#: as "a link that leaves the engine", and asked for the top result for
#: "blender", the readout clicked it.
CHROME_LINK = re.compile(r"^\s*(?:skip\s+to\s+(?:main\s+)?content|accessibility(?:\s+help)?|feedback|"
                         r"privacy(?:\s+(?:policy|statement))?|terms(?:\s+of\s+(?:use|service))?|cookies?|"
                         r"help|settings|about(?:\s+us)?|advertis\w*|report\s+\w+)\s*$", re.I)


def results_page(url: str) -> bool:
    """A web search engine's results page: an engine host with a query in the URL."""
    p = urlparse(url or "")
    if not ENGINES.search((p.hostname or "") + "."):
        return False
    q = parse_qs(p.query)
    return any(q.get(k) for k in ("q", "p", "query", "text", "wd"))


def search_url(home: str, query: str) -> str:
    """The results page for `query` on the engine `home` is (Bing when it is none of these)."""
    from urllib.parse import quote_plus
    host, q = (urlparse(home).hostname or ""), quote_plus(query)
    if "duckduckgo" in host:
        return "https://duckduckgo.com/?q=" + q
    if "google" in host:
        return "https://www.google.com/search?q=" + q
    if "brave" in host:
        return "https://search.brave.com/search?q=" + q
    return "https://www.bing.com/search?q=" + q


#: Web search engines: their one box takes the whole request as the query.
ENGINES = re.compile(r"(^|\.)(duckduckgo|bing|google|brave|startpage|ecosia|yahoo|qwant|kagi|yandex|"
                     r"baidu|mojeek)\.", re.I)


def site_words(url: str) -> set[str]:
    """The words of a URL's host: "https://en.wikipedia.org/x" -> {"en", "wikipedia", "org"}."""
    host = re.sub(r"^[a-z]+://", "", url or "", flags=re.I).split("/")[0].split(":")[0]
    return {p.lower() for p in host.split(".") if p}


def search_query(goal: str, url: str = "") -> str | None:
    """The whole request, as a web search engine's box should receive it - or None off an engine.

    Round 5 on Bing: asked which span to type for "search for the python
    programming language", the readout chose "python". A web search engine's
    box takes the request itself, so there is nothing to choose.
    """
    host = re.sub(r"^[a-z]+://", "", url or "", flags=re.I).split("/")[0]
    if not ENGINES.search(host + "."):
        return None
    spans = candidate_spans(goal, site=url)
    return spans[0] if spans else None


def candidate_spans(goal: str, *, max_len: int = 6, limit: int = 16, site: str = "") -> list[str]:
    """Contiguous spans of the goal that could plausibly be a field's value.

    Ordered so the readout sees the likeliest search value first - the goal
    with its leading command words removed - then SHORT spans before long ones.
    Capped at 16, the letter protocol's width, so the choice never needs
    narrowing - and the cap is why the order matters: longest-first filled all
    16 slots with 4-6 word spans of "flights from Zurich to London on 20
    September" and dropped "Zurich" and "London", the two values a flight form
    actually needs.
    """
    spans: list[str] = []

    def add(s):
        s = " ".join(s.split())
        if s and s.lower() not in {x.lower() for x in spans}:
            spans.append(s)

    # Values come from the clauses that name one: in "search for X, then open the
    # top one" nothing after the comma is ever typed.
    parts = [c for c in clauses(goal) if not pointing(c)] or clauses(goal) or [goal]
    here = site_words(site) if site else set()
    for part in parts:
        # 1. the clause minus its leading command/stop words: "search for the latest news on
        #    github" -> "latest news on github". The most common search value by far. On a
        #    site, a leading mention of the site itself goes too: "search wikipedia for the
        #    eiffel tower" on wikipedia.org -> "eiffel tower".
        words = _words(part)
        lower = [w.lower() for w in words]
        i = 0
        while i < len(words) and (lower[i] in COMMAND or lower[i] in STOP or lower[i] in here):
            i += 1
        if i < len(words):
            add(" ".join(words[i:]))
    for part in parts:
        # 2. every span that neither starts nor ends on a stop/command word, short first
        words = _words(part)
        lower = [w.lower() for w in words]
        for n in range(1, max_len + 1):
            for start in range(0, len(words) - n + 1):
                seg = lower[start:start + n]
                if seg[0] in STOP or seg[-1] in STOP or seg[0] in COMMAND or seg[-1] in COMMAND:
                    continue
                if all(w in STOP or w in COMMAND for w in seg):
                    continue
                add(" ".join(words[start:start + n]))
    return spans[:limit]


def has_value(goal: str) -> bool:
    """Does the goal contain anything worth typing at all?

    A span counts only if it holds a word that is not a stop, command or
    reference word - "nike" does, "the top result" does not.
    """
    skip = STOP | COMMAND | REFERENCE
    return any(w.lower() not in skip for w in _words(goal))


# --------------------------------------------- signing in: never on its own

#: Fields that take credentials or contact details. A search box never counts.
#: 2026-09-24: "open blender" on GitHub's login page typed "blender" into
#: "Username or email address", then clicked "Continue with Google".
CREDENTIAL_FIELD = re.compile(
    r"\b(?:user\s*name|e-?mail|password|passcode|phone\s+number|mobile\s+number|account\s+number|"
    r"card\s+number|cvv|cvc|security\s+code|one[\s-]time\s+(?:code|password)|otp|verification\s+code|"
    r"2fa|log\s*in|sign\s*in)\b", re.I)
#: Controls that start signing in, signing up, or an OAuth flow.
SIGN_IN_CONTROL = re.compile(
    r"\b(?:sign\s*in|log\s*in|login|sign\s*up|register|create\s+(?:an\s+|your\s+)?account|"
    r"continue\s+with\s+(?:google|apple|microsoft|facebook|github|email|phone)|sign\s+in\s+with|"
    r"forgot\s+(?:your\s+)?password|use\s+(?:a\s+)?passkey)\b", re.I)
_WANTS_SIGN_IN = re.compile(r"\b(?:sign\s*in|log\s*in|login|sign\s*up|register|account|password|"
                            r"user\s*name|e-?mail|subscribe)\b", re.I)


def wants_sign_in(goal: str) -> bool:
    """Only a request that says so may touch sign-in fields and buttons."""
    return bool(_WANTS_SIGN_IN.search(goal))


def sign_in_thing(action: dict) -> bool:
    """A credential field (as typed into, or as clicked into) or a sign-in control."""
    label = action.get("label") or ""
    if action.get("role") in ("textbox", "searchbox", "combobox", "spinbutton") or action.get("kind") == "fill":
        return bool(CREDENTIAL_FIELD.search(label)) and "search" not in label.lower()
    return action.get("kind") == "click" and bool(SIGN_IN_CONTROL.search(label))


# ------------------------------------------------- controls the goal names

#: Words a control's label adds that a goal need not repeat: "Free cancellation only" -
#: and the conversational words of a request, which name nothing on the page.
FILLER = {"only", "show", "include", "includes", "including", "all", "any", "filter", "by", "with",
          "some", "can", "could", "would", "want", "need", "like", "just", "now", "you", "us", "we",
          "let's", "lets", "i'd", "i'm", "what", "where", "when", "how", "which", "who", "there",
          "here", "do", "does", "should", "will", "be", "have", "has", "if", "so", "as", "than"}
#: A goal that negates may be asking to UNset something - the guard stays out of it.
NEGATION = {"not", "no", "without", "don't", "dont", "exclude", "excluding", "remove", "untick",
            "uncheck", "clear", "except", "never", "off"}
_TOGGLES = {"checkbox", "radio", "switch", "menuitemcheckbox", "menuitemradio"}


def _norm(w: str) -> str:
    w = w.lower()
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def named_words(s: str) -> set[str]:
    """The words of `s` that carry meaning: no stop, command, reference or filler words."""
    skip = STOP | COMMAND | REFERENCE | FILLER
    return {_norm(w) for w in _words(s) if w.lower() not in skip}


def unset_controls(goal: str, actions: list[dict], typed=()) -> list[str]:
    """Checkboxes, radios and dropdown values the goal names that the page has not set.

    Round 4: "find a Lisbon stay with free cancellation" typed Lisbon, submitted,
    and called DONE with "Free cancellation" unticked beside it. A 4B readout
    reads "the results are showing" as the goal met; whether a control the goal
    NAMES is set is not a judgement, it is on the page.

    Deliberately narrow, so that it only ever withholds DONE when it is right to:
    every meaning word of the control's label must be in the goal; words already
    typed into a field are spent ("search nike" does not then demand a "Nike"
    brand filter); a dropdown's own name ("Stay category") does not count as
    naming one of its values; and a goal with a negation is left alone.
    """
    if {w.lower() for w in _words(goal)} & NEGATION:
        return []
    words = named_words(goal)
    for t in typed:
        words -= named_words(t or "")
    if not words:
        return []
    out: list[str] = []
    selects: dict = {}
    for a in actions:
        kind, role, label = a.get("kind"), a.get("role"), a.get("label") or ""
        if kind == "click" and role in _TOGGLES and str(a.get("checked")).lower() == "false":
            need = named_words(label)
            if need and need <= words:
                out.append(label)
        elif kind == "select":
            name, _, option = label.rpartition(" → ")
            mine = words - named_words(name)
            need = named_words(option)
            if need and need <= mine:
                current = named_words(a.get("current_value") or "")
                if not (current and current <= mine):       # the current value is not a named one
                    selects.setdefault(a.get("node"), label)
    return out + list(selects.values())


def unused_words(goal: str, actions: list[dict], *, typed=(), clicked=(), title: str = "",
                 url: str = "") -> list[str]:
    """Goal words nothing has used yet - while an empty field is still there to type them into.

    Round 5, the other half of round 4's failure: with the "Free cancellation"
    guard in place, the agent ticked the box and called DONE without ever
    typing Lisbon. A word of the goal is USED when it was typed, is on an
    element that was clicked, is on a control that is now set (a ticked box, a
    dropdown's name or chosen value), or names the site itself (page title,
    host). Page text does not count - "Lisbon" is on the unfiltered page too.

    Only while some text field on the page is still empty: once every field
    holds something there is nothing left to type into, and the guard stands
    down rather than hold DONE hostage to a word no control can take.
    """
    if not any(a.get("kind") == "fill" and not str(a.get("value") or "").strip() for a in actions):
        return []
    words = named_words(goal)
    if not words:
        return []
    used: set[str] = set()
    for t in list(typed) + list(clicked):
        used |= named_words(t or "")
    host = re.sub(r"^https?://", "", url or "").split("/")[0]
    used |= named_words(title or "") | named_words(host.replace(".", " "))
    for a in actions:
        kind, role, label = a.get("kind"), a.get("role"), a.get("label") or ""
        if kind == "click" and role in _TOGGLES and str(a.get("checked")).lower() == "true":
            used |= named_words(label)
        elif kind == "select":
            used |= named_words(label.rpartition(" → ")[0]) | named_words(a.get("current_value") or "")
        elif kind == "fill":
            used |= named_words(str(a.get("value") or ""))
    return sorted(words - used)

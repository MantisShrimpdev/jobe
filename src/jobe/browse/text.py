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
CLAUSES = re.compile(r",\s*(?:and\s+|then\s+)?|\s+(?:and then|then|and)\s+", re.I)


def last_clause(goal: str) -> str:
    return CLAUSES.split(goal)[-1]


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


def results_page(url: str) -> bool:
    """A web search engine's results page: an engine host with a query in the URL."""
    p = urlparse(url or "")
    if not ENGINES.search((p.hostname or "") + "."):
        return False
    q = parse_qs(p.query)
    return any(q.get(k) for k in ("q", "p", "query", "text", "wd"))


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
    words = _words(goal)
    lower = [w.lower() for w in words]
    spans: list[str] = []

    def add(s):
        s = " ".join(s.split())
        if s and s.lower() not in {x.lower() for x in spans}:
            spans.append(s)

    # 1. the goal minus its leading command/stop words: "search for the latest news on github"
    #    -> "latest news on github". The most common search value by far. On a site, a
    #    leading mention of the site itself goes too: "search wikipedia for the eiffel
    #    tower" on wikipedia.org -> "eiffel tower".
    here = site_words(site) if site else set()
    i = 0
    while i < len(words) and (lower[i] in COMMAND or lower[i] in STOP or lower[i] in here):
        i += 1
    if i < len(words):
        add(" ".join(words[i:]))

    # 2. every span that neither starts nor ends on a stop/command word, short first
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

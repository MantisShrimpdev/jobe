# -*- coding: utf-8 -*-
"""The browser agent's rules, each pinned to the failure that produced it.

No model and no browser: every rule below is plain code, and each test names the
suite round whose failure it came from, so a later "simplification" that drops
one has to argue with a measurement.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from jobe.browse import agent, policy
from jobe.browse.browser import Browser
from jobe.browse.text import (candidate_spans, has_value, leaves_site, normalise, results_page,
                              search_first, search_query, site_request, typeable, unset_controls,
                              unused_words, wants_a_result)

# A page as snapshot.js returns it, trimmed to what the rules read.
PAGE = {
    "url": "https://stays.example/", "title": "Stays", "w": 1180, "h": 820,
    "text": "Find a place\nCasa Flora\n€145 / night\nSerra Lodge\n€120 / night",
    "actions": [
        {"id": "e1", "node": 11, "kind": "fill", "role": "searchbox", "label": "Destination", "value": ""},
        {"id": "e2", "node": 11, "kind": "click", "role": "searchbox", "label": "Open Destination", "value": ""},
        {"id": "e3", "node": 12, "kind": "click", "role": "button", "label": "Find stays"},
        {"id": "e4", "node": 13, "kind": "select", "role": "combobox", "label": "Stay category → Nature",
         "value": "Nature", "current_value": "All stays"},
        {"id": "e5", "node": 13, "kind": "select", "role": "combobox", "label": "Stay category → Coastal",
         "value": "Coastal", "current_value": "All stays"},
        {"id": "e6", "node": 14, "kind": "click", "role": "button", "label": "View Casa Flora"},
        {"id": "e7", "node": 15, "kind": "click", "role": "button", "label": "View Serra Lodge"},
        {"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 560},
        {"id": "wait", "kind": "wait", "label": "Wait"},
    ],
    "guards": {"14": [0] * 13 + ["LISBON · DESIGN Casa Flora €145 / night View Casa Flora"],
               "15": [0] * 13 + ["LISBON · NATURE Serra Lodge €120 / night View Serra Lodge"],
               "11": [0] * 13 + ["x" * 400]},
}


# ----------------------------------------------------------------- text


def test_the_search_value_is_the_goal_without_its_command():
    assert candidate_spans("search nike") == ["nike"]
    assert candidate_spans("search for the latest news on github")[0] == "latest news on github"


def test_a_web_search_box_gets_the_whole_request():
    """Round 5 on Bing: the readout chose 'python' for 'search for the python programming language'."""
    assert search_query("search for the python programming language",
                        "https://www.bing.com/") == "python programming language"
    assert search_query("search nike", "https://duckduckgo.com/") == "nike"
    assert search_query("find a stay in Lisbon", "file:///stays.html") is None   # a site's own form: choose


def test_a_leading_mention_of_the_site_is_not_part_of_the_value():
    spans = candidate_spans("search wikipedia for the eiffel tower", site="https://en.wikipedia.org/wiki/Main_Page")
    assert spans[0] == "eiffel tower"


def test_short_values_survive_the_sixteen_slot_cap():
    """Round 1: longest-first filled every slot and dropped 'Zurich' and 'London'."""
    spans = candidate_spans("find one-way flights from Zurich to London on 20 September")
    assert "Zurich" in spans and "London" in spans and len(spans) <= 16


@pytest.mark.parametrize("goal,value", [("open the top result", False), ("click the first link", False),
                                        ("open the cheapest one", False), ("search nike", True),
                                        ("find a stay in Lisbon", True)])
def test_a_goal_with_nothing_to_type_offers_no_typing(goal, value):
    """Round 1 typed 'top result' into DuckDuckGo fourteen times."""
    assert has_value(goal) is value


# ----------------------------------------------------------------- page


def test_one_index_per_node_and_operation_specific_targets():
    els, targets, controls = policy.action_space(PAGE["actions"], PAGE["guards"])
    assert [e["label"] for e in els][:2] == ["Destination", "Find stays"]
    assert list(targets["TYPE_TEXT"]) == ["1"]                      # only the text field
    assert set(targets["SELECT"]) == {"3:1", "3:2"}
    assert "1" in targets["CLICK"] and set(controls) == {"SCROLL_DOWN", "WAIT"}


def test_a_card_button_carries_its_card():
    """Round 3 opened the €145 stay for 'the cheapest one' - prices were not on the buttons."""
    els, _, _ = policy.action_space(PAGE["actions"], PAGE["guards"])
    serra = next(e for e in els if e["label"] == "View Serra Lodge")
    assert "€120" in policy.element_line(serra)
    field = next(e for e in els if e["label"] == "Destination")
    assert "(in:" not in policy.element_line(field)                 # a huge scope is noise, dropped


def test_evidence_states_unsubmitted_text_plainly():
    els, _, _ = policy.action_space(PAGE["actions"], PAGE["guards"])
    ev = policy.render_state("find a stay in Lisbon", PAGE, els, [],
                             pending={"text": "Lisbon", "label": "Destination"})
    assert 'NOT YET SUBMITTED: "Lisbon" was typed into "Destination".' in ev
    assert ev.startswith("GOAL: find a stay in Lisbon")


# ------------------------------------------------------------- operations


class _NoModel(policy.Policy):
    def __init__(self):                                            # no backbone needed for these
        pass


def ops(goal, pending=None, allow_done=True):
    _, targets, controls = policy.action_space(PAGE["actions"], PAGE["guards"])
    return _NoModel().operations(goal, targets, controls, pending, allow_done)


def test_done_is_not_on_offer_while_text_is_unsubmitted():
    """Round 1's only false DONE: 'Lisbon' typed, a box ticked, DONE on an unfiltered page."""
    got = ops("find a stay in Lisbon", pending={"text": "Lisbon", "label": "Destination", "node": 11})
    assert "DONE" not in got and "SUBMIT" in got


def test_no_way_to_stop_before_the_first_attempt():
    """Round 2 called DONE on step 1; round 3 then called BLOCKED instead."""
    got = ops("show only nature stays", allow_done=False)
    assert "DONE" not in got and "BLOCKED" not in got


def test_operations_are_described_by_what_they_would_touch():
    """Round 3: SELECT 0.17 vs CLICK 0.41 for 'show only nature stays'."""
    els, targets, _ = policy.action_space(PAGE["actions"], PAGE["guards"])
    d = _NoModel().describe_ops(["SELECT", "TYPE_TEXT"], els, targets, None)
    assert "Nature" in d["SELECT"] and "Destination" in d["TYPE_TEXT"]


# ------------------------------------------------------------ loop guards


def test_open_goals_are_recognised_by_their_last_clause():
    last = lambda g: agent._CLAUSES.split(g)[-1]
    assert agent._OPEN.search(last("find a stay in Copenhagen and open it"))
    assert agent._OPEN.search(last("search for Lisbon, then open the cheapest one"))
    assert not agent._OPEN.search(last("search nike"))
    assert not agent._OPEN.search(last("reopen the tab"))          # a word, not a substring


# ------------------------------------------- the first real session, 2026-09-24


def test_sign_in_fields_and_buttons_are_never_offered_unasked():
    """'open blender' on GitHub's login page typed into the username box, then 'Continue with Google'."""
    login = [{"kind": "fill", "role": "textbox", "label": "Username or email address", "node": 1},
             {"kind": "click", "role": "textbox", "label": "Open Username or email address", "node": 1},
             {"kind": "click", "role": "button", "label": "Sign in", "node": 2},
             {"kind": "click", "role": "button", "label": "Continue with Google", "node": 3},
             {"kind": "click", "role": "link", "label": "Download — Blender", "node": 4},
             {"kind": "fill", "role": "searchbox", "label": "Search mobile phones", "node": 5}]
    kept = [a["label"] for a in policy.offered("open blender", login)]
    assert kept == ["Download — Blender", "Search mobile phones"]
    assert len(policy.offered("sign in to github", login)) == len(login)


@pytest.mark.parametrize("text,want", [
    ("go to github", ("go", "github")), ("got to github", ("go", "github")),
    ("open blender", ("open", "blender")), ("visit the blender website", ("go", "blender")),
    ("open the top one", None), ("open it", None), ("open browser", None), ("open a new tab", None)])
def test_a_site_by_name_is_recognised(text, want):
    """'got to github' was searched on Bing; 'open blender' typed the name into a login form."""
    assert site_request(text) == want


@pytest.mark.parametrize("text,opens,closes", [
    ("hi jobe, open browser", True, False), ("open browser", True, False), ("open chrome", True, False),
    ("close the browser", False, True), ("open blender", False, False), ("open browser and search nike", False, False)])
def test_browser_commands_are_rules_not_guesses(text, opens, closes):
    """A second 'open browser' with the browser already open was read as a task."""
    assert bool(agent._OPEN_BROWSER.fullmatch(text)) is opens
    assert bool(agent._CLOSE_BROWSER.fullmatch(text)) is closes


def test_a_run_on_request_is_split_before_the_pointing_end():
    """All of 'search for the latest news on github open the top one' was typed into Bing."""
    goal = normalise("search for the latest news on github open the top one")
    assert goal == "search for the latest news on github, then open the top one"
    assert search_query(goal, "https://www.bing.com/") == "latest news on github"
    assert normalise("search for open source tools") == "search for open source tools"
    assert candidate_spans("search for rock and roll")[0] == "rock and roll"     # no split inside a value


def test_a_click_command_has_nothing_to_type_and_ends_when_clicked():
    """After clicking Image creator it typed 'image creator' into the prompt box."""
    assert not typeable("click image creator") and typeable("find a stay in Copenhagen and open it")
    s = agent.Session.__new__(agent.Session)
    clicked = [{"step": 1, "operation": "CLICK", "changed": True, "label": "Image creator"}]
    assert s._pointing_done("click image creator", clicked)
    assert not s._pointing_done("click image creator", [dict(clicked[0], changed=False)])
    assert not s._pointing_done("open the top one", clicked)          # names nothing: the readout decides


def test_a_site_looked_up_by_name_prefers_its_own_results_and_never_page_furniture():
    """Round 9: asked for the top result for 'blender', it clicked Bing's 'Accessibility Help'."""
    import base64
    from jobe.browse.text import CHROME_LINK, destination_host, names_host
    ck = lambda d: "https://www.bing.com/ck/a?!&&p=1&u=a1" + base64.urlsafe_b64encode(d.encode()).decode().rstrip("=")
    serp = "https://www.bing.com/search?q=blender"
    assert destination_host(ck("https://www.blender.org/download/"), serp) == "www.blender.org"
    assert names_host("blender", "www.blender.org") and not names_host("blender", "go.microsoft.com")
    assert names_host("the new york times", "www.nytimes.com")
    assert CHROME_LINK.match("Accessibility Help") and not CHROME_LINK.match("Download — Blender")


@pytest.mark.parametrize("said,meant", [
    ("Can you search Adidas and a white tennis shoe?", "search Adidas and a white tennis shoe"),
    ("could you please go to github", "go to github"),
    ("I want you to search for flights to Perth", "search for flights to Perth"),
    ("hi jobe, can you search nike", "search nike"),
    ("search nike", "search nike")])
def test_the_polite_way_in_is_not_part_of_the_request(said, meant):
    """The first spoken request typed its polite opening into Bing, 'Can you' and all."""
    from jobe.browse.text import strip_preamble
    assert strip_preamble(said) == meant


def test_a_search_request_types_before_it_clicks():
    """'search github' on a results page clicked a GitHub link instead of searching."""
    assert search_first("search github") and search_first("find a stay in Lisbon")
    assert not search_first("find the cheapest one") and not search_first("open the top result")


def test_the_top_result_is_a_link_that_leaves_the_engine():
    """Round 6: 'open the top one' clicked DuckDuckGo's own 'News for ...' header."""
    serp = "https://duckduckgo.com/?t=h_&q=latest+news+on+github"
    assert results_page(serp) and not results_page("https://duckduckgo.com/")
    assert wants_a_result("open the top one") and not wants_a_result("open the github blog")
    assert not leaves_site("/?q=latest+news+on+github&iar=news", serp)
    assert leaves_site("https://github.blog/news-insights/", serp)
    # Bing sends result clicks through its own redirect - that still leaves
    assert leaves_site("https://www.bing.com/ck/a?!&&p=abc", "https://www.bing.com/search?q=python")
    assert not leaves_site("https://www.bing.com/images/search?q=python", "https://www.bing.com/search?q=python")


def test_a_click_tracker_is_judged_by_where_it_really_goes():
    """Round 7: Bing wraps its own tabs and related searches in /ck/a too - one was 'the top result'."""
    import base64
    ck = lambda dest: ("https://www.bing.com/ck/a?!&&p=1f&u=a1%s&ntb=1"
                       % base64.urlsafe_b64encode(dest.encode()).decode().rstrip("="))
    serp = "https://www.bing.com/search?q=python"
    assert leaves_site(ck("https://www.python.org/"), serp)
    assert not leaves_site(ck("/search?q=python+official+website"), serp)
    assert not leaves_site(ck("/images/search?q=python&FORM=HDRSC2"), serp)


def test_setting_a_value_does_not_count_as_opening_something():
    """Round 6: for 'open Serra Lodge' a ticked checkbox let DONE through on the search page."""
    s = agent.Session.__new__(agent.Session)
    s.history = [{"step": 1, "operation": "CLICK", "changed": True, "role": "checkbox", "label": "Free"}]
    assert not s._done_allowed("open Serra Lodge")
    s.history.append({"step": 2, "operation": "CLICK", "changed": True, "role": "button",
                      "label": "View Serra Lodge"})
    assert s._done_allowed("open Serra Lodge")


def test_meaning_ignores_node_identity():
    """SPAs re-render with fresh node ids; round 1 never saw 'nothing changed'."""
    other = json.loads(json.dumps(PAGE))
    for a in other["actions"]:
        if "node" in a:
            a["node"] += 1000
    assert agent.meaning(other) == agent.meaning(PAGE)


FIXTURE_CONTROLS = [
    {"kind": "fill", "role": "searchbox", "label": "Destination", "node": 1, "value": ""},
    {"kind": "click", "role": "checkbox", "label": "Free cancellation", "node": 2, "checked": "false"},
    {"kind": "select", "role": "combobox", "label": "Stay category → Nature", "node": 3,
     "value": "Nature", "current_value": "All stays"},
    {"kind": "select", "role": "combobox", "label": "Stay category → Design", "node": 3,
     "value": "Design", "current_value": "All stays"},
]


def test_a_control_the_goal_names_must_be_set_before_done():
    """Round 4: Lisbon typed and submitted, DONE called with 'Free cancellation' unticked."""
    assert unset_controls("find a Lisbon stay with free cancellation", FIXTURE_CONTROLS,
                          typed=["Lisbon"]) == ["Free cancellation"]
    assert unset_controls("show only nature stays", FIXTURE_CONTROLS) == ["Stay category → Nature"]


def test_a_set_control_or_a_goal_that_does_not_name_one_is_left_alone():
    ticked = [dict(a, checked="true") if a["role"] == "checkbox" else a for a in FIXTURE_CONTROLS]
    assert unset_controls("find a Lisbon stay with free cancellation", ticked, typed=["Lisbon"]) == []
    assert unset_controls("find a stay in Lisbon", FIXTURE_CONTROLS, typed=["Lisbon"]) == []
    # "stays" is in the dropdown's own name; it does not name the value "All stays"
    assert unset_controls("show stays", FIXTURE_CONTROLS) == []


def test_typed_words_are_spent_and_negations_are_not_second_guessed():
    brand = [{"kind": "click", "role": "checkbox", "label": "Nike", "node": 9, "checked": "false"}]
    assert unset_controls("search nike", brand, typed=["nike"]) == []
    assert unset_controls("show stays without free cancellation", FIXTURE_CONTROLS) == []


def test_a_goal_word_nothing_has_used_holds_done_while_a_field_is_empty():
    """Round 5: the box ticked, DONE called, 'Lisbon' never typed."""
    ticked = [dict(a, checked="true") if a["role"] == "checkbox" else a for a in FIXTURE_CONTROLS]
    goal = "find a Lisbon stay with free cancellation"
    assert unused_words(goal, ticked, title="Stays") == ["lisbon"]
    assert unused_words(goal, ticked, typed=["Lisbon"], title="Stays") == []


def test_used_words_come_from_clicks_set_controls_and_the_site_itself():
    assert unused_words("open Serra Lodge", FIXTURE_CONTROLS, clicked=["View Serra Lodge"]) == []
    chosen = [dict(a, current_value="Nature") for a in FIXTURE_CONTROLS]
    assert unused_words("show only nature stays", chosen) == []
    assert unused_words("search wikipedia for the eiffel tower", FIXTURE_CONTROLS,
                        typed=["eiffel tower"], title="Eiffel Tower - Wikipedia") == []


def test_with_every_field_filled_the_word_guard_stands_down():
    full = [dict(a, value="Lisbon") if a["kind"] == "fill" else a for a in FIXTURE_CONTROLS]
    assert unused_words("find a stay in Porto", full) == []


def test_adverts_are_not_offered_unless_asked_for():
    """2026-09-23: DuckDuckGo's first ten result links were two adverts and their sitelinks."""
    acts = [{"kind": "click", "role": "link", "label": "Udemy", "node": 1, "ad": True},
            {"kind": "click", "role": "link", "label": "Welcome to Python.org", "node": 2}]
    assert [a["label"] for a in policy.offered("open the top result", acts)] == ["Welcome to Python.org"]
    assert len(policy.offered("open the sponsored link", acts)) == 2


class _Frame:
    def __init__(self, url, box):
        self.url, self._box = url, box

    def frame_element(self):
        return type("El", (), {"bounding_box": lambda _self: self._box})()


def _wall(text, frames=(), title=""):
    fake = type("B", (), {"page": type("P", (), {"frames": list(frames)})()})()
    return Browser.challenge(fake, {"title": title, "text": text})


def test_a_human_verification_page_is_recognised():
    """Round 4: DuckDuckGo answered with a duck-picking challenge and the agent clicked Submit."""
    assert _wall("Unfortunately, bots use DuckDuckGo too. Please complete the following challenge "
                 "to confirm this search was made by a human.")
    assert _wall("Our systems have detected unusual traffic from your computer network.")
    assert _wall("", [_Frame("https://www.google.com/recaptcha/api2/anchor?k=x&size=normal",
                             {"width": 304, "height": 78})])
    # headless probe, 2026-09-23: Mojeek and Startpage refused instead of challenging
    assert "refusing" in _wall("Contact us", title="403 - Forbidden")
    assert "refusing" in _wall("Contact support", title="Access Denied - Startpage")


def test_ordinary_pages_are_not_walls():
    long_article = "CAPTCHA is a type of challenge-response test. " + "History of the test. " * 120
    assert _wall(long_article, title="CAPTCHA - Wikipedia") is None
    assert _wall("Nike. Just Do It. Shop the latest shoes.") is None
    # the invisible reCAPTCHA many ordinary forms embed
    assert _wall("Sign in", [_Frame("https://www.google.com/recaptcha/api2/anchor?k=x&size=invisible",
                                    {"width": 256, "height": 60})]) is None


def test_cancelling_a_paused_action_does_not_start_a_goal_called_cancel():
    """The window's Cancel button sends "cancel"; it once became the next goal."""
    events = []
    s = agent.Session.__new__(agent.Session)
    s.emit = lambda kind, data: events.append(kind)
    s.pending = {"goal": "buy it"}
    s.say("cancel")
    assert events == ["note", "done"] and s.pending is None


@pytest.mark.parametrize("text,match", [("continue", True), ("ok", True), ("I did it", True),
                                        ("done!", True), ("search nike", False), ("open the top one", False)])
def test_continuing_after_a_wall_is_a_short_reply_only(text, match):
    assert bool(agent._CONTINUE.fullmatch(text)) is match


def test_what_changed_names_the_new_url_and_text():
    after = dict(PAGE, url="https://stays.example/#serra-lodge", text=PAGE["text"] + "\nA hillside retreat")
    change = agent.what_changed(PAGE, after)
    assert "URL now https://stays.example/#serra-lodge" in change and "hillside" in change


# ------------------------------------------------------------ app security


@pytest.fixture()
def app_server():
    from http.server import ThreadingHTTPServer
    from jobe.browse import app
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield "http://127.0.0.1:%d" % httpd.server_address[1], app.TOKEN
    httpd.shutdown()
    httpd.server_close()


def _post(url, headers):
    req = urllib.request.Request(url, data=b'{"text":"search nike"}', method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        return urllib.request.urlopen(req, timeout=5).status
    except urllib.error.HTTPError as e:
        return e.code


def test_a_command_without_the_token_is_refused(app_server):
    base, _ = app_server
    assert _post(base + "/say", {}) == 403


def test_a_command_from_another_origin_is_refused_even_with_the_token(app_server):
    """Any page you visit can POST to 127.0.0.1 - it must not be able to drive the browser."""
    base, token = app_server
    assert _post(base + "/say", {"X-Jobe-Token": token, "Origin": "https://evil.example"}) == 403


def test_a_command_that_is_not_json_is_refused(app_server):
    """A cross-site 'simple' POST (text/plain) skips the CORS preflight - it must not get through."""
    base, token = app_server
    req = urllib.request.Request(base + "/say", data=b'{"text":"search nike"}', method="POST",
                                 headers={"Content-Type": "text/plain", "X-Jobe-Token": token})
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(req, timeout=5)
    assert err.value.code == 415


def test_a_request_the_browser_marks_cross_site_is_refused(app_server):
    base, token = app_server
    assert _post(base + "/say", {"X-Jobe-Token": token, "Sec-Fetch-Site": "cross-site"}) == 403


def test_the_browser_opens_beside_the_chat_not_under_another_window():
    """2026-09-24: it opened behind the Claude window and the person could not find it."""
    from jobe.browse.app import place_beside
    work = (1920, 0, 4480, 1392)                                   # a 2560-wide right screen
    got = place_beside((2069, 63, 2875, 1051), work, 1180, 820)    # the chat, where it was
    assert got == {"x": 2887, "y": 63, "width": 1180, "height": 820}
    assert got["x"] + got["width"] <= work[2]
    # chat hard against the right edge: the browser goes to its left
    left = place_beside((3900, 40, 4460, 820), work, 1180, 820)
    assert left["x"] + left["width"] <= 3900 and left["x"] >= work[0]
    # no room either side: keep the default placement
    assert place_beside((2200, 0, 4200, 900), work, 1180, 820) == {}


def test_the_agent_browser_is_kept_off_the_chat_server():
    from jobe.browse import browser
    browser.protect(7900)
    assert {"127.0.0.1:7900", "localhost:7900", "[::1]:7900"} <= browser.PROTECTED
    assert browser._SECRET_ENV.search("OPENROUTER_API_KEY") and not browser._SECRET_ENV.search("PATH")


# ------------------------------------------------------------ a hosted brain


@pytest.fixture()
def fake_openrouter():
    """An OpenAI-compatible endpoint that always puts its mass on the first letter."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    seen = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append({"auth": self.headers.get("Authorization"), "model": body["model"],
                         "logprobs": body.get("logprobs"), "max_tokens": body.get("max_tokens")})
            out = json.dumps({"choices": [{"logprobs": {"content": [{"top_logprobs": [
                {"token": "A", "logprob": -0.05}, {"token": "B", "logprob": -3.0}]}]}}],
                "usage": {"prompt_tokens": 42}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    cfg = {"model": "test/model", "key": "k-test", "top_logprobs": 20, "timeout": 5,
           "endpoint": "http://127.0.0.1:%d/v1/chat/completions" % httpd.server_address[1]}
    yield cfg, seen
    httpd.shutdown()
    httpd.server_close()


def test_a_hosted_brain_narrows_wide_choices_one_level_at_a_time(fake_openrouter):
    from jobe.prompt import Option
    cfg, seen = fake_openrouter
    brain = policy.RemotePolicy(cfg)
    options = [Option("e%d" % i, 'link "Result %d"' % i) for i in range(20)]
    r = brain._ask("target:CLICK", "a results page", "Which element?", options)
    assert r.choice == "e0" and r.passes == 2            # 20 options: a group question, then 10
    assert len(seen) == 2 and all(s["logprobs"] and s["max_tokens"] == 1 for s in seen)
    assert seen[0]["auth"] == "Bearer k-test"


def test_one_option_is_not_sent_to_the_provider(fake_openrouter):
    from jobe.prompt import Option
    cfg, seen = fake_openrouter
    r = policy.RemotePolicy(cfg)._ask("t", "page", "Which?", [Option("only", "the search box")])
    assert r.choice == "only" and not seen


def test_switching_brains_checks_the_key_and_the_id_first(app_server, monkeypatch):
    from jobe.browse import app
    base, token = app_server
    monkeypatch.setitem(app.STATE, "ready", True)
    monkeypatch.setattr(app, "openrouter_key", lambda: None)

    def brain(model):
        req = urllib.request.Request(base + "/brain", data=json.dumps({"model": model}).encode(),
                                     method="POST", headers={"Content-Type": "application/json",
                                                             "X-Jobe-Token": token})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    assert brain("rm -rf /")[0] == 400
    code, body = brain("mistralai/mistral-nemo")
    assert code == 400 and "OPENROUTER_API_KEY" in body["error"]


def test_a_probed_hosted_model_is_queued_for_the_worker(app_server, monkeypatch, fake_openrouter):
    from jobe.browse import app
    base, token = app_server
    cfg, _ = fake_openrouter
    monkeypatch.setitem(app.STATE, "ready", True)
    monkeypatch.setattr(app, "remote_cfg", lambda model: dict(cfg, model=model))
    q = app.STATE["commands"]
    while not q.empty():
        q.get_nowait()
    req = urllib.request.Request(base + "/brain", data=b'{"model": "test/model"}', method="POST",
                                 headers={"Content-Type": "application/json", "X-Jobe-Token": token})
    with urllib.request.urlopen(req, timeout=10) as r:
        assert r.status == 200
    cmd, arg = q.get_nowait()
    assert cmd == "brain" and isinstance(arg, policy.RemotePolicy) and arg.name == "test/model"


# ------------------------------------------------------ quality of life, 2026-09-24


def _cmd(base, token, path, body=b"{}", kind="application/json"):
    req = urllib.request.Request(base + path, data=body, method="POST",
                                 headers={"Content-Type": kind, "X-Jobe-Token": token})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def test_new_chat_and_answers_are_queued_for_the_worker(app_server):
    from jobe.browse import app
    base, token = app_server
    q = app.STATE["commands"]
    while not q.empty():
        q.get_nowait()
    assert _cmd(base, token, "/clear")[0] == 202
    assert _cmd(base, token, "/pick", b'{"id": "7"}')[0] == 202
    assert [q.get_nowait(), q.get_nowait()] == [("clear", None), ("pick", "7")]


def test_the_hub_forgets_a_cleared_conversation_but_keeps_counting():
    from jobe.browse.app import Hub
    hub = Hub()
    hub.emit("user", {"text": "search nike"})
    hub.clear()
    hub.emit("cleared", {})
    _, backlog = hub.subscribe()
    assert [e["kind"] for e in backlog] == ["cleared"] and backlog[0]["seq"] == 2


def test_audio_must_be_wav_and_reaches_speech_to_text(app_server, monkeypatch):
    """The microphone posts WAV; anything else is refused before it is read."""
    from jobe.browse import voice
    base, token = app_server
    assert _cmd(base, token, "/transcribe", b"RIFF....", kind="text/plain")[0] == 415
    monkeypatch.setattr(voice, "ensure", lambda wait=90.0: None)
    assert "error" in voice.transcribe(b"not a wav")
    seen = []

    class Reply:
        def __init__(self, data): self.data = data
        def read(self): return self.data
        def __enter__(self): return self
        def __exit__(self, *a): return False

    real_urlopen = urllib.request.urlopen

    def fake_urlopen(req, timeout=0):
        if not getattr(req, "full_url", str(req)).startswith(voice.STT_URL):
            return real_urlopen(req, timeout=timeout)            # the test's own requests
        seen.append((req.full_url, req.data[:4]))
        return Reply(b'{"text": "search nike", "ms": 40}')

    monkeypatch.setattr(voice.urllib.request, "urlopen", fake_urlopen)
    code, body = _cmd(base, token, "/transcribe", b"RIFF" + b"\0" * 40, kind="audio/wav")
    assert code == 200 and body["text"] == "search nike"
    assert seen == [(voice.STT_URL + "/transcribe", b"RIFF")]


def test_the_server_shuts_itself_down_once_the_window_has_gone(monkeypatch):
    """Closing the window used to leave the model holding ~9 GB of the card."""
    from jobe.browse import app
    calls = []
    monkeypatch.setattr(app, "shutdown", lambda reason: calls.append(reason) or
                        app.STATE.__setitem__("stopping", True))
    monkeypatch.setitem(app.STATE, "stopping", False)
    monkeypatch.setitem(app.STATE, "busy", False)
    monkeypatch.setattr(app.HUB, "last_seen", time.time() - 10)
    monkeypatch.setattr(app.HUB, "watching", lambda: False)
    app.watchdog(linger=1, every=0.01)
    assert calls == ["the chat window was closed"]


def test_close_calls_are_asked_not_guessed():
    s = agent.Session.__new__(agent.Session)
    ch = policy.Choice(operation="CLICK", op_probs={}, op_confidence=0.5,
                       target_probs={"1": 0.46, "2": 0.32, "3": 0.1},
                       candidates={"1": ('[1] link "Sign in"', {}, {}), "2": ('[2] link "Blender"', {}, {}),
                                   "3": ('[3] link "Help"', {}, {})})
    assert s._unsure(ch)
    ch.target_probs = {"1": 0.77, "2": 0.04, "3": 0.02}
    assert not s._unsure(ch)                                    # a clear winner is just done
    ch.target_probs = {"1": 0.46, "2": 0.32}
    ch.operation = "TYPE_TEXT"
    assert not s._unsure(ch)                                    # only clicks and selects are asked


def test_the_page_serves_the_token_only_to_itself(app_server):
    base, token = app_server
    with urllib.request.urlopen(base + "/", timeout=5) as r:
        assert token in r.read().decode()
    req = urllib.request.Request(base + "/", headers={"Origin": "https://evil.example"})
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(req, timeout=5)

# The browser agent behind the chat window

`jobe.browse` turns the letter readout into a browser agent you talk to. This
page is the design, every rule it enforces with the failure that produced it,
the measurements, and what was taken from two outside projects.

## Running it

```bash
desktop\Jobe.cmd                                   # double-click: server + always-on-top window
python desktop/jobe_chat.py                        # the same, from a shell
PYTHONPATH=src python -m jobe.browse.app           # just the server, on 127.0.0.1:7900
PYTHONPATH=src python -m jobe.browse "search nike" "open the top result"   # no window, prints decisions
PYTHONPATH=src python bench/browse_suite.py        # the evaluation (fixture headless, live sites headed)
```

The window is a Chrome app window with a profile of its own, pinned on top.
Not Edge: on a PC signed in with a Microsoft account, Edge signs every new
profile into that account and turns on sync, and the first version of the
window opened on "We are now syncing your browsing data across all your
devices". Without Chrome, the launcher uses the Chromium Playwright installed.

## One process

Model, browser and chat live in one Python process. A single worker thread owns
the model and the browser (Playwright's sync API must stay on the thread that
created it); HTTP threads only queue commands and stream events. The version
before this was three processes - a decision server, a Node console and a Tk
widget - and most of an evening's failures were the seams between them.

Everything the loop does is an event (`intent`, `decision`, `typing`, `acted`,
`confirm`, `done`, ...), streamed to the window over `fetch` so the access token
travels as a header, never in a URL. A reopened window replays the last 300
events; only the newest ten keep their screenshots.

## One step

1. **Observe.** `snapshot.js` (vendored unchanged from jev-ultrafast) lists the
   actionable elements in the viewport with code-owned node ids, plus the page
   text. Adverts are marked (see the rules below).
2. **Evidence**, rendered once as compact lines -
   `[4] combobox "Search with DuckDuckGo" = ""` - with the goal, what the last
   actions changed, and anything still unsubmitted or unset. The evidence is
   encoded once; every question below is a suffix off that prefix.
3. **Operation head**: CLICK, TYPE_TEXT, SELECT, SUBMIT, SCROLL, WAIT, DONE or
   BLOCKED - only the ones this page and this moment allow, each described by
   what it would touch here.
4. **Target head**, only for the chosen operation, over only the elements that
   operation can act on.
5. **Text**, for TYPE_TEXT: chosen among spans of the person's own words. Nothing
   is ever typed that they did not say.
6. **Act**, after proving the element is still connected, visible, enabled and
   not covered - then wait for the page to finish filling in.

A decision costs a median 262 ms on this card (round 8, 56 decisions, median
482 evidence tokens).

## Rules, and the failure each came from

Each rule is plain code, pinned by a test in `tests/test_browse.py` that names
the round it came from. The suite rounds are `runs/browse-suite-r*.json`.

| Rule | What failed without it |
|---|---|
| No DONE or BLOCKED before the first action of a goal | Round 2 called DONE on step 1 of "show only nature stays"; round 3 then called BLOCKED instead |
| No DONE while typed text is unsubmitted; SUBMIT offered instead | Round 1: "Lisbon" typed, a box ticked, DONE on a page still listing Copenhagen |
| A goal ending in "open ..." needs a link or button click that changed the page | Round 2 searched and stopped short of opening; round 6 counted a ticked checkbox as "opening" |
| A checkbox, radio or dropdown value the goal names must be set before DONE | Round 4: DONE with "Free cancellation" unticked |
| A goal word nothing has used holds DONE while a text field is still empty | Round 5: the box ticked, DONE called, "Lisbon" never typed |
| ...but that last rule is a DONE guard, not advice shown on step 1 | Round 6: "NOT YET USED: serra, lodge" sent "open Serra Lodge" into the search box |
| No TYPE_TEXT when the goal has nothing to type | Round 1 typed "top result" into DuckDuckGo fourteen times |
| On a web search engine, the box gets the whole request | Round 5 on Bing typed "python" for "search for the python programming language" |
| After a click or Enter, wait for network idle and a still DOM (bounded) | Round 5: DuckDuckGo's `load` fires on an empty page; five of six failures decided on it |
| Adverts are not offered unless the goal asks for them | DuckDuckGo's first ten result links were two adverts and their sitelinks |
| "Open the top one" on a results page is offered only links that leave the engine | Round 6 clicked DuckDuckGo's own "News for ..." header |
| ...judging a click-tracker by where it really goes | Round 7: Bing wraps its own tabs and related searches in `/ck/a` too |
| A click that changed nothing is not offered again in that goal | Round 3 re-clicked an open item until the loop gave up |
| Stop on repeats, an A-B-A-B oscillation, or three actions that changed nothing | Round 1 clicked for fourteen steps on a page it had already reached |
| "Changed" is judged by meaning, not node identity | Single-page apps re-render with fresh node ids on every click |
| A human-verification page stops the goal; "continue" resumes it | Round 4: DuckDuckGo's duck-picking challenge, and the agent clicked its Submit |
| Sign-in fields and buttons are never offered unless the request says to sign in | 2026-09-24, first real use: "open blender" on GitHub's login page typed into the username box, then clicked "Continue with Google" |
| "go to X", or "open X" when nothing on the page carries X, looks X up and opens the top result | Same session: "got to github" was searched as text; "open blender" is above |
| "open browser" and "close browser" are rules, not readout guesses | Same session: a second "open browser" was read as a task |
| A run-on request is split before its pointing end, and only value clauses are typed | Same session: all of "search for the latest news on github open the top one" was typed into Bing |
| A click command offers no typing, and ends once the thing it names was clicked | Same session: after clicking Image creator it typed "image creator" into the image prompt |
| A search request types before it clicks | Same session: "search github" clicked a GitHub link instead |
| An element that fails the pre-click check is not offered again; a covered link or button is clicked directly | Same session: Bing's streaming answer failed the check three times and the goal gave up |
| An action judged hard to undo waits for the person's OK | This project's own boundary, not a measured failure |

## Measured

| Round | Passed | False DONE | Walls | Decisions | Median ms | Median evidence tokens |
|---|---|---|---|---|---|---|
| 4 | 10 / 12 | 1 (and the live passes were checked by URL alone) | not counted | 42 | 211 | 457 |
| 5 | 8 / 14 | 6 | 0 | 54 | 216 | 457 |
| 6 | 12 / 14 | 2 | 0 | 59 | 253 | 506 |
| 7 | 13 / 14 | 1 | 0 | 56 | 254 | 482 |
| 8 | 14 / 14 | 0 | 0 | 56 | 262 | 482 |
| 9 | 19 / 20 | 1 | 0 | 62 | 293 | 570 |

Round 9 added six requests from the first real session, word for word, and
passed five. The sixth, "open blender", looked blender up and then clicked
Bing's "Accessibility Help", which links off-site. With page furniture excluded
and a site's own results preferred, the seven tasks that open a result re-ran
7 / 7.

Round 8 is the code as committed, including the security hardening of the
agent's browser. One run each, one seed of live pages: a pass rate on 14 tasks
says the loop works end to end, not how often it will on a new site.

Round 5 looks like a regression and is not: from round 5 on, a live pass also
needs result links on the page, and round 4's DuckDuckGo "passes" were
challenge pages with `q=nike` in their URL. The stretch task, "find the best
flight from melbourne to perth in october", passes as a search - it lands on
results, not on a booking.

**Headless browsers are walled.** One results page per engine, 2026-09-23:

| Engine | Headless | Headed |
|---|---|---|
| DuckDuckGo | challenge | results |
| Google | "unusual traffic" | results |
| Bing | results | results |
| Brave | captcha | results |
| Ecosia | challenge | results |
| Startpage | access denied | results |
| Mojeek | 403 | captcha |

So the window's browser is headed, and the suite runs its live tasks headed.
Jobe never solves a challenge and adds no evasion beyond the one Chromium flag
agent browsers commonly set.

## A hosted brain instead of Jobe

The model button in the window's header switches what answers the questions:
Jobe on this card, or any OpenRouter model that returns logprobs. The loop,
the rules and the evidence stay exactly the same; only the question-answering
moves off the card (`RemotePolicy`, through `jobe.remote`).

- The key comes only from `OPENROUTER_API_KEY` - the server's environment, or
  the Windows user environment, read when you switch - and never reaches the
  page. `OPENROUTER_ENDPOINT` points it at any OpenAI-compatible gateway.
- A model is tested with one real decision before it is accepted: no logprobs,
  or a reasoning preamble before the answer, and it cannot serve the protocol.
  The current brain stays until then, and a switch requested mid-goal waits for
  the goal to end.
- `--brain vendor/model` starts on a hosted model and does not load Jobe at all,
  which leaves the card free for other work.
- What it costs: no prefix cache, so every question re-sends the page; only a
  top-20 window comes back; wide choices are narrowed a level at a time with the
  same balanced tree as `jobe.wide`; and the provider sees every page.

Verified 2026-09-24 against a local stand-in endpoint (unit tests, the agent
loop, and the window clicked through end to end). **Not yet run against
OpenRouter itself** - no key was set on this machine.

## Security

A local server that drives a browser is a target: any page you visit can send a
request to 127.0.0.1.

- Commands need a per-launch token that only this app's own page can read.
- Host must be 127.0.0.1 or localhost on this port (DNS rebinding), Origin must
  be this app's, and a request the browser labels cross-site is refused.
- Every command must be declared `application/json` (415 otherwise): a
  cross-site "simple" POST skips the CORS preflight, and cannot be JSON.
- The page cannot be framed (`X-Frame-Options`, `frame-ancestors 'none'`).
- The agent's browser can never load the chat server itself - a page could
  otherwise lead the agent to its own control panel - downloads nothing, and is
  launched without API keys or tokens in its environment.

## DeepSeek Harness: what was worth taking

[deepseek-ai/deepseek-harness](https://github.com/deepseek-ai/deepseek-harness)
(MIT, TypeScript) is a plugin-kernel harness on Cordis for tool-calling models.
Its browser-use and computer-use packages are deliberately thin: each is a
registry that admits exactly one provider by name and owns no browser, no action
type and no dispatch. The providers - Playwright MCP, Chrome DevTools MCP,
Stagehand, Cua Driver - expose their own native tools to the model.

**Taken, in `app.py` and `browser.py`**, from its design notes:

1. A media-type fence on every command, and refusing `Sec-Fetch-Site:
   cross-site` outright (their "carrier-level browser trust" note).
2. Keeping the agent's browser off the host's own endpoint, loopback aliases
   included (their desktop webview's request filter).
3. Launching the browser with a scrubbed environment and no downloads (their
   Stagehand launcher inherited its environment; their guest policy denies
   downloads).

**Deliberately not taken: their refusal to unify actions.** It is right for
them - a generalist model can call whatever tool a provider exposes - and wrong
here: a letter readout can only choose among declared options, so the finite
action vocabulary (`snapshot.js`'s element list, one head per operation) is the
whole method, not a limitation.

**Fork it for a Jobe-like model? No.** Its model contract is an LLM adapter that
streams text and tool calls; a readout that never generates cannot satisfy it,
and forcing it to would throw away what the readout is good at. The fit runs the
other way: their browser-use slot takes a provider whose tools arrive over MCP,
so a Jobe browser MCP server - one tool, `browse(goal)`, running this loop
locally - would plug into DeepSeek Harness, Claude Code or anything else that
speaks MCP. A large model plans; Jobe does the clicking, fast and on this card.
That is the next integration worth building.

Their attach mode is also worth knowing: they can attach to your existing,
logged-in browser, reserving it exclusively and disconnecting without closing it
afterwards. Jobe does not do that - it is the one thing here that would act
inside your real sessions - and should only ever be an explicit opt-in.

## jev-ultrafast: what came from it

`browser-use/jev-ultrafast` (MIT) is the design this adapts: `snapshot.js`
unchanged, one operation question plus per-operation target questions,
pre-input freshness checks, select-all-then-insert typing, the 200 ms
combobox settle. What differs: it attaches to your own Chrome over remote
debugging and asks a hosted API every target question speculatively; this
launches an isolated Chromium, runs in process, asks only the head the chosen
operation needs, and chooses typed text among the goal's own words instead of
generating it.

## Known limits

- Only elements inside the viewport are offered; below the fold means SCROLL.
- Values the person never said cannot be typed - no dates inferred from
  "october", no addresses.
- Multi-field forms (flights) are the weak spot: the stretch task succeeds as a
  search, not as a form.
- The always-on-top pin needs the chat window's process to be allowed into the
  foreground, which Windows grants when you launch it yourself. From a
  background process, while a security prompt held the foreground, Windows
  accepted the call and ignored it.

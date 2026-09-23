# A browser agent where every decision is Jobe

The split people ship with Jev: a plan says, in words, what outcome is wanted,
and the model decides. Nothing is generated. Each loop takes a DOM snapshot,
renders it as a numbered table of interactive controls, and asks the readout
which element, which operation and what the status is, then acts with
Playwright.

Needs `playwright` in the same environment as jobe (`pip install playwright`; the browser download is skipped if one is cached).

```bash
python demo/site_serve.py          # or any static server on demo/site
python demo/agent.py               # headed, so you can watch it
```

`--url` and `--goal` point it at anything else. `--headless` for CI, `--slow`
for the dwell between steps, `--shot out.png` for a final frame, `--out
trace.json` for the decision trace.

![the end of a run](run.png)

## What a run looks like

Five steps, fifteen decisions, no generation anywhere:

| step | decision | choice | p |
|---|---|---|---|
| 1 | which element | `[12] checkbox Direct flights only` | 0.98 |
| 2 | which element | `[14] button Search flights` | 0.99 |
| | | *nothing changed; set aside for this page state* | |
| 3 | which element | `[7] dropdown To` | 0.98 |
| 3 | which value | `Rome (FCO)` | 0.99 |
| 4 | which element | `[14] button Search flights` | 0.99 |
| 5 | which element | `[7] Select — Ryanair, 06:25, 2h 30m, Direct, £98` | 0.69 |
| 5 | step status | `done` | 0.98 |

Step 5 is the one worth watching. Five fares are on screen, the cheapest is
£98 with one stop... no: the cheapest *direct* is £98 and the cheapest overall
is also £98, while a £112 fare has a stop and a £214 fare is direct. Picking it
means reading price and stops together, which is a real judgement over a table,
and the readout gets it in one forward pass.

## Five things the build taught, each of which cost a wrong run

**Do not ask a question the DOM has already answered.** Offering TYPE_TEXT for a
`<select>` produced a confident-sounding 0.41 and a crash. Operations are now
filtered by what the control can take, and when only one is legal the loop says
so instead of asking.

**DONE is not an operation.** An operation is what you do *to* an element.
Offering "done" in the same answer set let a 0.62-confidence DONE stop the run
one click short of the goal. Whether the task is finished is a separate
question with a separate answer set.

**"Already tried" has to be keyed to the page state, not the element.** Search
before a destination is chosen does nothing; the same button two steps later is
exactly right. And a checkbox toggled back and forth changes the page every
time, so a plain no-change test never catches that loop. Pairing the state with
the element catches both.

**The status decision needs the page's words, not its controls.** A confirmation
page's controls are just the nav bar. "Booking held — Ryanair" is the entire
signal and it is not a control.

**A plan that never says what finished looks like cannot expect the model to
infer it.** Adding one sentence — "the task is complete once the page confirms a
booking is held" — moved the status confidences from 0.99 / 0.75 / 0.63 to
1.00 / 0.86 / 0.98, and it is the difference between stopping at the
confirmation page and wandering into the nav bar.

## The honest numbers

**2,851 ms per decision, on prompts averaging 433 tokens.** Not the 80 ms in the
README, and the gap is prompt length rather than anything else: 152 tokens costs
80 ms and 500 tokens costs about 2.8 seconds, with the GPU verified at 1,965 MHz
throughout, so it is not a power-state artefact. Three quarters of this
backbone's layers are linear-attention, `causal_conv1d` is not installed on this
machine, and transformers says so at every load — it runs those layers through
reference PyTorch, which scales badly with sequence length. Installing the
kernel is the first optimisation and it is not done here.

So: a real agent-loop prompt costs seconds on this install, against the roughly
300 ms per call that published Jev-based browser agents report. The decisions
are good. The speed claim belongs to short prompts and does not survive a
19-control page.

**Menus over sixteen are split rather than scored as text.** The letter protocol
stops at sixteen options and scoring option text instead is about eighteen times
slower, so a 19-control page becomes a group decision then an element decision.
Two decisions, no fallback.

**The gate.** Any step status under 0.80 hands over instead of acting, because
status is the decision this readout is measurably weakest at — 4 of 6 in
`bench/computer_use.py`, where every error in that set sat below 0.80.

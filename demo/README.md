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
| 5 | which element | `[7] Select — Ryanair, 06:25, 2h 30m, Direct, £98` | **0.43** |
| 5 | step status | `done` | 0.98 |

Step 5 is the one worth watching, and it is deliberately a trap. Five fares
are on screen and **the cheapest is £89 on Wizz Air, with a stop**. The cheapest
*direct* is £98 on Ryanair. Answering needs price and stops read together, so a
model that only sorts on price gets it wrong.

It gets it right, and it is honest about how close it was: **p = 0.43**, against
0.98 and 0.99 on every easy decision in the run. That is the one number here
that should give you pause. The confidence gate measured in
`bench/computer_use.py` sits at 0.80, and if it were applied to element choices
rather than only to status, this correct answer would have been escalated. On
the 39-decision set a 0.80 gate caught 3 errors out of 3 and kept 30 decisions
that were all right; on a 16-row table of fares it would hand over a right
answer. The gate needs tuning per decision type, and that is not done.

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

**The speed numbers in this file are not trustworthy and are being re-measured.**
A first pass reported about 3 seconds a decision on 433-token prompts against
80 ms on a 126-token one, and I attributed the gap to the missing
`causal_conv1d` kernel. That attribution was wrong. Checked by inspection, the
expensive path is already fast:

| hook | fast? | bound to |
|---|---|---|
| gated delta rule, chunked | **yes** | `fla.ops.gated_delta_rule.chunk` |
| gated delta rule, recurrent | **yes** | `fla.ops.gated_delta_rule.fused_recurrent` |
| `causal_conv1d_fn` | no | torch fallback |

and the fallback that remains is one `F.conv1d`, a grouped depthwise
convolution of kernel width four. That is a cuDNN call linear in sequence
length; it cannot cost seconds.

What the slow readings almost certainly are is memory pressure. This model is
about 8.6 GB on a 10 GB card, so anything else resident makes it spill, and a
later run against 6.6 GB of someone else's job reported **11 seconds** a
decision. `require_headroom()` now refuses to start without room, because a
timing taken on a shared card is not a timing. The decision-quality results are
unaffected: what was chosen, and at what confidence, does not depend on how long
it took.

**Menus over sixteen are split rather than scored as text.** The letter protocol
stops at sixteen options and scoring option text instead is about eighteen times
slower, so a 19-control page becomes a group decision then an element decision.
Two decisions, no fallback.

**The gate.** Any step status under 0.80 hands over instead of acting, because
status is the decision this readout is measurably weakest at — 4 of 6 in
`bench/computer_use.py`, where every error in that set sat below 0.80.

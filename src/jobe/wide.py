# -*- coding: utf-8 -*-
"""Choices wider than the answer-slot protocol can hold.

The letter readout scores one token per option, so it stops at 16. Real agent
loops ask much wider questions: `jev-browser` picks a DOM element out of up to
240 candidates (its own constant says the hosted model accepts 255), and on the
25 public tasks in its bench, 10 exceed 16 on the very first decision —
`webform` by exactly one option, `wiki-link` by 2,509.

This narrows instead. Options are grouped into a tree with branching factor at
most `cap`; one readout picks among the groups, the likeliest groups are opened,
and the probabilities multiply down the path. Every pass shares one piece of
evidence, so the whole thing runs off a single primed prefix — the page is
encoded once and each narrowing pass is a short suffix.

    P(option) = P(group) * P(option | group)

**This is a different estimator, not a cheaper route to the same number, and it
is measured.** `bench/narrowing.py` forces 142 labelled JevBench tasks of 4-6
options through a deliberately lowered `cap` and scores both paths against the
same gold:

    flat        0.732        —
    cap3-full   0.704   net -4 of 142   (4 fixed, 8 broken, McNemar p = 0.39)
    cap3-prune  0.697   net -5          (2.0 passes instead of 3.0)
    cap2-full   0.683   net -7          (p = 0.12) - the deepest tree, the worst

So: **not significant, and not free either.** Every arm's point estimate is
negative, the ordering is stable, and the honest reading is that there is no
evidence narrowing is lossless and suggestive evidence it costs a little. Use it
because the alternative is refusing the decision, not because it is equivalent.

Two things the measurement did settle. **Depth costs**: cap 2 builds one more
level than cap 3 and loses twice as much, so the shallowest tree that fits is
the right one — at cap 16 a 240-option choice is depth 2, the good regime.
**Pruning is nearly free**: opening only the likeliest group costs 0.7 points
against opening all of them and saves a third of the passes.

What is structural, and worth stating plainly: a narrowed decision is not a
harder problem for the model than the flat one. Each individual readout still
sees at most `cap` options. What it loses is the chance to weigh a late
candidate against an early one directly — they only ever meet as groups, and
that is where the missing points are.

Two design points that came from the caller, not from taste:

  * **A full distribution, not a winner.** `jev-browser`'s `resolve()` sorts
    `target.probabilities` and, when the top element does not fit the chosen
    tool, walks down for one that does with a `>= 0.1` floor. A winner alone
    would break that.
  * **Unopened groups keep their mass**, spread evenly over their members, so
    the result is still a distribution and an unexamined option reads as
    uncertain rather than as impossible. Zeroing them would sum to less than 1
    and would state, falsely, that the option was ruled out.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .prefix import PrefixScorer
from .prompt import Decision, DecisionError, Option
from .readout import Readout, score
from .slots import MAX_OPTIONS

#: Open groups until this much probability mass is covered.
EXPAND_MASS = 0.9
#: Never open more than this many groups at one level, whatever the mass.
MAX_EXPAND = 4
#: Characters of a group's summary shown to the model. 700 is `jev-browser`'s
#: own figure for summarising 30 elements - a value taken from a system that
#: works rather than invented here. It matters: at 220 a 16-member group of
#: ordinary DOM labels already overflows, so members get dropped from the
#: summary at depth 2, where nothing should need dropping.
SUMMARY_LIMIT = 700


@dataclass(frozen=True)
class WideReadout:
    """A distribution over every declared option, however many there were."""

    decision_id: str
    option_ids: tuple[str, ...]
    probabilities: tuple[float, ...]
    #: Forward passes actually run. 1 when the decision fitted and nothing narrowed.
    passes: int = 1
    #: Levels of the tree; 1 means it fitted in one flat readout.
    depth: int = 1
    #: True when the flat readout answered it and this module did nothing.
    flat: bool = True
    #: Options whose probability is a spread-out group mass, never examined
    #: individually. Their ranking among themselves is not informative.
    estimated: frozenset[str] = field(default_factory=frozenset)
    input_tokens: int = 0
    total_seconds: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def choice(self) -> str:
        best = max(range(len(self.option_ids)), key=lambda i: self.probabilities[i])
        return self.option_ids[best]

    @property
    def scores(self) -> dict[str, float]:
        return dict(zip(self.option_ids, self.probabilities))

    def confidence(self) -> float:
        """Chance-corrected against the FULL option set.

        Deliberately against all N options, not against the `cap` the winning
        readout actually saw: the caller asked an N-way question, and a number
        rescaled against 16 would read far more confident than the decision was.
        """
        k = len(self.probabilities)
        if k < 2:
            return 0.0
        return (max(self.probabilities) - 1.0 / k) / (1.0 - 1.0 / k)


# ------------------------------------------------------------------ the tree


@dataclass
class _Node:
    """A group of options, or a single option when `leaf` is set."""

    children: list["_Node"] = field(default_factory=list)
    leaf: Option | None = None

    @property
    def options(self) -> list[Option]:
        if self.leaf is not None:
            return [self.leaf]
        out: list[Option] = []
        for child in self.children:
            out.extend(child.options)
        return out


def build_tree(options: tuple[Option, ...], cap: int) -> list[_Node]:
    """Group options into at most `cap` nodes, preserving presentation order.

    Built bottom-up and BALANCED: each fold picks the fewest groups that respect
    the cap, `ceil(len/cap)`, then splits into that many groups of near-equal
    size. Both halves of that matter, and each cost a measurement:

      * Taking `ceil(len/cap)` as the group SIZE instead caps the top level while
        leaving every node with `ceil(len/cap)` children, so descending one level
        asks a question wider than the protocol allows - 2,525 options that way
        gives 16 groups of 158.
      * Folding into fixed runs of `cap` respects the cap but leaves a ragged
        remainder, and the remainder is not harmless. At cap 3 a 4-option
        decision became a group of 3 against a LONE SINGLETON, and that singleton
        competes at the top level against a whole group on nothing but the shape
        of the fold. Measured over 70 four-option JevBench tasks, it cost 10
        points of accuracy (0.629 -> 0.529), while the balanced 5- and 6-option
        cases lost 1.8 and gained 6.2.

    Order is preserved because a DOM's neighbours are usually related, which is
    the same reason `jev-browser` groups consecutive elements rather than
    clustering them.
    """
    nodes = [_Node(leaf=o) for o in options]
    while len(nodes) > cap:
        groups = -(-len(nodes) // cap)          # fewest groups that respect the cap
        size, extra = divmod(len(nodes), groups)
        folded, at = [], 0
        for g in range(groups):
            take = size + (1 if g < extra else 0)   # spread the remainder, one each
            folded.append(_Node(children=nodes[at:at + take]))
            at += take
        nodes = folded
    return nodes


def summarise(node: _Node, limit: int = SUMMARY_LIMIT) -> str:
    """What a group looks like to the model: its members, sampled across the span.

    Sampled, not truncated, and the first and last members are always kept so
    the group's span is visible. The reason is structural: a top-level group of
    a 2,525-option decision holds 256 options, and cutting the joined text at
    `limit` describes only its first 5%, so the level that decides whether to
    open the group cannot see most of what is in it.

    **Which of the two is better is NOT measured.** At the shipping budget the
    obvious probe - a fake scorer looking for a marker string - scores them
    identically: 2/5 each on a 2,525-option decision. It cannot settle the
    question in either direction, because it rewards an exact substring rather
    than judging what a group is about, which is what a real model would do.
    Sampling is chosen on the principle that a summary should represent the
    whole group rather than its first 5%. Settling it needs the backbone.

    Where the budget does bite is depth 3 and only depth 3. At 700 characters,
    recovery on this probe is 5/5 at 230, 256 and 275 options, and 2/5 at 2,525,
    where one group holds 256 members. `summary_limit` is the knob, and raising
    it recovers all of them.
    """
    parts = [o.description for o in node.options]
    text = " | ".join(parts)
    if len(text) <= limit:
        return text
    avg = max(1, len(text) // len(parts))
    keep = max(2, min(len(parts), limit // (avg + 3)))
    step = (len(parts) - 1) / (keep - 1)
    idx = sorted({round(i * step) for i in range(keep)} | {0, len(parts) - 1})
    out: list[str] = []
    for n, i in enumerate(idx):
        if n and i != idx[n - 1] + 1:
            out.append("…")          # says plainly that members were skipped
        out.append(parts[i])
    return " | ".join(out)[:limit]


# --------------------------------------------------------------- the readout


def _pick(readout: Readout, nodes: list[_Node], expand_mass: float, max_expand: int):
    """Which groups to open, and what every group's probability is."""
    probs = readout.scores
    ranked = sorted(range(len(nodes)), key=lambda i: -probs.get(str(i), 0.0))
    opened, mass = [], 0.0
    for i in ranked:
        if opened and (mass >= expand_mass or len(opened) >= max_expand):
            break
        opened.append(i)
        mass += probs.get(str(i), 0.0)
    return opened, [probs.get(str(i), 0.0) for i in range(len(nodes))]


def score_wide(
    model,
    tokenizer,
    decision: Decision,
    *,
    cap: int = MAX_OPTIONS,
    expand_mass: float = EXPAND_MASS,
    max_expand: int = MAX_EXPAND,
    summary_limit: int = SUMMARY_LIMIT,
    scorer: PrefixScorer | None = None,
) -> WideReadout:
    """Score a decision of any width, narrowing only when it does not fit.

    A decision within `cap` is passed straight to the flat readout and comes back
    byte-identical - no tree, no extra pass, no change in behaviour. Narrowing is
    what happens when there is no alternative.

    Args:
        cap: options one readout may carry. Defaults to the letter protocol's 16.
        expand_mass: open groups until this much mass is covered.
        max_expand: never open more than this many groups per level.
        summary_limit: characters of each group summary. Raising it makes deep
            options visible at higher levels, at the cost of prompt length.
        scorer: a `PrefixScorer` to reuse. One is built and primed if absent;
            pass your own when the caller already primed this same evidence.

    Raises:
        DecisionError: the decision is malformed, or `cap` is below 2.
    """
    started = time.perf_counter()
    if cap < 2:
        raise DecisionError(f"{decision.id}: cap must be at least 2, got {cap}")
    decision.validate(max_options=None)      # width is this module's problem, not an error

    if len(decision.options) <= cap:
        flat = score(model, tokenizer, decision)
        return WideReadout(
            decision_id=flat.decision_id, option_ids=flat.option_ids,
            probabilities=flat.probabilities, passes=1, depth=1, flat=True,
            input_tokens=flat.input_tokens, total_seconds=flat.total_seconds,
            meta={"prompt_sha256": flat.prompt_sha256})

    if scorer is None:
        scorer = PrefixScorer(model, tokenizer)
    scorer.prime(decision.evidence)

    # Each level: open the likeliest groups, carrying their mass down. A group
    # nobody opens keeps its mass, shared evenly among its members.
    frontier: list[tuple[_Node, float, str]] = [
        (_Node(children=build_tree(decision.options, cap)), 1.0, "")]
    settled: dict[str, float] = {}
    estimated: set[str] = set()
    passes = depth = tokens = 0

    while frontier:
        depth += 1
        level_next: list[tuple[_Node, float, str]] = []
        asks: list[tuple[_Node, float, str, Decision]] = []

        for node, prob, path in frontier:
            kids = node.children
            if len(kids) == 1:                       # nothing to choose between
                level_next.append((kids[0], prob, path + "0"))
                continue
            leaf_level = all(k.leaf is not None for k in kids)
            asks.append((node, prob, path, Decision(
                id=f"{decision.id}#{path or 'r'}",
                evidence=decision.evidence,
                criterion=decision.criterion if leaf_level else (
                    f"{decision.criterion}\n\n"
                    "Each option below is a GROUP of candidates, summarised. "
                    "Choose the group that contains the answer."),
                options=tuple(Option(str(i), summarise(k, summary_limit) if not leaf_level
                                     else k.leaf.description)
                              for i, k in enumerate(kids)))))

        if asks:
            readouts = (scorer.score_batch([a[3] for a in asks], max_batch=2)
                        if len(asks) > 1 else [scorer.score(asks[0][3])])
            passes += len(asks)
            for (node, prob, path, _), readout in zip(asks, readouts):
                tokens += readout.input_tokens
                opened, group_probs = _pick(readout, node.children, expand_mass, max_expand)
                for i, kid in enumerate(node.children):
                    share = prob * group_probs[i]
                    if i in opened:
                        if kid.leaf is not None:
                            settled[kid.leaf.id] = settled.get(kid.leaf.id, 0.0) + share
                        else:
                            level_next.append((kid, share, f"{path}{i}."))
                    else:
                        # Never looked inside. Spread the mass and say so.
                        members = kid.options
                        for o in members:
                            settled[o.id] = settled.get(o.id, 0.0) + share / len(members)
                            estimated.add(o.id)

        frontier = [(n, p, q) for n, p, q in level_next if n.leaf is None]
        for node, prob, _ in level_next:
            if node.leaf is not None:
                settled[node.leaf.id] = settled.get(node.leaf.id, 0.0) + prob

    total = sum(settled.values()) or 1.0
    ids = tuple(o.id for o in decision.options)      # presentation order, always
    probs = tuple(settled.get(i, 0.0) / total for i in ids)
    return WideReadout(
        decision_id=decision.id, option_ids=ids, probabilities=probs,
        passes=passes, depth=depth, flat=False, estimated=frozenset(estimated),
        input_tokens=tokens, total_seconds=time.perf_counter() - started,
        meta={"cap": cap, "expand_mass": expand_mass, "max_expand": max_expand,
              "summary_limit": summary_limit,
              "n_options": len(ids), "examined": len(ids) - len(estimated)})

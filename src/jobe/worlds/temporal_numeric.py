# -*- coding: utf-8 -*-
"""Exact-law worlds for the temporal_numeric family.

Every record is a decision whose answer is computed by code from rules stated
in full in the evidence: day counts, month-end roll, leap years, time zones
and clock changes, inclusive/exclusive boundaries, unit conversions, rounding,
and thresholds with strict or non-strict comparison. The wrong options are not
random: each is the answer a NAMED mistake produces (wall-clock time across a
clock change, an inclusive end date, a 365-day basis, tax counted in net cost,
...), recorded per option in `mistakes`, so the training signal is the law and
not the surface.

Nothing here is JevBench text. The family is the same; the scenarios, names,
rule wording and numbers are ours, and `--check-contamination <jevbench dir>`
verifies that no 10-word run is shared with the public task files.

  python -m jobe.worlds.temporal_numeric --n 2000 --seed 0 --out temporal_numeric.jsonl
  python -m jobe.worlds.temporal_numeric --n 200 --check-contamination <jevbench clone>
"""
from __future__ import annotations

import argparse
import calendar
import datetime as dt
import hashlib
import json
import random
import re
import sys
from dataclasses import dataclass, field
from decimal import ROUND_CEILING, ROUND_HALF_UP, Decimal

FAMILY = "temporal_numeric"
CENT = Decimal("0.01")
KG_PER_LB = Decimal("0.45359237")
TIB = 1024 ** 4
GB = 10 ** 9


def money(x: Decimal) -> Decimal:
    return Decimal(x).quantize(CENT, rounding=ROUND_HALF_UP)


def money_id(cur: str, x: Decimal) -> str:
    return f"{cur.lower()}_{str(money(x)).replace('.', '_').replace('-', 'neg')}"


def fmt(cur: str, x: Decimal) -> str:
    return f"{cur} {money(x):,.2f}"


def longdate(d: dt.date) -> str:
    return f"{d.day} {calendar.month_name[d.month]} {d.year}"


def dayname(d: dt.date) -> str:
    return calendar.day_name[d.weekday()]


def wall(dtm: dt.datetime) -> str:
    return f"{dtm.strftime('%H:%M')} on {dayname(dtm.date())} {longdate(dtm.date())}"


def add_months(d: dt.date, n: int, *, month_end_rule: bool = True) -> dt.date:
    """d + n months. With the month-end rule a day that does not exist in the
    target month becomes that month's last day; without it the surplus days
    spill into the following month (the naive mistake)."""
    y, m0 = divmod(d.month - 1 + n, 12)
    y, m = d.year + y, m0 + 1
    last = calendar.monthrange(y, m)[1]
    if d.day <= last:
        return dt.date(y, m, d.day)
    return dt.date(y, m, last) if month_end_rule else dt.date(y, m, last) + dt.timedelta(days=d.day - last)


# ---------------------------------------------------------------------------
# Time zones with explicit clock-change laws (stdlib only; nothing is looked up)
# ---------------------------------------------------------------------------
def _nth_sunday(year: int, month: int, n: int) -> dt.date:
    first = dt.date(year, month, 1)
    return first + dt.timedelta(days=(6 - first.weekday()) % 7 + 7 * (n - 1))


def _last_sunday(year: int, month: int) -> dt.date:
    last = dt.date(year, month, calendar.monthrange(year, month)[1])
    return last - dt.timedelta(days=(last.weekday() - 6) % 7)


@dataclass(frozen=True)
class Zone:
    name: str
    std: int                 # standard UTC offset, whole hours
    rule: str | None = None  # "us" | "eu" | None (no clock change)

    def transitions(self, year: int) -> tuple[dt.datetime, dt.datetime] | None:
        """(wall time at which clocks go forward, wall time at which they go back)."""
        if self.rule == "us":
            return (dt.datetime.combine(_nth_sunday(year, 3, 2), dt.time(2)),
                    dt.datetime.combine(_nth_sunday(year, 11, 1), dt.time(2)))
        if self.rule == "eu":
            return (dt.datetime.combine(_last_sunday(year, 3), dt.time(1 + self.std)),
                    dt.datetime.combine(_last_sunday(year, 10), dt.time(2 + self.std)))
        return None

    def is_dst(self, w: dt.datetime) -> bool:
        t = self.transitions(w.year)
        return bool(t) and t[0] <= w < t[1]

    def offset(self, w: dt.datetime) -> dt.timedelta:
        return dt.timedelta(hours=self.std + (1 if self.is_dst(w) else 0))

    def to_utc(self, w: dt.datetime) -> dt.datetime:
        return w - self.offset(w)

    def change_instants(self, year: int) -> tuple[dt.datetime, dt.datetime] | None:
        """The two clock changes as UTC instants (clocks go forward from standard
        time; they go back from daylight time)."""
        t = self.transitions(year)
        if not t:
            return None
        fwd, back = t
        return fwd - dt.timedelta(hours=self.std), back - dt.timedelta(hours=self.std + 1)

    def from_utc(self, u: dt.datetime) -> dt.datetime:
        ci = self.change_instants(u.year)
        dst = bool(ci) and ci[0] <= u < ci[1]
        return u + dt.timedelta(hours=self.std + (1 if dst else 0))

    def near_transition(self, w: dt.datetime, hours: int = 3) -> bool:
        """Within `hours` of a clock change, measured as an instant — a wall reading
        alone cannot tell the repeated hour from the hour before it."""
        ci = self.change_instants(w.year)
        u = self.to_utc(w)
        return bool(ci) and any(abs((u - c).total_seconds()) < hours * 3600 for c in ci)

    def change_sentence(self, first: dt.date, last: dt.date) -> str:
        """The clock-change law in words, for the evidence. Only the change that
        falls inside [first, last] is stated; otherwise the absence is stated."""
        t = self.transitions(first.year)
        if t:
            fwd, back = t
            if first <= fwd.date() <= last:
                return (f"Clocks in {self.name} go forward on {dayname(fwd.date())} {longdate(fwd.date())}: "
                        f"at {fwd.strftime('%H:%M')} they jump to {(fwd + dt.timedelta(hours=1)).strftime('%H:%M')}.")
            if first <= back.date() <= last:
                return (f"Clocks in {self.name} go back on {dayname(back.date())} {longdate(back.date())}: "
                        f"at {back.strftime('%H:%M')} they return to {(back - dt.timedelta(hours=1)).strftime('%H:%M')}.")
        return f"No clock change occurs in {self.name} during this period."

    def utc_label(self, w: dt.datetime) -> str:
        h = int(self.offset(w).total_seconds() // 3600)
        return f"UTC{h:+d}"


ZONES = {
    "New York": Zone("New York", -5, "us"), "Chicago": Zone("Chicago", -6, "us"),
    "Denver": Zone("Denver", -7, "us"), "Los Angeles": Zone("Los Angeles", -8, "us"),
    "London": Zone("London", 0, "eu"), "Amsterdam": Zone("Amsterdam", 1, "eu"),
    "Berlin": Zone("Berlin", 1, "eu"), "Madrid": Zone("Madrid", 1, "eu"), "Helsinki": Zone("Helsinki", 2, "eu"),
    "Tokyo": Zone("Tokyo", 9), "Singapore": Zone("Singapore", 8), "Dubai": Zone("Dubai", 4),
}
US_ZONES = [z for z in ZONES.values() if z.rule == "us"]
EU_ZONES = [z for z in ZONES.values() if z.rule == "eu"]
FIXED_ZONES = [z for z in ZONES.values() if z.rule is None]

# ---------------------------------------------------------------------------
# Scenario skins — all fictional, all ours
# ---------------------------------------------------------------------------
FIRM_A = ["Harrow", "Quillon", "Bexley", "Marlow", "Tessaro", "Ridgeway", "Corvin", "Ashgrove", "Pellston", "Ravelle",
          "Norquist", "Ibex", "Calloway", "Stroud", "Lindqvist", "Okonkwo", "Meridian", "Tolbert", "Sablewood", "Kirkham"]
FIRM_B = ["Freight", "Analytics", "Cold Chain", "Marine Supply", "Robotics", "Foods", "Logistics", "Diagnostics",
          "Print Works", "Software", "Textiles", "Energy", "Optics", "Dental", "Interiors", "Timber", "Instruments",
          "Studios", "Aviation Services", "Laboratories"]
FIRST = ["Mara", "Tobias", "Ingrid", "Kwame", "Sofia", "Rahul", "Elena", "Marcus", "Yuki", "Dario", "Amara", "Piet",
         "Noor", "Callum", "Lucia", "Henrik", "Zara", "Owen", "Beatriz", "Tarek"]
LAST = ["Sandoval", "Whitcombe", "Nakamura", "Okafor", "Lindgren", "Petrov", "Duarte", "Haddad", "Fairweather",
        "Oyelaran", "Bergstrom", "Castellano", "Mwangi", "Ferreira", "Kowalski", "Abernathy", "Sato", "Quintero"]


def firm(rng: random.Random) -> str:
    return f"{rng.choice(FIRM_A)} {rng.choice(FIRM_B)}"


def person(rng: random.Random) -> str:
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def ref(rng: random.Random, prefix: str) -> str:
    return f"{prefix}-{rng.randint(10, 99)}-{rng.randint(1000, 9999)}"


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class World:
    id: str
    law: str
    kind: str                      # "choice" | "noul" | "score"
    evidence: object               # str, or dict for the structured style
    criterion: str
    options: tuple[tuple[str, str], ...]   # (id, description), gold included
    expected: str
    rationale: str
    mistakes: dict[str, str] = field(default_factory=dict)   # option id -> mistake that produces it
    params: dict = field(default_factory=dict)
    ordinal: bool = False

    @property
    def split(self) -> str:
        return "heldout" if int(hashlib.sha1(self.id.encode()).hexdigest(), 16) % 10 == 0 else "train"

    def record(self) -> dict:
        return {"id": self.id, "family": FAMILY, "law": self.law, "type": self.kind, "split": self.split,
                "evidence": self.evidence, "criterion": self.criterion,
                "options": [{"id": i, "description": d} for i, d in self.options],
                "expected": self.expected, "ordinal": self.ordinal, "mistakes": self.mistakes,
                "rationale": self.rationale, "params": self.params}


def _finish_money_options(cur: str, truth: Decimal, mistakes: dict[str, Decimal], describe, rng: random.Random,
                          *, want: int = 5) -> tuple[tuple[tuple[str, str], ...], str, dict[str, str]]:
    """Gold + every distinct mistake value, padded with plausible neighbours to `want`, shuffled."""
    vals: dict[str, tuple[Decimal, str]] = {money_id(cur, truth): (money(truth), "")}
    for name, v in mistakes.items():
        k = money_id(cur, v)
        if k not in vals:
            vals[k] = (money(v), name)
    pad = 0
    while len(vals) < want:
        pad += 1
        v = money(truth * (Decimal(1) + Decimal(rng.choice([-7, -4, -2, 2, 4, 7]) + pad) / Decimal(100)))
        k = money_id(cur, v)
        if k not in vals:
            vals[k] = (v, f"pad_{pad}")
    ids = list(vals)
    rng.shuffle(ids)
    options = tuple((i, describe(vals[i][0])) for i in ids)
    return options, money_id(cur, truth), {i: vals[i][1] for i in ids if vals[i][1] and not vals[i][1].startswith("pad_")}


def _yes_no(truth: bool, yes: str, no: str) -> tuple[tuple[tuple[str, str], ...], str]:
    return (("yes", yes), ("no", no)), ("yes" if truth else "no")


# ---------------------------------------------------------------------------
# Laws
# ---------------------------------------------------------------------------
def law_interest_period(rng: random.Random, idx: int) -> World:
    """Interest = principal x rate x days / basis; days from-and-including start
    to-but-excluding end; one rounding at the end."""
    principal = Decimal(rng.randint(50, 4000) * 1000)
    rate = Decimal(rng.randint(26, 78)) / Decimal(8) / Decimal(100)      # 3.25 % .. 9.75 % in 1/8 steps
    start = dt.date(rng.randint(2026, 2028), rng.randint(1, 12), rng.randint(1, 28))
    end = start + dt.timedelta(days=rng.choice([28, 29, 30, 31, 89, 90, 91, 92, 181, 182, 183, 184]))
    basis = rng.choice(["ACT/360", "ACT/365", "30/360"])
    n = rng.randint(2, 14)

    def days_30_360(a: dt.date, b: dt.date) -> int:
        d1, d2 = min(a.day, 30), b.day
        if d2 == 31 and d1 == 30:
            d2 = 30
        return 360 * (b.year - a.year) + 30 * (b.month - a.month) + (d2 - d1)

    def interest(days: int, denom: int) -> Decimal:
        return money(principal * rate * Decimal(days) / Decimal(denom))

    act = (end - start).days
    thirty = days_30_360(start, end)
    days, denom = (thirty, 360) if basis == "30/360" else (act, 360 if basis == "ACT/360" else 365)
    truth = interest(days, denom)
    daily_rounded = money(principal * rate / Decimal(denom)) * days       # rounding each day, then summing
    mistakes = {
        "inclusive_end_date": interest(days + 1, denom),
        "wrong_basis_365_vs_360": interest(days, 365 if denom == 360 else 360),
        "rounded_each_day": money(daily_rounded),
        "act_vs_30_360": interest(act if basis == "30/360" else thirty, 360),
    }
    basis_text = {
        "ACT/360": "the actual number of days elapsed, divided by 360",
        "ACT/365": "the actual number of days elapsed, divided by 365",
        "30/360": "a 30-day month and a 360-day year (each month counts as 30 days; a 31st is treated as the 30th)",
    }[basis]
    statement = rng.choice(list(mistakes.values()))
    evidence = (
        f"{firm(rng).upper()} — LENDING OPERATIONS: INTEREST PERIOD REVIEW\n"
        f"Loan {ref(rng, 'TL')}, borrower {firm(rng)}. Currency USD.\n"
        f"Loan agreement, clause 6 (extract)\n"
        f"6.1 Interest is charged for every day on which principal is outstanding at the start of that day. "
        f"The first day of an Interest Period is charged; the last day of an Interest Period is not.\n"
        f"6.2 Day count convention: {basis_text}.\n"
        f"6.3 The rate for the period is {rate * 100:.3f} % per annum. Interest for the whole period is computed in one "
        f"step and rounded to the nearest cent, with a half cent rounded upwards. Daily amounts are never rounded.\n"
        f"Interest Period {n}: {longdate(start)} to {longdate(end)}. Principal outstanding on every day: {fmt('USD', principal)}.\n"
        f"The legacy statement run printed 'Interest Period {n}: {fmt('USD', statement)}'. The review checks that figure "
        f"against clause 6; the treasury spreadsheet, which uses a different convention, is not authoritative."
    )
    options, expected, mm = _finish_money_options(
        "USD", truth, mistakes, lambda v: f"Interest for Interest Period {n} of {fmt('USD', v)}.", rng)
    return World(
        id=f"tn-interest-{idx:05d}", law="interest_period", kind="choice", evidence=evidence,
        criterion=f"What is the interest payable for Interest Period {n} under clause 6?",
        options=options, expected=expected, mistakes=mm,
        rationale=f"{days} chargeable days ({basis}); {principal} x {rate} x {days}/{denom} = {truth}.",
        params={"principal": str(principal), "rate": str(rate), "start": start.isoformat(), "end": end.isoformat(),
                "basis": basis, "days": days})


def law_term_end(rng: random.Random, idx: int) -> World:
    """Start + N months with the month-end rule, ending at 24:00 in the seller's
    zone; the claim is timestamped in the customer's zone."""
    months = rng.choice([12, 18, 24, 30, 36])
    y, m = rng.randint(2025, 2027), rng.randint(1, 12)
    start = dt.date(y, m, min(rng.choice([28, 29, 30, 31, 15, 1]), calendar.monthrange(y, m)[1]))
    end_day = add_months(start, months)                     # last covered day
    naive_end = add_months(start, months, month_end_rule=False)
    seller, customer = rng.choice(EU_ZONES), rng.choice(US_ZONES + FIXED_ZONES)
    end_utc = seller.to_utc(dt.datetime.combine(end_day + dt.timedelta(days=1), dt.time(0)))
    # a submission within a day either side of the cutoff, expressed in the customer's zone
    sub_utc = end_utc + dt.timedelta(hours=rng.choice([-30, -20, -9, -5, -3, -1, 1, 3, 5, 9, 20, 30]),
                                     minutes=rng.choice([0, 15, 30, 45]))
    sub_local = customer.from_utc(sub_utc)
    if customer.near_transition(sub_local) or seller.near_transition(seller.from_utc(sub_utc)):
        return law_term_end(rng, idx)
    truth = sub_utc < end_utc
    wall_truth = sub_local < dt.datetime.combine(end_day + dt.timedelta(days=1), dt.time(0))
    surface = rng.choice(["naive", "wall", "none"])
    surface_line = {
        "naive": f"The desk's screen shows the cover as ending on {longdate(naive_end)}, a date produced by adding "
                 f"{months} months to the day number without applying section 2.",
        "wall": f"The desk compared the claim time against midnight on {longdate(end_day)} on the customer's own clock, "
                f"without converting it.",
        "none": "The desk has not yet recorded a view.",
    }[surface]
    evidence = (
        f"{firm(rng).upper()} — SERVICE PLAN CERTIFICATE {ref(rng, 'SP')} AND CLAIM NOTE\n"
        f"Customer: {person(rng)}, {customer.name}. Plan issued by our {seller.name} office.\n"
        f"Section 1. Cover begins on the activation date, {longdate(start)}, and runs for {months} months.\n"
        f"Section 2. The last covered day is the day in the final month that carries the same day number as the "
        f"activation date. If that month has no such day, the last covered day is the final day of that month.\n"
        f"Section 3. Cover ends at 24:00 {seller.name} time on the last covered day. A claim counts as made at the "
        f"moment it is received; times are compared as moments in time, not as clock readings in different places.\n"
        f"{seller.change_sentence(end_day - dt.timedelta(days=3), end_day + dt.timedelta(days=3))} "
        f"{customer.change_sentence(sub_local.date() - dt.timedelta(days=3), sub_local.date() + dt.timedelta(days=3))}\n"
        f"At the moment of the claim {seller.name} is {seller.utc_label(seller.from_utc(sub_utc))} and "
        f"{customer.name} is {customer.utc_label(sub_local)}.\n"
        f"Claim {ref(rng, 'CL')} received at {wall(sub_local)}, {customer.name} time.\n"
        f"{surface_line}"
    )
    options, expected = _yes_no(truth, "The claim was received before cover ended.", "The claim was received after cover ended.")
    mistakes = {}
    if wall_truth != truth:
        mistakes["yes" if wall_truth else "no"] = "compared_wall_clocks_without_converting"
    if naive_end != end_day and (sub_utc < seller.to_utc(dt.datetime.combine(naive_end + dt.timedelta(days=1), dt.time(0)))) != truth:
        mistakes["no" if truth else "yes"] = "no_month_end_rule"
    return World(
        id=f"tn-termend-{idx:05d}", law="term_end", kind="noul", evidence=evidence,
        criterion="Was the claim received within the cover period of the certificate?",
        options=options, expected=expected, mistakes=mistakes,
        rationale=f"Last covered day {end_day}; cutoff {end_utc} UTC; claim {sub_utc} UTC -> {'within' if truth else 'after'}.",
        params={"start": start.isoformat(), "months": months, "end_day": end_day.isoformat(), "seller": seller.name,
                "customer": customer.name, "submitted_local": sub_local.isoformat()})


def law_threshold_day(rng: random.Random, idx: int) -> World:
    """First day on which month-to-date NET cost (charges minus credits, before
    tax) is at or above a percentage of the budget."""
    budget = Decimal(rng.randint(5, 60) * 1000)
    pct = rng.choice([70, 75, 80, 85, 90])
    year, month = rng.randint(2026, 2028), rng.randint(1, 12)
    ndays = calendar.monthrange(year, month)[1]
    tax_rate = Decimal(rng.choice([8, 10, 15, 19, 20, 21])) / 100
    target_day = rng.randint(12, ndays - 3)
    base = budget * Decimal(pct) / 100 / Decimal(target_day)
    rows = []
    for d in range(1, ndays + 1):
        charge = money(base * Decimal(rng.randint(80, 125)) / 100)
        credit = money(charge * Decimal(rng.randint(10, 40)) / 100) if rng.random() < 0.3 else Decimal(0)
        rows.append((d, charge, credit, money(charge * tax_rate)))
    threshold = budget * Decimal(pct) / 100

    def first_day(select) -> int | None:
        total = Decimal(0)
        for d, charge, credit, tax in rows:
            total += select(charge, credit, tax)
            if total >= threshold:
                return d
        return None

    truth = first_day(lambda c, cr, t: c - cr)
    mistakes = {
        "ignored_credits": first_day(lambda c, cr, t: c),
        "counted_tax_in_net": first_day(lambda c, cr, t: c - cr + t),
        "counted_tax_and_ignored_credits": first_day(lambda c, cr, t: c + t),
    }
    if truth is None:
        return law_threshold_day(rng, idx)
    # the whole month is shown: a table that stopped near the answer would leak it
    shown = ndays
    mon = calendar.month_abbr[month].lower()
    table = "\n".join(f"  {d:2d} {calendar.month_abbr[month]} | {fmt('USD', c):>14s} | {fmt('USD', cr):>12s} | {fmt('USD', t):>12s}"
                      for d, c, cr, t in rows[:shown])
    evidence = (
        f"{firm(rng).upper()} — CLOUD SPEND ALERTS, ACCOUNT {ref(rng, 'ACC')}, {calendar.month_name[month].upper()} {year}, USD\n"
        f"Budget: {fmt('USD', budget)} for the month.\n"
        f"Alert rule R1 triggers a single time: when a day closes with the cumulative net cost for the month at "
        f"{pct} % of the budget or more ({fmt('USD', threshold)}), the trigger is attributed to that day.\n"
        f"Definition: net cost = usage charges minus credits applied. Taxes are listed for information and are not "
        f"part of net cost.\n"
        f"Daily export (each line is that day alone, not a running total):\n"
        f"  day    | usage charges  | credits      | taxes\n{table}\n"
        f"Amounts are exact; nothing is omitted."
    )
    lo = max(1, min(truth - rng.randint(0, 4), ndays - 4))   # the answer is not always the middle option
    days = list(range(lo, min(ndays, lo + 4) + 1))
    for v in mistakes.values():
        if v and v not in days and len(days) < 7:
            days.append(v)
    days.sort()
    options = [(f"{mon}_{d:02d}", f"At the end of {d} {calendar.month_name[month]}.") for d in days] + [
        ("never", f"It does not fire in {calendar.month_name[month]}.")]
    rng.shuffle(options)
    mm = {}
    for name, v in mistakes.items():
        oid = f"{mon}_{v:02d}" if v else "never"
        if oid != f"{mon}_{truth:02d}" and oid in dict(options) and oid not in mm:
            mm[oid] = name
    return World(
        id=f"tn-threshold-{idx:05d}", law="threshold_day", kind="choice", evidence=evidence,
        criterion="On which day does alert rule R1 fire?",
        options=tuple(options), expected=f"{mon}_{truth:02d}", mistakes=mm,
        rationale=f"Net (charges - credits, no tax) first reaches {threshold} at end of day {truth}.",
        params={"budget": str(budget), "pct": pct, "month": f"{year}-{month:02d}", "truth_day": truth})


def law_first_declined(rng: random.Random, idx: int) -> World:
    """A cycle limit with a foreign-transaction fee; declined purchases do not
    count; a purchase dated outside the cycle does not count."""
    limit = Decimal(rng.choice([2500, 4000, 5000, 6000, 8000, 10000]))
    fee_pct = Decimal(rng.choice([2, 3])) / 100
    year, month = rng.randint(2026, 2028), rng.randint(1, 12)
    ndays = calendar.monthrange(year, month)[1]
    merchants = [("office chair", 620), ("conference fee", 1450), ("rail tickets", 210), ("client dinner", 310),
                 ("laptop dock", 385), ("hotel deposit", 900), ("book set", 240), ("taxi", 48), ("parking", 18),
                 ("software licence", 1200), ("catering", 760), ("courier", 95)]
    picks = rng.sample(merchants, rng.randint(6, 8))
    total_list = sum(Decimal(a) for _, a in picks)
    scale = limit / total_list * Decimal(rng.randint(105, 140)) / 100          # ensure the limit is hit
    day = 0
    purchases = []
    for name, amt in picks:
        day = min(ndays, day + rng.randint(1, 4))
        foreign = rng.random() < 0.35
        purchases.append((dt.date(year, month, day), name, money(Decimal(amt) * scale), foreign))
    # one purchase dated the day BEFORE the cycle: outside it, and first in date order,
    # so counting it shifts every later decision
    stray_i = rng.randrange(len(purchases))
    prev = dt.date(year, month, 1) - dt.timedelta(days=1)
    purchases[stray_i] = (prev, purchases[stray_i][1], purchases[stray_i][2], purchases[stray_i][3])
    purchases.sort(key=lambda p: p[0])

    def first_declined(*, with_fee: bool, in_cycle_only: bool) -> int | None:
        total = Decimal(0)
        for i, (d, name, amt, foreign) in enumerate(purchases):
            if in_cycle_only and d.month != month:
                continue
            charge = money(amt * (1 + fee_pct)) if (foreign and with_fee) else amt
            if total + charge <= limit:
                total += charge
            else:
                return i
        return None

    truth = first_declined(with_fee=True, in_cycle_only=True)
    mistakes = {"ignored_foreign_fee": first_declined(with_fee=False, in_cycle_only=True),
                "counted_out_of_cycle_purchase": first_declined(with_fee=True, in_cycle_only=False)}
    lines = "\n".join(f"  {longdate(d)} — {name}{' (foreign merchant)' if foreign else ''} — {fmt('USD', amt)}"
                      for d, name, amt, foreign in purchases)
    holder = person(rng)
    evidence = (
        f"{firm(rng).upper()} — CORPORATE CARD DESK, CARDHOLDER {holder.upper()} (CARD ENDING {rng.randint(1000, 9999)})\n"
        f"Programme terms, schedule C\n"
        f"C1 Spending in one statement cycle may not go beyond {fmt('USD', limit)}. The {calendar.month_name[month]} "
        f"cycle covers {longdate(dt.date(year, month, 1))} to {longdate(dt.date(year, month, ndays))} inclusive.\n"
        f"C2 Each purchase is tested in turn: take the purchases already approved in this cycle, add this purchase "
        f"together with any C3 fee, and approve it when that sum is within the C1 limit. Reaching the limit exactly "
        f"is within it.\n"
        f"C3 A purchase at a foreign merchant charges the amount plus a {fee_pct * 100:.0f} % foreign transaction fee, "
        f"rounded to the cent.\n"
        f"C4 A declined purchase charges nothing and does not count towards the cycle total. Purchases are assessed "
        f"in date order.\n"
        f"Purchase record (amounts as presented by the merchant, before any fee):\n{lines}\n"
        f"{holder} asks which purchase was the first to be declined."
    )
    def oid(i: int) -> str:
        d, name, _, _ = purchases[i]
        return f"{calendar.month_abbr[d.month].lower()}_{d.day:02d}_{name.replace(' ', '_')}"
    options = [(oid(i), f"The {longdate(p[0])} {p[1]} ({fmt('USD', p[2])}) was the first purchase declined.")
               for i, p in enumerate(purchases)] + [("none_declined", "No listed purchase was declined.")]
    rng.shuffle(options)
    expected = oid(truth) if truth is not None else "none_declined"
    mm = {}
    for name, v in mistakes.items():
        o = oid(v) if v is not None else "none_declined"
        if o != expected and o not in mm:
            mm[o] = name
    return World(
        id=f"tn-declined-{idx:05d}", law="first_declined", kind="choice", evidence=evidence,
        criterion="Under schedule C, which listed purchase was the first one declined?",
        options=tuple(options), expected=expected, mistakes=mm,
        rationale=f"Running approved total with fees, in-cycle only; first decline index {truth}.",
        params={"limit": str(limit), "fee_pct": str(fee_pct), "month": f"{year}-{month:02d}"})


def law_rest_violations(rng: random.Random, idx: int) -> World:
    """Count rest periods under 11 elapsed hours in a roster that spans a clock change."""
    zone = rng.choice(US_ZONES + EU_ZONES)
    year = rng.randint(2026, 2028)
    fwd, back = zone.transitions(year)
    change = rng.choice([fwd, back])
    week_start = change.date() - dt.timedelta(days=rng.randint(2, 4))
    n_shifts = rng.randint(5, 7)
    change_utc = zone.change_instants(year)[0 if change == fwd else 1]
    shifts: list[tuple[dt.datetime, dt.datetime]] = []
    t = dt.datetime.combine(week_start, dt.time(rng.choice([6, 7, 8, 14, 19])))
    for i in range(n_shifts):
        length = rng.choice([8, 9, 10, 12])
        end_utc = zone.to_utc(t) + dt.timedelta(hours=length)
        end = zone.from_utc(end_utc)
        shifts.append((t, end))
        if i == n_shifts - 1:
            break
        rest = rng.choice([10, 10.5, 11, 11, 11.5, 12, 13, 14])
        gap = (change_utc - end_utc).total_seconds() / 3600
        if 0 < gap <= 9:
            # a rest that spans the clock change, chosen so the wall clock misreads it: clocks set
            # BACK make the rest look an hour shorter than it was, clocks set FORWARD an hour longer
            rest = rng.choice([11, 11.5]) if change == back else rng.choice([10, 10.5])
        nxt = zone.from_utc(end_utc + dt.timedelta(hours=rest))
        # keep every roster time out of the repeated / missing hour itself
        if zone.near_transition(nxt, 1) or zone.near_transition(end, 1):
            nxt = zone.from_utc(end_utc + dt.timedelta(hours=rest + 3))
        t = nxt
    if not any(s[1] <= change <= shifts[i + 1][0] for i, s in enumerate(shifts[:-1])):
        return law_rest_violations(rng, idx)   # the change must fall inside a rest period

    def count(*, elapsed: bool, strict: bool) -> int:
        v = 0
        for (s1, e1), (s2, e2) in zip(shifts, shifts[1:]):
            hours = ((zone.to_utc(s2) - zone.to_utc(e1)) if elapsed else (s2 - e1)).total_seconds() / 3600
            if hours < 11 or (strict and hours == 11):
                v += 1
        return v

    truth = min(3, count(elapsed=True, strict=False))
    mistakes = {"wall_clock_across_change": min(3, count(elapsed=False, strict=False)),
                "exactly_11h_counted_as_violation": min(3, count(elapsed=True, strict=True))}
    if mistakes["wall_clock_across_change"] == truth and rng.random() < 0.7:
        return law_rest_violations(rng, idx)    # most rosters should be ones where the wall clock misleads
    roster = "\n".join(f"  shift {i + 1}: {wall(s)} to {wall(e)}" for i, (s, e) in enumerate(shifts))
    who = person(rng)
    evidence = (
        f"{firm(rng).upper()} — CREW WORKING-TIME RULE W3 AND ROSTER EXTRACT ({who.upper()})\n"
        f"W3 Rest between shifts: a crew member's next shift may begin no sooner than 11 hours after the previous "
        f"shift ended, counted as real elapsed hours. Precisely 11 hours satisfies W3. Every gap shorter than that "
        f"is a separate breach.\n"
        f"All times below are {zone.name} local clock times. {zone.change_sentence(week_start, week_start + dt.timedelta(days=8))}\n"
        f"Roster:\n{roster}"
    )
    labels = ["No breach of W3", "One breach of W3", "Two breaches of W3", "Three or more breaches of W3"]
    options = tuple((str(i), lab) for i, lab in enumerate(labels))
    mm = {str(v): name for name, v in mistakes.items() if v != truth}
    return World(
        id=f"tn-rest-{idx:05d}", law="rest_violations", kind="score", evidence=evidence, ordinal=True,
        criterion="How many breaches of W3 does the roster contain? Answer 0, 1, 2, or 3 for three or more.",
        options=options, expected=str(truth), mistakes=mm,
        rationale=f"Elapsed rests computed via UTC across the {zone.name} clock change; {truth} under 11 h.",
        params={"zone": zone.name, "change": change.isoformat(), "shifts": [(s.isoformat(), e.isoformat()) for s, e in shifts]})


def law_window_bands(rng: random.Random, idx: int) -> World:
    """A response window opens at 09:00 on the first business day after a notice
    and lasts N elapsed hours, with a clock change stated. Structured style."""
    zone = rng.choice(US_ZONES + EU_ZONES)
    year = rng.randint(2026, 2028)
    change = rng.choice(zone.transitions(year))
    received = dt.datetime.combine(change.date() - dt.timedelta(days=rng.randint(1, 4)),
                                   dt.time(rng.choice([8, 11, 15, 17, 18]), rng.choice([0, 10, 40])))
    hours = rng.choice([24, 36, 48, 60, 72, 96])
    d = received.date() + dt.timedelta(days=1)
    while d.weekday() >= 5:
        d += dt.timedelta(days=1)
    opens = dt.datetime.combine(d, dt.time(9))
    close_utc = zone.to_utc(opens) + dt.timedelta(hours=hours)
    close_wall_naive = opens + dt.timedelta(hours=hours)
    event_utc = close_utc + dt.timedelta(minutes=rng.choice([-2000, -600, -90, -30, 0, 20, 50, 90, 130, 400]))
    event = zone.from_utc(event_utc)
    if zone.near_transition(event, 2) or zone.near_transition(opens, 2):
        return law_window_bands(rng, idx)

    def band(ev_utc: dt.datetime, cl_utc: dt.datetime, op_utc: dt.datetime) -> str:
        if ev_utc < op_utc:
            return "before_open"
        if ev_utc <= cl_utc:
            return "within_window"
        return "late_by_under_2h" if (ev_utc - cl_utc) < dt.timedelta(hours=2) else "late_by_2h_or_more"

    open_utc = zone.to_utc(opens)
    truth = band(event_utc, close_utc, open_utc)
    weekend_pause = sum(1 for k in range(hours // 24 + 2)
                        if (opens + dt.timedelta(days=k)).weekday() >= 5 and opens + dt.timedelta(days=k) < close_wall_naive) * 24
    mistakes = {
        "wall_clock_across_change": band(event, close_wall_naive, opens),
        "weekend_paused_the_window": band(event_utc, close_utc + dt.timedelta(hours=weekend_pause), open_utc),
        "opened_on_the_notice_day": band(event_utc, zone.to_utc(dt.datetime.combine(received.date(), dt.time(9))) + dt.timedelta(hours=hours),
                                         zone.to_utc(dt.datetime.combine(received.date(), dt.time(9)))),
    }
    evidence = {
        "event": f"The reply was received at {wall(event)}, {zone.name} local time.",
        "rules": (f"Clock readings are {zone.name} local time. The reply period starts at 09:00 on the first weekday "
                  f"following the day the notice arrived, and it runs for {hours} elapsed hours from that start; "
                  f"weekends and clock changes never stop it running and matter only for choosing the start day. "
                  f"Saturday and Sunday are not weekdays; no holidays apply. The notice arrived at {wall(received)}, "
                  f"{zone.name} time. {zone.change_sentence(received.date(), event.date() + dt.timedelta(days=1))}"),
    }
    options = (("before_open", "The reply arrived before the window opened."),
               ("within_window", "The reply arrived between the opening and the exact closing moment, inclusive."),
               ("late_by_under_2h", "The reply arrived after closing, but less than two hours late."),
               ("late_by_2h_or_more", "The reply arrived two hours or more after closing."))
    mm = {}
    for name, b in mistakes.items():
        if b != truth and b not in mm:
            mm[b] = name
    return World(
        id=f"tn-window-{idx:05d}", law="window_bands", kind="choice", evidence=evidence,
        criterion="Apply every stated timing rule and select the band that describes when the reply arrived.",
        options=options, expected=truth, mistakes=mm,
        rationale=f"Opens {opens} local; closes {close_utc} UTC; event {event_utc} UTC -> {truth}.",
        params={"zone": zone.name, "received": received.isoformat(), "hours": hours, "event": event.isoformat()})


def law_unit_limit(rng: random.Random, idx: int) -> World:
    """Mixed lb/kg weights, an exact conversion factor, one rounding of the
    total (up, to the next 10 kg), then a non-strict limit."""
    limit = Decimal(rng.choice([1500, 2000, 2500, 3000, 4000]))
    # within reach of every named mistake: per-item rounding adds up to 20 kg, the 0.45
    # factor removes ~30 kg on a 4-tonne load, exact equality tests the non-strict rule
    target = limit + Decimal(rng.choice([-19, -12, -5, 0, 0, 3, 8, 15, 25]))
    tare = Decimal(rng.randint(80, 140))
    restraint_lb = Decimal(rng.randint(40, 120))
    net_lb = ((target - tare - restraint_lb * KG_PER_LB) / KG_PER_LB).quantize(Decimal("1"))
    items = [("net cargo", net_lb, "lb"), ("container tare", tare, "kg"), ("restraint equipment", restraint_lb, "lb")]

    def total(*, factor: Decimal, per_item: bool, mode: str) -> Decimal:
        def rnd(x: Decimal) -> Decimal:
            if mode == "up10":
                return (x / 10).quantize(Decimal("1"), rounding=ROUND_CEILING) * 10
            if mode == "nearest":
                return x.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
            return x
        kgs = [(v * factor if u == "lb" else v) for _, v, u in items]
        if per_item:
            return sum(rnd(k) for k in kgs)
        return rnd(sum(kgs))

    truth_kg = total(factor=KG_PER_LB, per_item=False, mode="up10")
    truth = truth_kg <= limit
    mistakes = {
        "rounded_each_item": total(factor=KG_PER_LB, per_item=True, mode="up10") <= limit,
        "rounded_to_nearest": total(factor=KG_PER_LB, per_item=False, mode="nearest") <= limit,
        "used_0_45_factor": total(factor=Decimal("0.45"), per_item=False, mode="up10") <= limit,
        "strict_comparison": truth_kg < limit,
    }
    evidence = (
        f"{firm(rng).upper()} — LOAD CONTROL, MAIN-DECK POSITION {rng.randint(11, 44)}{rng.choice('LPR')}, "
        f"FLIGHT {rng.choice('KQZ')}{rng.choice('XVN')} {rng.randint(100, 999)}\n"
        f"Load rule L4: the gross weight of a unit loaded at this position must not exceed {limit:,.0f} kg. "
        f"Gross weight is the net cargo weight plus the container tare plus the restraint equipment. Convert pounds "
        f"to kilograms at exactly 1 lb = {KG_PER_LB} kg, add the three weights, and only then round the total UP to "
        f"the next whole 10 kg. A rounded total equal to the limit complies.\n"
        f"Unit {ref(rng, 'ULD')}: net cargo {net_lb:,.0f} lb; container tare {tare} kg; restraint equipment {restraint_lb} lb.\n"
        f"The ramp agent's tablet, which rounds each line separately, shows a different figure."
    )
    options, expected = _yes_no(truth, f"Its gross weight, computed as L4 prescribes, is at or below {limit:,.0f} kg.",
                                f"Its gross weight, computed as L4 prescribes, exceeds {limit:,.0f} kg.")
    mm = {}
    for name, ok in mistakes.items():
        if ok != truth:
            mm.setdefault("yes" if ok else "no", name)
    return World(
        id=f"tn-unit-{idx:05d}", law="unit_limit", kind="noul", evidence=evidence,
        criterion="Does the unit comply with load rule L4 at this position?",
        options=options, expected=expected, mistakes=mm,
        rationale=f"Exact total rounded up to 10 kg = {truth_kg} kg vs limit {limit} -> {'complies' if truth else 'exceeds'}.",
        params={"limit": str(limit), "net_lb": str(net_lb), "tare_kg": str(tare), "restraint_lb": str(restraint_lb),
                "total_kg": str(truth_kg)})


def law_usage_allowance(rng: random.Random, idx: int) -> World:
    """Binary allowance, decimal usage, a correction, strict exceedance. Structured style."""
    tib = Decimal(rng.choice(["1.50", "2.00", "2.40", "3.25", "4.00", "5.00"]))
    allowance = int(tib * TIB)
    # aim the final decimal-GB total near the allowance so binary vs decimal decides the band
    aim = Decimal(allowance) / GB + Decimal(rng.choice([-180, -60, -8, 0, 5, 40, 150]))
    parts = [money(aim * Decimal(rng.randint(15, 30)) / 100) for _ in range(3)]   # 45-90 %, so the 4th part stays positive
    correction = money(aim * Decimal(rng.randint(1, 3)) / 100)
    parts.append(money(aim - sum(parts) + correction))       # so that sum(parts) - correction == aim
    used_bytes = int(sum(parts) * GB) - int(correction * GB)

    def band(used: int, allow: int, *, strict: bool = True) -> str:
        if used > allow or (not strict and used == allow):
            return "over_limit"
        if used == allow:
            return "exactly_equal"
        return "under_100gb_below" if allow - used < 100 * GB else "at_least_100gb_below"

    truth = band(used_bytes, allowance)
    mistakes = {
        "tib_read_as_decimal_tb": band(used_bytes, int(tib * 10 ** 12)),
        "ignored_the_correction": band(used_bytes + int(correction * GB), allowance),
        "equality_counted_as_exceedance": band(used_bytes, allowance, strict=False),
    }
    seq = " then ".join(f"used {p:,.2f} GB" for p in parts[:2]) + f", then received a correction removing {correction:,.2f} GB, then used {parts[2]:,.2f} GB, then used {parts[3]:,.2f} GB"
    evidence = {
        "event": "Classify the final cumulative use against the allowance.",
        "rules": (f"The plan's transfer allowance is {tib} TiB per cycle; one TiB means {TIB:,} bytes (binary). Meter "
                  f"readings are in decimal GB, one GB meaning 10^9 bytes. In this cycle the account {seq}. Going over "
                  f"the allowance requires the cumulative byte count to be larger than the allowance; a count equal to "
                  f"it is not over."),
    }
    options = (("over_limit", "The cumulative count is above the allowance."),
               ("exactly_equal", "The cumulative count equals the allowance."),
               ("under_100gb_below", "The cumulative count is under the allowance, with a shortfall smaller than 100 decimal GB."),
               ("at_least_100gb_below", "The cumulative count is under the allowance by 100 decimal GB or more."))
    mm = {}
    for name, b in mistakes.items():
        if b != truth and b not in mm:
            mm[b] = name
    return World(
        id=f"tn-usage-{idx:05d}", law="usage_allowance", kind="choice", evidence=evidence,
        criterion="Apply every stated unit, correction and threshold rule and select the exact band.",
        options=options, expected=truth, mistakes=mm,
        rationale=f"used {used_bytes:,} B vs allowance {allowance:,} B -> {truth}.",
        params={"tib": str(tib), "parts_gb": [str(p) for p in parts], "correction_gb": str(correction)})


def law_fx_reimbursement(rng: random.Random, idx: int) -> World:
    """Hotel nights converted at the reference rate for the checkout date — or the
    previous business day's rate when that date has none — capped per night,
    excluding non-lodging lines."""
    nights = rng.randint(2, 4)
    checkout = dt.date(rng.randint(2026, 2028), rng.randint(1, 12), rng.randint(3, 27))
    while checkout.weekday() not in (0, 5, 6) and rng.random() < 0.6:     # bias to weekend checkouts
        checkout += dt.timedelta(days=1)
    checkin = checkout - dt.timedelta(days=nights)
    rate_per_night = Decimal(rng.randint(180, 420) * 100)                    # JPY
    breakfast = Decimal(rng.randint(2, 4) * 1000)
    cap = Decimal(rng.randint(120, 220))                                       # EUR per night
    rates: dict[dt.date, Decimal] = {}
    d = checkin - dt.timedelta(days=4)
    while d <= checkout + dt.timedelta(days=4):                                # always spans a weekend either side
        if d.weekday() < 5:
            rates[d] = Decimal(rng.randint(15800, 17400)) / 100                # JPY per EUR
        d += dt.timedelta(days=1)

    def rate_for(day: dt.date, *, direction: int) -> tuple[dt.date, Decimal]:
        for _ in range(7):
            if day in rates:
                return day, rates[day]
            day += dt.timedelta(days=direction)
        raise AssertionError("rate table does not span the stay")

    used_day, rate = rate_for(checkout, direction=-1)
    room_jpy = rate_per_night * nights

    def amount(r: Decimal, *, include_breakfast: bool, apply_cap: bool, per_night_rounding: bool) -> Decimal:
        jpy = room_jpy + (breakfast * nights if include_breakfast else 0)
        if per_night_rounding:
            eur = money(rate_per_night / r) * nights + (money(breakfast / r) * nights if include_breakfast else 0)
        else:
            eur = jpy / r
        if apply_cap:
            eur = min(eur, cap * nights)
        return money(eur)

    truth = amount(rate, include_breakfast=False, apply_cap=True, per_night_rounding=False)
    next_day, next_rate = rate_for(checkout + dt.timedelta(days=1), direction=+1)
    in_day, in_rate = rate_for(checkin, direction=-1)
    mistakes = {
        "used_next_business_day_rate": amount(next_rate, include_breakfast=False, apply_cap=True, per_night_rounding=False),
        "used_checkin_date_rate": amount(in_rate, include_breakfast=False, apply_cap=True, per_night_rounding=False),
        "included_breakfast": amount(rate, include_breakfast=True, apply_cap=True, per_night_rounding=False),
        "ignored_nightly_cap": amount(rate, include_breakfast=False, apply_cap=False, per_night_rounding=False),
        "rounded_each_night": amount(rate, include_breakfast=False, apply_cap=True, per_night_rounding=True),
    }
    table = "\n".join(f"  {longdate(k)} ({dayname(k)[:3]}): {v:.2f}" for k, v in sorted(rates.items()))
    who = person(rng)
    evidence = (
        f"{firm(rng).upper()} — EXPENSE AUDIT, CLAIM {ref(rng, 'EXP')}\n"
        f"Claimant: {who}, home currency EUR. Trip: supplier visit, Osaka, {longdate(checkin)} to {longdate(checkout)}.\n"
        f"Travel policy P7 (extract)\n"
        f"7.1 A foreign-currency lodging charge is converted to EUR at the treasury reference rate for the transaction "
        f"date (the checkout date on the hotel invoice). If the table has no rate for that date, use the most recent "
        f"earlier date that has one.\n"
        f"7.2 Lodging is reimbursed up to {fmt('EUR', cap)} per night; the reimbursable amount is the lower of the "
        f"converted room charge and the cap multiplied by the number of nights.\n"
        f"7.3 Breakfast, minibar and laundry are not lodging and are claimed separately.\n"
        f"7.4 Convert the total room charge in one step and round once, to the cent, half a cent upwards.\n"
        f"Hotel invoice (JPY): {nights} nights room charge {rate_per_night:,.0f} per night; breakfast {breakfast:,.0f} per "
        f"night; paid on checkout, {longdate(checkout)}.\n"
        f"Treasury reference table, JPY per EUR:\n{table}"
    )
    options, expected, mm = _finish_money_options(
        "EUR", truth, mistakes, lambda v: f"Reimbursable lodging amount {fmt('EUR', v)}.", rng)
    return World(
        id=f"tn-fx-{idx:05d}", law="fx_reimbursement", kind="choice", evidence=evidence,
        criterion="What is the reimbursable lodging amount for this hotel stay under policy P7?",
        options=options, expected=expected, mistakes=mm,
        rationale=f"Rate for {checkout} -> {used_day} at {rate}; room {room_jpy} JPY; cap {cap}x{nights}; {truth}.",
        params={"nights": nights, "checkout": checkout.isoformat(), "rate_day": used_day.isoformat(), "rate": str(rate),
                "room_jpy": str(room_jpy), "cap": str(cap)})


def law_refund_proration(rng: random.Random, idx: int) -> World:
    """Refund = amount actually paid x unused days / term days, minus a fee,
    where unused days run from the day after the effective date to the term end
    inclusive."""
    start = dt.date(rng.randint(2026, 2028), rng.randint(1, 12), 1)
    end = add_months(start, 12) - dt.timedelta(days=1)
    term_days = (end - start).days + 1
    list_price = money(Decimal(rng.randint(90000, 260000)) / 100)
    discount = Decimal(rng.choice([10, 12, 15, 20])) / 100
    paid = money(list_price * (1 - discount))
    fee = Decimal(rng.choice([0, 25, 40, 60]))
    effective = start + dt.timedelta(days=rng.randint(60, term_days - 40))

    def refund(base: Decimal, unused: int, days: int, f: Decimal) -> Decimal:
        return max(Decimal(0), money(base * Decimal(unused) / Decimal(days) - f))

    unused = (end - effective).days                    # day after effective .. end, inclusive
    truth = refund(paid, unused, term_days, fee)
    mistakes = {
        "list_price_instead_of_paid": refund(list_price, unused, term_days, fee),
        "counted_the_effective_date_as_unused": refund(paid, unused + 1, term_days, fee),
        "excluded_the_last_day_of_term": refund(paid, unused - 1, term_days, fee),
    }
    if term_days == 366:
        mistakes["365_day_term"] = refund(paid, unused, 365, fee)
    if fee:
        mistakes["forgot_the_fee"] = refund(paid, unused, term_days, Decimal(0))
    evidence = (
        f"{firm(rng).upper()} — SUBSCRIPTION CANCELLATION WORKSHEET, ACCOUNT {ref(rng, 'AC')}\n"
        f"Plan: annual, prepaid. Current term {longdate(start)} to {longdate(end)}, both days included "
        f"({term_days} days).\n"
        f"Invoice: list price {fmt('EUR', list_price)}; prepayment discount {discount * 100:.0f} % "
        f"({fmt('EUR', list_price - paid)} off); amount paid {fmt('EUR', paid)}.\n"
        f"Terms, section 9 (extract)\n"
        f"9.1 On cancellation the customer receives a refund of the amount actually paid for the term, in proportion "
        f"to the unused days.\n"
        f"9.2 Unused days are the days from the day after the effective cancellation date up to and including the "
        f"last day of the term. The effective date itself is a used day.\n"
        f"9.3 An administration fee of {fmt('EUR', fee)} is deducted from the refund; a refund cannot be negative.\n"
        f"9.4 The refund is computed in one step and rounded once, to the cent, half a cent upwards.\n"
        f"Cancellation notice received and effective {longdate(effective)}."
    )
    options, expected, mm = _finish_money_options("EUR", truth, mistakes, lambda v: f"Refund of {fmt('EUR', v)}.", rng)
    return World(
        id=f"tn-refund-{idx:05d}", law="refund_proration", kind="choice", evidence=evidence,
        criterion="What refund does the customer receive under section 9?",
        options=options, expected=expected, mistakes=mm,
        rationale=f"{paid} x {unused}/{term_days} - {fee} = {truth}.",
        params={"paid": str(paid), "unused": unused, "term_days": term_days, "fee": str(fee),
                "effective": effective.isoformat()})


def law_credit_proration(rng: random.Random, idx: int) -> World:
    """Annual credits prorated by active calendar days, both boundary days
    included, minus suspended days; exact integer arithmetic. Structured style."""
    year = rng.choice([2028, 2032, 2027, 2029])
    ydays = 366 if calendar.isleap(year) else 365
    per_day = rng.randint(2, 9)
    annual = per_day * ydays
    act = dt.date(year, rng.randint(1, 12), rng.randint(1, 26))
    span = rng.randint(2, 9)
    canc = act + dt.timedelta(days=span)
    susp = act + dt.timedelta(days=rng.randint(1, span - 1)) if span > 1 else None
    active = span + 1 - (1 if susp else 0)
    truth = per_day * active
    mistakes = {
        "excluded_the_cancellation_date": per_day * (active - 1),
        "ignored_the_suspension": per_day * (span + 1),
        "excluded_both_boundary_days": per_day * max(0, active - 2),
        "365_day_year": round(annual / 365 * active) if ydays == 366 else round(annual / 366 * active),
    }
    vals = {truth}
    for v in mistakes.values():
        vals.add(v)
    while len(vals) < 4:
        vals.add(truth + per_day * rng.choice([2, 3, 4]))
    options = tuple((f"{v}_credits", f"The correctly calculated outcome is {v} credits.") for v in sorted(vals))
    mm = {}
    for name, v in mistakes.items():
        if v != truth:
            mm.setdefault(f"{v}_credits", name)
    susp_text = (f" The service was suspended on {longdate(susp)}, and a suspended day earns nothing; all other days "
                 f"in the span are active.") if susp else ""
    evidence = {
        "event": "Work out the service credit earned and pick the matching figure.",
        "rules": (f"For {year}, a {ydays}-day year, the full-year credit is {annual:,} credits and it accrues evenly "
                  f"per calendar day, so each active day earns a whole number of credits. Activation day and "
                  f"cancellation day both count as active. Activation: {longdate(act)}. Cancellation: "
                  f"{longdate(canc)}.{susp_text} Work out the exact figure first."),
    }
    return World(
        id=f"tn-credits-{idx:05d}", law="credit_proration", kind="choice", evidence=evidence,
        criterion="Apply every stated inclusion and correction rule and select the exact outcome.",
        options=options, expected=f"{truth}_credits", mistakes=mm,
        rationale=f"{per_day}/day x {active} active days = {truth}.",
        params={"year": year, "per_day": per_day, "activated": act.isoformat(), "cancelled": canc.isoformat(),
                "suspended": susp.isoformat() if susp else None})


LAWS = {
    "interest_period": law_interest_period,
    "term_end": law_term_end,
    "threshold_day": law_threshold_day,
    "first_declined": law_first_declined,
    "rest_violations": law_rest_violations,
    "window_bands": law_window_bands,
    "unit_limit": law_unit_limit,
    "usage_allowance": law_usage_allowance,
    "fx_reimbursement": law_fx_reimbursement,
    "refund_proration": law_refund_proration,
    "credit_proration": law_credit_proration,
}


# ---------------------------------------------------------------------------
# Generation, validation, contamination check
# ---------------------------------------------------------------------------
def generate(n: int, seed: int = 0, laws: list[str] | None = None) -> list[World]:
    """n worlds, round-robin over the laws, reproducible from the seed."""
    names = laws or list(LAWS)
    out: list[World] = []
    for i in range(n):
        name = names[i % len(names)]
        rng = random.Random(f"{seed}:{name}:{i}")
        w = LAWS[name](rng, i)
        check(w)
        out.append(w)
    return out


def check(w: World) -> None:
    ids = [i for i, _ in w.options]
    assert len(ids) == len(set(ids)), f"{w.id}: duplicate option ids"
    assert 2 <= len(ids) <= 16, f"{w.id}: {len(ids)} options"
    assert w.expected in ids, f"{w.id}: expected {w.expected!r} not among options"
    assert w.expected not in w.mistakes, f"{w.id}: gold option labelled as a mistake"
    assert set(w.mistakes) <= set(ids), f"{w.id}: mistake option not among options"
    assert all(re.fullmatch(r"[a-z0-9_]+", i) for i in ids), f"{w.id}: option ids must be snake_case"


def to_decision(record: dict):
    """A generated record as a jobe Decision (import deferred; the generator itself needs no torch)."""
    from jobe.prompt import Decision, Option
    return Decision(id=record["id"], evidence=record["evidence"], criterion=record["criterion"],
                    options=tuple(Option(o["id"], o["description"]) for o in record["options"]),
                    ordinal=record.get("ordinal", False))


def _grams(text: str, n: int = 10):
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def contamination(worlds: list[World], jevbench_dir: str, n: int = 10) -> list[tuple[str, str]]:
    """(world id, shared phrase) for every n-word run also present in a public task file."""
    import glob
    import os
    pool = set()
    for path in glob.glob(os.path.join(jevbench_dir, "datasets", "public", "*.jsonl")):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                t = json.loads(line)
                pool |= _grams(json.dumps(t.get("state"), ensure_ascii=False) + " " + json.dumps(t.get("question"), ensure_ascii=False), n)
    hits = []
    for w in worlds:
        text = json.dumps(w.evidence, ensure_ascii=False) + " " + w.criterion + " " + " ".join(d for _, d in w.options)
        for g in sorted(_grams(text, n) & pool):
            hits.append((w.id, g))
    return hits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--laws", nargs="*", choices=list(LAWS))
    ap.add_argument("--out", default="")
    ap.add_argument("--check-contamination", metavar="JEVBENCH_DIR", default="")
    args = ap.parse_args(argv)
    worlds = generate(args.n, args.seed, args.laws)
    by_law = {}
    for w in worlds:
        s = by_law.setdefault(w.law, [0, 0, 0, 0])
        s[0] += 1
        s[1] += w.split == "heldout"
        s[2] += len(w.mistakes)
        s[3] += len(w.options)
    print(f"{'law':18s} {'n':>5s} {'heldout':>7s} {'mistakes/rec':>12s} {'options/rec':>11s}")
    for law, (n, h, m, o) in by_law.items():
        print(f"{law:18s} {n:5d} {h:7d} {m / n:12.2f} {o / n:11.2f}")
    if args.check_contamination:
        hits = contamination(worlds, args.check_contamination)
        print(f"contamination: {len(hits)} shared 10-word runs against the public files")
        for wid, g in hits[:10]:
            print(f"  {wid}: {g!r}")
        if hits:
            return 1
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            for w in worlds:
                fh.write(json.dumps(w.record(), ensure_ascii=False) + "\n")
        print(f"wrote {len(worlds)} records to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""Exact-law worlds for the long_policy family.

A long policy pack (8-13k characters): purpose and scope, definitions with a
carve-out, numbered clauses, an amendment with its own applicability date, a
superseded or draft text left in the file, the case records, and a desk note
that applies the naive rule. The answer is computed by code from the stack of
devices the pack contains; the wrong options are the answers named mistakes
produce (the desk note's rule, the unamended clause, the superseded version,
the decoy record, the wrong start date for a time limit).

Six laws: approval routing by aggregated contract value, a water-damage claim
under exclusion, exception and endorsement, alert routing through ordered
runbook rules, payment-relief eligibility on a defined income, appeal
admissibility on working days with closures, and a commercial returns decision.

  python -m jobe.worlds.long_policy --n 600 --seed 0 --out long_policy.jsonl
  python -m jobe.worlds.long_policy --n 120 --check-contamination <jevbench clone>
"""
from __future__ import annotations

import datetime as dt
import random
import sys
from decimal import Decimal

from .base import (World, check, contamination, dayname, firm, fmt, generate as _generate, longdate,  # noqa: F401
                   money, person, ref, run_cli, to_decision)

FAMILY = "long_policy"


# ---------------------------------------------------------------------------
# Document assembly
# ---------------------------------------------------------------------------
class Doc:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def title(self, text: str) -> None:
        self.lines += [text.upper(), "=" * 72]

    def part(self, text: str) -> None:
        self.lines += ["", "=" * 72, text.upper(), "=" * 72]

    def section(self, text: str) -> None:
        self.lines += ["", text]

    def clause(self, num: str, text: str) -> None:
        self.lines.append(f"{num} {text}")

    def para(self, text: str) -> None:
        self.lines.append(text)

    def blank(self) -> None:
        self.lines.append("")

    def render(self) -> str:
        return "\n".join(self.lines)


GENERAL = [
    lambda P, O, D, C, n: f"Interpretation. Clause headings are for navigation only and do not affect meaning. A reference to a clause is a reference to a clause of {P} unless another document is named. The singular includes the plural and the plural the singular.",
    lambda P, O, D, C, n: f"Precedence between documents. Where {P} and a signed contract differ, the contract prevails only on the point on which it is expressly more specific; on every other point {P} applies. A quotation, order acknowledgement or email is not a signed contract.",
    lambda P, O, D, C, n: f"Records. The {D} keeps every decision file, including the version of {P} that was applied, for {n} years from the decision date. A file is produced on request to {O} or to internal audit within five business days.",
    lambda P, O, D, C, n: f"Escalation. A reviewer who cannot reach a decision within {n} business days escalates the file to {O} with a note of the clauses considered so far. Escalation does not extend any time limit that {P} imposes on a customer, an applicant or on {C}.",
    lambda P, O, D, C, n: f"Review of this document. {O} reviews {P} at least once a year and after any change in law that affects it. A review may produce an amendment; an amendment takes effect only from the date it states, and only for the matters it states.",
    lambda P, O, D, C, n: f"Training. Staff who apply {P} complete the {C} refresher course on it every year. A decision is not invalid merely because the reviewer's refresher was overdue at the time.",
    lambda P, O, D, C, n: f"Delegation. {O} may delegate a decision under {P} in writing. The delegate applies {P} in full; a delegation can neither enlarge nor narrow what {P} allows.",
    lambda P, O, D, C, n: f"Communicating decisions. Every decision is issued in writing, names the clause or clauses relied on and states how a review may be requested. A decision that omits the clause reference remains effective but is corrected within {n} business days.",
    lambda P, O, D, C, n: f"Review of decisions. A person affected by a decision may request a review within {n} business days of receiving it. The review is conducted by someone other than the original reviewer and applies the same version of {P} that governed the original decision.",
    lambda P, O, D, C, n: f"Personal data. Files under {P} contain personal data and are handled under the {C} data handling standard. Only staff with a role in the decision open a file, and every access is logged.",
    lambda P, O, D, C, n: f"Conflicts of interest. A reviewer with a personal or financial connection to a party declares it to {O} and takes no part in the decision. A decision taken in breach of this clause is re-taken by another reviewer.",
    lambda P, O, D, C, n: f"Errors. An arithmetic or clerical error in a decision may be corrected by {O} at any time. The corrected decision is dated on the day of correction and is treated as issued on that day for the purpose of any review period.",
    lambda P, O, D, C, n: f"Language. Where {P} is translated, the English text governs. Defined terms keep their defined meaning in every translation.",
    lambda P, O, D, C, n: f"Continuity. {P} applies to every matter not yet decided on its effective date, subject to any transitional rule that an amendment states expressly. A matter decided under an earlier version is not reopened by a later one.",
    lambda P, O, D, C, n: f"Notices to {C}. A notice is received on the business day it reaches the address in the file, or on the next business day if it arrives after {n % 2 + 4} p.m. A notice sent to any other address is not received until it is forwarded to the {D}.",
]


PROCEDURE = [
    lambda P, D, n: f"Opening a file. The {D} opens a file on the day a matter is received and records the receipt date, the documents received and the version of {P} in force on that day. Documents received later are added with their own receipt date.",
    lambda P, D, n: f"Completeness. A matter is assessed only when the file is complete. The {D} requests missing items once, in writing, and allows {n} business days; a matter still incomplete after that is closed and may be resubmitted as a new matter.",
    lambda P, D, n: f"Evidence. The {D} relies on the documents in the file, on the systems of record named in {P} and on nothing else. A statement in a covering email that conflicts with a document in the file does not displace the document.",
    lambda P, D, n: f"Worksheets. Every computation under {P} is written on the worksheet in the file, with the clause applied beside each step. A worksheet is a working record: where it and {P} differ, {P} governs and the worksheet is corrected.",
    lambda P, D, n: f"Timing of the decision. The {D} issues a decision within {n} business days of the file becoming complete, or explains in writing why it cannot. The time limit binds the {D}; it neither shortens nor extends any period that {P} sets for another party.",
    lambda P, D, n: f"Second check. A decision that departs from the desk's draft note, or from the rules engine's proposal where one exists, is countersigned by a second reviewer before it is issued. The countersignature confirms the clause reference, not the outcome.",
    lambda P, D, n: f"Superseded texts. Earlier editions and superseded clauses are kept in the file for reference and are marked as such. A decision that relies on a superseded text is defective and is re-taken.",
    lambda P, D, n: f"Templates. Standard letters and templates are maintained by the {D}. A figure printed on a template is a default only; the figure that governs is the one that {P}, the schedule or the amendment states for the matter in hand.",
    lambda P, D, n: f"Closing the file. A file is closed when the decision has been issued and the period for requesting a review has passed. A closed file is reopened only under the errors clause or on a review.",
]


def procedure(rng: random.Random, doc: Doc, num: str, P: str, D: str, k: int = 6) -> None:
    doc.section(f"{num}. Procedure and administration")
    for i, clause in enumerate(rng.sample(PROCEDURE, k), 1):
        doc.clause(f"{num}.{i}", clause(P, D, rng.randint(5, 15)))


def general_provisions(rng: random.Random, doc: Doc, num: str, P: str, O: str, D: str, C: str, k: int = 9) -> None:
    doc.section(f"{num}. General provisions")
    for i, clause in enumerate(rng.sample(GENERAL, k), 1):
        doc.clause(f"{num}.{i}", clause(P, O, D, C, rng.randint(3, 9)))


def yes_no(truth: bool, yes: str, no: str) -> tuple[tuple[tuple[str, str], ...], str]:
    return (("yes", yes), ("no", no)), ("yes" if truth else "no")


def working_days_after(start: dt.date, n: int, off: set[dt.date]) -> dt.date:
    """The n-th working day strictly after `start`."""
    d, k = start, 0
    while k < n:
        d += dt.timedelta(days=1)
        if d.weekday() < 5 and d not in off:
            k += 1
    return d


def eur(x) -> str:
    return fmt("EUR", x)


def usd(x) -> str:
    return fmt("USD", x)


def shuffled(rng: random.Random, options: list[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    opts = list(options)
    rng.shuffle(opts)
    return tuple(opts)


# ---------------------------------------------------------------------------
# Law 1 — approval level by aggregated contract value
# ---------------------------------------------------------------------------
def law_approval_level(rng: random.Random, idx: int) -> World:
    C, sup, other = firm(rng), firm(rng), firm(rng)
    P = f"PP-{rng.randint(10, 99)}"
    t1 = rng.choice([20000, 25000, 30000, 40000])
    t2, t3, t4 = t1 * 3, t1 * 6, t1 * 18
    fee = rng.randint(20, 70) * 1000 + rng.choice([0, 500])
    renewals = rng.choice([0, 1, 2])
    impl = rng.choice([0, 4800, 9800, 15000])
    training = rng.choice([0, 2400, 3600])
    disc = rng.choice([5, 8, 10])
    S = dt.date(rng.randint(2026, 2027), rng.randint(2, 12), rng.randint(3, 27))
    amend_applies = rng.random() < 0.7
    amend_eff = S - dt.timedelta(days=rng.randint(20, 200)) if amend_applies else S + dt.timedelta(days=rng.randint(5, 60))
    win = 90
    d_rel = rng.randint(25, 85) if rng.random() < 0.7 else rng.randint(100, 170)
    p_rel = rng.randint(30, 140) * 1000
    p_other = rng.randint(20, 90) * 1000
    p_old = rng.randint(40, 120) * 1000
    d_old = rng.randint(200, 320)
    d_other = rng.randint(10, 80)

    def level(x: int) -> int:
        return 1 if x <= t1 else 2 if x <= t2 else 3 if x <= t3 else 4 if x <= t4 else 5

    tcv = fee * (1 + renewals) + impl + training
    counts_rel = amend_applies and d_rel <= win
    agg = tcv + (p_rel if counts_rel else 0)
    truth = level(agg)
    mistakes = {
        "annual_fee_only": level(fee),
        "renewals_ignored": level(fee + impl + training + (p_rel if counts_rel else 0)),
        "discount_applied": level(round(fee * (1 + renewals) * (100 - disc) / 100) + impl + training + (p_rel if counts_rel else 0)),
        "no_aggregation" if counts_rel else "aggregated_although_not_required": level(tcv) if counts_rel else level(tcv + p_rel),
        "other_supplier_aggregated": level(agg + p_other),
        "old_purchase_aggregated": level(agg + p_old),
    }
    ids = ["budget_holder", "department_head", "vp_with_finance_director", "cfo", "executive_committee"]
    names = ["Budget Holder", "Department Head", "Vice President together with the Finance Director",
             "Chief Financial Officer", "Executive Committee"]
    bands = [f"up to {eur(t1)}", f"above {eur(t1)} up to {eur(t2)}", f"above {eur(t2)} up to {eur(t3)}",
             f"above {eur(t3)} up to {eur(t4)}", f"above {eur(t4)}"]
    owner, desk = "the Head of Procurement", "Procurement Desk"
    pr = f"PR-{S.year}-{rng.randint(1000, 9999)}"
    doc = Doc()
    doc.title(f"{C} — Procurement Approval Policy {P} (version {rng.randint(3, 9)}, effective {longdate(dt.date(S.year - 1, 3, 1))})")
    doc.para(f"Extract for approval routing, with Amendment {P}/A{rng.randint(2, 5)} and purchase request {pr}. Prepared for the {desk}. Page numbers omitted.")
    doc.part("Part I — Policy extract")
    doc.section("1. Purpose and scope")
    doc.clause("1.1", f"{P} governs every purchase of goods and services by {C} and its subsidiaries, other than raw materials bought on the trading desk, internal recharges between group companies, statutory taxes, salaries, and fees paid to government bodies.")
    doc.clause("1.2", "Its objectives are value for money, a clear record of who approved what, and one routing rule that every requester can apply before submitting.")
    doc.clause("1.3", "The routing question is always the same: which approval level is reached by the amount to be approved, as defined in section 2 and adjusted by section 5.")
    doc.section("2. Definitions")
    doc.clause("2.1", "\"Request\" — a purchase request submitted in the procurement system, identified by its PR number and its Submission Date.")
    doc.clause("2.2", "\"Submission Date\" — the date on which the requester presses Submit. The date on which the draft was created, and the date of any quotation, are irrelevant to routing.")
    doc.clause("2.3", "\"Contract Value\" — the total amount payable to the supplier over the Committed Term: every recurring fee for every year of the Committed Term, plus every one-time fee (implementation, set-up, training, data migration), before VAT.")
    doc.clause("2.4", "\"Committed Term\" — the initial term together with every renewal period that takes effect automatically unless a party gives notice. A renewal that requires a fresh signature or a new purchase order is not part of the Committed Term.")
    doc.clause("2.5", "Discounts. Only a discount written into the quotation or order form signed by the supplier reduces the Contract Value. A discount that is conditional — on signing by a date, on volume, on a reference call or a case study — or that appears only in correspondence is disregarded, however likely it is to be honoured.")
    doc.clause("2.6", "\"Supplier Group\" — a supplier together with every company under common control with it, as shown in the supplier master record.")
    doc.section("3. Approval levels")
    doc.clause("3.1", "The approval level for a Request is the first level in the table whose band contains the amount to be approved:")
    for i, (nm, band) in enumerate(zip(names, bands), 1):
        doc.para(f"    Level {i} — {nm} — {band}")
    doc.clause("3.2", "An amount exactly equal to the upper bound of a band belongs to that band, not to the next one.")
    doc.clause("3.3", "Approval at a level requires the approvers named for that level; a higher level may always approve in place of a lower one, but this does not change the routing.")
    doc.section("4. Process")
    doc.clause("4.1", "The requester attaches the supplier's quotation or order form, the business case and, for software, the information security assessment.")
    doc.clause("4.2", "The Procurement Desk checks the attachments, confirms the amount to be approved under sections 2 and 5, records it on the Request and routes the Request. The Desk's routing note is a working document; the policy governs.")
    doc.clause("4.3", "A Request returned for correction keeps its original Submission Date if resubmitted within ten business days; otherwise the resubmission is a new Request.")
    doc.section("5. Splitting and aggregation")
    doc.clause("5.1", "A purchase may not be split into several Requests in order to reach a lower level. A Request that appears to have been split is routed as if the parts were one Request.")
    doc.clause("5.2", f"[Inserted by Amendment {P}/A — see Part II] Related Purchases. The amount to be approved is the Contract Value of the Request plus the Contract Value of every Related Purchase. A Related Purchase is a Request to the same supplier or to a member of the same Supplier Group that was approved within the {win} days before the Submission Date, whichever department raised it. A Request to a different supplier, and a Request approved more than {win} days before the Submission Date, is not a Related Purchase.")
    procedure(rng, doc, "6", P, desk)
    general_provisions(rng, doc, "7", P, owner, desk, C)
    doc.part(f"Part II — Amendment {P}/A")
    doc.para(f"Adopted by the Executive Committee on {longdate(amend_eff - dt.timedelta(days=21))}. Inserts clause 5.2. Applies to every Request whose Submission Date is on or after {longdate(amend_eff)}. A Request with an earlier Submission Date is routed on its own Contract Value, without clause 5.2, even if it is still awaiting approval on that date.")
    doc.part(f"Part III — Purchase request {pr}")
    doc.para(f"Requester: {person(rng)}, {rng.choice(['Operations', 'Marketing', 'Quality', 'Engineering'])}. Supplier: {sup} (supplier master: group parent {sup}). Category: software subscription.")
    doc.para(f"Draft created: {longdate(S - dt.timedelta(days=rng.randint(6, 18)))}. Submission Date: {longdate(S)}.")
    doc.para(f"Quotation Q-{rng.randint(10000, 99999)}, signed by the supplier on {longdate(S - dt.timedelta(days=4))}:")
    doc.para(f"    Subscription fee {eur(fee)} per year. Initial term: 12 months from go-live.")
    doc.para("    Renewal: " + (f"the subscription renews automatically for {renewals} further period{'s' if renewals > 1 else ''} of 12 months each unless one party serves notice in writing at least 60 days ahead of the end of the running period." if renewals else "any renewal requires a new order form signed by both parties."))
    doc.para(f"    One-time fees: implementation {eur(impl)}; training package {eur(training)}. All amounts exclude VAT.")
    doc.para(f"Email from the supplier's account manager, {longdate(S - dt.timedelta(days=2))}: \"If the order is signed before the end of the month we can apply a {disc}% promotional discount to the subscription. Let me know and I will issue a revised quotation.\" No revised quotation was issued.")
    doc.section("Approval register extract (last 12 months, all departments)")
    rows = [(S - dt.timedelta(days=d_rel), sup, p_rel, "approved"),
            (S - dt.timedelta(days=d_other), other, p_other, "approved"),
            (S - dt.timedelta(days=d_old), sup, p_old, "approved")]
    rows.sort(key=lambda r: r[0])
    for d, s_, v, st in rows:
        doc.para(f"    {longdate(d)} — {s_} — Contract Value {eur(v)} — {st}")
    doc.section("Desk routing note (working document)")
    naive = rng.choice(["fee", "tcv"])
    doc.para(f"{person(rng)}, {desk}: \"" + (f"Subscription {eur(fee)} per year — Level {level(fee)}." if naive == "fee" else f"Contract Value {eur(tcv)} — Level {level(tcv)}.") + " Routed accordingly; please confirm.\"")
    options = tuple((i, f"Level {k} ({nm}) is the level reached: amount to be approved {band}.") for k, (i, nm, band) in enumerate(zip(ids, names, bands), 1))
    expected = ids[truth - 1]
    mm = {}
    for name, lv in mistakes.items():
        if lv != truth:
            mm.setdefault(ids[lv - 1], name)
    return World(
        id=f"lp-approval-{idx:05d}", law="approval_level", kind="choice", evidence=doc.render(),
        criterion=f"Determine the approval level required for purchase request {pr} under {P} as amended.",
        options=options, expected=expected, mistakes=mm,
        rationale=f"TCV = {fee} x {1 + renewals} + {impl} + {training} = {tcv}; related purchase {p_rel} {'counts' if counts_rel else 'does not count'} "
                  f"(amendment {'applies' if amend_applies else 'does not apply'}, {d_rel} days before); amount {agg} -> Level {truth}.",
        params={"fee": fee, "renewals": renewals, "impl": impl, "training": training, "tcv": tcv, "amend_applies": amend_applies,
                "d_rel": d_rel, "p_rel": p_rel, "amount": agg, "thresholds": [t1, t2, t3, t4]})


# ---------------------------------------------------------------------------
# Law 2 — water-damage claim: exclusion, exception, endorsement, occupancy
# ---------------------------------------------------------------------------
def law_water_damage(rng: random.Random, idx: int) -> World:
    insurer = f"{rng.choice(['Harborline', 'Westmoor', 'Calder', 'Tarnwood', 'Silverbeck'])} Mutual"
    form = f"HP-{rng.randint(2, 9)}"
    weeks = rng.choice([1, 1, 3, 4, 5, 6])
    concealed = rng.random() < 0.75
    knew = concealed and rng.random() < 0.2
    occupancy = rng.choice(["lived_in", "unoccupied", "unoccupied", "vacant"])
    away_days = rng.randint(62, 95)
    base_sub = rng.choice([8000, 10000, 12000])
    end_sub = rng.choice([15000, 20000])
    E = dt.date(rng.randint(2025, 2026), rng.randint(1, 12), 1)
    renewal = E + dt.timedelta(days=rng.choice([-120, -45, -10, 15, 60, 140]))
    endorsed = renewal >= E
    sub = end_sub if endorsed else base_sub
    estimate = rng.randint(22, 46) * 1000 + rng.choice([0, 350, 780])
    ded = rng.choice([500, 1000, 2500])
    loss_date = renewal + dt.timedelta(days=rng.randint(40, 300))
    if occupancy == "vacant":
        truth = "deny_vacancy_exclusion"
    elif weeks * 7 >= 14:
        truth = f"pay_subject_to_{sub}_sublimit" if (concealed and not knew) else "deny_repeated_seepage"
    else:
        truth = "pay_full_estimate_less_deductible"
    other_sub = base_sub if endorsed else end_sub
    mistakes = {
        "exception_ignored": "deny_repeated_seepage" if truth.startswith("pay_subject") else None,
        "unoccupied_treated_as_vacant": "deny_vacancy_exclusion" if occupancy == "unoccupied" else None,
        "endorsement_misapplied": f"pay_subject_to_{other_sub}_sublimit" if truth.startswith("pay_subject") else None,
        "seepage_exclusion_missed": "pay_full_estimate_less_deductible" if weeks * 7 >= 14 and occupancy != "vacant" else None,
        "sudden_leak_treated_as_seepage": "deny_repeated_seepage" if weeks * 7 < 14 and occupancy != "vacant" else None,
        "vacancy_days_miscounted": None if occupancy != "vacant" else (f"pay_subject_to_{sub}_sublimit" if weeks * 7 >= 14 and concealed and not knew else "pay_full_estimate_less_deductible"),
    }
    insured = person(rng)
    addr = f"{rng.randint(3, 98)} {rng.choice(['Linden', 'Aspen', 'Harbour', 'Quarry', 'Beacon'])} {rng.choice(['Road', 'Lane', 'Close', 'Rise'])}, {rng.choice(['Millbrook', 'Eastvale', 'Carrow', 'Northam'])}"
    owner, desk = "the Claims Manager", "Claims Desk"
    claim = f"CL-{loss_date.year}-{rng.randint(100000, 999999)}"
    doc = Doc()
    doc.title(f"{insurer} Insurance — Home Policy Form {form} (edition {E.year - 2}) — extracts, endorsements and claim file {claim}")
    doc.para("Prepared for coverage review. Bracketed notes are from the file clerk and are not policy text.")
    doc.part("Part A — Schedule for the current term")
    doc.para(f"Policyholder: {insured}. Insured address: {addr}. Policy number: {ref(rng, 'HP')}.")
    doc.para(f"Current term: {longdate(renewal)} to {longdate(renewal + dt.timedelta(days=365))}, renewed on {longdate(renewal)}. Original inception: {longdate(renewal - dt.timedelta(days=365 * rng.randint(2, 6)))}.")
    doc.para(f"Coverage A (Dwelling) limit: {usd(rng.randint(300, 650) * 1000)}. All-perils deductible: {usd(ded)}.")
    doc.para(f"Endorsements attached to the current term: {form}-E4 (ordinance or law), {form}-E12 (concealed water sublimit increase) — see Part C for applicability.")
    doc.part(f"Part B — Policy form {form} (extracts)")
    doc.section("1. Definitions")
    doc.clause("1.1", "\"Vacant\" — the residence premises contain neither furnishings nor personal property sufficient for a person to live there, whether or not anyone intends to return. A dwelling emptied for renovation or sale is vacant from the day the contents are removed.")
    doc.clause("1.2", "\"Unoccupied\" — the residence premises contain the furnishings and personal property needed to live there, but no person is living there. A dwelling whose occupants are travelling, in hospital or working away is unoccupied, not vacant, for as long as the contents remain.")
    doc.clause("1.3", "\"Concealed Water\" — water that escapes from a plumbing, heating or air-conditioning system and remains hidden within a wall, floor, ceiling or foundation, so that the escape is not known to any insured person until damage becomes visible.")
    doc.clause("1.4", "\"Sudden Escape\" — an escape of water that begins and is discovered within fourteen days.")
    doc.section("2. Coverage A — Dwelling")
    doc.clause("2.1", "We cover direct physical loss to the dwelling caused by a peril not excluded in section 4, subject to the limits in the Declarations, the sublimits in section 3 and the deductible.")
    doc.clause("2.2", "Payment is the cost of repair shown by an estimate we accept, less the deductible, and never more than the applicable limit or sublimit.")
    doc.section("3. Sublimits")
    doc.clause("3.1", f"Concealed Water sublimit: {usd(base_sub)} per occurrence, unless an endorsement attached to the term in which the loss occurs states a different amount.")
    doc.clause("3.2", "A sublimit applies after the deductible has been taken from the accepted estimate; the deductible is not taken again from the sublimit.")
    doc.section("4. Exclusions")
    doc.clause("4.1", "We do not cover loss caused by flood, surface water or water that backs up through sewers or drains.")
    doc.clause("4.2", "We do not cover loss caused by freezing of plumbing while the dwelling is unoccupied or vacant, unless heat was maintained or the water supply was shut off and the system drained.")
    doc.clause("4.3", "Repeated seepage. We do not cover loss caused by continuous or repeated seepage or leakage of water over a period of fourteen days or more, whether or not the insured was aware of it.")
    doc.clause("4.3.1", "Exception — Concealed Water. Where the escape is Concealed Water, exclusion 4.3 is disapplied and the loss is covered, limited by the Concealed Water sublimit in section 3 and reduced by the deductible. The exception is forfeited if an insured person knew of the escape before the damage became visible and did nothing.")
    doc.clause("4.4", "We do not cover loss caused by wear, rot, mould or corrosion, except the resulting damage from an escape of water that is otherwise covered.")
    doc.clause("4.5", "Vacancy. We do not cover loss caused by water escape, vandalism or glass breakage occurring while the dwelling has been vacant for more than sixty consecutive days. A dwelling that is unoccupied but not vacant is not subject to this exclusion.")
    procedure(rng, doc, "5", f"Form {form}", desk, 5)
    general_provisions(rng, doc, "6", f"Form {form}", owner, desk, insurer, 8)
    doc.part(f"Part C — Endorsement {form}-E12 (concealed water sublimit increase)")
    doc.para(f"This endorsement raises the Concealed Water sublimit in clause 3.1 to {usd(end_sub)} per occurrence. It applies to policies issued or renewed on or after {longdate(E)}; for a term that began before that date it takes effect at the next renewal on or after that date. Nothing else in the form is changed.")
    doc.part(f"Part D — Claim file {claim}")
    doc.para(f"Date of loss reported: {longdate(loss_date)}. Reported by: {insured}. Loss: water damage to the ground-floor wall, floor and joists behind the kitchen.")
    doc.para(f"Plumber's report ({longdate(loss_date + dt.timedelta(days=3))}): " + (
        f"the supply joint failed inside the wall cavity; from the staining and the calcification the leak had been active for {weeks} weeks; nothing was visible from the room until the skirting board swelled on {longdate(loss_date - dt.timedelta(days=1))}."
        if concealed else
        f"the flexible hose under the sink failed; water ran across the floor for about {weeks} weeks before the household noticed the buckling; the escape was in plain view under the cupboard."))
    if knew:
        doc.para(f"Insured's statement: \"I noticed damp on the wall about {max(2, weeks - 1)} weeks ago and meant to call someone.\"")
    else:
        doc.para("Insured's statement: \"We had no idea anything was leaking until the floor moved. Nothing was visible.\"")
    if occupancy == "lived_in":
        doc.para("Occupancy: the insured and family living at the premises throughout.")
    elif occupancy == "unoccupied":
        doc.para(f"Occupancy: the insured has been working abroad since {longdate(loss_date - dt.timedelta(days=away_days))} ({away_days} days before the loss). Furniture, appliances, clothing and food were left in place; a neighbour holds a key and collects post.")
    else:
        doc.para(f"Occupancy: the dwelling was emptied for sale on {longdate(loss_date - dt.timedelta(days=away_days))} ({away_days} days before the loss); the contents went to storage and the estate agent holds the key.")
    doc.para(f"Accepted repair estimate: {usd(estimate)}. Deductible per Declarations: {usd(ded)}.")
    doc.para(f"[Clerk: the file template shows the Concealed Water sublimit as {usd(base_sub)}; the template has not been updated for {form}-E12.]")
    notes = (["seepage"] if weeks * 7 >= 14 else ["sudden"]) + (["vacancy"] if occupancy != "lived_in" else []) + (["sublimit_template"] if truth.startswith("pay_subject") else [])
    note = rng.choice(notes)
    doc.section("Trainee adjuster's note")
    doc.para({
        "seepage": f"\"Leak ran {weeks} weeks — that is over fourteen days, so exclusion 4.3 applies and the building portion is denied.\"",
        "vacancy": "\"Nobody has lived there for over sixty days — vacancy exclusion 4.5, deny.\"",
        "sudden": f"\"Leak ran only {weeks} week{'s' if weeks > 1 else ''}; treat as a plain covered escape and pay the estimate less the deductible.\"",
        "sublimit_template": f"\"Covered under the exception but capped at the {usd(base_sub)} shown on the template.\"",
    }[note])
    options = shuffled(rng, [
        ("deny_repeated_seepage", "Excluded under clause 4.3 (repeated seepage or leakage): nothing is payable for the building damage."),
        ("deny_vacancy_exclusion", "Excluded under the vacancy clause 4.5: nothing is payable for the building damage."),
        ("pay_full_estimate_less_deductible", "Covered with no sublimit: the accepted estimate minus the deductible is payable."),
        (f"pay_subject_to_{base_sub}_sublimit", f"Covered; the building payment is limited to a {usd(base_sub)} Concealed Water sublimit."),
        (f"pay_subject_to_{end_sub}_sublimit", f"Covered; the building payment is limited to a {usd(end_sub)} Concealed Water sublimit."),
    ])
    mm = {}
    for name, opt in mistakes.items():
        if opt and opt != truth:
            mm.setdefault(opt, name)
    return World(
        id=f"lp-water-{idx:05d}", law="water_damage_claim", kind="choice", evidence=doc.render(),
        criterion=f"Decide the settlement of the building water-damage element of claim {claim} under form {form} and the endorsements in force for the current term.",
        options=options, expected=truth, mistakes=mm,
        rationale=f"weeks={weeks} concealed={concealed} knew={knew} occupancy={occupancy} endorsed={endorsed} (renewal {renewal} vs {E}) -> {truth}.",
        params={"weeks": weeks, "concealed": concealed, "knew": knew, "occupancy": occupancy, "renewal": renewal.isoformat(),
                "endorsement_date": E.isoformat(), "sublimit": sub})


# ---------------------------------------------------------------------------
# Law 3 — alert routing through ordered runbook rules
# ---------------------------------------------------------------------------
def law_alert_routing(rng: random.Random, idx: int) -> World:
    C = f"{rng.choice(['Pelagos', 'Northwind', 'Quarry', 'Beacon', 'Larkspur'])} {rng.choice(['Streaming', 'Payments', 'Commerce', 'Health'])}"
    rb = f"RB-ROUTE-{rng.randint(3, 9)}"
    rev = rng.randint(3, 7)
    night = dt.date(rng.randint(2026, 2027), rng.randint(1, 12), rng.randint(2, 27))
    comp_self = f"C-{rng.randint(1000, 1999)}"
    comp_vendor = f"C-{rng.randint(1000, 1999)}"
    comp_gw = f"C-{rng.randint(2000, 2999)}"
    comp_bill = f"C-{rng.randint(2000, 2999)}"
    vendor = rng.choice(["DataHarbor", "Cloudreef", "Stratalog"])
    source = rng.choice(["self", "self", "self", "vendor", "gateway"])
    comp = {"self": comp_self, "vendor": comp_vendor, "gateway": comp_gw}[source]
    sev = rng.choice(["S2", "S2", "S3", "S1"])
    postponed = rng.random() < 0.6
    window_comp = comp if rng.random() < 0.8 else comp_bill
    alr = rng.randint(4000, 6999)
    t_alert = dt.time(rng.choice([23, 0, 1]), rng.randint(0, 59))
    if source == "vendor":
        truth = "vendor_support"
    elif window_comp == comp and not postponed:
        truth = "no_page_ticket_only"
    elif source == "self":
        truth = "database_oncall" if sev in ("S1", "S2") else "no_page_ticket_only"
    else:
        truth = "platform_oncall" if sev != "S4" else "no_page_ticket_only"
    mistakes = {
        "postponed_window_treated_as_active": "no_page_ticket_only" if (window_comp == comp and postponed and truth != "no_page_ticket_only") else None,
        "security_by_association": "security_incident_response",
        "vendor_by_similar_name": "vendor_support" if source == "self" else None,
        "default_route_taken": "platform_oncall" if truth != "platform_oncall" else None,
        "severity_threshold_ignored": "database_oncall" if (source == "self" and sev == "S3" and truth == "no_page_ticket_only") else None,
        "suppression_ignored": ("database_oncall" if source == "self" and sev in ("S1", "S2") else "platform_oncall") if truth == "no_page_ticket_only" and window_comp == comp and not postponed else None,
    }
    owner, desk = "the Head of Operations", "Operations Router"
    doc = Doc()
    doc.title(f"{C} — Operations alert routing runbook {rb} (revision {rev}), with the inventory extract, the change-calendar extract and the overnight alert stream for {longdate(night)}")
    doc.part(f"{rb} revision {rev}")
    doc.section("1. Purpose")
    doc.para(f"Every production alert is routed by the overnight router — a person on the rota, plus the rules engine that implements this runbook — using the rules below. The runbook replaced ad-hoc paging after the {night.year - 1} incidents in which the wrong team was paged three times.")
    doc.section("2. Definitions")
    doc.clause("2.1", "\"Component\" — an entry in the component inventory, identified by its C-number. The inventory, not the alert text, decides what a component is and who manages it.")
    doc.clause("2.2", "\"Vendor-managed\" — a component whose inventory record names an external vendor as the managing party. A component that merely uses vendor software is not vendor-managed.")
    doc.clause("2.3", "\"Security signal\" — an alert emitted by the intrusion detection sensors, the audit-anomaly detector or the secrets scanner. No other alert is a security signal, whatever else is happening at the time.")
    doc.clause("2.4", "\"Maintenance window\" — a change-calendar entry naming one or more components and a start and end time. A window is Active from its start to its end only while its status is Approved. A window whose status is Postponed, Cancelled or Draft is not Active, whatever the calendar grid shows.")
    doc.clause("2.5", "\"Database component\" — a component whose inventory type is a database instance, replica or cluster.")
    doc.clause("2.6", "Severity S1 to S4 is set by the emitting monitor and is not changed by the router.")
    doc.section("3. Routing rules — applied in order; the first rule that matches decides")
    doc.clause("3.1", "Rule 1 — Security signals go to the Security Incident Response Team on-call, at any severity. A concurrent alert from another source is routed on its own merits and is not a security signal by association.")
    doc.clause("3.2", "Rule 2 — Maintenance suppression. An alert from a component named in an Active maintenance window is not paged; the router creates a next-business-day ticket. The status history in the change tool is authoritative; the calendar grid is a copy synchronised hourly and may lag.")
    doc.clause("3.3", "Rule 3 — Vendor-managed components: the router opens a priority case with the vendor and a platform ticket; no on-call is paged for the component itself.")
    doc.clause("3.4", "Rule 4 — Database components at severity S1 or S2 go to the Data Platform on-call. Database components at S3 or S4 get a next-business-day ticket only.")
    doc.clause("3.5", "Rule 5 — Every other alert at S1, S2 or S3 goes to the Core Platform on-call (the default). S4 alerts get a ticket only.")
    doc.clause("3.6", "A rule that does not match is skipped; the router does not choose between two matching rules — the earlier one governs.")
    procedure(rng, doc, "4", rb, desk, 6)
    general_provisions(rng, doc, "5", rb, owner, desk, C, 9)
    doc.part("Component inventory extract")
    doc.para(f"    {comp_self} — orders-db-replica-2 — type: database replica (PostgreSQL) — managed by: Data Platform team (self-managed) — notes: uses {vendor} backup agent")
    doc.para(f"    {comp_vendor} — orders-db-replica-3 — type: database replica ({vendor} managed service) — managed by: {vendor} (vendor-managed) — notes: vendor support contract V-{rng.randint(100, 999)}")
    doc.para(f"    {comp_gw} — edge-gateway — type: API gateway — managed by: Core Platform team")
    doc.para(f"    {comp_bill} — billing-worker — type: batch worker — managed by: Core Platform team")
    doc.part("Change calendar extract (grid, synchronised hourly)")
    mw = f"MW-{rng.randint(100, 999)}"
    w_start = dt.datetime.combine(night, dt.time(22, 0))
    w_end = w_start + dt.timedelta(hours=4)
    doc.para(f"    {mw} — components: {window_comp} — {w_start.strftime('%H:%M')} {longdate(night)} to {w_end.strftime('%H:%M')} {longdate(night + dt.timedelta(days=1))} — status shown: Approved")
    doc.para(f"Status history for {mw} (change tool, authoritative):")
    doc.para(f"    {longdate(night - dt.timedelta(days=6))} 10:12 — Draft -> Approved ({person(rng)})")
    if postponed:
        doc.para(f"    {longdate(night)} 17:{rng.randint(10, 59)} — Approved -> Postponed to {longdate(night + dt.timedelta(days=7))} (\"vendor engineer unavailable\", {person(rng)})")
        doc.para(f"    [grid last synchronised {longdate(night)} 16:{rng.randint(10, 59)}]")
    else:
        doc.para(f"    [grid last synchronised {longdate(night)} 21:{rng.randint(10, 59)}; no later entries]")
    doc.part(f"Alert stream, night of {longdate(night)}")
    t2 = (dt.datetime.combine(night, t_alert) + dt.timedelta(minutes=2)).time()
    doc.para(f"    ALR-{alr} — {t_alert.strftime('%H:%M')} — component {comp} — monitor: replication-lag — severity {sev} — \"replica lag above threshold for 10 minutes\"")
    doc.para(f"    ALR-{alr + 1} — {t2.strftime('%H:%M')} — source: secrets scanner — repository orders-service — severity S3 — \"pattern match: test fixture key (allowlisted entry AL-{rng.randint(10, 99)})\"")
    doc.para(f"    ALR-{alr + 2} — {(dt.datetime.combine(night, t_alert) + dt.timedelta(minutes=9)).strftime('%H:%M')} — component {comp_gw} — monitor: p95 latency — severity S4 — \"informational\"")
    doc.section("Rules-engine proposal (advisory, generated from the calendar grid)")
    engine = "no_page_ticket_only" if window_comp == comp else ("vendor_support" if source == "vendor" else "platform_oncall")
    doc.para(f"    ALR-{alr}: {engine.replace('_', ' ')} — reason: " + ("component in maintenance window per grid" if window_comp == comp else "default routing"))
    options = shuffled(rng, [
        ("database_oncall", "Page the Data Platform on-call."),
        ("no_page_ticket_only", "Page nobody; raise a ticket for the next business day."),
        ("platform_oncall", "Page the Core Platform on-call (the default route)."),
        ("security_incident_response", "Page the Security Incident Response Team on-call."),
        ("vendor_support", "Open a vendor priority case and a platform ticket; page no on-call."),
    ])
    mm = {}
    for name, opt in mistakes.items():
        if opt and opt != truth:
            mm.setdefault(opt, name)
    return World(
        id=f"lp-alert-{idx:05d}", law="alert_routing", kind="choice", evidence=doc.render(),
        criterion=f"Under {rb} at revision {rev}, which destination does the router choose for alert ALR-{alr}?",
        options=options, expected=truth, mistakes=mm,
        rationale=f"source={source} sev={sev} window_on_component={window_comp == comp} postponed={postponed} -> {truth}.",
        params={"source": source, "severity": sev, "window_on_component": window_comp == comp, "postponed": postponed})


# ---------------------------------------------------------------------------
# Law 4 — payment-relief eligibility on a defined income
# ---------------------------------------------------------------------------
def law_relief_eligibility(rng: random.Random, idx: int) -> World:
    lender = f"{rng.choice(['Brackford', 'Ellsmere', 'Wexcombe', 'Tarrow', 'Kingsmead'])} Building Society"
    P = f"RLF-{rng.randint(2, 9)}"
    hardship = dt.date(rng.randint(2026, 2027), rng.randint(3, 12), 1)
    base_g = rng.randint(38, 72) * 100                  # regular gross per month
    bonus = rng.randint(15, 45) * 100
    bonus_offset = rng.randint(1, 6)                      # months before hardship month in which the bonus was paid
    target = rng.choice([0.22, 0.235, 0.245, 0.25, 0.25, 0.26, 0.28])
    cur = round(base_g * (1 - target) / 10) * 10
    cur2 = cur + rng.choice([-40, -20, 0, 20, 40])
    prior_plan_months = rng.choice([None, None, 9, 14, 20])
    months = [(hardship.replace(day=1) - dt.timedelta(days=1)).replace(day=1)]
    for _ in range(7):
        months.append((months[-1] - dt.timedelta(days=1)).replace(day=1))
    months = months[::-1]                                  # 8 months before the hardship month, oldest first
    rows = []
    for i, m in enumerate(months):
        back = len(months) - i                             # 1 = the month immediately before hardship
        g = base_g + rng.choice([-60, -30, 0, 0, 30, 60])
        one_off = bonus if back == bonus_offset else 0
        rows.append((m, g, one_off, round((g + one_off) * 0.77)))
    cur_months = [hardship + dt.timedelta(days=rng.randint(60, 120))]
    cur_months = [(cur_months[0].replace(day=1) - dt.timedelta(days=1)).replace(day=1), cur_months[0].replace(day=1)]
    cur_rows = [(cur_months[0], cur, 0, round(cur * 0.77)), (cur_months[1], cur2, 0, round(cur2 * 0.77))]

    def reduction(window: int, *, include_one_off: bool) -> float:
        base_rows = rows[-window:]
        baseline = sum(g + (o if include_one_off else 0) for _, g, o, _ in base_rows) / window
        current = (cur + cur2) / 2
        return 1 - current / baseline

    red = reduction(6, include_one_off=False)
    ratio_ok = red >= 0.25 - 1e-12
    plan_ok = prior_plan_months is None or prior_plan_months >= 12
    truth = ratio_ok and plan_ok
    mistakes = {
        "bonus_included_in_baseline": (reduction(6, include_one_off=True) >= 0.25) and plan_ok,
        "unamended_three_month_baseline": (reduction(3, include_one_off=False) >= 0.25) and plan_ok,
        "strict_comparison": (red > 0.25 + 1e-12) and plan_ok,
        "prior_plan_condition_ignored": ratio_ok,
    }
    owner, desk = "the Head of Collections", "Collections Desk"
    app = f"RA-{hardship.year}-{rng.randint(10000, 99999)}"
    cust = person(rng)
    doc = Doc()
    doc.title(f"{lender} — Collections & Recoveries — Payment Relief Policy {P} with amendment {P}/M1, and the file for application {app}")
    doc.part(f"Payment relief policy {P} (version {rng.randint(2, 5)})")
    doc.section("1. Purpose")
    doc.para(f"{P} sets the conditions on which {lender} offers a Payment Relief Plan, under which the monthly payment is reduced for at most six months and the shortfall is added to the balance, to residential mortgage customers whose income has fallen. Customers who do not qualify are offered the standard forbearance options instead.")
    doc.section("2. Definitions")
    doc.clause("2.1", "\"Hardship Month\" — the calendar month in which, according to the application, the fall in income began.")
    doc.clause("2.2", "\"Gross Monthly Income\" — the customer's regular pay for a calendar month before tax and other deductions. It excludes one-off payments: bonuses, back pay, arrears of pay, redundancy or ex gratia payments, and any payment the payslip marks as non-recurring. Net pay is never used.")
    doc.clause("2.3", "\"Baseline Income\" — the average Gross Monthly Income over the Baseline Period.")
    doc.clause("2.4", "\"Baseline Period\" — the six complete calendar months that immediately precede the Hardship Month. [Wording per amendment M1; before it the clause said three complete calendar months.]")
    doc.clause("2.5", "\"Current Income\" — Gross Monthly Income averaged across the last two complete calendar months preceding the application.")
    doc.clause("2.6", "\"Income Reduction\" — one minus Current Income divided by Baseline Income, expressed as a percentage.")
    doc.clause("2.7", "\"Plan\" — a Payment Relief Plan under this policy or its predecessor; a standard forbearance arrangement is not a Plan.")
    doc.section("3. Eligibility")
    doc.clause("3.1", "An application is eligible only if every condition below is met on the application date:")
    doc.para("    (a) the Income Reduction is 25% or more (exactly 25% qualifies);")
    doc.para("    (b) no Plan was in force at any time in the twelve months before the application date;")
    doc.para("    (c) the account is not the subject of litigation or a possession order;")
    doc.para("    (d) the customer has supplied payslips or equivalent evidence for every month in the Baseline Period and the Current Income months.")
    doc.clause("3.2", "A customer who does not meet 3.1 is not refused help: the handler offers the forbearance options in the Arrears Policy.")
    doc.section("4. Assessment")
    doc.clause("4.1", "The handler computes Baseline Income and Current Income from the payslips, applying the definitions in section 2, records the arithmetic on the worksheet and then tests each condition in 3.1.")
    doc.clause("4.2", "Where the payslips show both gross and net figures, only the gross figures enter the computation. Where a payslip itemises a one-off payment, that item is removed before averaging.")
    doc.clause("4.3", "Rounding: the Income Reduction is computed to one decimal place; a result of 24.95% or above rounds to 25.0% and qualifies.")
    procedure(rng, doc, "5", P, desk, 5)
    general_provisions(rng, doc, "6", P, owner, desk, lender, 8)
    doc.part(f"Amendment {P}/M1")
    doc.para(f"Effective for applications received on or after {longdate(hardship - dt.timedelta(days=200))}. Replaces the Baseline Period in clause 2.4: six full calendar months instead of three, because three months produced results driven by a single unusual month. Nothing else changes.")
    doc.part(f"Application file {app}")
    doc.para(f"Customer: {cust}. Account: {ref(rng, 'MTG')}. Application received: {longdate(cur_months[1] + dt.timedelta(days=rng.randint(32, 45)))}.")
    doc.para(f"Customer's statement: \"My hours were cut from {longdate(hardship)}; the drop started that month.\" Hardship Month: {hardship.strftime('%B %Y')}.")
    doc.section("Payslip summary (gross regular pay / one-off items / net pay)")
    for m, g, o, net in rows + cur_rows:
        doc.para(f"    {m.strftime('%B %Y')}: gross regular {fmt('GBP', g)}" + (f"; one-off: annual bonus {fmt('GBP', o)} (marked non-recurring)" if o else "; one-off: none") + f"; net pay {fmt('GBP', net)}")
    doc.para("Payslips supplied for every month listed. Account status: no litigation, no possession proceedings.")
    doc.section("Plan history")
    doc.para("    " + (f"Payment Relief Plan {ref(rng, 'PLAN')}: in force until {longdate((hardship.replace(day=1) - dt.timedelta(days=30 * prior_plan_months)).replace(day=1))} ({prior_plan_months} months before the application)." if prior_plan_months else "No previous Plan. One standard forbearance arrangement (payment holiday) three years ago."))
    doc.section("Handler's worksheet (draft)")
    inc = reduction(6, include_one_off=True)
    doc.para(f"    Baseline (six months, gross incl. all items): {fmt('GBP', sum(g + o for _, g, o, _ in rows[-6:]) / 6)}; current {fmt('GBP', (cur + cur2) / 2)}; reduction {inc * 100:.1f}% — " + ("qualifies on (a)." if inc >= 0.25 else "fails (a)."))
    options, expected = yes_no(truth, f"Each condition of clause 3.1, read with amendment M1, is satisfied: the application qualifies for a Plan.",
                               "One or more conditions of clause 3.1 fail: the application does not qualify for a Plan.")
    mm = {}
    for name, ok in mistakes.items():
        if ok != truth:
            mm.setdefault("yes" if ok else "no", name)
    if not mm and rng.random() < 0.6:
        return law_relief_eligibility(rng, idx)   # most files should be ones where a named mistake flips the verdict
    return World(
        id=f"lp-relief-{idx:05d}", law="relief_eligibility", kind="noul", evidence=doc.render(),
        criterion=f"Is application {app} eligible for a Payment Relief Plan under {P} together with amendment {P}/M1?",
        options=options, expected=expected, mistakes=mm,
        rationale=f"reduction (6-month gross, one-offs excluded) = {red * 100:.1f}% -> {ratio_ok}; prior plan {prior_plan_months} months -> {plan_ok}; eligible={truth}.",
        params={"reduction": round(red, 4), "reduction_with_bonus": round(reduction(6, include_one_off=True), 4),
                "reduction_3m": round(reduction(3, include_one_off=False), 4), "prior_plan_months": prior_plan_months})


# ---------------------------------------------------------------------------
# Law 5 — appeal admissibility on working days with closures
# ---------------------------------------------------------------------------
def law_appeal_deadline(rng: random.Random, idx: int) -> World:
    uni = f"{rng.choice(['Harwood', 'Calderbank', 'Ashworth', 'Merton Vale', 'Stanlow'])} Metropolitan University"
    reg = "AAC"
    year = rng.randint(2026, 2027)
    old_limit, new_limit = 10, 15
    commence = dt.date(year, 11, 1)
    release = dt.date(year, rng.choice([11, 12, 12, 12]), rng.randint(2, 20))
    while release.weekday() >= 5:
        release += dt.timedelta(days=1)
    letter = release + dt.timedelta(days=rng.choice([-3, 2, 4]))
    holidays = {dt.date(year, 12, 25), dt.date(year, 12, 26), dt.date(year + 1, 1, 1)}
    closure = {dt.date(year, 12, 24) + dt.timedelta(days=i) for i in range(9)}       # 24 Dec - 1 Jan
    off = holidays | closure
    limit = new_limit if release >= commence else old_limit
    deadline = working_days_after(release, limit, off)
    lodged = deadline + dt.timedelta(days=rng.choice([-3, -1, 0, 0, 1, 2, 4]))
    ground = rng.choice(["procedural", "procedural", "judgement", "new_evidence"])
    ground_ok = ground != "judgement"
    on_time = lodged <= deadline
    truth = on_time and ground_ok
    mistakes = {
        "calendar_days_counted": (lodged <= release + dt.timedelta(days=limit)) and ground_ok,
        "unamended_ten_day_limit": (lodged <= working_days_after(release, old_limit, off)) and ground_ok,
        "counted_from_the_letter": (lodged <= working_days_after(letter, limit, off)) and ground_ok,
        "closure_days_ignored": (lodged <= working_days_after(release, limit, holidays)) and ground_ok,
        "ground_not_checked": on_time,
    }
    owner, desk = "the Academic Registrar", "Appeals Office"
    student = person(rng)
    apl = f"APL-{year + 1}-{rng.randint(1000, 9999)}"
    doc = Doc()
    doc.title(f"{uni} — Academic Registry, Appeals Office — admissibility screening: appeal {apl}")
    doc.para(f"This file holds: (1) the Academic Appeals Code ({reg}), {year}/{str(year + 1)[2:]} edition, with Senate Resolution S/{year}/{rng.randint(20, 60)} amending it; (2) an Academic Calendar extract; (3) the student record extract; (4) the appeal as submitted, with correspondence.")
    doc.part(f"(1) Academic Appeals Code, {year}/{str(year + 1)[2:]} edition — extract")
    doc.section("1. Definitions")
    doc.clause("1.1", "\"Result\" — a mark, grade or progression decision confirmed by a Board of Examiners.")
    doc.clause("1.2", "\"Publication\" — the moment a Result is released to the student in the results portal. A letter, transcript or email that repeats the Result later, or announces it earlier, is not the Publication.")
    doc.clause("1.3", "\"Working Day\" — a day on which the University is open for business: not a Saturday or Sunday, not a public holiday, and not a University Closure Day listed in the Academic Calendar.")
    doc.clause("1.4", "\"Lodged\" — an appeal is lodged on the day the completed appeal form is received by the Appeals Office through the appeals portal.")
    doc.section("2. Grounds")
    doc.clause("2.1", "The permitted grounds are these, and no others:")
    doc.para("    (a) a procedural irregularity in the assessment or examination process;")
    doc.para("    (b) new evidence that the student could not reasonably have produced before the Board met;")
    doc.para("    (c) bias or the reasonable appearance of bias on the part of an examiner.")
    doc.clause("2.2", "A challenge to the examiners' academic judgement is, on its own, outside every permitted ground. An appeal resting on that alone is inadmissible however promptly it is lodged.")
    doc.section("3. Time limit")
    doc.clause("3.1", f"The time allowed for lodging an appeal is {old_limit} Working Days from the Publication of the Result. [Amended: see the Senate Resolution below.]")
    doc.clause("3.2", "The count begins on the first Working Day after the Publication; the day of Publication is not counted. An appeal Lodged on the last day of the period is in time.")
    doc.clause("3.3", "The Appeals Office may accept a late appeal only where the student shows good cause outside their control; the screening officer records the reason. Pressure of other deadlines is not good cause.")
    doc.section("4. Screening")
    doc.clause("4.1", "Within five Working Days of lodging, the screening officer decides admissibility: whether the appeal was Lodged in time and whether at least one stated ground falls within clause 2.1.")
    doc.clause("4.2", "An appeal admissible on one ground proceeds on that ground only; inadmissible grounds are struck out.")
    doc.clause("4.3", "A screening decision names the Publication date, the last Working Day of the period and the grounds accepted or struck out.")
    procedure(rng, doc, "5", reg, desk, 6)
    general_provisions(rng, doc, "6", reg, owner, desk, uni, 9)
    doc.part(f"Senate Resolution S/{year}/{rng.randint(20, 60)}")
    doc.para(f"R1. Clause 3.1 is amended to read \"{new_limit} Working Days\". R2. The amendment commences on {longdate(commence)}. R3. It applies to every appeal against a Result whose Publication is on or after the commencement date; a Result published before that date remains subject to the {old_limit}-Working-Day limit, whenever the appeal is lodged.")
    doc.part("(2) Academic Calendar extract")
    doc.para(f"Public holidays: {', '.join(longdate(d) for d in sorted(holidays))}.")
    doc.para(f"University Closure Days: {longdate(min(closure))} to {longdate(max(closure))} inclusive (the University is closed; these are not Working Days).")
    doc.para("Term dates and examination periods omitted.")
    doc.part("(3) Student record extract")
    doc.para(f"Student: {student}, {ref(rng, 'STU')}. Programme: BSc. Module {rng.choice(['ECON', 'PHYS', 'HIST', 'CHEM'])}{rng.randint(200, 399)}: Result confirmed by the Board of Examiners on {longdate(release - dt.timedelta(days=rng.randint(3, 9)))}.")
    doc.para(f"Results portal log: Result released to the student on {dayname(release)} {longdate(release)} at {rng.randint(9, 16)}:{rng.randint(10, 59):02d}.")
    doc.para(f"Results letter: dated {longdate(letter)}, posted second class.")
    doc.part("(4) Appeal submission")
    doc.para(f"Appeals portal: completed form received {dayname(lodged)} {longdate(lodged)} at {rng.randint(8, 17)}:{rng.randint(10, 59):02d}.")
    doc.para("Ground stated by the student: " + {
        "procedural": "\"The examination paper contained a question on material that was withdrawn from the syllabus in week 4, and the invigilators refused to issue the corrected paper — a procedural irregularity.\"",
        "judgement": "\"I believe the mark is too low for the quality of my answers and ask for it to be raised.\"",
        "new_evidence": "\"A medical report dated after the Board met shows I was unwell during the examination; I could not have obtained it earlier.\"",
    }[ground])
    doc.section("Screening officer's draft note")
    doc.para(f"    \"Letter dated {longdate(letter)}; counting {limit} days from there the deadline is {longdate(letter + dt.timedelta(days=limit))} — " + ("in time." if lodged <= letter + dt.timedelta(days=limit) else "out of time.") + "\"")
    options, expected = yes_no(truth, "Lodged within the time allowed, with at least one stated ground inside clause 2.1: the appeal passes screening, at least in part.",
                               "The appeal fails screening: lodged after the time allowed, or no stated ground inside clause 2.1.")
    mm = {}
    for name, ok in mistakes.items():
        if ok != truth:
            mm.setdefault("yes" if ok else "no", name)
    if not mm and rng.random() < 0.6:
        return law_appeal_deadline(rng, idx)      # most files should be ones where a named mistake flips the verdict
    return World(
        id=f"lp-appeal-{idx:05d}", law="appeal_deadline", kind="noul", evidence=doc.render(),
        criterion=f"Does appeal {apl} pass admissibility screening — lodged in time, and on at least one permitted ground — under the {reg} as amended?",
        options=options, expected=expected, mistakes=mm,
        rationale=f"Publication {release}; limit {limit} working days (commencement {commence}); deadline {deadline}; lodged {lodged}; ground {ground} -> {truth}.",
        params={"release": release.isoformat(), "limit": limit, "deadline": deadline.isoformat(), "lodged": lodged.isoformat(), "ground": ground})


# ---------------------------------------------------------------------------
# Law 6 — commercial returns decision
# ---------------------------------------------------------------------------
def law_returns_decision(rng: random.Random, idx: int) -> World:
    C = f"{rng.choice(['Corvid', 'Halden', 'Marrable', 'Ostrova', 'Penrose'])} Industrial Controls"
    P = f"RET-{rng.randint(4, 9)}"
    invoice = dt.date(rng.randint(2026, 2027), rng.randint(1, 12), rng.randint(2, 24))
    delivery = invoice + dt.timedelta(days=rng.choice([-3, 2, 5, 8]))
    window = rng.choice([30, 45])
    request = delivery + dt.timedelta(days=window + rng.choice([-9, -3, -1, 0, 1, 3, 7]))
    draft = request - dt.timedelta(days=rng.randint(2, 6))
    configured = rng.random() < 0.35            # true configured product: an X- option code
    configurator_used = True
    defect = rng.random() < 0.3
    tier_now = rng.choice(["silver", "silver", "bronze", "gold"])
    tier_before = "gold" if tier_now != "gold" else "silver"
    fee_pct = {"gold": 0, "silver": 15, "bronze": 20}[tier_now]
    in_window = request <= delivery + dt.timedelta(days=window)
    if defect:
        truth = "repair_or_replace_only"
    elif configured or not in_window:
        truth = "reject_return"
    else:
        truth = "full_credit" if fee_pct == 0 else "credit_minus_restocking_fee"

    def decide(*, conf: bool, in_win: bool, fee: int, dfct: bool) -> str:
        if dfct:
            return "repair_or_replace_only"
        if conf or not in_win:
            return "reject_return"
        return "full_credit" if fee == 0 else "credit_minus_restocking_fee"

    mistakes = {
        "window_counted_from_invoice_date": decide(conf=configured, in_win=request <= invoice + dt.timedelta(days=window), fee=fee_pct, dfct=defect),
        "superseded_classification_rule": decide(conf=configurator_used, in_win=in_window, fee=fee_pct, dfct=defect),
        "tier_taken_from_history": decide(conf=configured, in_win=in_window, fee={"gold": 0, "silver": 15, "bronze": 20}[tier_before], dfct=defect),
        "draft_date_used": decide(conf=configured, in_win=draft <= delivery + dt.timedelta(days=window), fee=fee_pct, dfct=defect),
        "customer_fault_claim_believed": "repair_or_replace_only" if not defect else None,
    }
    owner, desk = "the Customer Service Manager", "Returns Desk"
    rq = f"RQ-{request.year}-{rng.randint(1000, 9999)}"
    cust = firm(rng)
    model = f"{rng.choice(['CX', 'MZ', 'TK'])}-{rng.randint(200, 900)}"
    doc = Doc()
    doc.title(f"{C} GmbH — Customer Service / Returns Desk — return authorisation assessment, RMA request {rq}")
    doc.para(f"Pack: (1) Commercial Returns Policy {P} with Amendment 1; (2) an extract from the superseded policy; (3) the Partner Programme tier rules (extract); (4) the order, the delivery record and the invoice; (5) the return request with the inspection findings; (6) the account's tier history.")
    doc.part(f"(1) Commercial Returns Policy {P}")
    doc.section("1. Scope")
    doc.clause("1.1", f"{P} covers product returns from business customers who bought from {C} GmbH on its standard terms. Consumer purchases are outside it.")
    doc.section("2. Definitions")
    doc.clause("2.1", "\"Delivery Date\" — the date on which the carrier's proof of delivery is signed at the customer's address. The invoice date, the dispatch date and the order date are not the Delivery Date.")
    doc.clause("2.2", "\"Standard Product\" — a product ordered by catalogue number, with or without catalogue option codes. [Amended: see Amendment 1.]")
    doc.clause("2.3", "\"Configured Product\" — a product built to the customer's specification. [Amended: see Amendment 1.]")
    doc.clause("2.4", "\"RMA Request Date\" — the date on which a complete return request (product, serial numbers, reason, invoice reference) is received by the Returns Desk. An incomplete draft is not a request.")
    doc.clause("2.5", "\"Restocking Fee\" — the percentage of the invoiced net value deducted from a credit, at the rate set by the customer's Partner Programme tier on the RMA Request Date.")
    doc.section("3. Returns of non-defective products")
    doc.clause("3.1", f"A Standard Product qualifies for a credit return when the RMA Request Date falls within {window} days after the Delivery Date and the product is unused and complete.")
    doc.clause("3.2", "A Configured Product is not returnable for credit. A request to return one is rejected unless section 5 applies.")
    doc.clause("3.3", "A return under 3.1 is credited at the invoiced net value less the Restocking Fee. A tier with a Restocking Fee of zero receives full credit.")
    doc.clause("3.4", f"A Standard Product returned after the {window}-day period is not accepted for credit.")
    doc.section("4. Inspection")
    doc.clause("4.1", "Every returned product is inspected. A report of \"no fault found\" means the product is not defective for the purposes of section 5, whatever the customer's request states.")
    doc.section("5. Defective products")
    doc.clause("5.1", "A product that inspection confirms to be defective within the 12-month warranty is repaired or replaced at our option. No credit is issued for a defective product; sections 3.1 to 3.4 do not apply to it.")
    doc.clause("5.2", "A request that claims a defect but whose inspection reports no fault found is handled under section 3, using the RMA Request Date of the original request.")
    procedure(rng, doc, "6", P, desk, 5)
    general_provisions(rng, doc, "7", P, owner, desk, f"{C} GmbH", 8)
    doc.part(f"Amendment 1 to {P}")
    doc.para(f"Effective {longdate(invoice - dt.timedelta(days=rng.randint(60, 300)))}. Clause 2.2 now reads: \"Standard Product — a product ordered by catalogue number with or without catalogue option codes, whether ordered by email, by purchase order or through the online configurator.\" Clause 2.3 now reads: \"Configured Product — a product whose order carries at least one X- option code, denoting customer-specific engineering.\" The use of the online configurator no longer makes a product a Configured Product.")
    doc.part(f"(2) Extract from the superseded policy {P[:-1]}{int(P[-1]) - 1} (retained for reference only)")
    doc.para("\"Configured Product — any product ordered through the online configurator or carrying an option code.\" This definition ceased to apply on the effective date of Amendment 1 above.")
    doc.part("(3) Partner Programme tier rules (extract)")
    doc.para("Restocking Fee by tier: Gold 0%; Silver 15%; Bronze 20%. A tier is assessed quarterly and applies from the first day of the quarter following the assessment. The tier in force on the RMA Request Date governs a return, regardless of the tier when the goods were bought.")
    doc.part("(4) The order, the delivery record and the invoice")
    codes = "option codes 04, 11" + (", X-CC7" if configured else "")
    doc.para(f"Order {ref(rng, 'ORD')} placed through the online configurator by {cust}: {rng.randint(2, 6)} x {model} controller, catalogue number {model}-{rng.randint(10, 99)}, {codes}.")
    doc.para(f"Invoice {ref(rng, 'INV')} dated {longdate(invoice)}, net value {eur(rng.randint(4, 30) * 1000)}. Dispatched {longdate(min(invoice, delivery) - dt.timedelta(days=1))}. Proof of delivery signed at the customer's address on {longdate(delivery)}.")
    doc.part(f"(5) Return request {rq} and the inspection findings")
    doc.para(f"Draft return request opened in the portal on {longdate(draft)} (serial numbers missing; not submitted). Complete request received {longdate(request)}. Reason stated: \"" + ("units fail to power on" if defect or rng.random() < 0.5 else "over-ordered; units unused in original packaging") + "\".")
    doc.para("Inspection report: " + ("defect confirmed — power supply board failure, within warranty." if defect else "no fault found; units power on and pass the functional test; packaging unopened on three units."))
    doc.part("(6) Tier history for the account")
    q_start = (request.replace(day=1) - dt.timedelta(days=rng.randint(20, 80))).replace(day=1)
    doc.para(f"Partner tier: {tier_before.capitalize()} until {longdate(q_start - dt.timedelta(days=1))}; {tier_now.capitalize()} from {longdate(q_start)} (quarterly assessment). The order was placed while the account was {tier_before.capitalize()}.")
    doc.section("Desk pre-assessment (draft)")
    doc.para(f"    \"Invoice {longdate(invoice)}; request {longdate(request)} — " + ("inside" if request <= invoice + dt.timedelta(days=window) else "outside") + f" the {window} days. Configurator order, so treat as configured. Tier on file: {tier_before.capitalize()}.\"")
    options = shuffled(rng, [
        ("credit_minus_restocking_fee", "Return accepted; credit note for the invoiced net value minus the Restocking Fee of the customer's tier."),
        ("full_credit", "Return accepted; credit note for the whole invoiced net value."),
        ("reject_return", "Return refused: no credit note, no repair, no replacement."),
        ("repair_or_replace_only", "No credit note; the units go to repair or replacement under warranty."),
    ])
    mm = {}
    for name, opt in mistakes.items():
        if opt and opt != truth:
            mm.setdefault(opt, name)
    return World(
        id=f"lp-returns-{idx:05d}", law="returns_decision", kind="choice", evidence=doc.render(),
        criterion=f"What decision must the Returns Desk issue on {rq} under {P} as amended?",
        options=options, expected=truth, mistakes=mm,
        rationale=f"delivery {delivery}, request {request} ({'in' if in_window else 'out of'} window of {window}); configured={configured}; defect={defect}; tier {tier_now} fee {fee_pct}% -> {truth}.",
        params={"delivery": delivery.isoformat(), "invoice": invoice.isoformat(), "request": request.isoformat(), "window": window,
                "configured": configured, "defect": defect, "tier": tier_now})


LAWS = {
    "approval_level": law_approval_level,
    "water_damage_claim": law_water_damage,
    "alert_routing": law_alert_routing,
    "relief_eligibility": law_relief_eligibility,
    "appeal_deadline": law_appeal_deadline,
    "returns_decision": law_returns_decision,
}


def generate(n: int, seed: int = 0, laws: list[str] | None = None) -> list[World]:
    """n worlds, round-robin over the laws, reproducible from the seed."""
    return _generate(LAWS, FAMILY, n, seed, laws)


def main(argv=None) -> int:
    return run_cli(LAWS, FAMILY, __doc__, argv)


if __name__ == "__main__":
    sys.exit(main())

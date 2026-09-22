# -*- coding: utf-8 -*-
"""Exact-law worlds for the multi_hop family.

Identifier discipline. A root record names two identifiers; each resolves
through a directory, an inventory, a class table, a control register, a
revision log and a status board — six lookups — and only the record that the
exact chain reaches supplies the rule that decides the case. The file also
contains, at every hop, the things that defeat a surface reader: a near-miss
code (a letter O for a zero, a trailing -R or -X), a pencil annotation that
was never issued, a reversed cross-reference, a cancelled line, a superseded
revision and a draft one. Every one of those paths ends at a different
disposition, and the record names which mistake reaches it.

Three laws: the identifier chain (five domain skins), invoice approval routing
through alias, risk override, FX and aggregation, and on-call paging through
process alias, tier-adjusted severity, a team merger and a time-zone handover.

  python -m jobe.worlds.multi_hop --n 600 --seed 0 --out multi_hop.jsonl
  python -m jobe.worlds.multi_hop --n 120 --check-contamination <jevbench clone>
"""
from __future__ import annotations

import datetime as dt
import random
import sys
from decimal import Decimal

from .base import (World, check, contamination, dayname, firm, fmt, generate as _generate, longdate,  # noqa: F401
                   money, person, ref, run_cli, to_decision)
from .long_policy import Doc, general_provisions, procedure, shuffled

FAMILY = "multi_hop"


# ---------------------------------------------------------------------------
# Law 1 — the identifier chain, in five skins
# ---------------------------------------------------------------------------
SKINS = {
    "grant": dict(
        binder="grant finance workbook", root_kind="Draw request", root_prefix="DRAW",
        a=("award", "GF", "AWARD REGISTER", ["Coastal Sensors programme", "Upland Hydrology programme", "Harbour Birds programme", "Peat Restoration programme"]),
        b=("ledger object", "LO", "LEDGER DIRECTORY", ["field equipment", "survey vehicles", "laboratory consumables", "volunteer training"]),
        c=("budget row", "E", "BUDGET MAP", None),
        w=("amendment", "AM", "AMENDMENT REGISTER", None),
        v=("board minute", "BM", "MINUTE LOG", None),
        quantity=("posted spending on the row", "the amount requested by the draw", "EUR"),
        cap_word="ceiling", tol_word="the tolerance",
        dispositions=[("release_draw", "Release the draw in full."), ("partial_release", "Release the draw up to the ceiling only."),
                      ("programme_review", "Refer the draw to programme review."), ("reject_draw", "Reject the draw.")],
        frozen_word="frozen",
    ),
    "lot": dict(
        binder="assembly deviation book", root_kind="Production lot", root_prefix="LOT",
        a=("bill code", "BOM", "BILL DIRECTORY", ["valve assembly VA-2", "manifold assembly MA-7", "sensor module SM-3", "housing set HS-9"]),
        b=("machine cell", "MC", "CELL REGISTER", ["press line East", "press line West", "finishing bay", "clean room"]),
        c=("inspection family", "IF", "INSPECTION MAP", None),
        w=("control plan", "CP", "CONTROL PLAN REGISTER", None),
        v=("plan revision", "REV", "REVISION LOG", None),
        quantity=("the measured deviation", "the gauge allowance", "µm"),
        cap_word="limit", tol_word="the rework band",
        dispositions=[("normal_release", "Release the lot normally."), ("rework", "Send the lot to rework."),
                      ("scrap_lot", "Scrap the lot."), ("engineering_review", "Hold the lot for engineering review.")],
        frozen_word="on hold",
    ),
    "claim": dict(
        binder="claims coverage binder", root_kind="Claim", root_prefix="CLM",
        a=("policy token", "NP", "POLICY DECODER", ["policy household_plus", "policy household_basic", "policy landlord_std", "policy travel_annual"]),
        b=("asset schedule item", "AS", "ASSET SCHEDULE", ["portable camera", "road bicycle", "laptop computer", "hearing aid"]),
        c=("property class", "P", "CLASS TABLE", None),
        w=("endorsement", "EN", "ENDORSEMENT REGISTER", None),
        v=("endorsement version", "VER", "VERSION LOG", None),
        quantity=("the amount claimed", "the claims already paid in the policy year", "EUR"),
        cap_word="limit", tol_word="the referral band",
        dispositions=[("coverage_confirmed", "Coverage confirmed; pay the claim."), ("adjuster_referral", "Refer the claim to an adjuster."),
                      ("coverage_excluded", "Coverage excluded; decline the claim."), ("documentation_needed", "Hold the claim for documentation.")],
        frozen_word="lapsed",
    ),
    "timetable": dict(
        binder="working timetable desk copy", root_kind="Train", root_prefix="MV",
        a=("origin code", "OAK", "LOCATION CODEBOOK", ["Oakmere", "Oakfield", "Oak Halt", "Oakridge"]),
        b=("consist class", "C", "CONSIST TABLE", ["five-car regional", "three-car shuttle", "eight-car express", "engineering train"]),
        c=("path column", "P", "PATH TABLE", None),
        w=("engineering circular", "EC", "CIRCULAR REGISTER", None),
        v=("circular issue", "ISS", "ISSUE LOG", None),
        quantity=("the consist length", "the buffer allowance", "metres"),
        cap_word="platform limit", tol_word="the dispensation band",
        dispositions=[("call_fen_cross", "Call at Fen Cross as booked."), ("pass_fen_cross", "Run through Fen Cross without calling."),
                      ("bus_substitution", "Bus substitution from Oakmere."), ("terminate_oakmere", "Terminate at Oakmere.")],
        frozen_word="withdrawn",
    ),
    "archive": dict(
        binder="archives desk file", root_kind="Request", root_prefix="REQ",
        a=("collection call", "COL", "FINDING AID", ["Marlow papers", "Harding letters", "Tern Society minutes", "Quarry survey plates"]),
        b=("unit", "U", "UNIT INVENTORY", ["correspondence 1952", "ledgers 1931", "photographs 1968", "maps 1904"]),
        c=("handling tier", "T", "TIER TABLE", None),
        w=("handling route", "HR", "ROUTE REGISTER", None),
        v=("handling notice", "HN", "NOTICE LOG", None),
        quantity=("the folios requested", "the folios already issued today", "folios"),
        cap_word="daily limit", tol_word="the supervision band",
        dispositions=[("supervised_delivery", "Deliver the unit under supervision."), ("digital_copy_only", "Supply a digital copy only."),
                      ("deny_request", "Deny the request."), ("conservation_hold", "Place the unit on conservation hold.")],
        frozen_word="on conservation hold",
    ),
}


def _near_miss(code: str, rng: random.Random) -> str:
    if "0" in code and rng.random() < 0.5:
        i = code.rindex("0")
        return code[:i] + "O" + code[i + 1:]
    return code + rng.choice(["-R", "-X", "A"])


def law_identifier_chain(rng: random.Random, idx: int) -> World:
    skin_name = rng.choice(list(SKINS))
    sk = SKINS[skin_name]
    a_word, a_pre, a_reg, a_names = sk["a"]
    b_word, b_pre, b_reg, b_names = sk["b"]
    c_word, c_pre, c_reg, _ = sk["c"]
    w_word, w_pre, w_reg, _ = sk["w"]
    v_word, v_pre, v_reg, _ = sk["v"]
    q1_word, q2_word, unit = sk["quantity"]
    dispo = sk["dispositions"]
    ids = [d for d, _ in dispo]
    desk_date = dt.date(rng.randint(2026, 2027), rng.randint(1, 12), rng.randint(2, 27))

    root = f"{sk['root_prefix']}-{rng.randint(100, 999)}"
    a_code = f"{a_pre}-{rng.randint(100, 999)}" if a_pre != "OAK" else "OAK"
    b_code = f"{b_pre}-{rng.randint(10, 99)}"
    a_name, a_alt = rng.sample(a_names, 2)
    b_name, b_alt = rng.sample(b_names, 2)
    c_code, c_alt = f"{c_pre}-{rng.randint(2, 9)}", f"{c_pre}-{rng.randint(10, 19)}"
    w_code, w_alt, w_alt2 = f"{w_pre}-{rng.randint(10, 39)}", f"{w_pre}-{rng.randint(40, 69)}", f"{w_pre}-{rng.randint(70, 99)}"
    rev_names = ["A", "B", "C", "D"]
    cur_i = rng.randint(1, 2)
    v_cur, v_old, v_draft = rev_names[cur_i], rev_names[cur_i - 1], rev_names[cur_i + 1]

    # the operative rule: q1 + q2 against a cap with a tolerance band; a frozen status short-circuits
    cap = rng.randint(20, 90) * 1000 if unit == "EUR" else rng.randint(30, 120)
    tol = rng.choice([10, 15, 20])
    frozen = rng.random() < 0.15
    q1 = round(cap * rng.choice([0.55, 0.7, 0.86, 0.95, 1.02, 1.08, 1.13, 1.3]) * rng.uniform(0.6, 0.8))
    q2 = max(1, round(cap * rng.choice([0.55, 0.7, 0.86, 0.95, 1.02, 1.08, 1.13, 1.3])) - q1)
    total = q1 + q2

    def band(total_, cap_, tol_, frozen_) -> str:
        if frozen_:
            return ids[3]
        if total_ <= cap_:
            return ids[0]
        if total_ <= cap_ * (100 + tol_) / 100:
            return ids[1]
        return ids[2]

    truth = band(total, cap, tol, frozen)

    def rule_for(target: str) -> tuple[int, int, bool]:
        """A (cap, tol, frozen) under which the case's quantities land on `target`."""
        if target == ids[3]:
            return cap, tol, True
        if target == ids[0]:
            return int(total * 1.05) + 1, tol, False
        if target == ids[1]:
            return max(1, int(total / (1 + tol / 200))), tol, False
        return max(1, int(total / (1 + tol / 100 + 0.25))), tol, False

    wrong = [d for d in ids if d != truth]
    rng.shuffle(wrong)
    # four alternative rules: the near-miss record, the record behind the annotation / reversed / cancelled paths,
    # the superseded issue and the draft issue — spread over the three wrong dispositions
    t_alt, t_alt2, t_old, t_draft = wrong[0], wrong[1 % len(wrong)], wrong[2 % len(wrong)], rng.choice(wrong)
    r_alt, r_alt2, r_old, r_draft = rule_for(t_alt), rule_for(t_alt2), rule_for(t_old), rule_for(t_draft)
    for name, target in (("followed_near_miss_identifier", t_alt), ("followed_unissued_annotation", t_alt2),
                         ("followed_reversed_reference", t_alt2), ("followed_cancelled_line", t_alt2),
                         ("used_superseded_revision", t_old), ("used_draft_revision", t_draft)):
        assert band(total, *rule_for(target)) == target, (name, target)

    fq = (lambda x: fmt("EUR", x)) if unit == "EUR" else (lambda x: f"{x} {unit}")
    cap_word, tol_word = sk["cap_word"], sk["tol_word"]

    def rule_text(label: str, c: int, t: int, fz: bool) -> str:
        if fz:
            return f"{label}: the record is {sk['frozen_word']} — every case under it takes the {ids[3].replace('_', ' ')} disposition."
        return (f"{label}: {cap_word} {fq(c)}. When {q1_word} plus {q2_word} is at or under the {cap_word}, the disposition is "
                f"{ids[0].replace('_', ' ')}; when the sum exceeds the {cap_word} by no more than {t} % ({tol_word}), "
                f"{ids[1].replace('_', ' ')}; beyond that, {ids[2].replace('_', ' ')}.")

    # padding identifiers never collide with the live codes, the near-miss, or each other:
    # a register that maps one code two ways would contradict itself
    used = {a_code, b_code}
    nm = _near_miss(a_code, rng)
    used.add(nm)

    def fresh(make) -> str:
        for _ in range(50):
            c = make()
            if c not in used:
                used.add(c)
                return c
        raise AssertionError("identifier pool exhausted")

    def pad_a() -> str:
        if a_pre == "OAK":
            return fresh(lambda: "".join(rng.choice("ABCDEFGHJKLMNPRSTUVWY") for _ in range(3)))
        return fresh(lambda: f"{a_pre}-{rng.randint(100, 999)}")

    def pad_b() -> str:
        return fresh(lambda: f"{b_pre}-{rng.randint(10, 99)}")

    lines: list[str] = []
    P = lines.append
    P(f"{firm(rng)} {sk['binder']}")
    P(f"Desk copy assembled: {longdate(desk_date)}")
    P("")
    P(f"{sk['root_kind'].upper()} RECORD")
    P(f"{sk['root_kind']} {root} cites {a_word} {a_code} and {b_word} {b_code}. Quantities recorded on the record: {q1_word} {fq(q1)}; {q2_word} {fq(q2)}.")
    P(f"Received {longdate(desk_date - dt.timedelta(days=rng.randint(1, 6)))}; logged by {person(rng)}; countersignature pending.")
    P("")
    P(a_reg)
    P(f"{a_code} -> {a_name}; entry current; issued {longdate(desk_date - dt.timedelta(days=rng.randint(40, 400)))}.")
    P(f"{nm} -> {a_alt}; a different subject. Note the identifier: it is not {a_code}. Matching is on the exact string, character for character.")
    P(f"A pencil annotation on this page proposes {a_name}-aux for {a_code}. The approval box beside it is empty and the annotation was never issued; an unissued annotation maps nothing.")
    for _ in range(rng.randint(5, 7)):
        P(f"{pad_a()} -> {rng.choice(a_names)}; entry current; issued {longdate(desk_date - dt.timedelta(days=rng.randint(40, 900)))}.")
    P("")
    P(b_reg)
    P(f"{b_code} -> {b_name}; entry current.")
    P(f"The archive preserves a reversed cross-reference reading '{b_name} -> {b_code}-R', and an archive row '{b_code}-R -> {b_alt}'. A reversed reference records where a lookup came from; it does not replace the forward entry above, and {b_code}-R is not {b_code}.")
    P(f"A cancelled line reads '{b_name}-C -> {c_alt}'. The cancellation stamp is dated {longdate(desk_date - dt.timedelta(days=rng.randint(30, 300)))}; a cancelled line controls no current transaction.")
    for _ in range(rng.randint(5, 7)):
        P(f"{pad_b()} -> {rng.choice(b_names)}; entry current.")
    P("")
    P(c_reg)
    P(f"{b_name} -> {c_word} {c_code}.")
    P(f"{b_alt} -> {c_word} {c_alt}.")
    P(f"{a_name}-aux -> {c_word} {c_alt}.   [reached only through the unissued annotation]")
    for nm_ in b_names:
        if nm_ not in (b_name, b_alt):
            P(f"{nm_} -> {c_word} {c_pre}-{rng.randint(20, 29)}.")
    P("")
    P(w_reg)
    P(f"{a_name} / {c_code} -> {w_word} {w_code}; current.")
    P(f"{a_alt} / {c_code} -> {w_word} {w_alt}; current.")
    P(f"{a_name} / {c_alt} -> {w_word} {w_alt2}; current.")
    P(f"{a_alt} / {c_alt} -> {w_word} {w_alt2}; current.")
    for _ in range(3):
        P(f"{rng.choice(a_names)} / {c_pre}-{rng.randint(20, 29)} -> {w_word} {w_pre}-{rng.randint(100, 140)}; current.")
    P("")
    P(v_reg)
    P(f"{w_code}: issue {v_old} — superseded on {longdate(desk_date - dt.timedelta(days=rng.randint(20, 200)))}; issue {v_cur} — in force on the desk-copy date; issue {v_draft} — draft, circulated for comment, lifecycle not advanced.")
    P(f"{w_alt}: issue {v_cur} — in force.")
    P(f"{w_alt2}: issue {v_cur} — in force.")
    P("")
    P("STATUS BOARD AND RULE TEXT")
    P(rule_text(f"{w_code} issue {v_cur} (in force)", cap, tol, frozen))
    P(rule_text(f"{w_code} issue {v_old} (superseded)", *r_old))
    P(rule_text(f"{w_code} issue {v_draft} (draft)", *r_draft))
    P(rule_text(f"{w_alt} issue {v_cur} (in force; the record for {a_alt})", *r_alt))
    P(rule_text(f"{w_alt2} issue {v_cur} (in force; the record for {c_alt})", *r_alt2))
    P("")
    P("CASE NOTES")
    P(f"A colleague's summary sheet, undated, lists {root} against {w_alt2}. Summary sheets are not sources; the registers above are.")
    P(f"The requester telephoned on {longdate(desk_date - dt.timedelta(days=1))} to ask that the {v_draft} wording be applied 'since it is about to be issued'. Drafts are not in force until their lifecycle says so.")
    P(f"Audit sample {rng.randint(2, 9)} copied the label {a_name} from this file without its identifier; it is evidence that a check happened, not an operative mapping.")
    P("")
    P("HANDLING NOTES")
    P("Live registers, issued amendments and dated case records outrank annotations, drafts, summaries and archive copies. A superseded issue is read for history only. Identifiers match on the exact string; a trailing suffix, a substituted character or an -aux, -R or -C variant denotes a different record. When two registers disagree, the forward entry in the register that owns the identifier governs.")
    P(f"The desk-copy date is {longdate(desk_date)}; 'in force' and 'superseded' are read as of that date. A record with no in-force issue on that date decides nothing.")
    P("")
    P("OTHER OPEN ITEMS ON THE DESK (not part of this case)")
    for _ in range(rng.randint(11, 15)):
        P(f"{sk['root_kind']} {sk['root_prefix']}-{rng.randint(100, 999)} cites {a_word} {pad_a()} and {b_word} {pad_b()}; awaiting {rng.choice(['countersignature', 'a status update', 'the next issue', 'return of the file', 'the requester', 'a second signature'])}.")
    evidence = "\n".join(lines)
    mm = {}
    for name, target in (("followed_near_miss_identifier", t_alt), ("followed_unissued_annotation", t_alt2),
                         ("followed_reversed_reference", t_alt2), ("followed_cancelled_line", t_alt2),
                         ("used_superseded_revision", t_old), ("used_draft_revision", t_draft)):
        mm.setdefault(target, name)
    options = shuffled(rng, dispo)
    return World(
        id=f"mh-chain-{idx:05d}", law="identifier_chain", kind="choice", evidence=evidence,
        criterion=f"Trace the exact identifiers and the material in force on the desk-copy date. What is the required disposition for {root}?",
        options=options, expected=truth, mistakes=mm,
        rationale=f"{a_code} -> {a_name}; {b_code} -> {b_name} -> {c_code}; {a_name}/{c_code} -> {w_code} issue {v_cur}; {q1}+{q2}={total} vs {cap_word} {cap} (tol {tol}%, frozen={frozen}) -> {truth}.",
        params={"skin": skin_name, "q1": q1, "q2": q2, "cap": cap, "tol": tol, "frozen": frozen})


# ---------------------------------------------------------------------------
# Law 2 — invoice approval routing: alias, risk override, FX, aggregation
# ---------------------------------------------------------------------------
def law_invoice_tier(rng: random.Random, idx: int) -> World:
    C = f"{firm(rng)} SE"
    base = rng.choice(["Nordwind", "Halvard", "Osterlund", "Brenner", "Kestrel"])
    kind = rng.choice(["Services", "Systems", "Logistics"])
    names = [f"{base} {kind} AG", f"{base} {kind} GmbH", f"{base} {kind[:-1]} AG", f"{base} {kind} Ltd"]
    vids = [f"V-{rng.randint(1000, 1999)}" for _ in range(4)]
    exact = names[0]
    risks = ["STANDARD", "ELEVATED", "STANDARD", "ELEVATED"]
    rng.shuffle(risks)
    risk_master = dict(zip(vids, risks))
    onboard_years = {v: rng.choice([2018, 2019, 2023, 2024, 2025]) for v in vids}
    approved_countries = ["DE", "FR", "NL", "AT", "BE", "IT", "ES", "DK", "SE", "PL"]
    payee_country = rng.choice(["DE", "NL", "CH", "GB", "NO", "AT"])
    exempt_year = 2021
    override = payee_country not in approved_countries and onboard_years[vids[0]] >= exempt_year
    risk = "ELEVATED" if override else risk_master[vids[0]]
    cur = rng.choice(["CHF", "GBP", "NOK"])
    rate = {"CHF": Decimal(rng.randint(9200, 9700)) / 10000, "GBP": Decimal(rng.randint(8300, 8800)) / 10000, "NOK": Decimal(rng.randint(1120, 1180)) / 100}[cur]
    amt_fx = Decimal(rng.randint(8, 60) * 500)
    inv_date = dt.date(rng.randint(2026, 2027), rng.randint(2, 12), rng.randint(3, 27))
    eur_now = money(amt_fx / rate)
    win = 30
    d_same = rng.choice([8, 17, 26, 34, 52])
    same_amt = Decimal(rng.randint(4, 40) * 500)
    other_amt = Decimal(rng.randint(4, 40) * 500)
    old_amt = Decimal(rng.randint(4, 40) * 500)
    agg = eur_now + (same_amt if d_same <= win else 0)
    thresholds = {"STANDARD": (10000, 50000, 250000), "ELEVATED": (5000, 25000, 100000)}

    def tier(risk_, amount) -> int:
        t1, t2, t3 = thresholds[risk_]
        return 1 if amount <= t1 else 2 if amount <= t2 else 3 if amount <= t3 else 4

    truth = tier(risk, agg)
    wrong_vid = vids[1]
    mistakes = {
        "alias_matched_without_suffix": tier("ELEVATED" if (payee_country not in approved_countries and onboard_years[wrong_vid] >= exempt_year) else risk_master[wrong_vid], agg),
        "country_override_ignored": tier(risk_master[vids[0]], agg),
        "fx_multiplied_instead_of_divided": tier(risk, money(amt_fx * rate) + (same_amt if d_same <= win else 0)),
        "aggregation_ignored" if d_same <= win else "stale_invoice_aggregated": tier(risk, eur_now) if d_same <= win else tier(risk, agg + same_amt),
        "other_vendor_aggregated": tier(risk, agg + other_amt),
        "old_invoice_aggregated": tier(risk, agg + old_amt),
    }
    tier_ids = ["tier_1_supervisor", "tier_2_controller", "tier_3_finance_director", "tier_4_audit_committee"]
    tier_names = ["Tier 1 (accounts payable supervisor)", "Tier 2 (financial controller)", "Tier 3 (finance director)", "Tier 4 (audit committee)"]
    inv = f"IN-{inv_date.year}-{rng.randint(100, 999)}"
    owner, desk = "the Head of Accounts Payable", "AP Desk"
    doc = Doc()
    doc.title(f"{C} — Accounts Payable Manual, chapter 7 \"Routing supplier invoices for approval\" (rev. {inv_date.year}-{rng.randint(1, 9):02d})")
    doc.section("7.1 Purpose")
    doc.para("Every supplier invoice is approved at the right tier before it is paid. The tier follows from (a) the vendor's risk grade and (b) the amount to be approved in EUR after aggregation (7.5). Getting the vendor wrong gets everything after it wrong.")
    doc.section("7.2 Identifying the vendor")
    doc.para("An invoice is tied to its vendor master record by looking the printed trading name up in Annex A. Some vendors trade under names that differ only in the legal-form suffix or by a single letter; the match is on the exact trading name printed on the invoice, suffix included. The vendor ID found in Annex A is carried into every later step; the vendor named in the purchase order or the email thread is not used.")
    doc.section("7.3 Risk grade")
    doc.para(f"(a) The starting grade is the master grade in Annex B. (b) Override: where the payee bank account on the invoice is held in a country outside Annex D, the grade for this invoice is ELEVATED whatever the master says — unless the vendor was onboarded before {exempt_year}, in which case the master grade stands. (c) A grade is never lowered by this section.")
    doc.section("7.4 Amount in EUR")
    doc.para("An invoice in a foreign currency is converted using Annex C's treasury rate for the invoice date; Annex C states each rate as units of the foreign currency per one EUR, so the EUR amount is the invoice amount divided by the rate. Round to the cent once.")
    doc.section("7.5 Aggregation")
    doc.para(f"The amount to be approved is the invoice's EUR amount plus the EUR amount of every invoice from the same vendor ID approved in the {win} days before the invoice date. Invoices from other vendor IDs — including vendors with similar trading names — and invoices older than {win} days are not added.")
    doc.section("7.6 Tier table (amount to be approved, EUR)")
    for r in ("STANDARD", "ELEVATED"):
        t1, t2, t3 = thresholds[r]
        doc.para(f"    {r}: Tier 1 up to {fmt('EUR', t1)}; Tier 2 above that up to {fmt('EUR', t2)}; Tier 3 above that up to {fmt('EUR', t3)}; Tier 4 above {fmt('EUR', t3)}.")
    doc.para("An amount equal to a boundary belongs to the lower tier.")
    doc.section("7.7 Credit notes and disputes")
    doc.para("A credit note is routed at the tier of the invoice it corrects, never lower. An invoice under dispute is routed when the dispute closes, using the vendor grade and the register as they stand on the original invoice date.")
    doc.section("7.8 Records")
    doc.para("The routing worksheet names the vendor ID, the grade with the clause that set it, the EUR amount with the rate used, every register line added under 7.5 and the tier reached. A worksheet missing any of these is returned before approval.")
    procedure(rng, doc, "7.9", "chapter 7", desk, 6)
    general_provisions(rng, doc, "7.10", "chapter 7", owner, desk, C, 9)
    doc.part("Annex A — trading-name list (extract)")
    for n, v in zip(names, vids):
        doc.para(f"    \"{n}\" -> {v}")
    doc.part("Annex B — vendor master (extract)")
    for v in vids:
        doc.para(f"    {v}: risk grade {risk_master[v]}; onboarded {onboard_years[v]}; payment terms 30 days")
    doc.part("Annex C — treasury rates on the invoice date (foreign units per EUR)")
    for c_, r_ in (("CHF", rate if cur == "CHF" else Decimal("0.9450")), ("GBP", rate if cur == "GBP" else Decimal("0.8560")), ("NOK", rate if cur == "NOK" else Decimal("11.42"))):
        doc.para(f"    {c_}: {r_}")
    doc.part("Annex D — approved payee bank countries")
    doc.para("    " + ", ".join(approved_countries))
    doc.part(f"Invoice {inv}")
    doc.para(f"Trading name printed: \"{exact}\". Purchase order names \"{names[1]}\" (raised by the requester from memory). Invoice date {longdate(inv_date)}. Amount {cur} {amt_fx:,.2f}. Payee bank: IBAN country {payee_country}.")
    doc.part("Approval register (last 90 days)")
    rows = [(inv_date - dt.timedelta(days=d_same), vids[0], same_amt), (inv_date - dt.timedelta(days=rng.randint(5, 25)), vids[1], other_amt),
            (inv_date - dt.timedelta(days=rng.randint(60, 88)), vids[0], old_amt)]
    rows.sort()
    for d, v, a in rows:
        doc.para(f"    {longdate(d)} — {v} — {fmt('EUR', a)} — approved")
    doc.section("AP clerk's note")
    doc.para(f"    \"PO says {names[1]}; master grade {risk_master[vids[1]]}; {cur} x {rate} = {fmt('EUR', amt_fx * rate)}. Suggest Tier {tier(risk_master[vids[1]], money(amt_fx * rate))}.\"")
    options = tuple((i, f"Route to {n}.") for i, n in zip(tier_ids, tier_names))
    expected = tier_ids[truth - 1]
    mm = {}
    for name, t in mistakes.items():
        if t != truth:
            mm.setdefault(tier_ids[t - 1], name)
    return World(
        id=f"mh-invoice-{idx:05d}", law="invoice_tier", kind="choice", evidence=doc.render(),
        criterion=f"Following chapter 7, to which approval tier must invoice {inv} be routed?",
        options=options, expected=expected, mistakes=mm,
        rationale=f"'{exact}' -> {vids[0]} ({risk_master[vids[0]]}, onboarded {onboard_years[vids[0]]}); payee {payee_country} -> override={override} -> {risk}; "
                  f"{amt_fx} {cur} / {rate} = {eur_now}; same-vendor {same_amt} {d_same} days ago {'added' if d_same <= win else 'not added'}; {agg} -> Tier {truth}.",
        params={"vendor": vids[0], "risk": risk, "override": override, "eur_now": str(eur_now), "aggregated": str(agg), "d_same": d_same})


# ---------------------------------------------------------------------------
# Law 3 — on-call paging: process alias, tier-adjusted severity, merger, handover
# ---------------------------------------------------------------------------
def law_oncall_page(rng: random.Random, idx: int) -> World:
    C = f"{rng.choice(['Fernway', 'Larkspur', 'Tidewater', 'Oxbow'])} Pay"
    people = ["Ana", "Bjorn", "Chen", "Dmitri", "Esra"]
    svc_core, svc_rep = "ledger-core", "ledger-reports"
    proc_core, proc_rep = "ledgerd", "ledgerrpt"
    alert_proc = rng.choice([proc_core, proc_core, proc_rep])
    rule = rng.choice(["ProcessCrashLoop", "DiskPressure", "QueueBacklog"])
    base_sev = {"ProcessCrashLoop": 3, "DiskPressure": 3, "QueueBacklog": 2}[rule]
    tier = {svc_core: 1, svc_rep: 3}
    team = {svc_core: "Payments-Core", svc_rep: "Reporting"}
    svc = svc_core if alert_proc == proc_core else svc_rep
    sev = max(1, min(4, base_sev - 1)) if tier[svc] == 1 else (min(4, base_sev + 1) if tier[svc] == 3 else base_sev)
    pages = sev <= 2
    merge_date = dt.date(rng.randint(2026, 2027), rng.randint(2, 11), 1)
    alert_date = merge_date + dt.timedelta(days=rng.randint(6, 40))
    while alert_date.weekday() != 0:                      # a Monday, around the 09:00 handover
        alert_date += dt.timedelta(days=1)
    utc_offset = 2 if 4 <= alert_date.month <= 9 else 1     # Berlin: CEST Apr-Sep (stated in the handbook), CET otherwise
    alert_utc = dt.time(rng.choice([6, 7]), rng.choice([20, 35, 50]))
    local = (dt.datetime.combine(alert_date, alert_utc) + dt.timedelta(hours=utc_offset)).time()
    after_handover = local >= dt.time(9, 0)
    week_old, week_new = rng.sample(people[:2] + people[3:4], 2)   # Ana, Bjorn, Dmitri as primaries
    primary = week_new if after_handover else week_old
    override_on = rng.random() < 0.3
    override_person = "Chen"
    override_window = (dt.time(6, 0), dt.time(10, 0)) if override_on else None
    in_override = bool(override_window) and override_window[0] <= local < override_window[1]
    reporting_person = "Esra"
    retired_primary = rng.choice([p for p in people if p != primary])
    if not pages:
        truth = "no_page_ticket_only"
    elif svc == svc_rep:
        truth = reporting_person.lower()
    elif in_override:
        truth = override_person.lower()
    else:
        truth = primary.lower()
    naive_local_after = alert_utc >= dt.time(9, 0)
    mistakes = {
        "process_alias_confused": (reporting_person.lower() if svc == svc_core else (override_person.lower() if in_override else primary.lower())) if pages else None,
        "severity_not_adjusted": ("no_page_ticket_only" if base_sev > 2 else None) if pages and tier[svc] == 1 else None,
        "retired_rota_sheet_used": retired_primary.lower() if pages and svc == svc_core else None,
        "utc_not_converted": ((week_new if naive_local_after else week_old).lower() if pages and svc == svc_core and not in_override and naive_local_after != after_handover else None),
        "override_window_ignored": primary.lower() if in_override and pages else None,
        "ticket_rule_ignored": (primary.lower() if svc == svc_core else reporting_person.lower()) if not pages else None,
    }
    owner, desk = "the SRE lead", "SRE desk"
    doc = Doc()
    doc.title(f"{C} — SRE paging handbook, version {rng.randint(11, 19)} (edited {longdate(alert_date - dt.timedelta(days=rng.randint(3, 20)))})")
    doc.section("1. How an alert becomes a page")
    doc.clause("1.1", "An alert names the failing process and its host, never the service. The process is mapped to its service through the service catalogue in section 3; one service may run several processes, and two services may run processes with similar names.")
    doc.clause("1.2", "Severity: the alert rule's base severity (section 4) is adjusted by the service tier (section 2). SEV1 and SEV2 page a human. SEV3 and SEV4 raise a ticket for the owning team and page no one, whatever the hour.")
    doc.clause("1.3", "The person paged is the primary on the owning team's current rota for the moment of the alert, unless an override window covers that moment, in which case the override person is paged.")
    doc.clause("1.4", f"Rota weeks run Monday 09:00 to Monday 09:00 Berlin time. Alerts are timestamped in UTC; Berlin is UTC+2 from April to September and UTC+1 otherwise. Convert before deciding which week's primary applies.")
    doc.section("2. Service tiers")
    doc.clause("2.1", "Tier 1: base severity is raised by one step (SEV3 becomes SEV2). Tier 2: no change. Tier 3: base severity is lowered by one step (SEV2 becomes SEV3).")
    doc.section("3. Service catalogue")
    doc.para(f"    process {proc_core} -> service {svc_core} — tier 1 — owning team: Payments-Core (see section 5)")
    doc.para(f"    process {proc_rep} -> service {svc_rep} — tier 3 — owning team: Reporting")
    doc.para(f"    process settlerd -> service settlement — tier 2 — owning team: Money Movement")
    doc.para(f"    process authz -> service auth-gateway — tier 1 — owning team: Identity")
    doc.section("4. Base severities")
    doc.para("    ProcessCrashLoop: SEV3. DiskPressure: SEV3. QueueBacklog: SEV2. CertificateExpiry: SEV4.")
    doc.section("5. Team changes")
    doc.clause("5.1", f"Payments-Core merged into Money Movement on {longdate(merge_date)}. From that date the Money Movement rota is the rota for every Payments-Core service. The former payments-core rota sheet is retired and is kept for audit only; it must not be used to page.")
    doc.section("6. Overrides")
    doc.clause("6.1", "An override names a person and a window in Berlin time. During the window the override person is paged instead of the primary, for the services the override lists.")
    doc.section("7. Acknowledgement and escalation")
    doc.clause("7.1", "A page is acknowledged within five minutes. An unacknowledged page repeats twice and then goes to the secondary; the secondary's unacknowledged page goes to the team lead. Escalation never changes who should have been paged first.")
    doc.clause("7.2", "A responder who believes the routing was wrong still handles the page, and files a routing correction afterwards.")
    doc.section("8. Quiet hours")
    doc.clause("8.1", "There are no quiet hours for SEV1 and SEV2. A SEV3 ticket raised between 22:00 and 07:00 Berlin time is picked up at the start of the next working day.")
    procedure(rng, doc, "9", "this handbook", desk, 6)
    general_provisions(rng, doc, "10", "this handbook", owner, desk, C, 9)
    doc.part("Money Movement rota (current)")
    wk = alert_date
    doc.para(f"    Week of Monday {longdate(wk - dt.timedelta(days=7))} 09:00 to Monday {longdate(wk)} 09:00 — primary {week_old}, secondary {rng.choice([p for p in people if p not in (week_old, week_new)])}")
    doc.para(f"    Week of Monday {longdate(wk)} 09:00 to Monday {longdate(wk + dt.timedelta(days=7))} 09:00 — primary {week_new}, secondary {rng.choice([p for p in people if p not in (week_old, week_new)])}")
    doc.para(f"    Overrides: " + (f"{override_person} covers {svc_core} and settlement on {dayname(alert_date)} {longdate(alert_date)} from {override_window[0].strftime('%H:%M')} to {override_window[1].strftime('%H:%M')} Berlin time." if override_on else "none this week."))
    doc.part("Reporting rota (current)")
    doc.para(f"    Primary this week and next: {reporting_person}.")
    doc.part("Retired payments-core rota sheet (audit copy, do not use)")
    doc.para(f"    Week of Monday {longdate(wk)}: primary {retired_primary}. [sheet retired on {longdate(merge_date)}]")
    doc.part("Alert")
    doc.para(f"    {rule} on host pay-{rng.randint(3, 19):02d}: process {alert_proc} — fired {longdate(alert_date)} {alert_utc.strftime('%H:%M')} UTC — base severity SEV{base_sev}")
    doc.section("Junior responder's note")
    doc.para(f"    \"SEV{base_sev} per the rule table; retired sheet says {retired_primary} this week; {alert_utc.strftime('%H:%M')} is before the 09:00 handover.\"")
    options = tuple([(p.lower(), f"Page {p}.") for p in people] + [("no_page_ticket_only", "No page; a ticket for the owning team only.")])
    mm = {}
    for name, opt in mistakes.items():
        if opt and opt != truth:
            mm.setdefault(opt, name)
    return World(
        id=f"mh-oncall-{idx:05d}", law="oncall_page", kind="choice", evidence=doc.render(),
        criterion="Under the handbook, which person is paged for this alert, if anyone?",
        options=options, expected=truth, mistakes=mm,
        rationale=f"{alert_proc} -> {svc} tier {tier[svc]}; SEV{base_sev} -> SEV{sev} pages={pages}; {alert_utc} UTC = {local} Berlin, after handover={after_handover}; override={in_override} -> {truth}.",
        params={"process": alert_proc, "service": svc, "severity": sev, "local_time": local.strftime("%H:%M"), "after_handover": after_handover, "override": in_override})


LAWS = {
    "identifier_chain": law_identifier_chain,
    "invoice_tier": law_invoice_tier,
    "oncall_page": law_oncall_page,
}


def generate(n: int, seed: int = 0, laws: list[str] | None = None) -> list[World]:
    """n worlds, round-robin over the laws, reproducible from the seed."""
    return _generate(LAWS, FAMILY, n, seed, laws)


def main(argv=None) -> int:
    return run_cli(LAWS, FAMILY, __doc__, argv)


if __name__ == "__main__":
    sys.exit(main())

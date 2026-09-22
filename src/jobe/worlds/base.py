# -*- coding: utf-8 -*-
"""Shared machinery for the exact-law worlds: the record, its checks, the
contamination guard, the name pools and the CLI. Each family module owns its
laws and calls `generate` / `run_cli` from here."""
from __future__ import annotations

import argparse
import calendar
import dataclasses
import datetime as dt
import glob
import hashlib
import json
import os
import random
import re
import sys
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal

CENT = Decimal("0.01")


def money(x) -> Decimal:
    return Decimal(x).quantize(CENT, rounding=ROUND_HALF_UP)


def money_id(cur: str, x) -> str:
    return f"{cur.lower()}_{str(money(x)).replace('.', '_').replace('-', 'neg')}"


def fmt(cur: str, x) -> str:
    return f"{cur} {money(x):,.2f}"


def longdate(d: dt.date) -> str:
    return f"{d.day} {calendar.month_name[d.month]} {d.year}"


def dayname(d: dt.date) -> str:
    return calendar.day_name[d.weekday()]


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


@dataclass(frozen=True)
class World:
    """One generated decision: the evidence, the criterion, the options with the
    gold among them, and the option each named mistake produces."""
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
    family: str = ""

    @property
    def split(self) -> str:
        return "heldout" if int(hashlib.sha1(self.id.encode()).hexdigest(), 16) % 10 == 0 else "train"

    def record(self) -> dict:
        return {"id": self.id, "family": self.family, "law": self.law, "type": self.kind, "split": self.split,
                "evidence": self.evidence, "criterion": self.criterion,
                "options": [{"id": i, "description": d} for i, d in self.options],
                "expected": self.expected, "ordinal": self.ordinal, "mistakes": self.mistakes,
                "rationale": self.rationale, "params": self.params}


def check(w: World) -> None:
    ids = [i for i, _ in w.options]
    assert len(ids) == len(set(ids)), f"{w.id}: duplicate option ids"
    assert 2 <= len(ids) <= 16, f"{w.id}: {len(ids)} options"
    assert w.expected in ids, f"{w.id}: expected {w.expected!r} not among options"
    assert w.expected not in w.mistakes, f"{w.id}: gold option labelled as a mistake"
    assert set(w.mistakes) <= set(ids), f"{w.id}: mistake option not among options"
    assert all(re.fullmatch(r"[a-z0-9_]+", i) for i in ids), f"{w.id}: option ids must be snake_case"


def to_decision(record: dict):
    """A generated record as a jobe Decision (import deferred; the generators need no torch)."""
    from jobe.prompt import Decision, Option
    return Decision(id=record["id"], evidence=record["evidence"], criterion=record["criterion"],
                    options=tuple(Option(o["id"], o["description"]) for o in record["options"]),
                    ordinal=record.get("ordinal", False))


def generate(laws: dict, family: str, n: int, seed: int = 0, only: list[str] | None = None) -> list[World]:
    """n worlds, round-robin over the laws, reproducible from the seed."""
    names = only or list(laws)
    out: list[World] = []
    for i in range(n):
        name = names[i % len(names)]
        rng = random.Random(f"{seed}:{name}:{i}")
        w = dataclasses.replace(laws[name](rng, i), family=family)
        check(w)
        out.append(w)
    return out


def _grams(text: str, n: int = 10):
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def contamination(worlds: list[World], jevbench_dir: str, n: int = 10) -> list[tuple[str, str]]:
    """(world id, shared phrase) for every n-word run also present in a public task file."""
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


def evidence_chars(w: World) -> int:
    return len(w.evidence) if isinstance(w.evidence, str) else len(json.dumps(w.evidence, ensure_ascii=False))


def run_cli(laws: dict, family: str, doc: str, argv=None) -> int:
    ap = argparse.ArgumentParser(description=doc.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--laws", nargs="*", choices=list(laws))
    ap.add_argument("--out", default="")
    ap.add_argument("--check-contamination", metavar="JEVBENCH_DIR", default="")
    args = ap.parse_args(argv)
    worlds = generate(laws, family, args.n, args.seed, args.laws)
    by_law: dict[str, list] = {}
    for w in worlds:
        s = by_law.setdefault(w.law, [0, 0, 0, 0, 0])
        s[0] += 1
        s[1] += w.split == "heldout"
        s[2] += len(w.mistakes)
        s[3] += len(w.options)
        s[4] += evidence_chars(w)
    print(f"{'law':22s} {'n':>5s} {'heldout':>7s} {'mistakes/rec':>12s} {'options/rec':>11s} {'chars/rec':>9s}")
    for law, (n, h, m, o, c) in by_law.items():
        print(f"{law:22s} {n:5d} {h:7d} {m / n:12.2f} {o / n:11.2f} {c / n:9.0f}")
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

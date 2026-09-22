"""A TheLab decision record as a jobe Decision.

The worlds generate records (`thelab.decisions.Record`, or their JSONL form);
this is the one place jobe turns one into something its readout can score.
"""
from __future__ import annotations

from .prompt import Decision, Option


def to_decision(record) -> Decision:
    """Accepts a `Record` or its dict form; validates through `Decision.validate()` semantics on use."""
    d = record.to_dict() if hasattr(record, "to_dict") else record
    return Decision(id=d["id"], evidence=d["evidence"], criterion=d["criterion"],
                    options=tuple(Option(o["id"], o["description"]) for o in d["options"]),
                    ordinal=d.get("ordinal", False))

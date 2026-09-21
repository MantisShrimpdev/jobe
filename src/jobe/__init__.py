"""Jobe - a local, frozen-backbone decision readout.

A decision is a classification over an option set the caller declares at request
time, read from the model's next-token distribution in one forward pass. Nothing
is generated, so there is no text to parse, nothing to repair, and no way for the
answer to be something other than one of the declared ids.

    from jobe import Decision, Option, load, score

    backbone = load("unsloth/llama-3.2-3b-instruct")
    r = score(backbone.model, backbone.tokenizer, Decision(
        id="ticket-1",
        evidence="The checkout page returns a 500 when applying a coupon.",
        criterion="Which team should own this ticket?",
        options=(
            Option("billing", "Payments, invoices and refunds"),
            Option("infra", "Outages, latency and deploys"),
        ),
    ))
    r.choice          # -> "infra"
    r.scores          # -> {"billing": 0.09, "infra": 0.91}
    r.confidence()    # chance-corrected; see the caveat on Readout.confidence

Protocol adapted from TheoLeeCJ/SemIf (MIT). See NOTICE.
"""

from .calibrate import (
    CalibrationReport,
    brier_score,
    calibration_report,
    fit_temperature,
    negative_log_likelihood,
    softmax_with_temperature,
)
from .model import LoadedModel, load, pick_device
from .orders import (
    AveragedReadout,
    average_readouts,
    order_variants,
    score_averaged,
    total_variation,
)
from .prompt import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    Decision,
    DecisionError,
    Option,
    build_messages,
    prompt_digest,
    render_prompt,
)
from .readout import Readout, restricted_softmax, score
from .slots import (
    LETTERS,
    MAX_OPTIONS,
    SlotError,
    resolve_slots,
    slot_token_ids,
    verify_boundary,
)

__all__ = [
    "total_variation",
    "softmax_with_temperature",
    "score_averaged",
    "order_variants",
    "negative_log_likelihood",
    "fit_temperature",
    "calibration_report",
    "brier_score",
    "average_readouts",
    "CalibrationReport",
    "AveragedReadout",
    "LETTERS",
    "MAX_OPTIONS",
    "PROMPT_VERSION",
    "SYSTEM_PROMPT",
    "Decision",
    "DecisionError",
    "LoadedModel",
    "Option",
    "Readout",
    "SlotError",
    "build_messages",
    "load",
    "pick_device",
    "prompt_digest",
    "render_prompt",
    "resolve_slots",
    "restricted_softmax",
    "score",
    "slot_token_ids",
    "verify_boundary",
]

"""Calibration lives in TheLab (`thelab.core.calibrate`); this keeps jobe's import path.

Temperature fitting and the reliability report are model-agnostic — they take
logits and gold indices — and axiom needed them too, so they moved. Nothing
about the API changed.
"""
from thelab.core.calibrate import (  # noqa: F401
    CalibrationReport,
    brier_score,
    calibration_report,
    fit_temperature,
    negative_log_likelihood,
    softmax_with_temperature,
)

__all__ = ["CalibrationReport", "brier_score", "calibration_report", "fit_temperature",
           "negative_log_likelihood", "softmax_with_temperature"]

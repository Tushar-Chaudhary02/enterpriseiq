"""Tests for probability calibration utilities."""

import numpy as np
import pandas as pd

from enterpriseiq.ml.calibration import (
    build_calibrated_model,
    calculate_probability_metrics,
)
from enterpriseiq.ml.pipeline import (
    build_logistic_pipeline,
)


def test_probability_metrics_reward_better_probabilities() -> None:
    """Better probabilities should produce lower losses."""

    target = pd.Series(
        [0, 0, 1, 1],
        dtype="int64",
    )

    strong_probabilities = np.asarray(
        [0.05, 0.10, 0.90, 0.95],
        dtype=np.float64,
    )

    weak_probabilities = np.asarray(
        [0.40, 0.45, 0.55, 0.60],
        dtype=np.float64,
    )

    strong_metrics = calculate_probability_metrics(
        target,
        strong_probabilities,
    )

    weak_metrics = calculate_probability_metrics(
        target,
        weak_probabilities,
    )

    assert strong_metrics["brier_score"] < weak_metrics["brier_score"]

    assert strong_metrics["log_loss"] < weak_metrics["log_loss"]


def test_calibrated_model_uses_sigmoid() -> None:
    """The calibrator should use the planned method and folds."""

    calibrated_model = build_calibrated_model(build_logistic_pipeline())

    assert calibrated_model.method == "sigmoid"
    assert calibrated_model.cv == 5

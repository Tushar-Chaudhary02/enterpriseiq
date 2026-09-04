"""Probability-calibration utilities for churn prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
)
from sklearn.pipeline import Pipeline

CALIBRATION_METHOD = "sigmoid"
CALIBRATION_FOLDS = 5


def build_calibrated_model(
    base_model: Pipeline,
) -> CalibratedClassifierCV:
    """Wrap a classifier with cross-validated sigmoid calibration."""

    return CalibratedClassifierCV(
        estimator=base_model,
        method=CALIBRATION_METHOD,
        cv=CALIBRATION_FOLDS,
        n_jobs=-1,
    )


def calculate_probability_metrics(
    y_true: pd.Series[int],
    probabilities: NDArray[np.float64],
) -> dict[str, float]:
    """Evaluate the reliability of predicted probabilities."""

    return {
        "brier_score": float(
            brier_score_loss(
                y_true,
                probabilities,
            )
        ),
        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=[0, 1],
            )
        ),
    }

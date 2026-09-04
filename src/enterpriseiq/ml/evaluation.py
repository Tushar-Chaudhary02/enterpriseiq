"""Evaluation utilities for binary churn classifiers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

MetricValue = float | list[list[int]]
Metrics = dict[str, MetricValue]


def calculate_binary_metrics(
    y_true: pd.Series[int],
    y_prediction: NDArray[np.int64],
    y_probability: NDArray[np.float64],
) -> Metrics:
    """Calculate classification metrics at the default 0.50 threshold."""

    matrix = confusion_matrix(y_true, y_prediction)

    matrix_as_list = [[int(value) for value in row] for row in matrix]

    return {
        "accuracy": float(accuracy_score(y_true, y_prediction)),
        "precision": float(
            precision_score(
                y_true,
                y_prediction,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_prediction,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_prediction,
                zero_division=0,
            )
        ),
        "roc_auc": float(roc_auc_score(y_true, y_probability)),
        "pr_auc": float(average_precision_score(y_true, y_probability)),
        "confusion_matrix": matrix_as_list,
    }

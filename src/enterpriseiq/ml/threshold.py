"""Business-aware probability-threshold analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

DEFAULT_FALSE_POSITIVE_COST = 50.0
DEFAULT_FALSE_NEGATIVE_COST = 500.0
DEFAULT_MINIMUM_RECALL = 0.75


def analyze_thresholds(
    y_true: pd.Series[int],
    probabilities: NDArray[np.float64],
    *,
    thresholds: NDArray[np.float64] | None = None,
    false_positive_cost: float = DEFAULT_FALSE_POSITIVE_COST,
    false_negative_cost: float = DEFAULT_FALSE_NEGATIVE_COST,
) -> pd.DataFrame:
    """Calculate performance and cost across probability thresholds."""

    if thresholds is None:
        thresholds = np.linspace(
            0.05,
            0.95,
            91,
            dtype=np.float64,
        )

    rows: list[dict[str, float | int]] = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(np.int64)

        matrix = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        )

        true_negatives, false_positives = matrix[0]
        false_negatives, true_positives = matrix[1]

        business_cost = (
            float(false_positives) * false_positive_cost
            + float(false_negatives) * false_negative_cost
        )

        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(
                    precision_score(
                        y_true,
                        predictions,
                        zero_division=0,
                    )
                ),
                "recall": float(
                    recall_score(
                        y_true,
                        predictions,
                        zero_division=0,
                    )
                ),
                "f1": float(
                    f1_score(
                        y_true,
                        predictions,
                        zero_division=0,
                    )
                ),
                "true_negatives": int(true_negatives),
                "false_positives": int(false_positives),
                "false_negatives": int(false_negatives),
                "true_positives": int(true_positives),
                "business_cost": business_cost,
            }
        )

    return pd.DataFrame(rows)


def select_operating_threshold(
    threshold_results: pd.DataFrame,
    *,
    minimum_recall: float = DEFAULT_MINIMUM_RECALL,
) -> float:
    """Select the lowest-cost threshold satisfying the recall requirement."""

    required_columns = {
        "threshold",
        "recall",
        "f1",
        "business_cost",
    }

    missing_columns = required_columns - set(threshold_results.columns)

    if missing_columns:
        raise ValueError(f"Threshold results are missing: {sorted(missing_columns)}")

    eligible_results = threshold_results[threshold_results["recall"] >= minimum_recall]

    if eligible_results.empty:
        raise ValueError("No threshold satisfies the minimum recall requirement.")

    selected_row = eligible_results.sort_values(
        by=[
            "business_cost",
            "f1",
            "recall",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    ).iloc[0]

    return float(selected_row["threshold"])

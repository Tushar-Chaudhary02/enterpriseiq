"""Cross-validation and model-selection utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline

from enterpriseiq.ml.data import DataSplit

METRIC_NAMES: tuple[str, ...] = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
)


def build_scoring() -> dict[str, object]:
    """Create metrics used during cross-validation."""

    return {
        "accuracy": "accuracy",
        "precision": make_scorer(
            precision_score,
            zero_division=0,
        ),
        "recall": make_scorer(
            recall_score,
            zero_division=0,
        ),
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }


def cross_validate_models(
    models: dict[str, Pipeline],
    data_split: DataSplit,
    *,
    folds: int = 5,
) -> pd.DataFrame:
    """Evaluate candidate pipelines on the training set only."""

    cross_validator = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=42,
    )

    rows: list[dict[str, str | float]] = []

    for model_name, model in models.items():
        print(f"Cross-validating: {model_name}")

        scores = cross_validate(
            estimator=model,
            X=data_split.x_train,
            y=data_split.y_train,
            cv=cross_validator,
            scoring=build_scoring(),
            return_train_score=False,
            n_jobs=-1,
            error_score="raise",
        )

        row: dict[str, str | float] = {
            "model": model_name,
        }

        for metric_name in METRIC_NAMES:
            values = np.asarray(
                scores[f"test_{metric_name}"],
                dtype=np.float64,
            )

            row[f"{metric_name}_mean"] = float(values.mean())
            row[f"{metric_name}_std"] = float(values.std(ddof=1))

        rows.append(row)

    results = pd.DataFrame(rows)

    return results.sort_values(
        by="pr_auc_mean",
        ascending=False,
    ).reset_index(drop=True)


def select_champion(
    results: pd.DataFrame,
    *,
    primary_metric: str = "pr_auc_mean",
) -> str:
    """Select the highest-scoring model using one CV metric."""

    if results.empty:
        raise ValueError("Model-comparison results are empty.")

    if primary_metric not in results.columns:
        raise ValueError(f"Selection metric is missing: {primary_metric}")

    champion = results.sort_values(
        primary_metric,
        ascending=False,
    ).iloc[0]["model"]

    return str(champion)

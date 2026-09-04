"""Tests for model comparison and selection."""

import pandas as pd
import pytest

from enterpriseiq.ml.model_selection import select_champion


def test_select_champion_uses_primary_metric() -> None:
    """The highest PR-AUC model should be selected."""

    results = pd.DataFrame(
        {
            "model": [
                "logistic_regression",
                "random_forest",
                "xgboost",
            ],
            "pr_auc_mean": [
                0.63,
                0.61,
                0.67,
            ],
        }
    )

    champion = select_champion(
        results,
        primary_metric="pr_auc_mean",
    )

    assert champion == "xgboost"


def test_select_champion_rejects_empty_results() -> None:
    """Empty model results should not produce a champion."""

    results = pd.DataFrame()

    with pytest.raises(
        ValueError,
        match="results are empty",
    ):
        select_champion(results)

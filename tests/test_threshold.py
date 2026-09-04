"""Tests for business threshold selection."""

import numpy as np
import pandas as pd
import pytest

from enterpriseiq.ml.threshold import (
    analyze_thresholds,
    select_operating_threshold,
)


def test_threshold_analysis_calculates_cost() -> None:
    """False positives and false negatives should create cost."""

    target = pd.Series(
        [0, 0, 1, 1],
        dtype="int64",
    )

    probabilities = np.asarray(
        [0.10, 0.80, 0.40, 0.90],
        dtype=np.float64,
    )

    results = analyze_thresholds(
        target,
        probabilities,
        thresholds=np.asarray(
            [0.50],
            dtype=np.float64,
        ),
        false_positive_cost=50.0,
        false_negative_cost=500.0,
    )

    row = results.iloc[0]

    assert row["false_positives"] == 1
    assert row["false_negatives"] == 1
    assert row["business_cost"] == 550.0


def test_threshold_selection_respects_recall_floor() -> None:
    """Thresholds below minimum recall must be excluded."""

    results = pd.DataFrame(
        {
            "threshold": [0.30, 0.50],
            "recall": [0.80, 0.70],
            "f1": [0.60, 0.70],
            "business_cost": [1000.0, 900.0],
        }
    )

    selected = select_operating_threshold(
        results,
        minimum_recall=0.75,
    )

    assert selected == pytest.approx(0.30)


def test_threshold_selection_uses_lowest_cost() -> None:
    """Among eligible thresholds, lowest cost should win."""

    results = pd.DataFrame(
        {
            "threshold": [0.20, 0.30, 0.40],
            "recall": [0.90, 0.85, 0.80],
            "f1": [0.55, 0.62, 0.64],
            "business_cost": [1200.0, 900.0, 1100.0],
        }
    )

    selected = select_operating_threshold(
        results,
        minimum_recall=0.75,
    )

    assert selected == pytest.approx(0.30)

"""Unit tests for prediction-service business logic."""

import pytest

from enterpriseiq.api.service import classify_risk


@pytest.mark.parametrize(
    ("probability", "threshold", "expected"),
    [
        (0.08, 0.08, "high_risk"),
        (0.079, 0.08, "low_risk"),
    ],
)
def test_classify_risk(
    probability: float,
    threshold: float,
    expected: str,
) -> None:
    assert classify_risk(probability, threshold) == expected

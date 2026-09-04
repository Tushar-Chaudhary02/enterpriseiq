"""Tests for machine-learning data preparation."""

import pandas as pd
import pytest

from enterpriseiq.ml.data import (
    create_train_test_split,
    prepare_modeling_data,
)


def create_customer(
    customer_id: str,
    churn: str,
    total_charges: str,
) -> dict[str, object]:
    """Create one minimal customer record for testing."""

    return {
        "customerID": customer_id,
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0,
        "TotalCharges": total_charges,
        "Churn": churn,
    }


def test_prepare_modeling_data_separates_target() -> None:
    """Identifiers and targets should not appear in model features."""

    raw_df = pd.DataFrame(
        [
            create_customer("CUSTOMER-1", "No", "840.00"),
            create_customer("CUSTOMER-2", "Yes", " "),
        ]
    )

    features, target = prepare_modeling_data(raw_df)

    assert "customerID" not in features.columns
    assert "Churn" not in features.columns
    assert target.tolist() == [0, 1]
    assert features["TotalCharges"].isna().sum() == 1
    assert features["SeniorCitizen"].tolist() == ["0", "0"]


def test_train_test_split_is_stratified() -> None:
    """Train and test sets should preserve target proportions."""

    features = pd.DataFrame(
        {
            "example_feature": range(100),
        }
    )

    target = pd.Series(
        [0] * 80 + [1] * 20,
        dtype="int64",
    )

    data_split = create_train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
    )

    assert len(data_split.x_train) == 80
    assert len(data_split.x_test) == 20
    assert data_split.y_train.mean() == pytest.approx(0.20)
    assert data_split.y_test.mean() == pytest.approx(0.20)

"""Load, prepare, and split customer data for machine learning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from enterpriseiq.data.download import DEFAULT_DATASET_PATH

ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"

NUMERIC_FEATURES: tuple[str, ...] = (
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
)

CATEGORICAL_FEATURES: tuple[str, ...] = (
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
)

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class DataSplit:
    """Container for one reproducible train/test split."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series[int]
    y_test: pd.Series[int]


def prepare_modeling_data(
    raw_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series[int]]:
    """Create model features and a binary target from the raw dataset."""

    required_columns = {
        ID_COLUMN,
        TARGET_COLUMN,
        *NUMERIC_FEATURES,
        *CATEGORICAL_FEATURES,
    }

    missing_columns = required_columns - set(raw_df.columns)

    if missing_columns:
        raise ValueError(f"Modeling data is missing columns: {sorted(missing_columns)}")

    model_df = raw_df.copy()

    model_df["TotalCharges"] = pd.to_numeric(
        model_df["TotalCharges"],
        errors="coerce",
    )

    # Although stored as 0/1, SeniorCitizen represents a category.
    model_df["SeniorCitizen"] = model_df["SeniorCitizen"].astype(str)

    target = model_df[TARGET_COLUMN].map(
        {
            "No": 0,
            "Yes": 1,
        }
    )

    if target.isna().any():
        unexpected_targets = sorted(model_df.loc[target.isna(), TARGET_COLUMN].astype(str).unique())
        raise ValueError(f"Unexpected target values: {unexpected_targets}")

    features = model_df.loc[:, MODEL_FEATURES].copy()

    return features, target.astype("int64")


def load_modeling_data(
    path: Path = DEFAULT_DATASET_PATH,
) -> tuple[pd.DataFrame, pd.Series[int]]:
    """Read raw customer data and prepare it for modeling."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Run the download module first.")

    raw_df = pd.read_csv(path)

    return prepare_modeling_data(raw_df)


def create_train_test_split(
    features: pd.DataFrame,
    target: pd.Series[int],
    *,
    test_size: float = 0.20,
    random_state: int = 42,
) -> DataSplit:
    """Create a reproducible, stratified train/test split."""

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    return DataSplit(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
    )

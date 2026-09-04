"""Build leakage-safe preprocessing and classification pipelines."""

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from enterpriseiq.ml.data import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)


def build_preprocessor() -> ColumnTransformer:
    """Create preprocessing for numerical and categorical features."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "one_hot_encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                list(NUMERIC_FEATURES),
            ),
            (
                "categorical",
                categorical_pipeline,
                list(CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
    )


def build_dummy_pipeline() -> Pipeline:
    """Create a no-skill benchmark pipeline."""

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                DummyClassifier(
                    strategy="prior",
                    random_state=42,
                ),
            ),
        ]
    )


def build_logistic_pipeline() -> Pipeline:
    """Create the initial Logistic Regression pipeline."""

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )

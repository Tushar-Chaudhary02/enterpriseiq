"""Build leakage-safe preprocessing and classification pipelines."""

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

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
    """Create the Logistic Regression pipeline."""

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


def build_decision_tree_pipeline() -> Pipeline:
    """Create an untuned Decision Tree pipeline."""

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                DecisionTreeClassifier(
                    random_state=42,
                ),
            ),
        ]
    )


def build_random_forest_pipeline() -> Pipeline:
    """Create an untuned Random Forest pipeline."""

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )


def build_xgboost_pipeline() -> Pipeline:
    """Create an initial XGBoost pipeline."""

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=250,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.80,
                    colsample_bytree=0.80,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=42,
                    n_jobs=1,
                    verbosity=0,
                ),
            ),
        ]
    )


def build_candidate_pipelines() -> dict[str, Pipeline]:
    """Create all pipelines used in Day 4 model comparison."""

    return {
        "dummy_classifier": build_dummy_pipeline(),
        "logistic_regression": build_logistic_pipeline(),
        "decision_tree": build_decision_tree_pipeline(),
        "random_forest": build_random_forest_pipeline(),
        "xgboost": build_xgboost_pipeline(),
    }

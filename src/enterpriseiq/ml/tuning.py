"""Hyperparameter-search configurations for finalist models."""

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)

from enterpriseiq.ml.pipeline import (
    build_logistic_pipeline,
    build_xgboost_pipeline,
)

CV_FOLDS = 5
RANDOM_STATE = 42


def build_cross_validator() -> StratifiedKFold:
    """Create the shared stratified cross-validator."""

    return StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def build_logistic_search() -> GridSearchCV:
    """Create a grid search for Logistic Regression."""

    parameter_grid = {
        "classifier__C": [
            0.01,
            0.05,
            0.10,
            0.50,
            1.00,
            2.00,
            10.00,
        ],
        "classifier__class_weight": [
            None,
            "balanced",
        ],
    }

    return GridSearchCV(
        estimator=build_logistic_pipeline(),
        param_grid=parameter_grid,
        scoring="average_precision",
        cv=build_cross_validator(),
        refit=True,
        n_jobs=-1,
        return_train_score=False,
        error_score="raise",
    )


def build_xgboost_search() -> RandomizedSearchCV:
    """Create a reproducible randomized search for XGBoost."""

    parameter_distributions = {
        "classifier__n_estimators": [
            150,
            250,
            400,
        ],
        "classifier__max_depth": [
            2,
            3,
            4,
            5,
        ],
        "classifier__learning_rate": [
            0.02,
            0.05,
            0.10,
        ],
        "classifier__min_child_weight": [
            1,
            3,
            5,
        ],
        "classifier__subsample": [
            0.70,
            0.85,
            1.00,
        ],
        "classifier__colsample_bytree": [
            0.70,
            0.85,
            1.00,
        ],
        "classifier__reg_alpha": [
            0.00,
            0.10,
            0.50,
        ],
        "classifier__reg_lambda": [
            1.00,
            2.00,
            5.00,
        ],
    }

    return RandomizedSearchCV(
        estimator=build_xgboost_pipeline(),
        param_distributions=parameter_distributions,
        n_iter=20,
        scoring="average_precision",
        cv=build_cross_validator(),
        refit=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        return_train_score=False,
        error_score="raise",
    )

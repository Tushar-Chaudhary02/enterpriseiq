"""Tests for finalist hyperparameter searches."""

from enterpriseiq.ml.tuning import (
    build_logistic_search,
    build_xgboost_search,
)


def test_logistic_search_uses_pr_auc() -> None:
    """Logistic Regression should optimize average precision."""

    search = build_logistic_search()

    assert search.scoring == "average_precision"
    assert search.refit is True


def test_xgboost_search_is_reproducible() -> None:
    """XGBoost randomized search should have fixed settings."""

    search = build_xgboost_search()

    assert search.scoring == "average_precision"
    assert search.n_iter == 20
    assert search.random_state == 42

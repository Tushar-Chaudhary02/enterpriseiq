"""Tests for candidate ML pipelines."""

from enterpriseiq.ml.pipeline import (
    build_candidate_pipelines,
)


def test_candidate_pipeline_names() -> None:
    """All planned candidate models should be available."""

    candidates = build_candidate_pipelines()

    assert set(candidates) == {
        "dummy_classifier",
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "xgboost",
    }


def test_candidate_pipelines_include_preprocessing() -> None:
    """Every model must include preprocessing inside its pipeline."""

    candidates = build_candidate_pipelines()

    for pipeline in candidates.values():
        assert "preprocessor" in pipeline.named_steps
        assert "classifier" in pipeline.named_steps

"""Calibrate, explain, track, and document the ML candidate."""

from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
from typing import cast

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline

from enterpriseiq.data.download import (
    DEFAULT_DATASET_PATH,
    download_dataset,
)
from enterpriseiq.ml.calibration import (
    CALIBRATION_METHOD,
    build_calibrated_model,
    calculate_probability_metrics,
)
from enterpriseiq.ml.data import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    DataSplit,
    create_train_test_split,
    load_modeling_data,
)
from enterpriseiq.ml.evaluation import (
    Metrics,
    calculate_binary_metrics,
)
from enterpriseiq.ml.pipeline import (
    build_candidate_pipelines,
)
from enterpriseiq.ml.threshold import (
    DEFAULT_FALSE_NEGATIVE_COST,
    DEFAULT_FALSE_POSITIVE_COST,
    DEFAULT_MINIMUM_RECALL,
    analyze_thresholds,
    select_operating_threshold,
)
from enterpriseiq.ml.tuning import build_cross_validator

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DAY5_RESULTS_PATH = PROJECT_ROOT / "reports" / "metrics" / "day5_tuning_results.json"

MODEL_DIRECTORY = PROJECT_ROOT / "artifacts" / "models"
METRICS_DIRECTORY = PROJECT_ROOT / "reports" / "metrics"
FIGURES_DIRECTORY = PROJECT_ROOT / "reports" / "figures"
MLFLOW_DIRECTORY = PROJECT_ROOT / "mlruns"

MODEL_PATH = MODEL_DIRECTORY / "enterpriseiq_churn_model_v1.joblib"

RESULTS_PATH = METRICS_DIRECTORY / "day6_calibration_results.json"

METADATA_PATH = METRICS_DIRECTORY / "day6_model_metadata.json"

THRESHOLD_RESULTS_PATH = METRICS_DIRECTORY / "day6_calibrated_threshold_analysis.csv"

IMPORTANCE_PATH = METRICS_DIRECTORY / "day6_feature_importance.csv"

MODEL_CARD_PATH = PROJECT_ROOT / "docs" / "model_card.md"

CALIBRATION_FIGURE_PATH = FIGURES_DIRECTORY / "day6_calibration_curve.png"

IMPORTANCE_FIGURE_PATH = FIGURES_DIRECTORY / "day6_feature_importance.png"

CONFUSION_FIGURE_PATH = FIGURES_DIRECTORY / "day6_calibrated_confusion_matrix.png"


def load_day5_candidate() -> tuple[str, Pipeline]:
    """Reconstruct the selected Day 5 model from committed metadata."""

    if not DAY5_RESULTS_PATH.exists():
        raise FileNotFoundError("Day 5 tuning results were not found.")

    payload = cast(
        dict[str, object],
        json.loads(DAY5_RESULTS_PATH.read_text(encoding="utf-8")),
    )

    selected_model_value = payload.get("selected_model")

    if not isinstance(selected_model_value, str):
        raise ValueError("Day 5 selected model is invalid.")

    all_parameters_value = payload.get("best_parameters")

    if not isinstance(all_parameters_value, dict):
        raise ValueError("Day 5 best parameters are invalid.")

    all_parameters = cast(
        dict[str, object],
        all_parameters_value,
    )

    selected_parameters_value = all_parameters.get(selected_model_value)

    if not isinstance(selected_parameters_value, dict):
        raise ValueError("Selected model parameters are missing.")

    selected_parameters = cast(
        dict[str, object],
        selected_parameters_value,
    )

    candidates = build_candidate_pipelines()

    if selected_model_value not in candidates:
        raise ValueError(f"Unknown selected model: {selected_model_value}")

    candidate = candidates[selected_model_value]

    candidate.set_params(**selected_parameters)

    return selected_model_value, candidate


def calculate_file_sha256(path: Path) -> str:
    """Calculate a reproducible SHA-256 file fingerprint."""

    digest = hashlib.sha256()

    with path.open("rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def get_metric(
    metrics: Metrics,
    metric_name: str,
) -> float:
    """Retrieve one scalar metric."""

    value = metrics[metric_name]

    if not isinstance(value, float):
        raise TypeError(f"{metric_name} is not a scalar metric.")

    return value


def calculate_business_cost(
    metrics: Metrics,
) -> float:
    """Calculate cost using the documented assumptions."""

    matrix = metrics["confusion_matrix"]

    if not isinstance(matrix, list):
        raise TypeError("Confusion matrix is invalid.")

    false_positives = matrix[0][1]
    false_negatives = matrix[1][0]

    return (
        false_positives * DEFAULT_FALSE_POSITIVE_COST
        + false_negatives * DEFAULT_FALSE_NEGATIVE_COST
    )


def evaluate_probabilities(
    y_true: pd.Series[int],
    probabilities: NDArray[np.float64],
    threshold: float,
) -> tuple[NDArray[np.int64], Metrics]:
    """Evaluate probabilities using one operating threshold."""

    predictions = (probabilities >= threshold).astype(np.int64)

    metrics = calculate_binary_metrics(
        y_true,
        predictions,
        probabilities,
    )

    return predictions, metrics


def save_calibration_figure(
    y_true: pd.Series[int],
    uncalibrated_probabilities: NDArray[np.float64],
    calibrated_probabilities: NDArray[np.float64],
) -> None:
    """Compare calibrated and uncalibrated reliability."""

    uncalibrated_fraction, uncalibrated_mean = calibration_curve(
        y_true,
        uncalibrated_probabilities,
        n_bins=10,
        strategy="quantile",
    )

    calibrated_fraction, calibrated_mean = calibration_curve(
        y_true,
        calibrated_probabilities,
        n_bins=10,
        strategy="quantile",
    )

    figure, axis = plt.subplots(figsize=(8, 7))

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="black",
        label="Perfect calibration",
    )

    axis.plot(
        uncalibrated_mean,
        uncalibrated_fraction,
        marker="o",
        label="Uncalibrated",
    )

    axis.plot(
        calibrated_mean,
        calibrated_fraction,
        marker="o",
        label="Sigmoid calibrated",
    )

    axis.set_title("Held-Out Probability Calibration")
    axis.set_xlabel("Mean Predicted Probability")
    axis.set_ylabel("Observed Churn Rate")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        CALIBRATION_FIGURE_PATH,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)


def calculate_feature_importance(
    model: object,
    data_split: DataSplit,
) -> pd.DataFrame:
    """Calculate held-out permutation importance."""

    result = permutation_importance(
        model,
        data_split.x_test,
        data_split.y_test,
        scoring="average_precision",
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )

    importance_mean = np.asarray(
        result.importances_mean,
        dtype=np.float64,
    )

    importance_std = np.asarray(
        result.importances_std,
        dtype=np.float64,
    )

    results = pd.DataFrame(
        {
            "feature": [str(column) for column in data_split.x_test.columns],
            "importance_mean": importance_mean,
            "importance_std": importance_std,
        }
    )

    return results.sort_values(
        "importance_mean",
        ascending=False,
    ).reset_index(drop=True)


def save_importance_figure(
    importance: pd.DataFrame,
) -> None:
    """Save the twelve most influential raw features."""

    top_features = importance.head(12).sort_values(
        "importance_mean",
        ascending=True,
    )

    figure, axis = plt.subplots(figsize=(9, 7))

    axis.barh(
        top_features["feature"],
        top_features["importance_mean"],
        xerr=top_features["importance_std"],
    )

    axis.set_title("Top Features by Permutation Importance")
    axis.set_xlabel("Decrease in Held-Out PR-AUC")
    axis.set_ylabel("Feature")

    figure.tight_layout()
    figure.savefig(
        IMPORTANCE_FIGURE_PATH,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_confusion_figure(
    y_true: pd.Series[int],
    predictions: NDArray[np.int64],
    threshold: float,
) -> None:
    """Save the calibrated candidate confusion matrix."""

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        display_labels=["Stay", "Churn"],
        cmap="Blues",
        colorbar=False,
    )

    plt.title(f"Calibrated Candidate Confusion Matrix (Threshold {threshold:.2f})")
    plt.tight_layout()
    plt.savefig(
        CONFUSION_FIGURE_PATH,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def write_model_card(
    model_name: str,
    selected_threshold: float,
    probability_metrics_before: dict[str, float],
    probability_metrics_after: dict[str, float],
    default_metrics: Metrics,
    selected_metrics: Metrics,
    feature_importance: pd.DataFrame,
) -> None:
    """Generate the committed EnterpriseIQ model card."""

    feature_lines_list: list[str] = []

    for row in feature_importance.head(10).itertuples():
        importance_mean = cast(float, row.importance_mean)

        feature_lines_list.append(f"- {row.feature}: {importance_mean:.5f}")

    feature_lines = "\n".join(feature_lines_list)

    default_cost = calculate_business_cost(default_metrics)
    selected_cost = calculate_business_cost(selected_metrics)

    calibration_lines = [
        "| Metric | Before calibration | After calibration |",
        "|---|---:|---:|",
        (
            f"| Brier score | "
            f"{probability_metrics_before['brier_score']:.4f} | "
            f"{probability_metrics_after['brier_score']:.4f} |"
        ),
        (
            f"| Log loss | "
            f"{probability_metrics_before['log_loss']:.4f} | "
            f"{probability_metrics_after['log_loss']:.4f} |"
        ),
    ]

    calibration_table = "\n".join(calibration_lines)

    classification_metrics = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1", "f1"),
        ("ROC-AUC", "roc_auc"),
        ("PR-AUC", "pr_auc"),
    ]

    performance_lines = [
        "| Metric | Threshold 0.50 | Operating threshold |",
        "|---|---:|---:|",
    ]

    for label, metric_name in classification_metrics:
        default_value = get_metric(
            default_metrics,
            metric_name,
        )
        selected_value = get_metric(
            selected_metrics,
            metric_name,
        )

        performance_lines.append(f"| {label} | {default_value:.3f} | {selected_value:.3f} |")

    performance_lines.append(f"| Simulated cost | ${default_cost:,.2f} | ${selected_cost:,.2f} |")

    performance_table = "\n".join(performance_lines)

    report = f"""# EnterpriseIQ Churn Model Card

## Model overview

- Model version: 1.0.0
- Base algorithm: {model_name}
- Calibration: sigmoid
- Operating threshold: {selected_threshold:.2f}
- Target: customer churn
- Positive class: churned customer

## Intended use

The model prioritizes telecommunications customers for human retention
review. It produces a churn probability and binary risk classification.

The model should support—not replace—human business decisions.

## Data

The model uses the public IBM Telco Customer Churn sample containing
7,043 fictional customer records. Features cover demographics, account
tenure, services, contract type, billing, and charges.

## Probability calibration

{calibration_table}

Lower values are better.

## Held-out classification performance

{performance_table}

## Business assumptions

- False-positive cost: ${DEFAULT_FALSE_POSITIVE_COST:.2f}
- False-negative cost: ${DEFAULT_FALSE_NEGATIVE_COST:.2f}
- Minimum recall target: {DEFAULT_MINIMUM_RECALL:.0%}

These are hypothetical portfolio assumptions and must be replaced with
validated organizational costs before real deployment.

## Permutation importance

{feature_lines}

Feature importance describes model reliance and does not establish causation.

## Limitations

- The dataset is fictional and relatively small.
- The data represents a static snapshot rather than customer history.
- No temporal or external validation has been performed.
- Business-cost assumptions are hypothetical.
- Feature importance is not causal.
- Performance and calibration may drift after deployment.
- Group-level fairness has not yet been fully evaluated.
- The model must not autonomously deny services or make financial decisions.
- Human review is required for retention actions.

## Monitoring requirements

A production implementation should monitor:

- feature distributions
- missing-value rates
- prediction distributions
- calibration
- precision and recall
- model drift
- subgroup performance
- latency and failures
"""

    MODEL_CARD_PATH.write_text(
        report,
        encoding="utf-8",
    )


def log_mlflow_experiment(
    model_name: str,
    model: object,
    selected_threshold: float,
    probability_metrics_before: dict[str, float],
    probability_metrics_after: dict[str, float],
    default_metrics: Metrics,
    selected_metrics: Metrics,
) -> str:
    """Record the finalized experiment in SQLite-backed local MLflow."""

    MLFLOW_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    mlflow_database = MLFLOW_DIRECTORY / "mlflow.db"
    tracking_uri = f"sqlite:///{mlflow_database.as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("enterpriseiq-customer-churn")

    with mlflow.start_run(run_name="calibrated-production-candidate-v1") as run:
        mlflow.log_params(
            {
                "base_model": model_name,
                "calibration_method": CALIBRATION_METHOD,
                "operating_threshold": selected_threshold,
                "false_positive_cost": (DEFAULT_FALSE_POSITIVE_COST),
                "false_negative_cost": (DEFAULT_FALSE_NEGATIVE_COST),
                "minimum_recall": (DEFAULT_MINIMUM_RECALL),
            }
        )

        mlflow.log_metrics(
            {
                "uncalibrated_brier": (probability_metrics_before["brier_score"]),
                "calibrated_brier": (probability_metrics_after["brier_score"]),
                "uncalibrated_log_loss": (probability_metrics_before["log_loss"]),
                "calibrated_log_loss": (probability_metrics_after["log_loss"]),
                "default_precision": get_metric(
                    default_metrics,
                    "precision",
                ),
                "default_recall": get_metric(
                    default_metrics,
                    "recall",
                ),
                "operating_precision": get_metric(
                    selected_metrics,
                    "precision",
                ),
                "operating_recall": get_metric(
                    selected_metrics,
                    "recall",
                ),
                "operating_f1": get_metric(
                    selected_metrics,
                    "f1",
                ),
                "operating_pr_auc": get_metric(
                    selected_metrics,
                    "pr_auc",
                ),
                "operating_roc_auc": get_metric(
                    selected_metrics,
                    "roc_auc",
                ),
                "operating_business_cost": (calculate_business_cost(selected_metrics)),
            }
        )

        mlflow.log_artifact(str(RESULTS_PATH))
        mlflow.log_artifact(str(METADATA_PATH))
        mlflow.log_artifact(str(IMPORTANCE_PATH))
        mlflow.log_artifact(str(MODEL_CARD_PATH))
        mlflow.log_artifact(str(CALIBRATION_FIGURE_PATH))
        mlflow.log_artifact(str(IMPORTANCE_FIGURE_PATH))
        mlflow.log_artifact(str(CONFUSION_FIGURE_PATH))

        skops_trusted_types: list[str] = [
            "numpy.dtype",
            "sklearn.calibration._CalibratedClassifier",
            "sklearn.calibration._SigmoidCalibration",
            "xgboost.core.Booster",
            "xgboost.sklearn.XGBClassifier",
        ]

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            skops_trusted_types=skops_trusted_types,
        )

        return str(run.info.run_id)


def main() -> None:
    """Finalize the EnterpriseIQ ML candidate."""

    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    METRICS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    FIGURES_DIRECTORY.mkdir(parents=True, exist_ok=True)

    dataset_path = download_dataset()

    features, target = load_modeling_data(dataset_path)

    data_split = create_train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
    )

    model_name, candidate = load_day5_candidate()

    uncalibrated_model = cast(
        Pipeline,
        clone(candidate),
    )

    uncalibrated_model.fit(
        data_split.x_train,
        data_split.y_train,
    )

    uncalibrated_test_probabilities = np.asarray(
        uncalibrated_model.predict_proba(data_split.x_test)[:, 1],
        dtype=np.float64,
    )

    calibrated_model = build_calibrated_model(
        cast(
            Pipeline,
            clone(candidate),
        )
    )

    out_of_fold_probabilities = np.asarray(
        cross_val_predict(
            calibrated_model,
            data_split.x_train,
            data_split.y_train,
            cv=build_cross_validator(),
            method="predict_proba",
            n_jobs=-1,
        ),
        dtype=np.float64,
    )[:, 1]

    threshold_results = analyze_thresholds(
        data_split.y_train,
        out_of_fold_probabilities,
    )

    selected_threshold = select_operating_threshold(
        threshold_results,
        minimum_recall=DEFAULT_MINIMUM_RECALL,
    )

    calibrated_model.fit(
        data_split.x_train,
        data_split.y_train,
    )

    calibrated_test_probabilities = np.asarray(
        calibrated_model.predict_proba(data_split.x_test)[:, 1],
        dtype=np.float64,
    )

    _, default_metrics = evaluate_probabilities(
        data_split.y_test,
        calibrated_test_probabilities,
        threshold=0.50,
    )

    selected_predictions, selected_metrics = evaluate_probabilities(
        data_split.y_test,
        calibrated_test_probabilities,
        threshold=selected_threshold,
    )

    probability_metrics_before = calculate_probability_metrics(
        data_split.y_test,
        uncalibrated_test_probabilities,
    )

    probability_metrics_after = calculate_probability_metrics(
        data_split.y_test,
        calibrated_test_probabilities,
    )

    feature_importance = calculate_feature_importance(
        calibrated_model,
        data_split,
    )

    joblib.dump(
        calibrated_model,
        MODEL_PATH,
    )

    threshold_results.to_csv(
        THRESHOLD_RESULTS_PATH,
        index=False,
    )

    feature_importance.to_csv(
        IMPORTANCE_PATH,
        index=False,
    )

    results = {
        "model_version": "1.0.0",
        "base_model": model_name,
        "calibration_method": CALIBRATION_METHOD,
        "selected_threshold": selected_threshold,
        "probability_metrics_before": (probability_metrics_before),
        "probability_metrics_after": (probability_metrics_after),
        "default_threshold_metrics": default_metrics,
        "operating_threshold_metrics": selected_metrics,
        "default_business_cost": (calculate_business_cost(default_metrics)),
        "operating_business_cost": (calculate_business_cost(selected_metrics)),
    }

    RESULTS_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    metadata = {
        "model_name": "enterpriseiq-churn",
        "model_version": "1.0.0",
        "base_algorithm": model_name,
        "calibration_method": CALIBRATION_METHOD,
        "operating_threshold": selected_threshold,
        "training_rows": len(data_split.x_train),
        "test_rows": len(data_split.x_test),
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "dataset_sha256": calculate_file_sha256(DEFAULT_DATASET_PATH),
        "python_version": platform.python_version(),
        "package_versions": {
            "mlflow": version("mlflow"),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scikit-learn": version("scikit-learn"),
            "xgboost": version("xgboost"),
        },
    }

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    save_calibration_figure(
        data_split.y_test,
        uncalibrated_test_probabilities,
        calibrated_test_probabilities,
    )

    save_importance_figure(feature_importance)

    save_confusion_figure(
        data_split.y_test,
        selected_predictions,
        selected_threshold,
    )

    write_model_card(
        model_name,
        selected_threshold,
        probability_metrics_before,
        probability_metrics_after,
        default_metrics,
        selected_metrics,
        feature_importance,
    )

    run_id = log_mlflow_experiment(
        model_name,
        calibrated_model,
        selected_threshold,
        probability_metrics_before,
        probability_metrics_after,
        default_metrics,
        selected_metrics,
    )

    print(f"Base model: {model_name}")
    print(f"Selected calibrated threshold: {selected_threshold:.2f}")

    print("\nProbability metrics before calibration:")
    print(
        json.dumps(
            probability_metrics_before,
            indent=2,
        )
    )

    print("\nProbability metrics after calibration:")
    print(
        json.dumps(
            probability_metrics_after,
            indent=2,
        )
    )

    print("\nOperating-threshold metrics:")
    print(
        json.dumps(
            selected_metrics,
            indent=2,
        )
    )

    print("\nTop ten features:")
    print(feature_importance.head(10).to_string(index=False))

    print(f"\nMLflow run ID: {run_id}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved model card: {MODEL_CARD_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()

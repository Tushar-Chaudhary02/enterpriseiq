"""Train and evaluate EnterpriseIQ baseline churn models."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline

from enterpriseiq.data.download import download_dataset
from enterpriseiq.ml.data import (
    DataSplit,
    create_train_test_split,
    load_modeling_data,
)
from enterpriseiq.ml.evaluation import (
    Metrics,
    calculate_binary_metrics,
)
from enterpriseiq.ml.pipeline import (
    build_dummy_pipeline,
    build_logistic_pipeline,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_DIRECTORY = PROJECT_ROOT / "artifacts" / "models"
METRICS_DIRECTORY = PROJECT_ROOT / "reports" / "metrics"
FIGURES_DIRECTORY = PROJECT_ROOT / "reports" / "figures"

MODEL_PATH = MODEL_DIRECTORY / "logistic_regression_baseline.joblib"

METRICS_PATH = METRICS_DIRECTORY / "day3_baseline_metrics.json"

REPORT_PATH = PROJECT_ROOT / "reports" / "day3_baseline_report.md"


def save_confusion_matrix(
    model_name: str,
    y_true: object,
    y_prediction: NDArray[np.int64],
) -> None:
    """Save a confusion-matrix visualization."""

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_prediction,
        display_labels=["Stay", "Churn"],
        cmap="Blues",
        colorbar=False,
    )

    plt.title(f"{model_name.replace('_', ' ').title()} Confusion Matrix")
    plt.tight_layout()

    output_path = FIGURES_DIRECTORY / f"day3_{model_name}_confusion_matrix.png"

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def train_and_evaluate(
    model_name: str,
    model: Pipeline,
    data_split: DataSplit,
) -> Metrics:
    """Train one pipeline and evaluate it on untouched test data."""

    model.fit(
        data_split.x_train,
        data_split.y_train,
    )

    predictions = np.asarray(
        model.predict(data_split.x_test),
        dtype=np.int64,
    )

    probabilities = np.asarray(
        model.predict_proba(data_split.x_test)[:, 1],
        dtype=np.float64,
    )

    metrics = calculate_binary_metrics(
        data_split.y_test,
        predictions,
        probabilities,
    )

    save_confusion_matrix(
        model_name,
        data_split.y_test,
        predictions,
    )

    return metrics


def get_metric(metrics: Metrics, metric_name: str) -> float:
    """Retrieve one scalar metric safely."""

    value = metrics[metric_name]

    if not isinstance(value, float):
        raise TypeError(f"{metric_name} is not a scalar metric.")

    return value


def write_report(
    metrics_by_model: dict[str, Metrics],
    data_split: DataSplit,
) -> None:
    """Write a human-readable model comparison report."""

    dummy = metrics_by_model["dummy_classifier"]
    logistic = metrics_by_model["logistic_regression"]

    f1_improvement = get_metric(logistic, "f1") - get_metric(dummy, "f1")

    dummy_row = (
        f"| Dummy classifier | {get_metric(dummy, 'accuracy'):.3f} | "
        f"{get_metric(dummy, 'precision'):.3f} | "
        f"{get_metric(dummy, 'recall'):.3f} | "
        f"{get_metric(dummy, 'f1'):.3f} | "
        f"{get_metric(dummy, 'roc_auc'):.3f} | "
        f"{get_metric(dummy, 'pr_auc'):.3f} |"
    )

    logistic_row = (
        f"| Logistic Regression | {get_metric(logistic, 'accuracy'):.3f} | "
        f"{get_metric(logistic, 'precision'):.3f} | "
        f"{get_metric(logistic, 'recall'):.3f} | "
        f"{get_metric(logistic, 'f1'):.3f} | "
        f"{get_metric(logistic, 'roc_auc'):.3f} | "
        f"{get_metric(logistic, 'pr_auc'):.3f} |"
    )

    report = f"""# Day 3 Baseline Model Report

## Dataset split

- Training customers: {len(data_split.x_train):,}
- Test customers: {len(data_split.x_test):,}
- Training churn rate: {data_split.y_train.mean():.2%}
- Test churn rate: {data_split.y_test.mean():.2%}
- Split strategy: stratified 80/20 split
- Random state: 42
- Classification threshold: 0.50

## Model comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
{dummy_row}
{logistic_row}

## Initial conclusion

Logistic Regression improved F1 by {f1_improvement:.3f} compared with
the dummy benchmark.

The dummy model demonstrates why accuracy is insufficient for this problem:
a model can achieve reasonable accuracy by favoring the majority class while
failing to identify customers who churn.

Logistic Regression provides the first meaningful predictive baseline. It is
not yet the final EnterpriseIQ model. Future experiments will compare additional
algorithms, class weighting, cross-validation, hyperparameter tuning, threshold
selection, calibration, and explainability.

## Business interpretation

- False positive: the company contacts a customer who would not have churned.
- False negative: the company misses a customer who actually churns.
- False negatives may represent lost retention opportunities.
- The final threshold should be selected using business costs, not accuracy alone.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    """Run the complete baseline training workflow."""

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

    models: dict[str, Pipeline] = {
        "dummy_classifier": build_dummy_pipeline(),
        "logistic_regression": build_logistic_pipeline(),
    }

    metrics_by_model: dict[str, Metrics] = {}

    for model_name, model in models.items():
        print(f"\nTraining: {model_name}")

        metrics = train_and_evaluate(
            model_name,
            model,
            data_split,
        )

        metrics_by_model[model_name] = metrics

        print(json.dumps(metrics, indent=2))

        if model_name == "logistic_regression":
            joblib.dump(model, MODEL_PATH)

    METRICS_PATH.write_text(
        json.dumps(metrics_by_model, indent=2),
        encoding="utf-8",
    )

    write_report(metrics_by_model, data_split)

    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()

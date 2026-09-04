"""Compare candidate churn models using cross-validation."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from enterpriseiq.ml.model_selection import (
    cross_validate_models,
    select_champion,
)
from enterpriseiq.ml.pipeline import (
    build_candidate_pipelines,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_DIRECTORY = PROJECT_ROOT / "artifacts" / "models"
METRICS_DIRECTORY = PROJECT_ROOT / "reports" / "metrics"
FIGURES_DIRECTORY = PROJECT_ROOT / "reports" / "figures"

CV_RESULTS_PATH = METRICS_DIRECTORY / "day4_cross_validation_results.csv"

METRICS_PATH = METRICS_DIRECTORY / "day4_model_comparison.json"

REPORT_PATH = PROJECT_ROOT / "reports" / "day4_model_comparison.md"

MODEL_PATH = MODEL_DIRECTORY / "day4_champion_model.joblib"

COMPARISON_FIGURE_PATH = FIGURES_DIRECTORY / "day4_cross_validation_comparison.png"

CONFUSION_FIGURE_PATH = FIGURES_DIRECTORY / "day4_champion_confusion_matrix.png"


def evaluate_champion(
    model: Pipeline,
    data_split: DataSplit,
) -> Metrics:
    """Fit the champion on training data and evaluate once on test data."""

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

    ConfusionMatrixDisplay.from_predictions(
        data_split.y_test,
        predictions,
        display_labels=["Stay", "Churn"],
        cmap="Blues",
        colorbar=False,
    )

    plt.title("Day 4 Champion Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        CONFUSION_FIGURE_PATH,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    return metrics


def save_comparison_chart(
    results: pd.DataFrame,
) -> None:
    """Save a chart comparing primary cross-validation metrics."""

    chart_data = results.set_index("model")[
        [
            "pr_auc_mean",
            "roc_auc_mean",
            "f1_mean",
        ]
    ]

    axis = chart_data.plot(
        kind="bar",
        figsize=(11, 6),
    )

    axis.set_title("Five-Fold Cross-Validation Model Comparison")
    axis.set_xlabel("Model")
    axis.set_ylabel("Mean Score")
    axis.set_ylim(0, 1)
    axis.legend(
        [
            "PR-AUC",
            "ROC-AUC",
            "F1",
        ]
    )

    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(
        COMPARISON_FIGURE_PATH,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def get_metric(
    metrics: Metrics,
    metric_name: str,
) -> float:
    """Retrieve one scalar metric."""

    value = metrics[metric_name]

    if not isinstance(value, float):
        raise TypeError(f"{metric_name} is not a scalar metric.")

    return value


def build_cv_table(results: pd.DataFrame) -> str:
    """Create a Markdown table from cross-validation results."""

    lines = [
        ("| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |"),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in results.iterrows():
        line = (
            f"| {row['model']} "
            f"| {float(row['accuracy_mean']):.3f} "
            f"| {float(row['precision_mean']):.3f} "
            f"| {float(row['recall_mean']):.3f} "
            f"| {float(row['f1_mean']):.3f} "
            f"| {float(row['roc_auc_mean']):.3f} "
            f"| {float(row['pr_auc_mean']):.3f} |"
        )
        lines.append(line)

    return "\n".join(lines)


def write_report(
    cv_results: pd.DataFrame,
    champion_name: str,
    test_metrics: Metrics,
) -> None:
    """Create the Day 4 model-comparison report."""

    matrix = test_metrics["confusion_matrix"]

    if not isinstance(matrix, list):
        raise TypeError("Confusion matrix is invalid.")

    true_negatives, false_positives = matrix[0]
    false_negatives, true_positives = matrix[1]

    table = build_cv_table(cv_results)

    report = f"""# Day 4 Model Comparison

## Selection methodology

Five-fold stratified cross-validation was performed using only the training
partition. Preprocessing was fitted independently inside every fold.

The champion was selected using mean cross-validation PR-AUC. The held-out
test set was evaluated only after model selection.

## Cross-validation results

{table}

## Selected model

**{champion_name}**

## Champion held-out test metrics

| Metric | Score |
|---|---:|
| Accuracy | {get_metric(test_metrics, "accuracy"):.3f} |
| Precision | {get_metric(test_metrics, "precision"):.3f} |
| Recall | {get_metric(test_metrics, "recall"):.3f} |
| F1 | {get_metric(test_metrics, "f1"):.3f} |
| ROC-AUC | {get_metric(test_metrics, "roc_auc"):.3f} |
| PR-AUC | {get_metric(test_metrics, "pr_auc"):.3f} |

## Confusion matrix interpretation

- True negatives: {true_negatives}
- False positives: {false_positives}
- False negatives: {false_negatives}
- True positives: {true_positives}

## Current conclusion

The selected model is the strongest initial candidate according to
cross-validation PR-AUC. It is not yet the final production model.

The next phase will tune promising models, analyze metric trade-offs, and
select a probability threshold using the business cost of false positives
and false negatives.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    """Run cross-validation, select a champion, and evaluate it."""

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

    candidate_models = build_candidate_pipelines()

    cv_results = cross_validate_models(
        candidate_models,
        data_split,
        folds=5,
    )

    champion_name = select_champion(
        cv_results,
        primary_metric="pr_auc_mean",
    )

    champion_model = candidate_models[champion_name]

    champion_test_metrics = evaluate_champion(
        champion_model,
        data_split,
    )

    joblib.dump(
        champion_model,
        MODEL_PATH,
    )

    cv_results.to_csv(
        CV_RESULTS_PATH,
        index=False,
    )

    results_payload = {
        "selection_metric": "pr_auc_mean",
        "champion_model": champion_name,
        "cross_validation": cv_results.to_dict(orient="records"),
        "champion_test_metrics": champion_test_metrics,
    }

    METRICS_PATH.write_text(
        json.dumps(results_payload, indent=2),
        encoding="utf-8",
    )

    save_comparison_chart(cv_results)

    write_report(
        cv_results,
        champion_name,
        champion_test_metrics,
    )

    print("\nCross-validation results:")
    print(cv_results.to_string(index=False))

    print(f"\nSelected champion: {champion_name}")

    print("\nChampion test metrics:")
    print(
        json.dumps(
            champion_test_metrics,
            indent=2,
        )
    )

    print(f"\nSaved champion model: {MODEL_PATH}")
    print(f"Saved CV results: {CV_RESULTS_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()

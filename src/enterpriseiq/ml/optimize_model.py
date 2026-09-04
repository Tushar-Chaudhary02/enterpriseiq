"""Tune finalist models and select a business operating threshold."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeAlias, cast

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_predict,
)
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
from enterpriseiq.ml.threshold import (
    DEFAULT_FALSE_NEGATIVE_COST,
    DEFAULT_FALSE_POSITIVE_COST,
    DEFAULT_MINIMUM_RECALL,
    analyze_thresholds,
    select_operating_threshold,
)
from enterpriseiq.ml.tuning import (
    build_cross_validator,
    build_logistic_search,
    build_xgboost_search,
)

SearchType: TypeAlias = GridSearchCV | RandomizedSearchCV

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_DIRECTORY = PROJECT_ROOT / "artifacts" / "models"
METRICS_DIRECTORY = PROJECT_ROOT / "reports" / "metrics"
FIGURES_DIRECTORY = PROJECT_ROOT / "reports" / "figures"

MODEL_PATH = MODEL_DIRECTORY / "day5_production_candidate.joblib"

TUNING_RESULTS_PATH = METRICS_DIRECTORY / "day5_tuning_results.json"

THRESHOLD_RESULTS_PATH = METRICS_DIRECTORY / "day5_threshold_analysis.csv"

REPORT_PATH = PROJECT_ROOT / "reports" / "day5_production_candidate_report.md"

THRESHOLD_FIGURE_PATH = FIGURES_DIRECTORY / "day5_threshold_metrics.png"

COST_FIGURE_PATH = FIGURES_DIRECTORY / "day5_business_cost.png"

CURVES_FIGURE_PATH = FIGURES_DIRECTORY / "day5_pr_roc_curves.png"

CONFUSION_FIGURE_PATH = FIGURES_DIRECTORY / "day5_optimized_confusion_matrix.png"


def make_parameters_json_safe(
    parameters: dict[str, object],
) -> dict[str, object]:
    """Convert NumPy parameter values into JSON-safe values."""

    safe_parameters: dict[str, object] = {}

    for name, value in parameters.items():
        if isinstance(value, np.generic):
            safe_parameters[name] = value.item()
        else:
            safe_parameters[name] = value

    return safe_parameters


def get_metric(
    metrics: Metrics,
    metric_name: str,
) -> float:
    """Retrieve one scalar model metric."""

    value = metrics[metric_name]

    if not isinstance(value, float):
        raise TypeError(f"{metric_name} is not a scalar metric.")

    return value


def calculate_business_cost(
    metrics: Metrics,
    *,
    false_positive_cost: float,
    false_negative_cost: float,
) -> float:
    """Calculate business cost from a confusion matrix."""

    matrix = metrics["confusion_matrix"]

    if not isinstance(matrix, list):
        raise TypeError("Confusion matrix is invalid.")

    false_positives = matrix[0][1]
    false_negatives = matrix[1][0]

    return false_positives * false_positive_cost + false_negatives * false_negative_cost


def save_threshold_figures(
    threshold_results: pd.DataFrame,
    selected_threshold: float,
) -> None:
    """Save threshold-performance and business-cost charts."""

    figure, axis = plt.subplots(figsize=(10, 6))

    axis.plot(
        threshold_results["threshold"],
        threshold_results["precision"],
        label="Precision",
    )
    axis.plot(
        threshold_results["threshold"],
        threshold_results["recall"],
        label="Recall",
    )
    axis.plot(
        threshold_results["threshold"],
        threshold_results["f1"],
        label="F1",
    )
    axis.axvline(
        selected_threshold,
        color="black",
        linestyle="--",
        label=f"Selected threshold: {selected_threshold:.2f}",
    )

    axis.set_title("Out-of-Fold Threshold Trade-offs")
    axis.set_xlabel("Probability Threshold")
    axis.set_ylabel("Metric Score")
    axis.set_ylim(0, 1)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        THRESHOLD_FIGURE_PATH,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 6))

    axis.plot(
        threshold_results["threshold"],
        threshold_results["business_cost"],
        color="darkred",
    )
    axis.axvline(
        selected_threshold,
        color="black",
        linestyle="--",
        label=f"Selected threshold: {selected_threshold:.2f}",
    )

    axis.set_title("Estimated Business Cost by Threshold")
    axis.set_xlabel("Probability Threshold")
    axis.set_ylabel("Estimated Cost ($)")
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        COST_FIGURE_PATH,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_test_curves(
    y_true: pd.Series[int],
    probabilities: NDArray[np.float64],
) -> None:
    """Save held-out precision-recall and ROC curves."""

    precision, recall, _ = precision_recall_curve(
        y_true,
        probabilities,
    )

    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_true,
        probabilities,
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
    )

    axes[0].plot(recall, precision)
    axes[0].set_title("Held-Out Precision-Recall Curve")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)

    axes[1].plot(
        false_positive_rate,
        true_positive_rate,
    )
    axes[1].plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
    )
    axes[1].set_title("Held-Out ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)

    figure.tight_layout()
    figure.savefig(
        CURVES_FIGURE_PATH,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)


def evaluate_threshold(
    y_true: pd.Series[int],
    probabilities: NDArray[np.float64],
    threshold: float,
) -> tuple[NDArray[np.int64], Metrics]:
    """Evaluate predicted probabilities at one threshold."""

    predictions = (probabilities >= threshold).astype(np.int64)

    metrics = calculate_binary_metrics(
        y_true,
        predictions,
        probabilities,
    )

    return predictions, metrics


def save_selected_confusion_matrix(
    y_true: pd.Series[int],
    predictions: NDArray[np.int64],
    selected_threshold: float,
) -> None:
    """Save the final candidate confusion matrix."""

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        display_labels=["Stay", "Churn"],
        cmap="Blues",
        colorbar=False,
    )

    plt.title(f"Production Candidate Confusion Matrix (Threshold {selected_threshold:.2f})")
    plt.tight_layout()
    plt.savefig(
        CONFUSION_FIGURE_PATH,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def build_tuning_table(
    scores: dict[str, float],
) -> str:
    """Create a Markdown table of tuning results."""

    lines = [
        "| Finalist | Best CV PR-AUC |",
        "|---|---:|",
    ]

    for model_name, score in sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        lines.append(f"| {model_name} | {score:.4f} |")

    return "\n".join(lines)


def build_test_comparison_table(
    default_metrics: Metrics,
    selected_metrics: Metrics,
    default_cost: float,
    selected_cost: float,
) -> str:
    """Create the held-out threshold comparison Markdown table."""

    metric_names = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1", "f1"),
        ("ROC-AUC", "roc_auc"),
        ("PR-AUC", "pr_auc"),
    ]

    lines = [
        "| Metric | Threshold 0.50 | Optimized threshold |",
        "|---|---:|---:|",
    ]

    for label, metric_name in metric_names:
        default_value = get_metric(default_metrics, metric_name)
        selected_value = get_metric(selected_metrics, metric_name)

        lines.append(f"| {label} | {default_value:.3f} | {selected_value:.3f} |")

    lines.append(f"| Estimated cost | ${default_cost:,.2f} | ${selected_cost:,.2f} |")

    return "\n".join(lines)


def write_report(
    best_scores: dict[str, float],
    best_parameters: dict[str, dict[str, object]],
    champion_name: str,
    selected_threshold: float,
    default_metrics: Metrics,
    selected_metrics: Metrics,
) -> None:
    """Write the production-candidate report."""

    tuning_table = build_tuning_table(best_scores)

    default_cost = calculate_business_cost(
        default_metrics,
        false_positive_cost=DEFAULT_FALSE_POSITIVE_COST,
        false_negative_cost=DEFAULT_FALSE_NEGATIVE_COST,
    )

    selected_cost = calculate_business_cost(
        selected_metrics,
        false_positive_cost=DEFAULT_FALSE_POSITIVE_COST,
        false_negative_cost=DEFAULT_FALSE_NEGATIVE_COST,
    )

    parameter_text = json.dumps(
        best_parameters[champion_name],
        indent=2,
        sort_keys=True,
    )

    comparison_table = build_test_comparison_table(
        default_metrics,
        selected_metrics,
        default_cost,
        selected_cost,
    )

    report = f"""# Day 5 Production ML Candidate

## Methodology

Logistic Regression and XGBoost were tuned using five-fold stratified
cross-validation on the training partition. Mean cross-validation PR-AUC
was used to select the finalist.

The operating threshold was selected using out-of-fold probabilities from
the training partition. The held-out test set was not used for model or
threshold selection.

## Tuning results

{tuning_table}

## Selected model

**{champion_name}**

## Selected hyperparameters

```json
{parameter_text}
```

## Business assumptions

- False-positive cost: ${DEFAULT_FALSE_POSITIVE_COST:.2f}
- False-negative cost: ${DEFAULT_FALSE_NEGATIVE_COST:.2f}
- Minimum recall requirement: {DEFAULT_MINIMUM_RECALL:.0%}

These are hypothetical and configurable portfolio assumptions.

## Selected operating threshold

**{selected_threshold:.2f}**

## Held-out test comparison

{comparison_table}

ROC-AUC and PR-AUC remain unchanged because they evaluate probability
ranking rather than one classification threshold.

## Production status

This artifact is the current production ML candidate. It still requires
experiment tracking, probability calibration, explainability, inference
testing, monitoring, API serving, and deployment before it can be called
a production model.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    """Tune finalists and generate the production candidate."""

    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    METRICS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    FIGURES_DIRECTORY.mkdir(parents=True, exist_ok=True)

    dataset_path = download_dataset()
    features, target = load_modeling_data(dataset_path)

    data_split: DataSplit = create_train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
    )

    searches: dict[str, SearchType] = {
        "logistic_regression": build_logistic_search(),
        "xgboost": build_xgboost_search(),
    }

    best_scores: dict[str, float] = {}
    best_parameters: dict[str, dict[str, object]] = {}
    best_estimators: dict[str, Pipeline] = {}

    for model_name, search in searches.items():
        print(f"\nTuning: {model_name}")

        search.fit(
            data_split.x_train,
            data_split.y_train,
        )

        best_scores[model_name] = float(search.best_score_)

        best_parameters[model_name] = make_parameters_json_safe(dict(search.best_params_))

        best_estimators[model_name] = cast(
            Pipeline,
            search.best_estimator_,
        )

        print(f"Best CV PR-AUC: {best_scores[model_name]:.6f}")
        print(f"Best parameters: {best_parameters[model_name]}")

    champion_name = max(
        best_scores,
        key=best_scores.__getitem__,
    )

    champion_model = best_estimators[champion_name]

    print(f"\nSelected tuned model: {champion_name}")

    out_of_fold_matrix = np.asarray(
        cross_val_predict(
            champion_model,
            data_split.x_train,
            data_split.y_train,
            cv=build_cross_validator(),
            method="predict_proba",
            n_jobs=-1,
        ),
        dtype=np.float64,
    )

    out_of_fold_probabilities = out_of_fold_matrix[:, 1]

    threshold_results = analyze_thresholds(
        data_split.y_train,
        out_of_fold_probabilities,
        false_positive_cost=DEFAULT_FALSE_POSITIVE_COST,
        false_negative_cost=DEFAULT_FALSE_NEGATIVE_COST,
    )

    selected_threshold = select_operating_threshold(
        threshold_results,
        minimum_recall=DEFAULT_MINIMUM_RECALL,
    )

    print(f"Selected operating threshold: {selected_threshold:.2f}")

    champion_model.fit(
        data_split.x_train,
        data_split.y_train,
    )

    test_probabilities = np.asarray(
        champion_model.predict_proba(data_split.x_test)[:, 1],
        dtype=np.float64,
    )

    _, default_metrics = evaluate_threshold(
        data_split.y_test,
        test_probabilities,
        threshold=0.50,
    )

    selected_predictions, selected_metrics = evaluate_threshold(
        data_split.y_test,
        test_probabilities,
        threshold=selected_threshold,
    )

    joblib.dump(
        champion_model,
        MODEL_PATH,
    )

    threshold_results.to_csv(
        THRESHOLD_RESULTS_PATH,
        index=False,
    )

    results_payload = {
        "business_assumptions": {
            "false_positive_cost": DEFAULT_FALSE_POSITIVE_COST,
            "false_negative_cost": DEFAULT_FALSE_NEGATIVE_COST,
            "minimum_recall": DEFAULT_MINIMUM_RECALL,
        },
        "best_cv_pr_auc": best_scores,
        "best_parameters": best_parameters,
        "selected_model": champion_name,
        "selected_threshold": selected_threshold,
        "held_out_default_threshold_metrics": default_metrics,
        "held_out_selected_threshold_metrics": selected_metrics,
    }

    TUNING_RESULTS_PATH.write_text(
        json.dumps(results_payload, indent=2),
        encoding="utf-8",
    )

    save_threshold_figures(
        threshold_results,
        selected_threshold,
    )

    save_test_curves(
        data_split.y_test,
        test_probabilities,
    )

    save_selected_confusion_matrix(
        data_split.y_test,
        selected_predictions,
        selected_threshold,
    )

    write_report(
        best_scores,
        best_parameters,
        champion_name,
        selected_threshold,
        default_metrics,
        selected_metrics,
    )

    print("\nDefault threshold metrics:")
    print(json.dumps(default_metrics, indent=2))

    print("\nOptimized threshold metrics:")
    print(json.dumps(selected_metrics, indent=2))

    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved tuning results: {TUNING_RESULTS_PATH}")
    print(f"Saved threshold results: {THRESHOLD_RESULTS_PATH}")
    print(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()

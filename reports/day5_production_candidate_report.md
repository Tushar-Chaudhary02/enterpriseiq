# Day 5 Production ML Candidate

## Methodology

Logistic Regression and XGBoost were tuned using five-fold stratified
cross-validation on the training partition. Mean cross-validation PR-AUC
was used to select the finalist.

The operating threshold was selected using out-of-fold probabilities from
the training partition. The held-out test set was not used for model or
threshold selection.

## Tuning results

| Finalist | Best CV PR-AUC |
|---|---:|
| xgboost | 0.6712 |
| logistic_regression | 0.6615 |

## Selected model

**xgboost**

## Selected hyperparameters

```json
{
  "classifier__colsample_bytree": 0.7,
  "classifier__learning_rate": 0.05,
  "classifier__max_depth": 2,
  "classifier__min_child_weight": 5,
  "classifier__n_estimators": 250,
  "classifier__reg_alpha": 0.5,
  "classifier__reg_lambda": 2.0,
  "classifier__subsample": 1.0
}
```

## Business assumptions

- False-positive cost: $50.00
- False-negative cost: $500.00
- Minimum recall requirement: 75%

These are hypothetical and configurable portfolio assumptions.

## Selected operating threshold

**0.08**

## Held-out test comparison

| Metric | Threshold 0.50 | Optimized threshold |
|---|---:|---:|
| Accuracy | 0.798 | 0.588 |
| Precision | 0.658 | 0.389 |
| Recall | 0.500 | 0.968 |
| F1 | 0.568 | 0.555 |
| ROC-AUC | 0.846 | 0.846 |
| PR-AUC | 0.661 | 0.661 |
| Estimated cost | $98,350.00 | $34,400.00 |

ROC-AUC and PR-AUC remain unchanged because they evaluate probability
ranking rather than one classification threshold.

## Production status

This artifact is the current production ML candidate. It still requires
experiment tracking, probability calibration, explainability, inference
testing, monitoring, API serving, and deployment before it can be called
a production model.

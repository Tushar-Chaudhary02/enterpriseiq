# Day 3 Baseline Model Report

## Dataset split

- Training customers: 5,634
- Test customers: 1,409
- Training churn rate: 26.54%
- Test churn rate: 26.54%
- Split strategy: stratified 80/20 split
- Random state: 42
- Classification threshold: 0.50

## Model comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Dummy classifier | 0.735 | 0.000 | 0.000 | 0.000 | 0.500 | 0.265 |
| Logistic Regression | 0.806 | 0.657 | 0.559 | 0.604 | 0.842 | 0.634 |

## Initial conclusion

Logistic Regression improved F1 by 0.604 compared with
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

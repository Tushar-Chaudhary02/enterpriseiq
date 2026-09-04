# Day 4 Model Comparison

## Selection methodology

Five-fold stratified cross-validation was performed using only the training
partition. Preprocessing was fitted independently inside every fold.

The champion was selected using mean cross-validation PR-AUC. The held-out
test set was evaluated only after model selection.

## Cross-validation results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| xgboost | 0.802 | 0.660 | 0.527 | 0.586 | 0.846 | 0.664 |
| logistic_regression | 0.803 | 0.655 | 0.543 | 0.593 | 0.846 | 0.661 |
| random_forest | 0.786 | 0.627 | 0.486 | 0.547 | 0.820 | 0.611 |
| decision_tree | 0.726 | 0.485 | 0.492 | 0.488 | 0.651 | 0.373 |
| dummy_classifier | 0.735 | 0.000 | 0.000 | 0.000 | 0.500 | 0.265 |

## Selected model

**xgboost**

## Champion held-out test metrics

| Metric | Score |
|---|---:|
| Accuracy | 0.803 |
| Precision | 0.658 |
| Recall | 0.535 |
| F1 | 0.590 |
| ROC-AUC | 0.843 |
| PR-AUC | 0.657 |

## Confusion matrix interpretation

- True negatives: 931
- False positives: 104
- False negatives: 174
- True positives: 200

## Current conclusion

The selected model is the strongest initial candidate according to
cross-validation PR-AUC. It is not yet the final production model.

The next phase will tune promising models, analyze metric trade-offs, and
select a probability threshold using the business cost of false positives
and false negatives.

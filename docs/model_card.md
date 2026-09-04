# EnterpriseIQ Churn Model Card

## Model overview

- Model version: 1.0.0
- Base algorithm: xgboost
- Calibration: sigmoid
- Operating threshold: 0.08
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

| Metric | Before calibration | After calibration |
|---|---:|---:|
| Brier score | 0.1356 | 0.1358 |
| Log loss | 0.4153 | 0.4223 |

Lower values are better.

## Held-out classification performance

| Metric | Threshold 0.50 | Operating threshold |
|---|---:|---:|
| Accuracy | 0.802 | 0.573 |
| Precision | 0.673 | 0.381 |
| Recall | 0.495 | 0.971 |
| F1 | 0.570 | 0.547 |
| ROC-AUC | 0.848 | 0.848 |
| PR-AUC | 0.666 | 0.666 |
| Simulated cost | $99,000.00 | $35,050.00 |

## Business assumptions

- False-positive cost: $50.00
- False-negative cost: $500.00
- Minimum recall target: 75%

These are hypothetical portfolio assumptions and must be replaced with
validated organizational costs before real deployment.

## Permutation importance

- tenure: 0.12672
- Contract: 0.07354
- InternetService: 0.01988
- TotalCharges: 0.01847
- OnlineSecurity: 0.01676
- MonthlyCharges: 0.01628
- TechSupport: 0.01417
- PaperlessBilling: 0.00572
- PaymentMethod: 0.00558
- MultipleLines: 0.00382

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

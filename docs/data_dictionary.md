# EnterpriseIQ Data Dictionary

## Dataset grain

Each row represents one telecommunications customer.

## Prediction target

`Churn` indicates whether the customer discontinued service.

| Column | Type | Role | Description |
|---|---|---|---|
| customerID | String | Identifier | Unique customer identifier |
| gender | Category | Feature | Customer gender |
| SeniorCitizen | Category | Feature | Whether the customer is a senior citizen |
| Partner | Category | Feature | Whether the customer has a partner |
| Dependents | Category | Feature | Whether the customer has dependents |
| tenure | Integer | Feature | Number of months with the company |
| PhoneService | Category | Feature | Whether phone service is active |
| MultipleLines | Category | Feature | Whether multiple phone lines are active |
| InternetService | Category | Feature | Type of internet service |
| OnlineSecurity | Category | Feature | Whether online security is active |
| OnlineBackup | Category | Feature | Whether online backup is active |
| DeviceProtection | Category | Feature | Whether device protection is active |
| TechSupport | Category | Feature | Whether technical support is active |
| StreamingTV | Category | Feature | Whether streaming TV is active |
| StreamingMovies | Category | Feature | Whether streaming movies are active |
| Contract | Category | Feature | Customer contract type |
| PaperlessBilling | Category | Feature | Whether paperless billing is enabled |
| PaymentMethod | Category | Feature | Customer payment method |
| MonthlyCharges | Float | Feature | Current monthly charge |
| TotalCharges | Float | Feature | Accumulated charges |
| Churn | Category | Target | Whether the customer left |

## Known data-quality considerations

- `TotalCharges` is initially read as text because some records contain blank strings.
- `SeniorCitizen` is stored numerically but represents a category.
- `customerID` is an identifier and must not be used as a predictive feature.
- `Churn` is the target and must be separated from model inputs.
- Model preprocessing must be fitted only on training data to prevent leakage.
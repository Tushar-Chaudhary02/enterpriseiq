# EnterpriseIQ

EnterpriseIQ is a production-oriented customer intelligence platform that combines
machine learning, structured enterprise data, retrieval-augmented generation, and
agentic AI workflows.

## Planned capabilities

- Predict customer churn with a reproducible ML pipeline
- Explain individual customer risk factors
- Query customer information through controlled read-only tools
- Retrieve retention policies using RAG
- Generate evidence-backed retention recommendations
- Require human approval for sensitive actions
- Track model, retrieval, agent, latency, and cost metrics
- Deploy through Docker, CI/CD, and AWS

## Planned technology stack

- Python
- Pandas and NumPy
- scikit-learn and XGBoost
- MLflow
- FastAPI and Pydantic
- PostgreSQL and pgvector
- LangChain and LangGraph
- RAG evaluation
- LLM observability
- Docker
- GitHub Actions
- AWS

## Current project status

### Completed

- Professional Python package and configuration
- Automated testing, linting, typing, and CI
- Reproducible IBM customer-churn dataset download
- Dataset schema validation
- Exploratory data analysis and visualizations
- Leakage-safe preprocessing pipeline
- Stratified train/test splitting
- Dummy-classifier benchmark
- Logistic Regression baseline
- Automated model metrics and confusion matrices
- Five-fold stratified cross-validation
- Logistic Regression, Decision Tree, Random Forest, and XGBoost comparison
- PR-AUC-based model selection
- Held-out champion-model evaluation
- Automated model-comparison reports

### Current model workflow

Run baseline training with:

```bash
python -m enterpriseiq.ml.train_baseline

## Local setup

Create the virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install the project:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Create local configuration:

```bash
cp .env.example .env
```

Run the application:

```bash
python -m enterpriseiq.main
```

Run the quality checks:

```bash
ruff check .
ruff format --check .
mypy src tests
python -m pytest
```

Run cross-validated model comparison with:

```bash
python -m enterpriseiq.ml.compare_models

## Project status

This project is under active development as part of an implementation-focused
AI/ML engineering program.
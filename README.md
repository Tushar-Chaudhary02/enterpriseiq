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

Day 1 establishes the project package, configuration, testing, code-quality checks,
continuous integration, and architecture documentation.

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

## Project status

This project is under active development as part of an implementation-focused
AI/ML engineering program.
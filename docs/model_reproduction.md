# Model Artifact Reproduction

The trained churn model is intentionally not committed to Git because binary
model artifacts can be large and depend on library versions.

## Prerequisites

Activate the project virtual environment and install the project dependencies:

```bash
source .venv/bin/activate
python -m pip install -e ".[dev]"
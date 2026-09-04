"""Tests for customer-analysis prompt construction."""

import json

from enterpriseiq.llm.prompts import (
    PROMPT_VERSION,
    build_customer_analysis_input,
)


def test_prompt_contains_only_supplied_evidence() -> None:
    """The prompt should preserve supplied customer and model evidence."""

    prompt = build_customer_analysis_input(
        customer={
            "tenure": 2,
            "Contract": "Month-to-month",
            "MonthlyCharges": 95.70,
        },
        prediction="high_risk",
        churn_probability=0.84,
        decision_threshold=0.08,
        support_summary="Customer reported intermittent service.",
        analysis_goal="Recommend retention actions.",
    )

    payload = json.loads(prompt)

    assert PROMPT_VERSION == "customer-retention-v1"
    assert payload["customer_attributes"]["tenure"] == 2
    assert payload["machine_learning_output"]["prediction"] == "high_risk"
    assert payload["machine_learning_output"]["churn_probability"] == 0.84
    assert payload["support_summary"] == ("Customer reported intermittent service.")


def test_prompt_identifies_missing_support_summary() -> None:
    """Missing support evidence must be represented explicitly."""

    prompt = build_customer_analysis_input(
        customer={"tenure": 2},
        prediction="high_risk",
        churn_probability=0.84,
        decision_threshold=0.08,
        support_summary=None,
        analysis_goal="Recommend retention actions.",
    )

    payload = json.loads(prompt)

    assert payload["support_summary"] == "No support summary was supplied."

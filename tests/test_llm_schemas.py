"""Tests for structured LLM response schemas."""

import pytest
from pydantic import ValidationError

from enterpriseiq.llm.schemas import (
    LLMCustomerAnalysis,
    RetentionAction,
)


def test_customer_analysis_accepts_valid_structured_output() -> None:
    """A complete structured response should pass validation."""

    analysis = LLMCustomerAnalysis(
        executive_summary=(
            "The customer has elevated churn risk and may benefit from proactive support."
        ),
        risk_factors=[
            "Month-to-month contract",
            "Short customer tenure",
        ],
        recommended_actions=[
            RetentionAction(
                priority="high",
                action="Contact the customer to understand current concerns.",
                rationale=(
                    "A proactive conversation can provide information "
                    "that is not present in the supplied data."
                ),
                requires_human_approval=False,
            ),
        ],
        customer_message_draft=(
            "We would appreciate the opportunity to learn more about "
            "your current service experience."
        ),
        requires_human_review=False,
        limitations=[
            "No recent support-interaction details were supplied.",
        ],
    )

    assert analysis.recommended_actions[0].priority == "high"
    assert len(analysis.risk_factors) == 2
    assert analysis.requires_human_review is False


def test_customer_analysis_rejects_unexpected_fields() -> None:
    """Unexpected LLM fields must be rejected."""

    invalid_response = {
        "executive_summary": "Elevated churn risk.",
        "risk_factors": ["Short tenure"],
        "recommended_actions": [
            {
                "priority": "high",
                "action": "Contact the customer.",
                "rationale": "More information is needed.",
                "requires_human_approval": False,
            },
        ],
        "customer_message_draft": "We would like to hear from you.",
        "requires_human_review": False,
        "limitations": ["Support history was not supplied."],
        "invented_discount": "$500",
    }

    with pytest.raises(ValidationError):
        LLMCustomerAnalysis.model_validate(invalid_response)

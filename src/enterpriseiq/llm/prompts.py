"""Versioned prompts for structured customer analysis."""

import json

PROMPT_VERSION = "customer-retention-v1"

SYSTEM_INSTRUCTIONS = """
You are an enterprise customer-retention decision-support assistant.

Create a concise analysis using only the evidence supplied in the input.

Rules:
1. Do not invent customer history, company policies, support interactions,
   discounts, credits, account details, or financial information.
2. Treat the machine-learning prediction as a risk estimate, not proof that
   the customer will churn.
3. Do not describe correlations as proven causes.
4. Clearly identify important missing information in the limitations field.
5. Recommendations involving credits, discounts, pricing, contract changes,
   refunds, or account changes must require human approval.
6. The customer-message draft must be professional and must not mention
   internal model scores or claim certainty.
7. Return a response matching the required structured-output schema.
""".strip()


def build_customer_analysis_input(
    *,
    customer: dict[str, str | int | float],
    prediction: str,
    churn_probability: float,
    decision_threshold: float,
    support_summary: str | None,
    analysis_goal: str,
) -> str:
    """Build a deterministic JSON input containing only supplied evidence."""

    payload = {
        "analysis_goal": analysis_goal,
        "customer_attributes": customer,
        "machine_learning_output": {
            "prediction": prediction,
            "churn_probability": churn_probability,
            "decision_threshold": decision_threshold,
        },
        "support_summary": (
            support_summary if support_summary else "No support summary was supplied."
        ),
        "constraints": [
            "Use only the supplied evidence.",
            "Do not invent company policy.",
            "Do not treat the prediction as certainty.",
            "Flag sensitive actions for human approval.",
        ],
    }

    return json.dumps(
        payload,
        indent=2,
        sort_keys=True,
    )

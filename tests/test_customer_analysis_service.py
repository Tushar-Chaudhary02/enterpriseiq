"""Tests for ML and LLM customer-analysis orchestration."""

from typing import Any

from enterpriseiq.api.analysis_service import CustomerAnalysisService
from enterpriseiq.api.schemas import (
    ChurnPredictionRequest,
    CustomerAnalysisRequest,
)
from enterpriseiq.api.service import PredictionResult
from enterpriseiq.llm.schemas import (
    LLMCustomerAnalysis,
    RetentionAction,
    StructuredGenerationResult,
)


class FakePredictionService:
    """Deterministic prediction service used by the unit test."""

    threshold = 0.08
    model_name = "test-xgboost"
    model_version = "test-version"

    def predict(
        self,
        request: ChurnPredictionRequest,
    ) -> PredictionResult:
        """Return a fixed prediction without loading a model."""

        return PredictionResult(
            prediction="high_risk",
            churn_probability=0.84,
        )


class FakeCustomerAnalysisProvider:
    """Record supplied evidence and return fixed structured output."""

    provider_name = "fake"
    model_name = "fake-structured-model"

    def __init__(self) -> None:
        self.received_arguments: dict[str, Any] = {}

    def generate_customer_analysis(
        self,
        *,
        customer: dict[str, str | int | float],
        prediction: str,
        churn_probability: float,
        decision_threshold: float,
        support_summary: str | None,
        analysis_goal: str,
    ) -> StructuredGenerationResult:
        """Return deterministic structured analysis."""

        self.received_arguments = {
            "customer": customer,
            "prediction": prediction,
            "churn_probability": churn_probability,
            "decision_threshold": decision_threshold,
            "support_summary": support_summary,
            "analysis_goal": analysis_goal,
        }

        return StructuredGenerationResult(
            analysis=LLMCustomerAnalysis(
                executive_summary="The customer has elevated churn risk.",
                risk_factors=[
                    "Short tenure",
                    "Month-to-month contract",
                ],
                recommended_actions=[
                    RetentionAction(
                        priority="high",
                        action="Contact the customer.",
                        rationale="Additional information is needed.",
                        requires_human_approval=False,
                    ),
                ],
                customer_message_draft=("We would appreciate your feedback about your service."),
                requires_human_review=False,
                limitations=[
                    "Only the supplied customer information was available.",
                ],
            ),
            provider=self.provider_name,
            model=self.model_name,
            response_id="fake-response-001",
            input_tokens=100,
            output_tokens=75,
            total_tokens=175,
            latency_ms=25.0,
        )


def build_customer_request() -> CustomerAnalysisRequest:
    """Create a valid customer-analysis request."""

    return CustomerAnalysisRequest(
        customer=ChurnPredictionRequest(
            customer_id="customer-001",
            gender="Female",
            senior_citizen=0,
            partner="No",
            dependents="No",
            tenure=2,
            phone_service="Yes",
            multiple_lines="No",
            internet_service="Fiber optic",
            online_security="No",
            online_backup="No",
            device_protection="No",
            tech_support="No",
            streaming_tv="Yes",
            streaming_movies="Yes",
            contract="Month-to-month",
            paperless_billing="Yes",
            payment_method="Electronic check",
            monthly_charges=95.70,
            total_charges=191.40,
        ),
        support_summary="Customer reported intermittent service.",
        analysis_goal="Recommend retention actions.",
    )


def test_service_passes_fixed_ml_evidence_to_llm() -> None:
    """The LLM provider must receive the deterministic ML result."""

    prediction_service = FakePredictionService()
    llm_provider = FakeCustomerAnalysisProvider()

    service = CustomerAnalysisService(
        prediction_service=prediction_service,
        llm_provider=llm_provider,
    )

    result = service.analyze(build_customer_request())

    assert result.prediction.prediction == "high_risk"
    assert result.prediction.churn_probability == 0.84
    assert result.generation.provider == "fake"

    assert llm_provider.received_arguments["prediction"] == "high_risk"
    assert llm_provider.received_arguments["churn_probability"] == 0.84
    assert llm_provider.received_arguments["decision_threshold"] == 0.08

    customer = llm_provider.received_arguments["customer"]

    assert customer["tenure"] == 2
    assert customer["Contract"] == "Month-to-month"

"""Orchestration service combining deterministic ML and structured LLM output."""

from dataclasses import dataclass
from typing import Protocol

from enterpriseiq.api.schemas import CustomerAnalysisRequest
from enterpriseiq.api.service import (
    PredictionResult,
    PredictionService,
)
from enterpriseiq.config import Settings
from enterpriseiq.llm.openai_provider import (
    OpenAICustomerAnalysisProvider,
)
from enterpriseiq.llm.provider import CustomerAnalysisProvider
from enterpriseiq.llm.schemas import StructuredGenerationResult


@dataclass(frozen=True)
class CustomerAnalysisResult:
    """Combined deterministic prediction and structured generation result."""

    prediction: PredictionResult
    generation: StructuredGenerationResult


class CustomerAnalysisServiceProtocol(Protocol):
    """Interface consumed by the FastAPI layer."""

    def analyze(
        self,
        request: CustomerAnalysisRequest,
    ) -> CustomerAnalysisResult:
        """Create one customer analysis."""


class CustomerAnalysisService:
    """Combine churn inference with provider-neutral structured generation."""

    def __init__(
        self,
        *,
        prediction_service: PredictionService,
        llm_provider: CustomerAnalysisProvider,
    ) -> None:
        self._prediction_service = prediction_service
        self._llm_provider = llm_provider

    def analyze(
        self,
        request: CustomerAnalysisRequest,
    ) -> CustomerAnalysisResult:
        """Predict churn, then generate analysis from that fixed evidence."""

        prediction = self._prediction_service.predict(request.customer)

        generation = self._llm_provider.generate_customer_analysis(
            customer=request.customer.to_model_record(),
            prediction=prediction.prediction,
            churn_probability=prediction.churn_probability,
            decision_threshold=self._prediction_service.threshold,
            support_summary=request.support_summary,
            analysis_goal=request.analysis_goal,
        )

        return CustomerAnalysisResult(
            prediction=prediction,
            generation=generation,
        )


def build_customer_analysis_service(
    *,
    settings: Settings,
    prediction_service: PredictionService,
) -> CustomerAnalysisService | None:
    """Build the service only when provider credentials are configured."""

    if settings.openai_api_key is None:
        return None

    provider = OpenAICustomerAnalysisProvider(
        api_key=settings.openai_api_key.get_secret_value(),
        model_name=settings.openai_model,
        timeout_seconds=settings.openai_timeout_seconds,
        max_retries=settings.openai_max_retries,
        max_output_tokens=settings.openai_max_output_tokens,
    )

    return CustomerAnalysisService(
        prediction_service=prediction_service,
        llm_provider=provider,
    )

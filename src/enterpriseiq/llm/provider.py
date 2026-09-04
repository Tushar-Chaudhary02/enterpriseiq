"""Provider-neutral interface for structured LLM generation."""

from typing import Protocol

from enterpriseiq.llm.schemas import StructuredGenerationResult


class LLMProviderError(RuntimeError):
    """Raised when a configured LLM provider cannot complete a request."""


class CustomerAnalysisProvider(Protocol):
    """Interface implemented by customer-analysis LLM providers."""

    provider_name: str
    model_name: str

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
        """Generate a validated customer analysis."""

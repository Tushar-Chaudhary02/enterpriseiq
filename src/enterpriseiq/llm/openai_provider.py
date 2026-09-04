"""OpenAI Responses API implementation of structured customer analysis."""

from time import perf_counter

from openai import OpenAI, OpenAIError

from enterpriseiq.llm.prompts import (
    SYSTEM_INSTRUCTIONS,
    build_customer_analysis_input,
)
from enterpriseiq.llm.provider import LLMProviderError
from enterpriseiq.llm.schemas import (
    LLMCustomerAnalysis,
    StructuredGenerationResult,
)


class OpenAICustomerAnalysisProvider:
    """Generate Pydantic-validated analysis with the OpenAI Responses API."""

    provider_name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str,
        timeout_seconds: float,
        max_retries: int,
        max_output_tokens: int,
    ) -> None:
        self.model_name = model_name
        self._max_output_tokens = max_output_tokens

        self._client = OpenAI(
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

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
        """Generate one validated structured customer analysis."""

        model_input = build_customer_analysis_input(
            customer=customer,
            prediction=prediction,
            churn_probability=churn_probability,
            decision_threshold=decision_threshold,
            support_summary=support_summary,
            analysis_goal=analysis_goal,
        )

        started_at = perf_counter()

        try:
            response = self._client.responses.parse(
                model=self.model_name,
                instructions=SYSTEM_INSTRUCTIONS,
                input=model_input,
                text_format=LLMCustomerAnalysis,
                max_output_tokens=self._max_output_tokens,
                store=False,
            )
        except OpenAIError as error:
            raise LLMProviderError("The OpenAI request could not be completed.") from error

        latency_ms = round(
            (perf_counter() - started_at) * 1000,
            2,
        )

        if response.output_parsed is None:
            raise LLMProviderError(
                "OpenAI did not return a completed structured customer analysis."
            )

        analysis = LLMCustomerAnalysis.model_validate(response.output_parsed)
        usage = response.usage

        input_tokens = usage.input_tokens if usage is not None else 0
        output_tokens = usage.output_tokens if usage is not None else 0
        total_tokens = usage.total_tokens if usage is not None else 0

        return StructuredGenerationResult(
            analysis=analysis,
            provider=self.provider_name,
            model=self.model_name,
            response_id=response.id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
        )

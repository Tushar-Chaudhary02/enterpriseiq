"""Validated schemas and metadata for structured LLM generation."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RetentionAction(BaseModel):
    """One proposed customer-retention action."""

    model_config = ConfigDict(extra="forbid")

    priority: Literal["high", "medium", "low"]
    action: str = Field(
        min_length=1,
        max_length=240,
    )
    rationale: str = Field(
        min_length=1,
        max_length=500,
    )
    requires_human_approval: bool


class LLMCustomerAnalysis(BaseModel):
    """Business analysis generated under a strict output schema."""

    model_config = ConfigDict(extra="forbid")

    executive_summary: str = Field(
        min_length=1,
        max_length=700,
    )
    risk_factors: list[str] = Field(
        min_length=1,
        max_length=5,
    )
    recommended_actions: list[RetentionAction] = Field(
        min_length=1,
        max_length=5,
    )
    customer_message_draft: str = Field(
        min_length=1,
        max_length=1000,
    )
    requires_human_review: bool
    limitations: list[str] = Field(
        min_length=1,
        max_length=4,
    )


@dataclass(frozen=True)
class StructuredGenerationResult:
    """Validated provider response with operational metadata."""

    analysis: LLMCustomerAnalysis
    provider: str
    model: str
    response_id: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float

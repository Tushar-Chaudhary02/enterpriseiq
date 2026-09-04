"""FastAPI application for EnterpriseIQ."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast

from fastapi import FastAPI, HTTPException, Request, status

from enterpriseiq.api.analysis_service import (
    CustomerAnalysisServiceProtocol,
    build_customer_analysis_service,
)
from enterpriseiq.api.schemas import (
    ChurnPredictionRequest,
    ChurnPredictionResponse,
    CustomerAnalysisRequest,
    CustomerAnalysisResponse,
    LivenessResponse,
    LLMResponseMetadata,
    ReadinessResponse,
)
from enterpriseiq.api.service import (
    ChurnPredictionService,
    PredictionService,
)
from enterpriseiq.config import get_settings
from enterpriseiq.llm.prompts import PROMPT_VERSION
from enterpriseiq.llm.provider import LLMProviderError

LOGGER = logging.getLogger(__name__)


def create_app(
    prediction_service: PredictionService | None = None,
    customer_analysis_service: CustomerAnalysisServiceProtocol | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        ml_service = (
            prediction_service if prediction_service is not None else ChurnPredictionService.load()
        )
        application.state.prediction_service = ml_service

        if customer_analysis_service is not None:
            ai_service = customer_analysis_service
        elif prediction_service is not None:
            # Dependency-injected application instances are deterministic
            # test fixtures and should not create external provider clients.
            ai_service = None
        else:
            ai_service = build_customer_analysis_service(
                settings=get_settings(),
                prediction_service=ml_service,
            )

        application.state.customer_analysis_service = ai_service
        yield

    settings = get_settings()

    application = FastAPI(
        title="EnterpriseIQ API",
        description=(
            "Production-style API for customer churn prediction and enterprise AI workflows."
        ),
        version=settings.app_version,
        lifespan=lifespan,
    )

    @application.get(
        "/health/live",
        response_model=LivenessResponse,
        tags=["health"],
    )
    def liveness() -> LivenessResponse:
        return LivenessResponse(
            status="ok",
            service="enterpriseiq-api",
        )

    @application.get(
        "/health/ready",
        response_model=ReadinessResponse,
        tags=["health"],
    )
    def readiness(request: Request) -> ReadinessResponse:
        service = cast(
            PredictionService,
            request.app.state.prediction_service,
        )

        return ReadinessResponse(
            status="ready",
            model_name=service.model_name,
            model_version=service.model_version,
            llm_configured=(request.app.state.customer_analysis_service is not None),
        )

    @application.post(
        "/api/v1/ml/churn/predict",
        response_model=ChurnPredictionResponse,
        status_code=status.HTTP_200_OK,
        tags=["machine-learning"],
    )
    def predict_churn(
        payload: ChurnPredictionRequest,
        request: Request,
    ) -> ChurnPredictionResponse:
        service = cast(
            PredictionService,
            request.app.state.prediction_service,
        )

        try:
            prediction = service.predict(payload)
        except Exception as error:
            LOGGER.exception("Churn prediction failed.")

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Model prediction failed.",
            ) from error

        return ChurnPredictionResponse(
            customer_id=payload.customer_id,
            prediction=prediction.prediction,
            churn_probability=prediction.churn_probability,
            decision_threshold=service.threshold,
            model_name=service.model_name,
            model_version=service.model_version,
        )

    @application.post(
        "/api/v1/ai/customer-analysis",
        response_model=CustomerAnalysisResponse,
        status_code=status.HTTP_200_OK,
        tags=["generative-ai"],
    )
    def analyze_customer(
        payload: CustomerAnalysisRequest,
        request: Request,
    ) -> CustomerAnalysisResponse:
        service = cast(
            CustomerAnalysisServiceProtocol | None,
            request.app.state.customer_analysis_service,
        )

        if service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The LLM provider is not configured.",
            )

        try:
            result = service.analyze(payload)
        except LLMProviderError as error:
            LOGGER.warning(
                "Customer analysis provider failed: %s",
                error,
            )

            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=("The customer-analysis provider is temporarily unavailable."),
            ) from error
        except Exception as error:
            LOGGER.exception("Customer analysis failed.")

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Customer analysis failed.",
            ) from error

        prediction_service_state = cast(
            PredictionService,
            request.app.state.prediction_service,
        )

        churn_prediction = ChurnPredictionResponse(
            customer_id=payload.customer.customer_id,
            prediction=result.prediction.prediction,
            churn_probability=result.prediction.churn_probability,
            decision_threshold=prediction_service_state.threshold,
            model_name=prediction_service_state.model_name,
            model_version=prediction_service_state.model_version,
        )

        return CustomerAnalysisResponse(
            customer_id=payload.customer.customer_id,
            churn_prediction=churn_prediction,
            analysis=result.generation.analysis,
            metadata=LLMResponseMetadata(
                provider=result.generation.provider,
                model=result.generation.model,
                prompt_version=PROMPT_VERSION,
                response_id=result.generation.response_id,
                input_tokens=result.generation.input_tokens,
                output_tokens=result.generation.output_tokens,
                total_tokens=result.generation.total_tokens,
                latency_ms=result.generation.latency_ms,
            ),
        )

    return application


app = create_app()

"""FastAPI application for EnterpriseIQ."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast

from fastapi import FastAPI, HTTPException, Request, status

from enterpriseiq.api.schemas import (
    ChurnPredictionRequest,
    ChurnPredictionResponse,
    LivenessResponse,
    ReadinessResponse,
)
from enterpriseiq.api.service import (
    ChurnPredictionService,
    PredictionService,
)

LOGGER = logging.getLogger(__name__)


def create_app(
    prediction_service: PredictionService | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        service = (
            prediction_service if prediction_service is not None else ChurnPredictionService.load()
        )
        application.state.prediction_service = service
        yield

    application = FastAPI(
        title="EnterpriseIQ API",
        description=(
            "Production-style API for customer churn prediction and enterprise AI workflows."
        ),
        version="0.7.0",
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

    return application


app = create_app()

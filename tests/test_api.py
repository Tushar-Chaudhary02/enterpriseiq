"""Tests for EnterpriseIQ API endpoints."""

from fastapi.testclient import TestClient

from enterpriseiq.api.analysis_service import CustomerAnalysisResult
from enterpriseiq.api.main import create_app
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

VALID_PAYLOAD: dict[str, object] = {
    "customer_id": "demo-001",
    "gender": "Female",
    "senior_citizen": 0,
    "partner": "No",
    "dependents": "No",
    "tenure": 2,
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "streaming_movies": "Yes",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 95.70,
    "total_charges": 191.40,
}


class FakePredictionService:
    """Deterministic prediction service used by API tests."""

    model_name = "xgboost"
    model_version = "1.0.0"
    threshold = 0.08

    def predict(
        self,
        request: ChurnPredictionRequest,
    ) -> PredictionResult:
        return PredictionResult(
            prediction="high_risk",
            churn_probability=0.91,
        )


class FakeCustomerAnalysisService:
    """Deterministic structured-generation workflow used by API tests."""

    def analyze(
        self,
        request: CustomerAnalysisRequest,
    ) -> CustomerAnalysisResult:
        return CustomerAnalysisResult(
            prediction=PredictionResult(
                prediction="high_risk",
                churn_probability=0.91,
            ),
            generation=StructuredGenerationResult(
                analysis=LLMCustomerAnalysis(
                    executive_summary=("The customer has elevated modeled churn risk."),
                    risk_factors=[
                        "Short tenure",
                        "Month-to-month contract",
                    ],
                    recommended_actions=[
                        RetentionAction(
                            priority="high",
                            action="Schedule a retention review.",
                            rationale=("Confirm needs before proposing an account change."),
                            requires_human_approval=True,
                        ),
                    ],
                    customer_message_draft=("We would like to discuss your service experience."),
                    requires_human_review=True,
                    limitations=[
                        "No company policy was supplied.",
                    ],
                ),
                provider="fake",
                model="fake-model",
                response_id="response-test-001",
                input_tokens=120,
                output_tokens=80,
                total_tokens=200,
                latency_ms=12.5,
            ),
        )


def test_health_endpoints() -> None:
    with TestClient(create_app(FakePredictionService())) as client:
        live_response = client.get("/health/live")
        ready_response = client.get("/health/ready")

    assert live_response.status_code == 200
    assert live_response.json()["status"] == "ok"

    assert ready_response.status_code == 200
    assert ready_response.json() == {
        "status": "ready",
        "model_name": "xgboost",
        "model_version": "1.0.0",
        "llm_configured": False,
    }


def test_churn_prediction_endpoint() -> None:
    with TestClient(create_app(FakePredictionService())) as client:
        response = client.post(
            "/api/v1/ml/churn/predict",
            json=VALID_PAYLOAD,
        )

    assert response.status_code == 200
    assert response.json() == {
        "customer_id": "demo-001",
        "prediction": "high_risk",
        "churn_probability": 0.91,
        "decision_threshold": 0.08,
        "model_name": "xgboost",
        "model_version": "1.0.0",
    }


def test_invalid_contract_is_rejected() -> None:
    invalid_payload = {
        **VALID_PAYLOAD,
        "contract": "Weekly",
    }

    with TestClient(create_app(FakePredictionService())) as client:
        response = client.post(
            "/api/v1/ml/churn/predict",
            json=invalid_payload,
        )

    assert response.status_code == 422


def test_unknown_fields_are_rejected() -> None:
    invalid_payload = {
        **VALID_PAYLOAD,
        "unknown_feature": "unexpected",
    }

    with TestClient(create_app(FakePredictionService())) as client:
        response = client.post(
            "/api/v1/ml/churn/predict",
            json=invalid_payload,
        )

    assert response.status_code == 422


def test_customer_analysis_endpoint_returns_structured_output() -> None:
    application = create_app(
        FakePredictionService(),
        FakeCustomerAnalysisService(),
    )

    with TestClient(application) as client:
        response = client.post(
            "/api/v1/ai/customer-analysis",
            json={
                "customer": VALID_PAYLOAD,
                "support_summary": ("The customer reported intermittent service."),
            },
        )

    assert response.status_code == 200

    payload = response.json()

    assert payload["churn_prediction"]["prediction"] == "high_risk"
    assert payload["analysis"]["requires_human_review"] is True
    assert payload["metadata"] == {
        "provider": "fake",
        "model": "fake-model",
        "prompt_version": "customer-retention-v1",
        "response_id": "response-test-001",
        "input_tokens": 120,
        "output_tokens": 80,
        "total_tokens": 200,
        "latency_ms": 12.5,
    }


def test_customer_analysis_requires_configured_provider() -> None:
    with TestClient(create_app(FakePredictionService())) as client:
        response = client.post(
            "/api/v1/ai/customer-analysis",
            json={"customer": VALID_PAYLOAD},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "The LLM provider is not configured."}

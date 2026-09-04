"""Pydantic schemas used by the EnterpriseIQ API."""

from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

YesNo: TypeAlias = Literal["Yes", "No"]
RiskLabel: TypeAlias = Literal["high_risk", "low_risk"]
InternetOption: TypeAlias = Literal[
    "Yes",
    "No",
    "No internet service",
]


class ChurnPredictionRequest(BaseModel):
    """Customer attributes required by the churn model."""

    model_config = ConfigDict(extra="forbid")

    customer_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
    )
    gender: Literal["Female", "Male"]
    senior_citizen: Literal[0, 1]
    partner: YesNo
    dependents: YesNo
    tenure: int = Field(ge=0, le=100)
    phone_service: YesNo
    multiple_lines: Literal[
        "Yes",
        "No",
        "No phone service",
    ]
    internet_service: Literal[
        "DSL",
        "Fiber optic",
        "No",
    ]
    online_security: InternetOption
    online_backup: InternetOption
    device_protection: InternetOption
    tech_support: InternetOption
    streaming_tv: InternetOption
    streaming_movies: InternetOption
    contract: Literal[
        "Month-to-month",
        "One year",
        "Two year",
    ]
    paperless_billing: YesNo
    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    monthly_charges: float = Field(ge=0, le=1000)
    total_charges: float = Field(ge=0)

    def to_model_record(self) -> dict[str, str | int | float]:
        """Convert API names into the original training-data columns."""

        return {
            "gender": self.gender,
            "SeniorCitizen": self.senior_citizen,
            "Partner": self.partner,
            "Dependents": self.dependents,
            "tenure": self.tenure,
            "PhoneService": self.phone_service,
            "MultipleLines": self.multiple_lines,
            "InternetService": self.internet_service,
            "OnlineSecurity": self.online_security,
            "OnlineBackup": self.online_backup,
            "DeviceProtection": self.device_protection,
            "TechSupport": self.tech_support,
            "StreamingTV": self.streaming_tv,
            "StreamingMovies": self.streaming_movies,
            "Contract": self.contract,
            "PaperlessBilling": self.paperless_billing,
            "PaymentMethod": self.payment_method,
            "MonthlyCharges": self.monthly_charges,
            "TotalCharges": self.total_charges,
        }


class ChurnPredictionResponse(BaseModel):
    """Structured model prediction returned to an API client."""

    customer_id: str | None
    prediction: RiskLabel
    churn_probability: float = Field(ge=0, le=1)
    decision_threshold: float = Field(ge=0, le=1)
    model_name: str
    model_version: str


class LivenessResponse(BaseModel):
    """Response proving that the API process is running."""

    status: Literal["ok"]
    service: str


class ReadinessResponse(BaseModel):
    """Response proving that the model is loaded."""

    status: Literal["ready"]
    model_name: str
    model_version: str

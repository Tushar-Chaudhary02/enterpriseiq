"""Loading and inference service for the churn model."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from enterpriseiq.api.schemas import (
    ChurnPredictionRequest,
    RiskLabel,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "enterpriseiq_churn_model_v1.joblib"
CALIBRATION_RESULTS_PATH = PROJECT_ROOT / "reports" / "metrics" / "day6_calibration_results.json"


class ModelLoadError(RuntimeError):
    """Raised when the production model cannot be loaded."""


class ProbabilityModel(Protocol):
    """Interface required from the trained model."""

    def predict_proba(
        self,
        features: pd.DataFrame,
    ) -> NDArray[np.float64]:
        """Return class probabilities."""


@dataclass(frozen=True)
class PredictionResult:
    """Internal result returned by the prediction service."""

    prediction: RiskLabel
    churn_probability: float


class PredictionService(Protocol):
    """Interface used by the API layer."""

    model_name: str
    model_version: str
    threshold: float

    def predict(
        self,
        request: ChurnPredictionRequest,
    ) -> PredictionResult:
        """Generate a churn prediction."""


def classify_risk(
    probability: float,
    threshold: float,
) -> RiskLabel:
    """Convert a churn probability into a business risk label."""

    if probability >= threshold:
        return "high_risk"

    return "low_risk"


class ChurnPredictionService:
    """Serve predictions from the finalized churn model."""

    def __init__(
        self,
        model: ProbabilityModel,
        threshold: float,
        model_name: str,
        model_version: str,
    ) -> None:
        self._model = model
        self.threshold = threshold
        self.model_name = model_name
        self.model_version = model_version

    @classmethod
    def load(cls) -> "ChurnPredictionService":
        """Load the finalized model and associated metadata."""

        if not MODEL_PATH.exists():
            raise ModelLoadError(f"Model artifact was not found: {MODEL_PATH}")

        if not CALIBRATION_RESULTS_PATH.exists():
            raise ModelLoadError(f"Calibration metadata was not found: {CALIBRATION_RESULTS_PATH}")

        loaded_artifact: object = joblib.load(MODEL_PATH)

        if isinstance(loaded_artifact, dict):
            artifact_dictionary = cast(
                dict[str, object],
                loaded_artifact,
            )
            model_object = artifact_dictionary.get("model")
        else:
            model_object = loaded_artifact

        if model_object is None or not callable(getattr(model_object, "predict_proba", None)):
            raise ModelLoadError("The saved artifact does not contain a valid probability model.")

        raw_metadata: object = json.loads(CALIBRATION_RESULTS_PATH.read_text(encoding="utf-8"))

        if not isinstance(raw_metadata, dict):
            raise ModelLoadError("Calibration metadata must be a JSON object.")

        metadata = cast(dict[str, object], raw_metadata)

        threshold_value = metadata.get("selected_threshold")
        model_name_value = metadata.get("base_model")
        model_version_value = metadata.get("model_version")

        if isinstance(threshold_value, bool) or not isinstance(threshold_value, (int, float)):
            raise ModelLoadError("Selected threshold is invalid.")

        if not isinstance(model_name_value, str):
            raise ModelLoadError("Base model name is invalid.")

        if not isinstance(model_version_value, str):
            raise ModelLoadError("Model version is invalid.")

        return cls(
            model=cast(ProbabilityModel, model_object),
            threshold=float(threshold_value),
            model_name=model_name_value,
            model_version=model_version_value,
        )

    def predict(
        self,
        request: ChurnPredictionRequest,
    ) -> PredictionResult:
        """Predict churn probability for one customer."""

        features = pd.DataFrame([request.to_model_record()])

        probabilities = np.asarray(
            self._model.predict_proba(features),
            dtype=np.float64,
        )

        if probabilities.shape != (1, 2):
            raise ValueError(
                f"The model returned an unexpected probability shape: {probabilities.shape}"
            )

        churn_probability = float(probabilities[0, 1])

        return PredictionResult(
            prediction=classify_risk(
                churn_probability,
                self.threshold,
            ),
            churn_probability=churn_probability,
        )

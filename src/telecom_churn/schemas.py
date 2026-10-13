"""Data schemas and Pydantic validation models for Telecom Churn Prediction."""

from enum import Enum
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator


class YesNo(str, Enum):
    YES = "yes"
    NO = "no"


class RiskTier(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class CustomerInput(BaseModel):
    """Schema for individual customer activity data."""

    # Optional metadata that might be in the source data
    state: Optional[str] = Field(default=None, description="US State abbreviation (optional)")
    area_code: Optional[Union[int, str]] = Field(default=None, description="Area code (optional)")
    phone_number: Optional[str] = Field(default=None, description="Phone number (optional)")

    # Account info
    account_length: int = Field(..., ge=1, le=500, description="Account duration in months / days")
    intl_plan: Union[str, bool, int] = Field(..., description="International plan ('yes'/'no' or boolean/int)")
    voice_mail_plan: Union[str, bool, int] = Field(..., description="Voice mail plan ('yes'/'no' or boolean/int)")
    number_vmail_messages: int = Field(default=0, ge=0, le=200, description="Number of voicemail messages")

    # Day activity
    total_day_minutes: float = Field(..., ge=0.0, le=1000.0, description="Total minutes of day calls")
    total_day_calls: int = Field(..., ge=0, le=500, description="Total number of day calls")
    total_day_charge: float = Field(..., ge=0.0, le=300.0, description="Total charge for day calls")

    # Evening activity
    total_eve_minutes: float = Field(..., ge=0.0, le=1000.0, description="Total minutes of evening calls")
    total_eve_calls: int = Field(..., ge=0, le=500, description="Total number of evening calls")
    total_eve_charge: float = Field(..., ge=0.0, le=300.0, description="Total charge for evening calls")

    # Night activity
    total_night_minutes: float = Field(..., ge=0.0, le=1000.0, description="Total minutes of night calls")
    total_night_calls: int = Field(..., ge=0, le=500, description="Total number of night calls")
    total_night_charge: float = Field(..., ge=0.0, le=300.0, description="Total charge for night calls")

    # International activity
    total_intl_minutes: float = Field(..., ge=0.0, le=500.0, description="Total minutes of international calls")
    total_intl_calls: int = Field(..., ge=0, le=100, description="Total number of international calls")
    total_intl_charge: float = Field(..., ge=0.0, le=100.0, description="Total charge for international calls")

    # Service calls
    number_customer_service_calls: int = Field(
        ..., ge=0, le=50, description="Number of customer service calls made"
    )

    @field_validator("intl_plan", "voice_mail_plan", mode="before")
    @classmethod
    def normalize_binary(cls, v: Any) -> str:
        """Normalize boolean, int, or string inputs to 'yes'/'no'."""
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, (int, float)):
            return "yes" if v == 1 else "no"
        if isinstance(v, str):
            val = v.strip().lower()
            if val in {"yes", "y", "true", "1"}:
                return "yes"
            if val in {"no", "n", "false", "0"}:
                return "no"
            raise ValueError(f"Value '{v}' cannot be parsed as yes/no.")
        return "no"

    model_config = {
        "json_schema_extra": {
            "example": {
                "account_length": 128,
                "intl_plan": "no",
                "voice_mail_plan": "yes",
                "number_vmail_messages": 25,
                "total_day_minutes": 265.1,
                "total_day_calls": 110,
                "total_day_charge": 45.07,
                "total_eve_minutes": 197.4,
                "total_eve_calls": 99,
                "total_eve_charge": 16.78,
                "total_night_minutes": 244.7,
                "total_night_calls": 91,
                "total_night_charge": 11.01,
                "total_intl_minutes": 10.0,
                "total_intl_calls": 3,
                "total_intl_charge": 2.70,
                "number_customer_service_calls": 1,
            }
        }
    }


class PredictionOutput(BaseModel):
    """Schema for prediction results for a single customer."""

    churn_prediction: bool = Field(..., description="True if customer is predicted to churn")
    churn_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of churn (0.0 to 1.0)")
    risk_tier: RiskTier = Field(..., description="Risk category: LOW, MEDIUM, or HIGH")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score")
    key_drivers: List[str] = Field(default_factory=list, description="Top factors driving the prediction")
    recommendations: List[str] = Field(
        default_factory=list, description="Actionable retention recommendations"
    )


class BatchPredictionOutput(BaseModel):
    """Schema for batch prediction results."""

    total_records: int
    predicted_churn_count: int
    churn_rate_percentage: float
    average_churn_probability: float
    risk_distribution: Dict[str, int]
    predictions: List[PredictionOutput]


class HealthResponse(BaseModel):
    """Service health response."""

    status: str = "healthy"
    version: str
    model_loaded: bool
    model_version: Optional[str] = None


class ModelInfoResponse(BaseModel):
    """Detailed model metadata response."""

    model_name: str
    algorithm: str
    scikit_learn_version: str
    trained_at: str
    features: List[str]
    metrics: Dict[str, Any]
    risk_thresholds: Dict[str, float]

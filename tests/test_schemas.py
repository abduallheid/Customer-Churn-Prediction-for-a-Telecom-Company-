"""Unit tests for Pydantic V2 schemas and input validation."""

import pytest
from pydantic import ValidationError
from telecom_churn.schemas import CustomerInput, PredictionOutput, RiskTier


def test_valid_customer_input(sample_customer_dict):
    """Verify standard valid customer input succeeds."""
    customer = CustomerInput(**sample_customer_dict)
    assert customer.account_length == 128
    assert customer.intl_plan == "no"
    assert customer.voice_mail_plan == "yes"


def test_normalize_binary_fields(sample_customer_dict):
    """Test flexible normalization of boolean and integer inputs."""
    data = sample_customer_dict.copy()

    # Boolean inputs
    data["intl_plan"] = True
    data["voice_mail_plan"] = False
    c1 = CustomerInput(**data)
    assert c1.intl_plan == "yes"
    assert c1.voice_mail_plan == "no"

    # Numeric inputs
    data["intl_plan"] = 0
    data["voice_mail_plan"] = 1
    c2 = CustomerInput(**data)
    assert c2.intl_plan == "no"
    assert c2.voice_mail_plan == "yes"

    # Case insensitive string
    data["intl_plan"] = "YES "
    data["voice_mail_plan"] = " No"
    c3 = CustomerInput(**data)
    assert c3.intl_plan == "yes"
    assert c3.voice_mail_plan == "no"


def test_invalid_negative_minutes(sample_customer_dict):
    """Verify negative call minutes trigger validation errors."""
    data = sample_customer_dict.copy()
    data["total_day_minutes"] = -10.5
    with pytest.raises(ValidationError):
        CustomerInput(**data)


def test_invalid_negative_customer_service_calls(sample_customer_dict):
    """Verify negative customer service calls trigger validation errors."""
    data = sample_customer_dict.copy()
    data["number_customer_service_calls"] = -1
    with pytest.raises(ValidationError):
        CustomerInput(**data)


def test_prediction_output_schema():
    """Verify PredictionOutput serialization."""
    out = PredictionOutput(
        churn_prediction=True,
        churn_probability=0.785,
        risk_tier=RiskTier.HIGH,
        confidence=0.57,
        key_drivers=["Critical customer service friction"],
        recommendations=["Initiate VIP retention outreach"],
    )
    assert out.risk_tier == RiskTier.HIGH
    assert out.churn_probability == 0.785
    data_json = out.model_dump_json()
    assert "HIGH" in data_json

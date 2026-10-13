"""Pytest shared fixtures for Telecom Churn testing."""

import pytest
import pandas as pd
from telecom_churn.schemas import CustomerInput


@pytest.fixture
def sample_customer_dict():
    """A valid dictionary representing customer activity."""
    return {
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


@pytest.fixture
def high_risk_customer_dict():
    """A customer dictionary designed to trigger high risk."""
    return {
        "account_length": 95,
        "intl_plan": "yes",
        "voice_mail_plan": "no",
        "number_vmail_messages": 0,
        "total_day_minutes": 320.5,
        "total_day_calls": 120,
        "total_day_charge": 54.49,
        "total_eve_minutes": 230.1,
        "total_eve_calls": 98,
        "total_eve_charge": 19.56,
        "total_night_minutes": 220.0,
        "total_night_calls": 85,
        "total_night_charge": 9.90,
        "total_intl_minutes": 14.2,
        "total_intl_calls": 2,
        "total_intl_charge": 3.83,
        "number_customer_service_calls": 5,
    }


@pytest.fixture
def sample_customer_input(sample_customer_dict):
    """A validated CustomerInput instance."""
    return CustomerInput(**sample_customer_dict)


@pytest.fixture
def sample_dataframe(sample_customer_dict, high_risk_customer_dict):
    """A sample DataFrame with multiple rows."""
    return pd.DataFrame([sample_customer_dict, high_risk_customer_dict])

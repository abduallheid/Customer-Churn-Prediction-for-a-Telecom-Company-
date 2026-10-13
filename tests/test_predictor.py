"""Unit tests for ChurnPredictor inference engine."""

from telecom_churn.models.predictor import get_predictor
from telecom_churn.schemas import RiskTier, PredictionOutput, BatchPredictionOutput


def test_predictor_singleton_loaded():
    """Verify predictor singleton loads pipeline successfully."""
    predictor = get_predictor()
    assert predictor.pipeline is not None
    assert isinstance(predictor.metadata, dict)


def test_predict_single_customer(sample_customer_input):
    """Test single customer inference returns valid PredictionOutput."""
    predictor = get_predictor()
    result = predictor.predict_single(sample_customer_input)

    assert isinstance(result, PredictionOutput)
    assert isinstance(result.churn_prediction, bool)
    assert 0.0 <= result.churn_probability <= 1.0
    assert result.risk_tier in [RiskTier.LOW, RiskTier.MEDIUM, RiskTier.HIGH]
    assert len(result.key_drivers) > 0
    assert len(result.recommendations) > 0


def test_predict_high_risk_customer(high_risk_customer_dict):
    """Verify customer with high friction triggers high risk or medium risk."""
    predictor = get_predictor()
    result = predictor.predict_single(high_risk_customer_dict)

    assert result.risk_tier in [RiskTier.HIGH, RiskTier.MEDIUM]
    assert result.churn_probability > 0.40


def test_predict_batch(sample_dataframe):
    """Test batch prediction produces enriched DataFrame and BatchPredictionOutput."""
    predictor = get_predictor()
    scored_df, batch_out = predictor.predict_batch(sample_dataframe)

    assert len(scored_df) == len(sample_dataframe)
    assert "churn_prediction" in scored_df.columns
    assert "churn_probability" in scored_df.columns
    assert "risk_tier" in scored_df.columns
    assert isinstance(batch_out, BatchPredictionOutput)
    assert batch_out.total_records == len(sample_dataframe)

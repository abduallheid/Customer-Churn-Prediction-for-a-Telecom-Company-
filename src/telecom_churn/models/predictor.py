"""Inference engine for Telecom Churn Prediction."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd

from telecom_churn.config import config
from telecom_churn.schemas import (
    CustomerInput,
    PredictionOutput,
    BatchPredictionOutput,
    RiskTier,
)

logger = logging.getLogger(__name__)


class ChurnPredictor:
    """Production inference engine for single and batch churn predictions."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
    ):
        self.model_path = model_path or config.model_artifact_path
        self.metadata_path = metadata_path or config.metadata_path
        self.pipeline = None
        self.metadata = {}
        self._load_model()

    def _load_model(self) -> None:
        """Load serialized pipeline and metadata from disk."""
        if not self.model_path.exists():
            logger.warning(
                f"Model artifact not found at {self.model_path}. Training initial model now..."
            )
            from telecom_churn.models.trainer import train_churn_model

            result = train_churn_model(save_artifact=True)
            self.pipeline = result["pipeline"]
            self.metadata = result["metadata"]
            return

        try:
            logger.info(f"Loading churn model pipeline from: {self.model_path}")
            self.pipeline = joblib.load(self.model_path)
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
            raise

        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load metadata from {self.metadata_path}: {e}")

    def predict_single(self, customer: Union[CustomerInput, dict]) -> PredictionOutput:
        """Score a single customer and provide churn risk analysis.

        Args:
            customer: CustomerInput instance or raw dictionary.

        Returns:
            PredictionOutput with prediction, probability, risk tier, drivers, and recommendations.
        """
        if isinstance(customer, CustomerInput):
            data_dict = customer.model_dump()
        else:
            data_dict = customer

        df = pd.DataFrame([data_dict])
        prob = float(self.pipeline.predict_proba(df)[0, 1])
        prediction = bool(prob >= 0.50)

        # Risk tier determination
        if prob < config.low_risk_threshold:
            risk_tier = RiskTier.LOW
        elif prob < config.high_risk_threshold:
            risk_tier = RiskTier.MEDIUM
        else:
            risk_tier = RiskTier.HIGH

        # Confidence (distance from decision boundary 0.5)
        confidence = float(abs(prob - 0.50) * 2.0)

        # Explainable risk drivers and recommendations
        key_drivers = self._identify_risk_drivers(data_dict, prob)
        recommendations = self._generate_recommendations(data_dict, risk_tier)

        return PredictionOutput(
            churn_prediction=prediction,
            churn_probability=round(prob, 4),
            risk_tier=risk_tier,
            confidence=round(confidence, 4),
            key_drivers=key_drivers,
            recommendations=recommendations,
        )

    def predict_batch(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, BatchPredictionOutput]:
        """Score a batch DataFrame and append prediction columns.

        Args:
            df: DataFrame containing customer activity records.

        Returns:
            Tuple of (enriched DataFrame, BatchPredictionOutput).
        """
        logger.info(f"Scoring batch of {len(df)} customer records...")
        scored_df = df.copy()

        # Vectorized probability prediction
        probs = self.pipeline.predict_proba(df)[:, 1]
        preds = (probs >= 0.50).astype(bool)

        # Vectorized risk tier
        risk_tiers = np.where(
            probs < config.low_risk_threshold,
            RiskTier.LOW.value,
            np.where(probs < config.high_risk_threshold, RiskTier.MEDIUM.value, RiskTier.HIGH.value),
        )

        scored_df["churn_prediction"] = preds
        scored_df["churn_probability"] = np.round(probs, 4)
        scored_df["risk_tier"] = risk_tiers

        # Compute summary stats
        churn_count = int(preds.sum())
        total_records = len(preds)
        churn_rate = round((churn_count / total_records) * 100.0, 2) if total_records > 0 else 0.0
        avg_prob = round(float(probs.mean()), 4) if total_records > 0 else 0.0

        risk_dist = {
            RiskTier.LOW.value: int((risk_tiers == RiskTier.LOW.value).sum()),
            RiskTier.MEDIUM.value: int((risk_tiers == RiskTier.MEDIUM.value).sum()),
            RiskTier.HIGH.value: int((risk_tiers == RiskTier.HIGH.value).sum()),
        }

        # Build list of individual PredictionOutputs
        predictions_list = []
        for i, row in df.iterrows():
            p = float(probs[i])
            tier = RiskTier(risk_tiers[i])
            conf = float(abs(p - 0.50) * 2.0)
            drivers = self._identify_risk_drivers(row.to_dict(), p)
            recs = self._generate_recommendations(row.to_dict(), tier)
            predictions_list.append(
                PredictionOutput(
                    churn_prediction=bool(preds[i]),
                    churn_probability=round(p, 4),
                    risk_tier=tier,
                    confidence=round(conf, 4),
                    key_drivers=drivers,
                    recommendations=recs,
                )
            )

        batch_output = BatchPredictionOutput(
            total_records=total_records,
            predicted_churn_count=churn_count,
            churn_rate_percentage=churn_rate,
            average_churn_probability=avg_prob,
            risk_distribution=risk_dist,
            predictions=predictions_list,
        )

        return scored_df, batch_output

    @staticmethod
    def _identify_risk_drivers(data: Dict[str, Any], prob: float) -> List[str]:
        """Generate human-readable insights into factors elevating churn risk."""
        drivers = []
        cs_calls = data.get("number_customer_service_calls", 0)
        day_mins = data.get("total_day_minutes", 0)
        day_charge = data.get("total_day_charge", 0)
        intl_plan = str(data.get("intl_plan", "no")).lower() in {"yes", "true", "1"}
        intl_calls = data.get("total_intl_calls", 0)
        intl_mins = data.get("total_intl_minutes", 0)
        vmail_plan = str(data.get("voice_mail_plan", "no")).lower() in {"yes", "true", "1"}

        if cs_calls >= 4:
            drivers.append(f"Critical customer service contact frequency ({cs_calls} calls indicates unresolved friction)")
        elif cs_calls >= 3:
            drivers.append(f"Elevated customer service activity ({cs_calls} calls)")

        if day_mins > 250 or day_charge > 45.0:
            drivers.append(f"High daytime bill exposure (${day_charge:.2f} across {day_mins:.1f} mins)")

        if intl_plan and intl_calls < 2:
            drivers.append("Paying for International Plan with near-zero international call volume")
        elif not intl_plan and intl_mins > 10.0:
            drivers.append(f"Frequent international calls ({intl_mins:.1f} mins) at standard per-minute tariff")

        if not vmail_plan:
            drivers.append("No voicemail package enabled (lower multi-service stickiness)")

        if not drivers:
            if prob < config.low_risk_threshold:
                drivers.append("Healthy usage across all call categories with minimal service friction")
            else:
                drivers.append("Aggregated multi-channel usage patterns closely align with historical churn profiles")

        return drivers

    @staticmethod
    def _generate_recommendations(data: Dict[str, Any], tier: RiskTier) -> List[str]:
        """Generate targeted retention playbooks based on risk profile."""
        recs = []
        cs_calls = data.get("number_customer_service_calls", 0)
        day_charge = data.get("total_day_charge", 0)
        intl_plan = str(data.get("intl_plan", "no")).lower() in {"yes", "true", "1"}
        intl_mins = data.get("total_intl_minutes", 0)

        if tier == RiskTier.HIGH:
            recs.append("Trigger VIP proactive account manager outreach within 24 hours.")
            if cs_calls >= 3:
                recs.append("Initiate dedicated case audit for unresolved customer service grievances.")
            if day_charge > 40.0:
                recs.append("Offer high-usage loyalty discount or flat-rate unlimited day bundle.")
            recs.append("Present 6-month contract extension incentive (e.g. 15% rate lock).")
        elif tier == RiskTier.MEDIUM:
            recs.append("Send satisfaction check-in survey with small customer appreciation credit.")
            if not intl_plan and intl_mins > 8.0:
                recs.append("Recommend enrollment in Discounted International Plan to reduce billing shocks.")
            recs.append("Include customer in digital retention nurturing campaign.")
        else:
            recs.append("Maintain standard service engagement; eligible for premium cross-sell offers.")

        return recs


# Global singleton instance
_predictor_instance: Optional[ChurnPredictor] = None


def get_predictor() -> ChurnPredictor:
    """Retrieve or initialize the global ChurnPredictor instance."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = ChurnPredictor()
    return _predictor_instance

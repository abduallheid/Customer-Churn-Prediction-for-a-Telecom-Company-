"""Models package for training, evaluation, pipeline assembly, and inference."""

from telecom_churn.models.pipeline import create_churn_pipeline
from telecom_churn.models.evaluator import evaluate_model
from telecom_churn.models.trainer import train_churn_model
from telecom_churn.models.predictor import ChurnPredictor, get_predictor

__all__ = [
    "create_churn_pipeline",
    "evaluate_model",
    "train_churn_model",
    "ChurnPredictor",
    "get_predictor",
]

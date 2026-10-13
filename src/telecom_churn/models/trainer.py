"""Model training and artifact serialization pipeline."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import joblib
import sklearn

from telecom_churn.config import config
from telecom_churn.data.loader import load_churn_data, split_churn_data
from telecom_churn.models.pipeline import create_churn_pipeline
from telecom_churn.models.evaluator import evaluate_model

logger = logging.getLogger(__name__)


def train_churn_model(
    classifier_type: str = "random_forest",
    classifier_params: Optional[dict] = None,
    save_artifact: bool = True,
    model_output_path: Optional[Path] = None,
    metadata_output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train, evaluate, and serialize customer churn prediction pipeline.

    Args:
        classifier_type: 'random_forest', 'gradient_boosting', or 'logistic_regression'.
        classifier_params: Optional hyperparameters dictionary.
        save_artifact: Whether to persist pipeline and metadata to disk.
        model_output_path: Target path for .joblib artifact.
        metadata_output_path: Target path for metadata .json.

    Returns:
        Dictionary containing trained pipeline and evaluation metrics.
    """
    logger.info(f"Initiating churn model training with algorithm: {classifier_type}")

    # 1. Load data
    X, y = load_churn_data(return_xy=True)

    # 2. Train-test split
    X_train, X_test, y_train, y_test = split_churn_data(X, y)
    logger.info(
        f"Train size: {len(X_train)} (churn rate: {y_train.mean():.2%}) | "
        f"Test size: {len(X_test)} (churn rate: {y_test.mean():.2%})"
    )

    # 3. Create pipeline
    pipeline = create_churn_pipeline(
        classifier_type=classifier_type,
        classifier_params=classifier_params,
        scale_numerics=True,
    )

    # 4. Train pipeline
    logger.info("Fitting pipeline on training data...")
    pipeline.fit(X_train, y_train)

    # 5. Evaluate pipeline
    logger.info("Evaluating pipeline on test hold-out...")
    metrics = evaluate_model(pipeline, X_test, y_test)

    # 6. Metadata preparation
    metadata = {
        "model_name": "Orange Telecom Churn Predictor",
        "algorithm": classifier_type,
        "scikit_learn_version": sklearn.__version__,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "target_column": config.target_column,
        "features": config.feature_columns,
        "categorical_columns": config.categorical_columns,
        "numerical_columns": config.numerical_columns,
        "metrics": metrics,
        "risk_thresholds": {
            "low_risk_max": config.low_risk_threshold,
            "high_risk_min": config.high_risk_threshold,
        },
    }

    # 7. Persist artifacts
    if save_artifact:
        model_path = model_output_path or config.model_artifact_path
        meta_path = metadata_output_path or config.metadata_path

        model_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving serialized pipeline to: {model_path}")
        joblib.dump(pipeline, model_path)

        logger.info(f"Saving model metadata to: {meta_path}")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Also update root RF_v1.ml for backward compatibility
        try:
            legacy_path = config.project_root / "RF_v1.ml"
            joblib.dump(pipeline, legacy_path)
            logger.info(f"Updated legacy artifact at: {legacy_path}")
        except Exception as e:
            logger.warning(f"Could not update legacy artifact: {e}")

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "metadata": metadata,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    train_churn_model()

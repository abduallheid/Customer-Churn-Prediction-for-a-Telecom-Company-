"""End-to-end Scikit-Learn Pipeline factory for Telecom Churn Prediction."""

from typing import Optional
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from telecom_churn.config import config
from telecom_churn.data.preprocessor import CleanDataFrameTransformer, build_preprocessor


def create_churn_pipeline(
    classifier_type: str = "random_forest",
    classifier_params: Optional[dict] = None,
    scale_numerics: bool = True,
) -> Pipeline:
    """Create a self-contained, end-to-end Scikit-Learn Pipeline.

    Pipeline stages:
      1. 'clean': CleanDataFrameTransformer (drops ignored columns, normalizes yes/no, handles NaNs)
      2. 'prep': ColumnTransformer (OrdinalEncoder for categoricals, StandardScaler for numerics)
      3. 'clf': Classifier (RandomForestClassifier, GradientBoostingClassifier, LogisticRegression)

    Args:
        classifier_type: One of 'random_forest', 'gradient_boosting', 'logistic_regression'.
        classifier_params: Optional dict of hyperparameters to pass to the classifier.
        scale_numerics: Whether to apply StandardScaler on numerical features.

    Returns:
        Complete Scikit-Learn Pipeline.
    """
    params = classifier_params or {}

    if classifier_type == "random_forest":
        default_params = {
            "n_estimators": config.n_estimators,
            "max_depth": config.max_depth,
            "min_samples_split": config.min_samples_split,
            "min_samples_leaf": config.min_samples_leaf,
            "class_weight": config.class_weight,
            "random_state": config.random_state,
            "n_jobs": -1,
        }
        default_params.update(params)
        clf = RandomForestClassifier(**default_params)

    elif classifier_type == "gradient_boosting":
        default_params = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 5,
            "subsample": 0.8,
            "random_state": config.random_state,
        }
        default_params.update(params)
        clf = GradientBoostingClassifier(**default_params)

    elif classifier_type == "logistic_regression":
        default_params = {
            "max_iter": 1000,
            "class_weight": "balanced",
            "random_state": config.random_state,
        }
        default_params.update(params)
        clf = LogisticRegression(**default_params)

    else:
        raise ValueError(
            f"Unsupported classifier_type '{classifier_type}'. Choose from 'random_forest', 'gradient_boosting', 'logistic_regression'."
        )

    pipeline = Pipeline(
        steps=[
            ("clean", CleanDataFrameTransformer()),
            ("prep", build_preprocessor(scale_numerics=scale_numerics)),
            ("clf", clf),
        ]
    )

    return pipeline

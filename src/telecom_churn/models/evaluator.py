"""Model evaluation metrics and reporting tools."""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

from telecom_churn.config import config

logger = logging.getLogger(__name__)


def evaluate_model(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics on test dataset.

    Args:
        pipeline: Fitted Scikit-Learn Pipeline.
        X_test: Test features.
        y_test: Test ground truth labels (0 or 1).

    Returns:
        Dictionary of performance metrics and confusion matrix values.
    """
    y_pred = pipeline.predict(X_test)

    # Probabilities
    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
    elif hasattr(pipeline, "decision_function"):
        y_prob = pipeline.decision_function(X_test)
    else:
        y_prob = y_pred.astype(float)

    # Metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    pr_auc = float(average_precision_score(y_test, y_prob))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Extract feature importance if available
    feature_importances = get_feature_importances(pipeline)

    results = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "classification_report": classification_report(
            y_test, y_pred, target_names=["Retained (0)", "Churned (1)"], output_dict=True
        ),
        "feature_importances": feature_importances,
    }

    logger.info(
        f"Evaluation Summary -> ROC-AUC: {roc_auc:.4f} | F1: {f1:.4f} | Recall: {rec:.4f} | Accuracy: {acc:.4f}"
    )

    return results


def get_feature_importances(pipeline: Any) -> List[Dict[str, Any]]:
    """Extract and sort feature importances from pipeline classifier."""
    try:
        clf = pipeline.named_steps.get("clf")
        if clf is None:
            return []

        importances = None
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            importances = np.abs(clf.coef_[0])

        if importances is None:
            return []

        # Canonical features in the order output by ColumnTransformer
        # Note: OrdinalEncoder keeps categorical columns first, followed by numerical columns
        feature_names = config.categorical_columns + config.numerical_columns

        fi_list = [
            {"feature": name, "importance": round(float(imp), 4)}
            for name, imp in zip(feature_names, importances)
        ]
        fi_list.sort(key=lambda x: x["importance"], reverse=True)
        return fi_list
    except Exception as e:
        logger.warning(f"Could not extract feature importances: {e}")
        return []

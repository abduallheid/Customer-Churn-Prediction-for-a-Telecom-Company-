"""Unit tests for Scikit-Learn Pipeline assembly and training."""

import pytest
import numpy as np
import pandas as pd
from telecom_churn.models.pipeline import create_churn_pipeline


def test_create_churn_pipeline_random_forest():
    """Verify Random Forest pipeline creation and components."""
    pipe = create_churn_pipeline(classifier_type="random_forest", classifier_params={"n_estimators": 10})
    assert "clean" in pipe.named_steps
    assert "prep" in pipe.named_steps
    assert "clf" in pipe.named_steps


def test_pipeline_fit_predict(sample_dataframe):
    """Verify end-to-end fit and predict on synthetic binary target."""
    pipe = create_churn_pipeline(classifier_type="random_forest", classifier_params={"n_estimators": 10, "random_state": 42})
    y = pd.Series([0, 1])

    pipe.fit(sample_dataframe, y)
    preds = pipe.predict(sample_dataframe)
    probs = pipe.predict_proba(sample_dataframe)

    assert len(preds) == 2
    assert probs.shape == (2, 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

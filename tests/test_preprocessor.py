"""Unit tests for transformers and preprocessing pipeline."""

import numpy as np
import pandas as pd
from telecom_churn.data.preprocessor import CleanDataFrameTransformer, build_preprocessor
from telecom_churn.config import config


def test_clean_dataframe_transformer_drops_unneeded_cols(sample_customer_dict):
    """Verify state, area_code, phone_number are dropped gracefully."""
    data = sample_customer_dict.copy()
    data["state"] = "KS"
    data["area_code"] = 415
    data["phone_number"] = "382-4657"
    df = pd.DataFrame([data])

    transformer = CleanDataFrameTransformer()
    transformer.fit(df)
    transformed = transformer.transform(df)

    assert "state" not in transformed.columns
    assert "area_code" not in transformed.columns
    assert "phone_number" not in transformed.columns
    assert list(transformed.columns) == config.feature_columns


def test_clean_dataframe_transformer_handles_missing_columns():
    """Verify transformer fills missing feature columns with default imputations."""
    partial_df = pd.DataFrame([{"account_length": 100, "total_day_minutes": 200.0}])
    transformer = CleanDataFrameTransformer()
    transformer.fit(partial_df)
    transformed = transformer.transform(partial_df)

    assert list(transformed.columns) == config.feature_columns
    assert transformed["intl_plan"].iloc[0] == "no"
    assert transformed["total_night_minutes"].iloc[0] == 0.0


def test_build_preprocessor_shape(sample_dataframe):
    """Verify ColumnTransformer transforms 17 features into numeric matrix."""
    cleaner = CleanDataFrameTransformer()
    cleaned = cleaner.fit_transform(sample_dataframe)

    preprocessor = build_preprocessor(scale_numerics=True)
    matrix = preprocessor.fit_transform(cleaned)

    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (len(sample_dataframe), 17)

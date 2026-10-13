"""Preprocessing pipeline components and transformers for Telecom Churn data."""

import logging
from typing import List, Optional, Union, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from telecom_churn.config import config

logger = logging.getLogger(__name__)


class CleanDataFrameTransformer(BaseEstimator, TransformerMixin):
    """Custom Scikit-Learn transformer to sanitize and align raw input DataFrames.

    - Drops optional/irrelevant columns ('state', 'area_code', 'phone_number').
    - Normalizes categorical binary fields ('intl_plan', 'voice_mail_plan') to 'yes'/'no'.
    - Casts numeric fields to standard float/int dtypes.
    - Fills any unexpected NaN values with median/mode.
    - Reorders columns to strictly match canonical feature ordering.
    """

    def __init__(
        self,
        feature_columns: Optional[List[str]] = None,
        drop_columns: Optional[List[str]] = None,
    ):
        self.feature_columns = feature_columns or config.feature_columns
        self.drop_columns = drop_columns or config.drop_columns
        self.impute_values_: dict = {}

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Any] = None):
        """Compute median/mode imputers on training data."""
        df = self._ensure_dataframe(X)

        # Drop non-feature columns
        cols_to_drop = [c for c in self.drop_columns if c in df.columns]
        cleaned = df.drop(columns=cols_to_drop, errors="ignore")

        # Record imputation statistics
        for col in self.feature_columns:
            if col in cleaned.columns:
                if col in config.categorical_columns:
                    self.impute_values_[col] = "no"
                else:
                    self.impute_values_[col] = float(cleaned[col].median())
            else:
                self.impute_values_[col] = "no" if col in config.categorical_columns else 0.0

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Sanitize and format input DataFrame."""
        df = self._ensure_dataframe(X).copy()

        # Drop non-feature columns if present
        cols_to_drop = [c for c in self.drop_columns if c in df.columns]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

        # Normalize categorical features
        for cat_col in config.categorical_columns:
            if cat_col in df.columns:
                df[cat_col] = df[cat_col].apply(self._normalize_binary)
            else:
                df[cat_col] = self.impute_values_.get(cat_col, "no")

        # Normalize numerical features
        for num_col in config.numerical_columns:
            if num_col in df.columns:
                df[num_col] = pd.to_numeric(df[num_col], errors="coerce")
                median_val = self.impute_values_.get(num_col, 0.0)
                df[num_col] = df[num_col].fillna(median_val)
            else:
                df[num_col] = self.impute_values_.get(num_col, 0.0)

        # Select canonical columns in exact order
        return df[self.feature_columns]

    @staticmethod
    def _normalize_binary(val: Any) -> str:
        """Map booleans, integers, and variations to 'yes'/'no'."""
        if pd.isna(val):
            return "no"
        if isinstance(val, bool):
            return "yes" if val else "no"
        if isinstance(val, (int, float)):
            return "yes" if val == 1 else "no"
        s = str(val).strip().lower()
        if s in {"yes", "y", "true", "1"}:
            return "yes"
        return "no"

    @staticmethod
    def _ensure_dataframe(X: Union[pd.DataFrame, np.ndarray, list, dict]) -> pd.DataFrame:
        """Ensure input is converted into a pandas DataFrame."""
        if isinstance(X, pd.DataFrame):
            return X
        if isinstance(X, dict):
            return pd.DataFrame([X])
        if isinstance(X, (list, tuple)):
            if len(X) > 0 and isinstance(X[0], dict):
                return pd.DataFrame(X)
            return pd.DataFrame(X, columns=config.feature_columns)
        if isinstance(X, np.ndarray):
            return pd.DataFrame(X, columns=config.feature_columns)
        raise ValueError(f"Unsupported input type for preprocessor: {type(X)}")


def build_preprocessor(scale_numerics: bool = True) -> ColumnTransformer:
    """Build Scikit-Learn ColumnTransformer for Orange Telecom features.

    - Encodes 'intl_plan' and 'voice_mail_plan' using OrdinalEncoder: ['no', 'yes'] -> [0.0, 1.0].
    - Scales numerical features with StandardScaler (optional).
    """
    cat_transformer = OrdinalEncoder(
        categories=[["no", "yes"], ["no", "yes"]],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    num_transformer = StandardScaler() if scale_numerics else "passthrough"

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, config.categorical_columns),
            ("num", num_transformer, config.numerical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor

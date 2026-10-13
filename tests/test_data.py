"""Unit tests for dataset loading and splitting."""

import pandas as pd
from telecom_churn.data.loader import load_churn_data, split_churn_data
from telecom_churn.config import config


def test_load_churn_data_raw():
    """Verify loading raw dataset."""
    df = load_churn_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5000
    assert config.target_column in df.columns


def test_load_churn_data_xy():
    """Verify loading X and y separated."""
    X, y = load_churn_data(return_xy=True)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y) == 5000
    assert config.target_column not in X.columns
    # Check y values are binary (0 and 1)
    assert set(y.unique()).issubset({0, 1})


def test_split_churn_data():
    """Verify stratified splitting proportions and labels."""
    X, y = load_churn_data(return_xy=True)
    X_train, X_test, y_train, y_test = split_churn_data(X, y, test_size=0.2, random_state=42)

    assert len(X_train) == 4000
    assert len(X_test) == 1000
    assert len(y_train) == 4000
    assert len(y_test) == 1000

    # Stratification check (similar churn rate)
    train_rate = y_train.mean()
    test_rate = y_test.mean()
    assert abs(train_rate - test_rate) < 0.01

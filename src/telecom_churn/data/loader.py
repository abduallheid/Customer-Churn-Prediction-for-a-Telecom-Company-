"""Data loader utilities for Telecom Churn datasets."""

import logging
from pathlib import Path
from typing import Tuple, Union, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

from telecom_churn.config import config

logger = logging.getLogger(__name__)


def load_churn_data(
    file_path: Optional[Union[str, Path]] = None,
    return_xy: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
    """Load and perform initial validation on Orange Telecom churn dataset.

    Args:
        file_path: Path to CSV data file. Defaults to config.raw_data_path.
        return_xy: If True, returns (X, y) tuple. If False, returns raw DataFrame.

    Returns:
        DataFrame or (X, y) tuple.
    """
    path = Path(file_path) if file_path else config.raw_data_path

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path}")

    logger.info(f"Loading churn dataset from: {path}")
    df = pd.read_csv(path)

    logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns.")

    if return_xy:
        if config.target_column not in df.columns:
            raise KeyError(
                f"Target column '{config.target_column}' not found in dataset columns: {list(df.columns)}"
            )

        # Normalize target column: boolean/string/int -> int (0 or 1)
        y_raw = df[config.target_column]
        if y_raw.dtype == bool:
            y = y_raw.astype(int)
        elif y_raw.dtype == object:
            y = y_raw.astype(str).str.strip().str.lower().map({"true": 1, "yes": 1, "1": 1, "false": 0, "no": 0, "0": 0}).fillna(0).astype(int)
        else:
            y = y_raw.astype(int)

        X = df.drop(columns=[config.target_column])
        return X, y

    return df


def split_churn_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: Optional[float] = None,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform stratified train/test split on churn features and labels.

    Args:
        X: Feature DataFrame.
        y: Target label Series.
        test_size: Proportion of dataset to include in test split (default: config.test_size).
        random_state: Random state for reproducibility (default: config.random_state).

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    ts = test_size if test_size is not None else config.test_size
    rs = random_state if random_state is not None else config.random_state

    logger.info(f"Splitting data: test_size={ts}, random_state={rs}, stratify=True")

    return train_test_split(
        X,
        y,
        test_size=ts,
        random_state=rs,
        stratify=y,
    )

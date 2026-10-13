"""Configuration settings for Telecom Churn Prediction project."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class ProjectConfig:
    """Project-wide configuration dataclass."""

    # Paths
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "dataset"
    raw_data_path: Path = PROJECT_ROOT / "dataset" / "Orange_Telecom_Churn_Data.csv"
    models_dir: Path = PROJECT_ROOT / "models"
    model_artifact_path: Path = PROJECT_ROOT / "models" / "churn_pipeline.joblib"
    metadata_path: Path = PROJECT_ROOT / "models" / "model_metadata.json"

    # Target
    target_column: str = "churned"

    # Ignored non-predictive columns
    drop_columns: List[str] = field(
        default_factory=lambda: ["state", "area_code", "phone_number"]
    )

    # Categorical features requiring encoding (yes/no -> 1/0)
    categorical_columns: List[str] = field(
        default_factory=lambda: ["intl_plan", "voice_mail_plan"]
    )

    # Numerical features
    numerical_columns: List[str] = field(
        default_factory=lambda: [
            "account_length",
            "number_vmail_messages",
            "total_day_minutes",
            "total_day_calls",
            "total_day_charge",
            "total_eve_minutes",
            "total_eve_calls",
            "total_eve_charge",
            "total_night_minutes",
            "total_night_calls",
            "total_night_charge",
            "total_intl_minutes",
            "total_intl_calls",
            "total_intl_charge",
            "number_customer_service_calls",
        ]
    )

    # Complete feature list in canonical order
    @property
    def feature_columns(self) -> List[str]:
        return self.categorical_columns + self.numerical_columns

    # Model training settings
    test_size: float = 0.20
    random_state: int = 42
    n_estimators: int = 200
    max_depth: int = 14
    min_samples_split: int = 4
    min_samples_leaf: int = 2
    class_weight: str = "balanced"

    # Risk thresholds
    low_risk_threshold: float = 0.35
    high_risk_threshold: float = 0.65


# Singleton configuration instance
config = ProjectConfig()

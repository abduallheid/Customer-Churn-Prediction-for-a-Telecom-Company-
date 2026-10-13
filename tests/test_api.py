"""Integration tests for FastAPI REST API endpoints."""

from fastapi.testclient import TestClient
from telecom_churn.api.app import app

client = TestClient(app)


def test_root_endpoint():
    """Verify API root endpoint returns service info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "Orange Telecom Churn Prediction API" in data["service"]


def test_health_endpoint():
    """Verify health check returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_model_info_endpoint():
    """Verify model info returns metadata and metrics."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert "features" in data
    assert len(data["features"]) == 17


def test_predict_single_endpoint(sample_customer_dict):
    """Verify POST /predict returns valid prediction."""
    response = client.post("/predict", json=sample_customer_dict)
    assert response.status_code == 200
    data = response.json()
    assert "churn_prediction" in data
    assert "churn_probability" in data
    assert "risk_tier" in data
    assert "key_drivers" in data
    assert "recommendations" in data


def test_predict_batch_endpoint(sample_customer_dict, high_risk_customer_dict):
    """Verify POST /predict/batch scores multiple customers."""
    payload = [sample_customer_dict, high_risk_customer_dict]
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] == 2
    assert "predicted_churn_count" in data
    assert "risk_distribution" in data


def test_predict_csv_upload_endpoint(sample_dataframe):
    """Verify POST /predict/upload-csv processes CSV file upload."""
    csv_bytes = sample_dataframe.to_csv(index=False).encode("utf-8")
    files = {"file": ("test.csv", csv_bytes, "text/csv")}
    response = client.post("/predict/upload-csv", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] == len(sample_dataframe)


def test_predict_csv_upload_download_as_csv(sample_dataframe):
    """Verify POST /predict/upload-csv with download_as_csv=true returns CSV content."""
    csv_bytes = sample_dataframe.to_csv(index=False).encode("utf-8")
    files = {"file": ("test.csv", csv_bytes, "text/csv")}
    response = client.post("/predict/upload-csv?download_as_csv=true", files=files)
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert "churn_probability" in response.text

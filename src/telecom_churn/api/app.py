"""FastAPI production REST API for Orange Telecom Customer Churn Prediction."""

import io
import logging
from typing import List, Dict, Any
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from telecom_churn import __version__
from telecom_churn.config import config
from telecom_churn.schemas import (
    CustomerInput,
    PredictionOutput,
    BatchPredictionOutput,
    HealthResponse,
    ModelInfoResponse,
)
from telecom_churn.models.predictor import get_predictor

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Orange Telecom Churn Prediction API",
    description=(
        "Production-grade Machine Learning API for predicting customer attrition, "
        "evaluating churn probability, categorizing risk tiers, and recommending retention strategies."
    ),
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
def root():
    """API Root summary."""
    return {
        "service": "Orange Telecom Churn Prediction API",
        "version": __version__,
        "documentation": "/docs",
        "health": "/health",
        "model_info": "/model-info",
        "prediction_endpoints": {
            "single": "POST /predict",
            "batch_json": "POST /predict/batch",
            "batch_csv_upload": "POST /predict/upload-csv",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    """Health check endpoint for Kubernetes / load balancer probes."""
    try:
        predictor = get_predictor()
        is_loaded = predictor.pipeline is not None
        algo = predictor.metadata.get("algorithm", "unknown")
        return HealthResponse(
            status="healthy",
            version=__version__,
            model_loaded=is_loaded,
            model_version=algo,
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="degraded",
            version=__version__,
            model_loaded=False,
            model_version=None,
        )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
def get_model_info():
    """Retrieve detailed model training metadata, features, and performance metrics."""
    predictor = get_predictor()
    meta = predictor.metadata

    if not meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model metadata is not available.",
        )

    return ModelInfoResponse(
        model_name=meta.get("model_name", "Orange Telecom Churn Predictor"),
        algorithm=meta.get("algorithm", "random_forest"),
        scikit_learn_version=meta.get("scikit_learn_version", "unknown"),
        trained_at=meta.get("trained_at", ""),
        features=meta.get("features", config.feature_columns),
        metrics=meta.get("metrics", {}),
        risk_thresholds=meta.get(
            "risk_thresholds",
            {
                "low_risk_max": config.low_risk_threshold,
                "high_risk_min": config.high_risk_threshold,
            },
        ),
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict_single_customer(customer: CustomerInput):
    """Predict churn risk and obtain actionable retention playbooks for a single customer."""
    try:
        predictor = get_predictor()
        return predictor.predict_single(customer)
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}",
        )


@app.post("/predict/batch", response_model=BatchPredictionOutput, tags=["Prediction"])
def predict_batch_customers(customers: List[CustomerInput]):
    """Predict churn risk for a batch of customers submitted as a JSON list."""
    if not customers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Customer list must contain at least one item.",
        )

    try:
        predictor = get_predictor()
        records = [c.model_dump() for c in customers]
        df = pd.DataFrame(records)
        _, batch_output = predictor.predict_batch(df)
        return batch_output
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failed: {str(e)}",
        )


@app.post("/predict/upload-csv", tags=["Prediction"])
async def predict_csv_upload(
    file: UploadFile = File(..., description="CSV file with customer activity data"),
    download_as_csv: bool = False,
):
    """Upload a CSV file to generate churn scores.

    - If download_as_csv=true, returns an annotated CSV download with predictions.
    - If download_as_csv=false, returns a JSON summary and list of predictions.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .csv files are supported.",
        )

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not parse uploaded CSV file: {str(e)}",
        )

    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded CSV file is empty.",
        )

    predictor = get_predictor()
    try:
        scored_df, batch_output = predictor.predict_batch(df)
    except Exception as e:
        logger.error(f"Error scoring uploaded CSV: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to score uploaded CSV: {str(e)}",
        )

    if download_as_csv:
        output_buffer = io.StringIO()
        scored_df.to_csv(output_buffer, index=False)
        output_buffer.seek(0)
        return StreamingResponse(
            io.BytesIO(output_buffer.getvalue().encode("utf-8")),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=scored_{file.filename}",
            },
        )

    return batch_output

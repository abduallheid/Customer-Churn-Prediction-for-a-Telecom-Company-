"""Unified Command-Line Interface (CLI) for Telecom Churn ML System."""

import argparse
import json
import logging
import sys
from pathlib import Path
import pandas as pd

from telecom_churn.config import config
from telecom_churn.models.trainer import train_churn_model
from telecom_churn.models.evaluator import evaluate_model
from telecom_churn.models.predictor import get_predictor
from telecom_churn.data.loader import load_churn_data, split_churn_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("telecom_churn.cli")


def cmd_train(args):
    """Execute model training workflow."""
    logger.info(f"Training churn model with algorithm: {args.algorithm}")
    result = train_churn_model(
        classifier_type=args.algorithm,
        save_artifact=True,
    )
    metrics = result["metrics"]
    print("\n" + "=" * 60)
    print("           TRAINING & EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Algorithm:       {args.algorithm}")
    print(f"ROC-AUC:         {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:          {metrics['pr_auc']:.4f}")
    print(f"F1-Score:        {metrics['f1_score']:.4f}")
    print(f"Recall:          {metrics['recall']:.4f}")
    print(f"Precision:       {metrics['precision']:.4f}")
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print("\nConfusion Matrix:")
    cm = metrics["confusion_matrix"]
    print(f"  TN: {cm['true_negatives']} | FP: {cm['false_positives']}")
    print(f"  FN: {cm['false_negatives']} | TP: {cm['true_positives']}")
    print("=" * 60)
    print(f"Model saved to:    {config.model_artifact_path}")
    print(f"Metadata saved to: {config.metadata_path}\n")


def cmd_evaluate(args):
    """Evaluate current saved model on holdout test set."""
    predictor = get_predictor()
    X, y = load_churn_data(return_xy=True)
    _, X_test, _, y_test = split_churn_data(X, y)
    metrics = evaluate_model(predictor.pipeline, X_test, y_test)
    print(json.dumps(metrics, indent=2))


def cmd_predict(args):
    """Predict churn on single record JSON or batch CSV."""
    predictor = get_predictor()
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
        scored_df, batch_out = predictor.predict_batch(df)
        out_path = args.output or "predictions_output.csv"
        scored_df.to_csv(out_path, index=False)
        print(f"Scored {len(df)} rows. Results saved to: {out_path}")
        print(f"Churn Count: {batch_out.predicted_churn_count} ({batch_out.churn_rate_percentage:.2f}%)")
    else:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        pred = predictor.predict_single(data)
        print(pred.model_dump_json(indent=2))


def cmd_serve_api(args):
    """Start production FastAPI server."""
    import uvicorn

    logger.info(f"Starting FastAPI server on {args.host}:{args.port}")
    uvicorn.run("telecom_churn.api.app:app", host=args.host, port=args.port, reload=args.reload)


def cmd_serve_ui(args):
    """Start Streamlit UI dashboard."""
    import subprocess

    app_path = config.project_root / "src" / "telecom_churn" / "ui" / "app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(args.port),
        "--server.address",
        args.host,
    ]
    logger.info(f"Starting Streamlit dashboard on {args.host}:{args.port}")
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(
        prog="telecom_churn",
        description="Production CLI for Orange Telecom Churn Prediction System",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train
    p_train = subparsers.add_parser("train", help="Train churn prediction pipeline")
    p_train.add_argument(
        "--algorithm",
        "-a",
        default="random_forest",
        choices=["random_forest", "gradient_boosting", "logistic_regression"],
        help="Model algorithm to train",
    )
    p_train.set_defaults(func=cmd_train)

    # Evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate current model on test holdout")
    p_eval.set_defaults(func=cmd_evaluate)

    # Predict
    p_pred = subparsers.add_parser("predict", help="Predict on CSV or JSON file")
    p_pred.add_argument("--input", "-i", required=True, help="Input CSV or JSON path")
    p_pred.add_argument("--output", "-o", default=None, help="Output CSV path for batch scoring")
    p_pred.set_defaults(func=cmd_predict)

    # Serve API
    p_api = subparsers.add_parser("serve-api", help="Launch FastAPI web service")
    p_api.add_argument("--host", default="0.0.0.0", help="Host address")
    p_api.add_argument("--port", type=int, default=8000, help="Port")
    p_api.add_argument("--reload", action="store_true", help="Enable live reload")
    p_api.set_defaults(func=cmd_serve_api)

    # Serve UI
    p_ui = subparsers.add_parser("serve-ui", help="Launch Streamlit web dashboard")
    p_ui.add_argument("--host", default="0.0.0.0", help="Host address")
    p_ui.add_argument("--port", type=int, default=8501, help="Port")
    p_ui.set_defaults(func=cmd_serve_ui)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

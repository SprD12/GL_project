import json
import os

import mlflow

from src.pipeline import train_and_evaluate, upload_files

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("wellness-tourism")
metrics = train_and_evaluate(
    "data/processed/train.csv", "data/processed/test.csv", "artifacts"
)
with mlflow.start_run():
    mlflow.log_params(metrics["best_params"])
    mlflow.log_metrics(
        {
            "cv_roc_auc": metrics["cv_roc_auc"],
            "test_accuracy": metrics["test_accuracy"],
            "test_roc_auc": metrics["test_roc_auc"],
        }
    )
    mlflow.log_artifact("artifacts/model.joblib")
upload_files(
    ["artifacts/model.joblib", "artifacts/metrics.json"],
    repo_id=os.environ["HF_MODEL_REPO_ID"],
    repo_type="model",
)
print(json.dumps(metrics, indent=2))

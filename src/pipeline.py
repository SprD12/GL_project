"""Data, training, and Hugging Face Hub helpers for the tourism project."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "ProdTaken"
ID_COLUMNS = {"CustomerID", "customerid", "Unnamed: 0"}


def _repo_id(kind: str) -> str:
    defaults = {
        "dataset": "sprd12/Great_Learning",
        "model": "sprd12/RandomForest",
    }
    return os.environ.get(f"HF_{kind.upper()}_REPO_ID", defaults[kind])


def load_source_csv(path: str | Path | None = None) -> pd.DataFrame:
    """Load the source data locally or download tourism.csv from the HF dataset repo."""
    if path and Path(path).exists():
        return pd.read_csv(path)
    filename = os.environ.get("HF_SOURCE_FILENAME", "tourism.csv")
    downloaded = hf_hub_download(
        repo_id=_repo_id("dataset"),
        filename=filename,
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN"),
    )
    return pd.read_csv(downloaded)


def load_split_from_hf(filename: str) -> pd.DataFrame:
    """Load a processed train/test CSV from the configured HF dataset repository."""
    downloaded = hf_hub_download(
        repo_id=_repo_id("dataset"),
        filename=filename,
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN"),
    )
    return pd.read_csv(downloaded)


def clean_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names, remove identifiers, and standardize obvious missing values."""
    data = frame.copy()
    data.columns = [str(column).strip() for column in data.columns]
    data = data.replace({"": np.nan, " ": np.nan, "null": np.nan, "NA": np.nan})
    drop_columns = [
        column
        for column in data.columns
        if column in ID_COLUMNS or column.lower().startswith("unnamed:")
    ]
    if drop_columns:
        data = data.drop(columns=drop_columns)
    if TARGET not in data:
        raise ValueError(f"Expected target column '{TARGET}' in the dataset.")
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data.dropna(subset=[TARGET])
    data[TARGET] = data[TARGET].astype(int)
    return data


def split_and_save(
    data: pd.DataFrame, output_dir: str | Path, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train, test = train_test_split(
        data, test_size=test_size, random_state=42, stratify=data[TARGET]
    )
    train.to_csv(output / "train.csv", index=False)
    test.to_csv(output / "test.csv", index=False)
    return train, test


def upload_files(paths: Iterable[str | Path], repo_id: str, repo_type: str) -> None:
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
    for path in paths:
        path = Path(path)
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=repo_id,
            repo_type=repo_type,
        )


def build_model(train: pd.DataFrame) -> GridSearchCV:
    features = train.drop(columns=[TARGET])
    numeric = features.select_dtypes(include=["number"]).columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1)),
        ]
    )
    parameters = {
        "classifier__n_estimators": [150, 300],
        "classifier__max_depth": [None, 12],
        "classifier__min_samples_leaf": [1, 2],
    }
    search = GridSearchCV(
        pipeline, parameters, scoring="roc_auc", cv=3, n_jobs=-1, refit=True
    )
    search.fit(features, train[TARGET])
    return search


def train_and_evaluate(
    train_path: str | Path, test_path: str | Path, output_dir: str | Path
) -> dict:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    search = build_model(train)
    predictions = search.predict(test.drop(columns=[TARGET]))
    probabilities = search.predict_proba(test.drop(columns=[TARGET]))[:, 1]
    metrics = {
        "best_params": search.best_params_,
        "cv_roc_auc": float(search.best_score_),
        "test_accuracy": float(accuracy_score(test[TARGET], predictions)),
        "test_roc_auc": float(roc_auc_score(test[TARGET], probabilities)),
        "classification_report": classification_report(
            test[TARGET], predictions, output_dict=True
        ),
    }
    joblib.dump(search.best_estimator_, output / "model.joblib")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics

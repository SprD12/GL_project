"""Train from train.csv and test.csv already uploaded to the HF dataset repo."""

import os
from pathlib import Path

from src.pipeline import load_split_from_hf, train_and_evaluate

Path("data/processed").mkdir(parents=True, exist_ok=True)
load_split_from_hf("train.csv").to_csv("data/processed/train.csv", index=False)
load_split_from_hf("test.csv").to_csv("data/processed/test.csv", index=False)
metrics = train_and_evaluate(
    "data/processed/train.csv", "data/processed/test.csv", "artifacts"
)
print(
    "Model trained from Hugging Face dataset:",
    os.environ.get("HF_DATASET_REPO_ID", "sprd12/Great_Learning"),
)
print(metrics)

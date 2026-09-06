import os
from pathlib import Path

from src.pipeline import upload_files

source = Path("data/tourism.csv")
if not source.exists():
    raise FileNotFoundError("Place tourism.csv in data/ before registering the dataset.")
upload_files([source], os.environ["HF_DATASET_REPO_ID"], "dataset")
print(f"Registered {source} in {os.environ['HF_DATASET_REPO_ID']}.")


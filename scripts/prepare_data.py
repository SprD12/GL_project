import os

from src.pipeline import clean_data, load_source_csv, split_and_save, upload_files

# The registered Hugging Face dataset is the source of truth for preparation.
cleaned = clean_data(load_source_csv())
train, test = split_and_save(cleaned, "data/processed")
upload_files(
    ["data/processed/train.csv", "data/processed/test.csv"],
    repo_id=os.environ.get("HF_DATASET_REPO_ID", "sprd12/Great_Learning"),
    repo_type="dataset",
)
print(f"Prepared {len(train)} training rows and {len(test)} test rows.")

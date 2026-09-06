import os
from pathlib import Path

from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
repo_id = os.environ["HF_SPACE_REPO_ID"]
api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
for path in ["app.py", "Dockerfile", "requirements.txt", "README.md"]:
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo=Path(path).name,
        repo_id=repo_id,
        repo_type="space",
    )
print(f"Deployed application files to https://huggingface.co/spaces/{repo_id}")

# Wellness Tourism MLOps Pipeline

This project predicts whether a customer will purchase the Wellness Tourism
Package before outreach. It covers data registration, cleaning, stratified
splitting, preprocessing, hyperparameter tuning, MLflow experiment tracking,
model registration, Streamlit deployment, and CI/CD.

## Repository structure

```text
data/                  Source and generated data files
src/pipeline.py        Reusable data and model code
scripts/               Registration, preparation, training, and deployment jobs
app.py                 Streamlit frontend
Dockerfile             Hugging Face Spaces container definition
.github/workflows/     End-to-end GitHub Actions workflow
```

## Configuration

Create a Hugging Face dataset repository, model repository, and Docker Space.
Set these GitHub repository variables:

* `HF_DATASET_REPO_ID`
* `HF_MODEL_REPO_ID`
* `HF_SPACE_REPO_ID`

Add a write-scoped `HF_TOKEN` as a GitHub secret. Put the original
`tourism.csv` in `data/` for the first run. The dataset repository is then the
source of truth for subsequent runs. Add `HF_MODEL_REPO_ID` as a Space secret
so the deployed app can download the registered model. The current public model
repository is `sprd12/RandomForest`; the app uses that as its default.

If a paid Hugging Face Space is not available, leave `HF_SPACE_REPO_ID`
unset. The GitHub Actions workflow will run registration, preparation, training,
evaluation, and model registration, then skip only the final Space deployment
job. The Dockerfile, Streamlit application, and hosting script remain available
for later deployment.

## GitHub submission checklist

1. Create a GitHub repository and upload this entire project, including
   `Learner_Template_Notebook_AML_and_MLOps_Project.ipynb`, `data/tourism.csv`,
   `src/`, `scripts/`, `app.py`, `Dockerfile`, `requirements.txt`, and
   `.github/workflows/pipeline.yml`.
2. Add `HF_TOKEN` as a GitHub Actions secret.
3. Add `HF_DATASET_REPO_ID=sprd12/Great_Learning` and
   `HF_MODEL_REPO_ID=sprd12/RandomForest` as repository variables.
4. Push to `main` and capture the repository structure and successful workflow
   run for the notebook's Output Evaluation section.
5. Add `HF_SPACE_REPO_ID` only if a public Space becomes available.

## Local execution

```powershell
pip install -r requirements.txt
$env:HF_TOKEN = "..."
$env:HF_DATASET_REPO_ID = "username/tourism-data"
$env:HF_MODEL_REPO_ID = "username/tourism-model"
$env:HF_SPACE_REPO_ID = "username/tourism-app"
python scripts/register_data.py
python scripts/prepare_data.py
python scripts/train_model.py
python scripts/deploy_space.py
```

The completed assignment notebook should link to the GitHub repository and the
public Space URL after the first successful workflow run. Do not commit tokens,
customer data, or generated model artifacts.

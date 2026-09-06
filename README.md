# Wellness Tourism Package Purchase Prediction

## Project submission

This repository implements an end-to-end MLOps workflow for predicting whether
a customer will purchase the Wellness Tourism Package before contact.

**Hugging Face dataset:** [sprd12/Great_Learning](https://huggingface.co/datasets/sprd12/Great_Learning/tree/main)  
**Hugging Face model:** [sprd12/RandomForest](https://huggingface.co/sprd12/RandomForest/tree/main)  
**Assignment notebook:** [Learner_Template_Notebook_AML_and_MLOps_Project.ipynb](./Learner_Template_Notebook_AML_and_MLOps_Project.ipynb)

## Rubric coverage

### Model deployment

- [Dockerfile](./Dockerfile) defines the Python 3.11 Streamlit container and
  exposes port `7860`.
- [app.py](./app.py) downloads `model.joblib` from
  `sprd12/RandomForest` using `hf_hub_download`.
- The Streamlit form collects customer attributes and creates a pandas
  DataFrame before prediction.
- [requirements.txt](./requirements.txt) lists deployment dependencies.
- [scripts/deploy_space.py](./scripts/deploy_space.py) creates a Docker Space
  and uploads the deployment files when a Space repository is available.

### MLOps pipeline

[.github/workflows/pipeline.yml](./.github/workflows/pipeline.yml) runs on every
push to `main` and supports manual execution. Its jobs are:

1. Register the source dataset in `sprd12/Great_Learning`.
2. Load data from Hugging Face, clean it, and create stratified train/test data.
3. Tune and evaluate a Random Forest and register the model in
   `sprd12/RandomForest`.
4. Deploy the Streamlit files to a Space when `HF_SPACE_REPO_ID` is configured.

The dataset and model IDs are already configured as defaults in the code and
workflow. Only `HF_TOKEN` is required for GitHub Actions authentication.

### Notebook and output evaluation

The notebook contains executable cells for data preparation, model training,
Docker configuration, Streamlit hosting, workflow documentation, repository
structure, and output links. After pushing to GitHub, add screenshots of:

1. The GitHub repository folder structure.
2. A successful GitHub Actions workflow run.
3. The deployed Streamlit application, if a public Space is available.

## Repository structure

```text
.
├── .github/workflows/pipeline.yml
├── data/tourism.csv
├── src/pipeline.py
├── scripts/
│   ├── register_data.py
│   ├── prepare_data.py
│   ├── train_model.py
│   ├── train_from_hf.py
│   └── deploy_space.py
├── app.py
├── Dockerfile
├── requirements.txt
└── Learner_Template_Notebook_AML_and_MLOps_Project.ipynb
```

## Run locally

```powershell
pip install -r requirements.txt
$env:HF_TOKEN = "your_huggingface_write_token"
python scripts/register_data.py
python scripts/prepare_data.py
python scripts/train_model.py
streamlit run app.py
```

The generated model metrics are saved in `artifacts/metrics.json`. Do not
commit access tokens.

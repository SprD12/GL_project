import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download
from sklearn.base import BaseEstimator

MODEL_PATH = Path("model.joblib")
MODEL_REPO_ID = os.environ.get("HF_MODEL_REPO_ID", "sprd12/RandomForest")


def _restore_sklearn_compatibility(model):
    """Restore attributes missing from tree estimators serialized by older sklearn."""
    visited = set()

    def visit(value):
        if id(value) in visited:
            return
        visited.add(id(value))

        if isinstance(value, BaseEstimator):
            if (
                value.__class__.__module__ == "sklearn.tree._classes"
                and not hasattr(value, "monotonic_cst")
            ):
                value.monotonic_cst = None
            for nested in vars(value).values():
                visit(nested)
        elif isinstance(value, (list, tuple, set)):
            for nested in value:
                visit(nested)
        elif isinstance(value, dict):
            for nested in value.values():
                visit(nested)

    visit(model)
    return model


@st.cache_resource
def load_model():
    path = MODEL_PATH
    if not path.exists():
        path = Path(
            hf_hub_download(
                repo_id=MODEL_REPO_ID,
                filename="model.joblib",
                repo_type="model",
                token=os.environ.get("HF_TOKEN"),
            )
        )
    return _restore_sklearn_compatibility(joblib.load(path))


st.set_page_config(page_title="Wellness Tourism Predictor", page_icon="🌿")
st.title("Wellness Tourism Package Predictor")
st.caption("Estimate purchase likelihood before contacting a customer.")

with st.form("prediction_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    city_tier = st.selectbox("City tier", [1, 2, 3], index=1)
    income = st.number_input("Monthly income", min_value=0, value=30000, step=1000)
    passport = st.selectbox("Has passport", [0, 1], format_func=lambda x: "Yes" if x else "No")
    own_car = st.selectbox("Owns a car", [0, 1], format_func=lambda x: "Yes" if x else "No")
    contact = st.selectbox("Type of contact", ["Self Inquiry", "Company Invited"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    product = st.selectbox("Product pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    property_star = st.selectbox("Preferred property star", [3, 4, 5], index=0)
    marital_status = st.selectbox("Marital status", ["Single", "Married", "Divorced", "Unmarried"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    trips = st.number_input("Number of trips", min_value=0, max_value=30, value=3)
    visitors = st.number_input("People visiting", min_value=1, max_value=10, value=2)
    children = st.number_input("Children visiting", min_value=0, max_value=5, value=0)
    pitch_score = st.slider("Pitch satisfaction score", 1, 5, 3)
    followups = st.number_input("Number of follow-ups", min_value=0, max_value=10, value=3)
    pitch_duration = st.number_input("Duration of pitch (minutes)", min_value=0, max_value=120, value=15)
    submitted = st.form_submit_button("Predict purchase likelihood")

if submitted:
    row = pd.DataFrame(
        [
            {
                "Age": age,
                "CityTier": city_tier,
                "MonthlyIncome": income,
                "Passport": passport,
                "OwnCar": own_car,
                "TypeofContact": contact,
                "Occupation": occupation,
                "Gender": gender,
                "ProductPitched": product,
                "PreferredPropertyStar": property_star,
                "MaritalStatus": marital_status,
                "Designation": designation,
                "NumberOfTrips": trips,
                "NumberOfPersonVisiting": visitors,
                "NumberOfChildrenVisiting": children,
                "PitchSatisfactionScore": pitch_score,
                "NumberOfFollowups": followups,
                "DurationOfPitch": pitch_duration,
            }
        ]
    )
    model = load_model()
    probability = float(model.predict_proba(row)[0, 1])
    st.metric("Purchase probability", f"{probability:.1%}")
    st.success("Prioritize this customer for outreach." if probability >= 0.5 else "Use a lower-cost nurture campaign.")

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from src.llm_explainer import build_prompt, generate_explanation


st.set_page_config(
    page_title="Parkinson’s Disease Detection Demo",
    layout="wide"
)

st.title("Parkinson’s Disease Detection Demo")

st.markdown(
    """
    This app demonstrates an explainable AI pipeline for Parkinson’s Disease prediction
    using MDVP voice biomarkers.

    This is a research demo, not a medical diagnostic tool.
    """
)

MODEL_PATH = BASE_DIR / "models" / "xgb_model.pkl"
SAMPLE_PATH = BASE_DIR / "app" / "sample_input.csv"


if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Please run `notebooks/06_app_model_export.ipynb` "
        "to generate `models/xgb_model.pkl` before using this app."
    )
    st.stop()

if not SAMPLE_PATH.exists():
    st.error(
        "Sample input file not found. Please run `notebooks/06_app_model_export.ipynb` "
        "to generate `app/sample_input.csv`."
    )
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_shap_explainer(_model):
    model_only = _model.named_steps["model"]
    return shap.TreeExplainer(model_only)


model = load_model()
explainer = load_shap_explainer(model)

sample_df = pd.read_csv(SAMPLE_PATH)
expected_features = sample_df.columns.tolist()


# =========================
# Input Options
# =========================
st.sidebar.header("Input Options")

input_method = st.sidebar.radio(
    "Choose input method:",
    ["Use sample data", "Upload CSV", "Manual input"]
)

if input_method == "Use sample data":
    st.info("Using safe demo input based on median feature values.")
    input_df = sample_df.copy()

elif input_method == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is None:
        st.warning("Please upload a CSV file.")
        st.stop()

    input_df = pd.read_csv(uploaded_file)

else:
    st.sidebar.markdown("Enter MDVP feature values manually.")

    manual_data = {}

    for feature in expected_features:
        default_value = float(sample_df[feature].iloc[0])
        manual_data[feature] = st.sidebar.number_input(
            feature,
            value=default_value,
            format="%.4f"
        )

    input_df = pd.DataFrame([manual_data])


# =========================
# Validate Input
# =========================
missing_cols = [col for col in expected_features if col not in input_df.columns]
extra_cols = [col for col in input_df.columns if col not in expected_features]

if missing_cols:
    st.error(f"Missing required columns: {missing_cols}")
    st.stop()

if extra_cols:
    st.warning(f"Extra columns detected and removed: {extra_cols}")

input_df = input_df[expected_features]


# =========================
# Reset Results if Input Changes
# =========================
current_input_signature = input_df.to_json()

if "last_input_signature" not in st.session_state:
    st.session_state["last_input_signature"] = current_input_signature

if st.session_state["last_input_signature"] != current_input_signature:
    st.session_state.pop("results", None)
    st.session_state.pop("shap_values", None)
    st.session_state.pop("input_df", None)
    st.session_state["last_input_signature"] = current_input_signature


# =========================
# Display Input
# =========================
st.subheader("Input Data")
st.dataframe(input_df, use_container_width=True)


# =========================
# Prediction
# =========================
if st.button("Predict"):
    predictions = model.predict(input_df)
    probabilities = model.predict_proba(input_df)[:, 1]

    results = input_df.copy()
    results["prediction"] = predictions
    results["parkinson_probability"] = probabilities.round(4)

    scaler = model.named_steps["scaler"]
    X_scaled = scaler.transform(input_df)
    shap_values = explainer.shap_values(X_scaled)

    st.session_state["results"] = results
    st.session_state["shap_values"] = shap_values
    st.session_state["input_df"] = input_df
    st.session_state["last_input_signature"] = current_input_signature


# =========================
# Display Results
# =========================
if "results" in st.session_state:
    results = st.session_state["results"]
    shap_values = st.session_state["shap_values"]
    input_df = st.session_state["input_df"]

    st.subheader("Prediction Results")
    st.dataframe(results, use_container_width=True)

    st.markdown(
        """
        **Interpretation note:**  
        `prediction = 1` means the model predicts the Parkinson’s Disease class.  
        `prediction = 0` means the model predicts the healthy/control class.
        """
    )

    # =========================
    # SHAP Explanation
    # =========================
    st.subheader("SHAP Explanation")

    sample_index = st.selectbox(
        "Select sample index to explain:",
        list(range(len(input_df)))
    )

    selected_prediction = results.loc[sample_index, "prediction"]
    selected_probability = results.loc[sample_index, "parkinson_probability"]

    st.markdown(
        f"""
        Showing SHAP explanation for **sample index {sample_index}**.

        Prediction: `{selected_prediction}`  
        Parkinson's probability: `{selected_probability}`

        SHAP values indicate how much each feature contributes to the model prediction.
        Larger absolute SHAP values indicate stronger influence.
        """
    )

    shap_df = pd.DataFrame({
        "feature": input_df.columns,
        "shap_value": shap_values[sample_index],
        "absolute_shap_value": abs(shap_values[sample_index])
    }).sort_values(by="absolute_shap_value", ascending=False)

    top_shap = shap_df.head(10)

    st.dataframe(top_shap, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_shap["feature"], top_shap["absolute_shap_value"])
    ax.invert_yaxis()
    ax.set_title("Top SHAP Features")
    ax.set_xlabel("Absolute SHAP Value")

    st.pyplot(fig)

    st.markdown(
        """
        **SHAP caution:**  
        SHAP explains model behavior, not medical causality.  
        These feature contributions should not be interpreted as clinical diagnosis.
        """
    )

    # =========================
    # AI Explanation
    # =========================
    st.subheader("AI Explanation")

    explanation_style = st.radio(
        "Choose explanation style:",
        ["Scientific + careful", "Simple + user-friendly"],
        horizontal=True
    )

    if st.button("Generate AI Explanation"):
        prompt = build_prompt(
            prediction=selected_prediction,
            probability=selected_probability,
            shap_df=top_shap,
            explanation_style=explanation_style
        )

        with st.spinner("Generating explanation..."):
            explanation = generate_explanation(prompt)

        st.markdown(explanation)
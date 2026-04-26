# 🧠 Parkinson’s Disease Detection with Explainable AI

This project presents an end-to-end machine learning system for detecting Parkinson’s Disease using MDVP voice biomarkers, enhanced with explainability (SHAP) and AI-generated interpretations (LLM).

---

## 🚀 Project Highlights

* 📊 Machine Learning Models: Gradient Boosting, XGBoost
* 🧠 Explainability: SHAP (feature contribution analysis)
* 🤖 AI Explanation Layer: OpenAI LLM (human-readable interpretation)
* 🌐 Interactive App: Streamlit interface for real-time prediction

---

## 🧠 Problem Motivation

Parkinson’s Disease affects speech patterns, which can be captured using voice biomarkers such as jitter, shimmer, and turbulence indices.

This project explores how machine learning can detect patterns in these features and provide interpretable predictions.

---

## 🏗️ System Architecture

```
Input (MDVP Features)
        ↓
ML Model (XGBoost Pipeline)
        ↓
Prediction + Probability
        ↓
SHAP (Feature Contribution)
        ↓
LLM (Natural Language Explanation)
```

---

## 📊 Model Performance

* Multiple models evaluated:

  * Logistic Regression
  * SVM
  * Random Forest
  * Gradient Boosting
  * XGBoost
  * LightGBM
  * CatBoost

* Final model selected: **XGBoost (robust performance + stability)**

Metrics include:

* Accuracy
* F1 Score
* ROC-AUC
* Cross-validation analysis

---

## 🔍 Explainability (SHAP)

SHAP is used to:

* Identify top contributing features per prediction
* Provide local interpretability
* Analyze model behavior

Example important features:

* VTI (Voice Turbulence Index)
* vAm (Amplitude Variation)
* vfo (Fundamental Frequency Variation)

---

## 🤖 AI Explanation Layer

An LLM is integrated to translate SHAP outputs into human-readable explanations.

Example:

> "The model predicts a higher likelihood of Parkinson’s Disease.
> Features such as VTI and vAm contributed strongly to this prediction."

⚠️ These explanations describe model behavior, not medical diagnosis.

---

## 🌐 Streamlit App

The app supports:

* ✔ Sample data input
* ✔ CSV upload
* ✔ Manual feature entry
* ✔ Interactive SHAP visualization
* ✔ AI-generated explanations

---

## ⚠️ Privacy & Data

* The original dataset is **not publicly available**
* A **synthetic sample input** is provided for demonstration
* Model artifacts are not uploaded and must be regenerated locally

---

## ▶️ How to Run

### 1. Clone repository

```
git clone https://github.com/your-username/parkinsons-disease-detection.git
cd parkinsons-disease-detection
```

---

### 2. Create environment

```
conda create -n parkinson_app python=3.9
conda activate parkinson_app
pip install streamlit pandas scikit-learn xgboost shap openai joblib matplotlib
```

---

### 3. Export model

Run:

```
notebooks/06_app_model_export.ipynb
```

---

### 4. Set API key

```
set OPENAI_API_KEY=your_key_here
```

---

### 5. Run app

```
streamlit run app/streamlit_app.py
```

---

## 📁 Project Structure

```
app/                → Streamlit app
src/                → LLM explanation module
notebooks/          → modeling and experiments
results/            → metrics and visualizations
models/             → saved models (ignored in Git)
data/               → dataset (private)
```

---

## 🎯 Key Takeaways

This project demonstrates:

* End-to-end ML pipeline development
* Model comparison and evaluation
* Explainable AI using SHAP
* Integration of LLM for interpretability
* Deployment via interactive application

---

## 📌 Future Work

* External dataset validation
* Feature selection optimization
* Lightweight deployment (API-based)
* Clinical collaboration for validation

---

## 👩‍💻 Author

**Danisa**

M.Sc. Electrical Engineering — National Central University (Taiwan)

Biomedical AI | Machine Learning | Explainable AI

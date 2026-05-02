# Parkinson’s Disease Detection with Explainable AI

An end-to-end **Biomedical AI system** for detecting Parkinson’s Disease from voice biomarkers, enhanced with **Explainable AI (SHAP)** and **LLM-based interpretation**.

---

## 🎯 Problem & Motivation

Parkinson’s Disease affects motor control, including **speech production**.  
Subtle variations in voice—such as **jitter, shimmer, and turbulence**— can act as early indicators.

Traditional diagnosis:
- Requires clinical expertise  
- Often occurs at later stages  

👉 This project explores a **non-invasive, data-driven approach** for early screening using voice biomarkers.

---

## System Overview

```
Voice Features (MDVP)
        ↓
Feature Processing
        ↓
XGBoost Model
        ↓
Prediction + Probability
        ↓
SHAP (Feature Contribution)
        ↓
LLM (Human-Readable Explanation)
```

---

## 🚀 Key Features

- 📊 **Machine Learning Pipeline**  
  Gradient Boosting, XGBoost, and model benchmarking  

- 🧠 **Explainable AI (XAI)**  
  SHAP for feature-level interpretability  

- 🤖 **AI Explanation Layer**  
  LLM-generated natural language explanations  

- 🌐 **Interactive Application**  
  Streamlit interface for real-time predictions  

---

## Model Development and Performance

* Multiple models evaluated:

  * Logistic Regression
  * SVM
  * Random Forest
  * Gradient Boosting
  * XGBoost
  * LightGBM
  * CatBoost

* Final model selected: **XGBoost (robust performance + stability)**
Chosen for its strong performance and robustness on structured biomedical data.

Metrics include:
* Accuracy
* F1 Score
* ROC-AUC
* Cross-validation analysis

---

## Explainability (SHAP)

SHAP is used to:

* Identify top contributing features per prediction
* Provide local interpretability
* Analyze model behavior

**Key contributing features:**

* VTI (Voice Turbulence Index)
* vAm (Amplitude Variation)
* vfo (Fundamental Frequency Variation)

---

## AI Explanation Layer

To improve usability, SHAP outputs are translated into **human-readable explanations**

Example:

> "The model predicts a higher likelihood of Parkinson’s Disease.
> Features such as VTI and vAm contributed strongly to this prediction."

These explanations describe model behavior, not medical diagnosis.

---

## Streamlit Application

The system includes an interactive app with:

- ✔ Manual feature input  
- ✔ CSV upload  
- ✔ Real-time prediction  
- ✔ SHAP visualization  
- ✔ AI-generated explanation
  
---

## Privacy & Data

- Original dataset is **not publicly available**  
- Synthetic sample data is provided for demonstration  
- Model artifacts must be generated locally  

---

## How to Run

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

## Project Structure

```
app/                → Streamlit app
src/                → LLM explanation module
notebooks/          → modeling and experiments
results/            → metrics and visualizations
models/             → saved models (ignored in Git)
data/               → dataset (private)
```

---

## Key Takeaways

This project demonstrates:

* End-to-end ML pipeline design
* Model benchmarking & evaluation
* Explainable AI integration (SHAP)
* Bridging ML + LLM for interpretability
* Deployment through an interactive system

---

## Future Work

* External dataset validation
* Feature selection optimization
* Lightweight deployment (API-based)
* Clinical collaboration for validation

---

## Author

**Danisa**

M.Sc. Electrical Engineering — National Central University (Taiwan)

Biomedical AI | Machine Learning | Explainable AI

# 🧠 Parkinson’s Disease Detection Using Voice Biomarkers

🎓 M.Sc. Electrical Engineering — National Central University (Taiwan)

🔬 Biomedical AI | Machine Learning | Explainable AI

---

## 🚀 Project Overview

This project develops a **machine learning pipeline for early detection of Parkinson’s Disease** using **voice-based biomedical features (MDVP)**.

Unlike typical ML projects, this work emphasizes:

* ✅ Robust model comparison
* ✅ Cross-validation for reliability
* ✅ Explainable AI (SHAP) for interpretability

---

## 📊 Problem Statement

Parkinson’s Disease affects speech patterns early, making **voice analysis a promising non-invasive diagnostic tool**.

The goal:

> Build a reliable and interpretable model to detect Parkinson’s Disease from voice features.

---

## ⚙️ Methodology

### Data Processing

* Removed identifier column (`name`)
* Removed duplicate samples
* Removed constant features
* Standardized features

---

### Models Evaluated

* Logistic Regression
* SVM (RBF Kernel)
* Random Forest
* Gradient Boosting
* XGBoost
* LightGBM
* CatBoost

---

### Evaluation Strategy

* Stratified Train/Test Split (80/20)
* 5-Fold Stratified Cross-Validation
* Metrics:

  * F1-score ⭐ (primary)
  * ROC-AUC
  * MCC (robust for imbalance)

---

## 🏆 Key Results

### 📌 Best Models

| Model             | F1 Score   | CV F1 (Mean ± Std) |
| ----------------- | ---------- | ------------------ |
| Gradient Boosting | **0.9737** | 0.924 ± 0.023      |
| XGBoost           | 0.9610     | **0.943 ± 0.018**  |

---

### 🧠 Insights

* Gradient Boosting achieved the **highest performance**
* XGBoost showed the **most stable generalization**
* Modern boosting models consistently outperformed classical ML

---

## 🔍 Explainable AI (SHAP)

To move beyond “black-box” predictions, SHAP was used to interpret model behavior.

### 🔑 Most Influential Features

* `vAm`
* `VTI`
* `vfo`
* `RAP`
* `SPI`

---

### 📈 Key Insight

Higher values of these features strongly push predictions toward:

```text
Parkinson’s Disease (class 1)
```

---

### 🧬 Example (Individual Prediction)

SHAP reveals that predictions are driven by **combined feature contributions**, not a single factor:

* VTI → strong positive impact
* RAP → moderate contribution
* vfo → meaningful influence

---

## ⚠️ Important Note

SHAP explains **model behavior**, not medical causality.

Further clinical validation is required before real-world application.

---

## 📁 Project Structure

```
notebooks/
├── 01_data_exploration.ipynb
├── 02_gradient_boosting.ipynb
├── 03_xgboost.ipynb
├── 04_model_comparison.ipynb
├── 05_interpretability_shap.ipynb

results/
├── figures/
├── metrics/
```

---

## 🧠 Key Contributions

* End-to-end biomedical ML pipeline
* Strong model comparison (7 models)
* Cross-validation for reliability
* Explainable AI integration (SHAP)

---

## 🚀 Future Work

* Larger dataset validation
* Class balancing techniques (SMOTE)
* Deep learning comparison (FastAI Tabular)
* Clinical collaboration

---

## 📬 Contact

📧 [mirnadanisat@gmail.com](mailto:mirnadanisat@gmail.com)

🔗 https://github.com/dans2303

---

⭐ This project demonstrates both **predictive performance and interpretability**, which are critical for real-world biomedical AI.

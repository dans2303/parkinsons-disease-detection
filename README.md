# 🧠 Parkinson’s Disease Detection Using Voice Features

🎓 M.Sc. Electrical Engineering — National Central University (Taiwan)

🔬 Biomedical AI | Machine Learning | Deep Learning

---

## 📌 Project Overview

This project develops a **machine learning pipeline for early detection of Parkinson’s Disease** using **MDVP (voice-based) acoustic features**.

The goal is to:

* Build accurate classification models
* Compare classical and modern machine learning methods
* Provide **interpretable insights** into vocal biomarkers using SHAP

---

## 📊 Dataset

* Type: Tabular biomedical data (MDVP voice features)
* Target: `status` (Parkinson’s = 1, Healthy = 0)
* Samples: ~200+
* Features: Jitter, Shimmer, Frequency-based biomarkers

⚠️ **Note:**
The original dataset is **not publicly available** due to privacy and ethical considerations.
A sample dataset is provided for demonstration purposes.

---

## ⚙️ Methodology

### 1. Data Processing

* Removed identifier column (`name`)
* Removed duplicate samples
* Removed constant features
* Standardized features using `StandardScaler`

---

### 2. Models Evaluated

| Category            | Models                      |
| ------------------- | --------------------------- |
| Linear              | Logistic Regression         |
| Kernel-based        | SVM (RBF)                   |
| Ensemble (Bagging)  | Random Forest               |
| Ensemble (Boosting) | Gradient Boosting           |
| Modern Boosting     | XGBoost, LightGBM, CatBoost |

---

### 3. Evaluation Strategy

* Stratified Train/Test Split (80/20)
* 5-Fold Stratified Cross-Validation
* Metrics:

  * Accuracy
  * Precision
  * Recall
  * F1-score
  * ROC-AUC
  * Matthews Correlation Coefficient (MCC)

📌 Due to class imbalance, **F1-score, ROC-AUC, and MCC** were prioritized.

---

## 🏆 Model Performance

### Key Findings

* **Gradient Boosting** achieved the highest test performance
* **XGBoost** showed the most stable cross-validation results
* Modern boosting models consistently outperformed classical methods

📊 Example (Cross-Validation):

```text
XGBoost F1-score ≈ 0.94 ± 0.02
```

---

## 🔍 Model Interpretability (SHAP)

SHAP analysis was applied to the **XGBoost model** to understand feature contributions.

### 🔑 Key Insights

* Most influential features:

  * `vAm`
  * `VTI`
  * `vfo`
  * `RAP`
  * `SPI`

* Higher values of these features tend to push predictions toward **Parkinson’s Disease classification**

---

### 📈 SHAP Visualizations

* Global feature importance
* Feature impact distribution
* Individual prediction explanation

📌 SHAP helps explain **model behavior**, not medical causality.

---

## 📁 Project Structure

```text
parkinsons-disease-detection/
│
├── data/
│   ├── raw/           # (ignored)
│   ├── processed/     # (ignored)
│   ├── sample/        # sample dataset
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_gradient_boosting.ipynb
│   ├── 03_xgboost.ipynb
│   ├── 04_model_comparison.ipynb
│   ├── 05_interpretability_shap.ipynb
│
├── results/
│   ├── figures/
│   ├── metrics/
│
├── src/
├── README.md
```

---

## 🧠 Key Contributions

* End-to-end machine learning pipeline for biomedical classification
* Comparative analysis of classical vs modern models
* Robust evaluation using cross-validation
* SHAP-based interpretability for clinical insight

---

## ⚠️ Limitations

* Small dataset size (~200 samples)
* Class imbalance
* No external validation dataset

---

## 🚀 Future Work

* Apply SMOTE or class balancing techniques
* Evaluate deep learning approaches (FastAI Tabular)
* Test on larger and external datasets
* Compare interpretability across multiple models

---

## 📬 Contact

📧 Email: [mirnadanisat@gmail.com](mailto:mirnadanisat@gmail.com)

🔗 GitHub: https://github.com/dans2303

---

## ⭐ Final Note

This project demonstrates not only predictive performance but also **interpretability and reproducibility**, which are essential for real-world biomedical AI applications.

# Project 4: Heart Disease Prediction (Multi-Class Classification)

An end-to-end Machine Learning classification project analyzing clinical parameters to diagnose and estimate the presence and severity of heart disease in patients using the UCI Heart Disease dataset.

---

## 📌 Project Overview

Cardiovascular diseases are among the leading causes of death worldwide. Early and accurate detection enables timely interventions that can save lives. 

This project explores clinical health metrics (such as chest pain type, resting blood pressure, cholesterol levels, and exercise-induced angina) to classify the presence and severity of heart disease across 5 classes (`0 = No Disease`, `1–4 = Increasing Severity`).

### Key Objectives:
- Perform detailed **Exploratory Data Analysis (EDA)** to surface clinical correlations.
- Implement robust **data preprocessing** handling missing values, feature scaling, and one-hot encoding without data leakage.
- Train and evaluate multiple predictive models:
  - **Logistic Regression** (Baseline)
  - **Random Forest Classifier** (Ensemble)
  - **Support Vector Machine (SVM)**
  - **K-Nearest Neighbors (KNN)**
- Assess performance across standard multi-class metrics (Accuracy, Precision, Recall, F1-Score, and Confusion Matrix).
- Extract **Feature Importance** to identify dominant clinical risk indicators.

---

## 📊 Dataset Details

The project utilizes the **UCI Heart Disease Dataset** (aggregated from Cleveland, Hungary, Switzerland, and VA Long Beach cohorts):
- **Samples:** 920 patients
- **Total Features:** 16 attributes (13 clinical features used for training)
- **Target Variable (`num`):**
  - `0`: No presence of heart disease
  - `1, 2, 3, 4`: Presence and severity stage of heart disease

### Feature Dictionary:
| Feature | Type | Description |
| :--- | :--- | :--- |
| `age` | Numerical | Age of the patient (years) |
| `sex` | Categorical | Sex (`Male`, `Female`) |
| `cp` | Categorical | Chest pain type (`typical angina`, `atypical angina`, `non-anginal`, `asymptomatic`) |
| `trestbps` | Numerical | Resting blood pressure (mm Hg on admission) |
| `chol` | Numerical | Serum cholesterol in mg/dl |
| `fbs` | Categorical | Fasting blood sugar > 120 mg/dl (`True`, `False`) |
| `restecg` | Categorical | Resting electrocardiographic results (`normal`, `lv hypertrophy`, `st-t abnormality`) |
| `thalch` | Numerical | Maximum heart rate achieved |
| `exang` | Categorical | Exercise-induced angina (`True`, `False`) |
| `oldpeak` | Numerical | ST depression induced by exercise relative to rest |
| `slope` | Categorical | Slope of the peak exercise ST segment (`upsloping`, `flat`, `downsloping`) |
| `ca` | Numerical | Number of major vessels (0–3) colored by fluoroscopy |
| `thal` | Categorical | Thalassemia status (`normal`, `fixed defect`, `reversable defect`) |
| `num` | Target | Diagnosis of heart disease (severity 0 to 4) |

---

## 🛠️ Data Preprocessing Pipeline

To ensure the models receive clean, unbiased representations:
1. **Feature Pruning:** Removed non-predictive identifiers (`id`, `dataset`).
2. **Train/Test Stratification:** 80/20 train-test split stratified on the target `num` to preserve class balance across sets.
3. **Imputation:**
   - **Numerical:** Mean imputation calculated strictly on the training partition.
   - **Categorical:** Mode imputation calculated strictly on the training partition.
4. **Encoding:** One-Hot Encoding (`drop='first'`) applied to all categorical predictors.
5. **Feature Scaling:** `StandardScaler` applied to standardize continuous numerical variables.

---

## 📈 Key Findings from EDA

- **Maximum Heart Rate (`thalch`):** Patients diagnosed with heart disease generally demonstrate significantly lower peak heart rates during stress testing.
- **Chest Pain (`cp`):** Patients presenting with asymptomatic chest pain (`asymptomatic`) showed a noticeably higher likelihood of heart disease compared to those reporting typical angina.
- **Sex Distribution:** A proportionally higher occurrence of severe stages was observed among male patients within this dataset cohort.
- **Fluoroscopy (`ca`):** The number of uncolored major vessels correlated directly with disease severity.

---

## 🤖 Models & Performance Comparison

The models were evaluated on the held-out test set ($N = 184$):


> **Analysis:**
> - All models perform reliably when identifying class `0` (healthy patients, achieving $\approx 80\%$ precision and recall).
> - Multi-class severity differentiation (classes `2`, `3`, and `4`) is constrained by smaller class representation, highlighting class imbalance in advanced disease stages.
> - **SVM** and **Logistic Regression** achieved the strongest balance between precision and sensitivity for positive disease detection.

---

## 🔍 Top Predictive Features (Random Forest)

Feature importance analysis identified the top contributors to the predictive models:
1. **`oldpeak`** (ST depression induced by exercise)
2. **`thalch`** (Maximum heart rate achieved)
3. **`age`** (Patient age)
4. **`chol`** (Serum cholesterol level)
5. **`trestbps`** (Resting blood pressure)
6. **`ca`** (Major vessels colored by fluoroscopy)
7. **`cp_asymptomatic`** (Presence of asymptomatic chest pain)

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook / Google Colab

### Installation
Clone the repository and install required packages:
```bash
git clone https://github.com/khushali06/Data-Science-Projects.git
cd Data-Science-Projects/4_AI_in_Healthcare_Building_Heart_Disease_Predictor
pip install -r requirements.txt
```

### Dependencies:
```txt
pandas
numpy
scikit-learn
matplotlib
seaborn
kagglehub
```

### Running the Notebook:
Launch Jupyter or open directly in VS Code / Google Colab:
```bash
jupyter notebook "4_AI_in_Healthcare__Building_a_Heart_Disease_Predictor.ipynb"
```

---


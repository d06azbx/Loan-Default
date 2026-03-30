# Loan-Default
# 🏦 Loan Default Prediction System

A complete end-to-end system for predicting loan defaults using machine learning.
Train models in Google Colab on any dataset, then deploy predictions here on GitHub + Streamlit.

---

## 🗂️ Project Structure

```
loan-default-prediction/
│
├── app.py                          ← Streamlit web app (run this)
├── requirements.txt                ← Python dependencies
├── README.md                       ← This file
│
└── models/                         ← Your trained models live here
    └── loan_default_model/         ← One folder per trained model
        ├── loan_default_model_xgb_model.pkl
        ├── loan_default_model_scaler.pkl
        ├── loan_default_model_label_encoders.pkl
        ├── loan_default_model_X_columns.pkl
        └── loan_default_model_config.pkl
```

---

## 🚀 Quick Start

### Step 1 — Train a model in Google Colab

1. Open `NOTEBOOK_1_Train_Model.ipynb` in Google Colab
2. Upload your dataset (`.xlsx` or `.csv`)
3. Fill in **CELL 5** only:
   - `TARGET_COLUMN` — name of your default/label column
   - `DEFAULT_VALUE` — what "defaulted" looks like (e.g. `'Yes'`, `1`)
   - `NO_DEFAULT_VALUE` — what "not defaulted" looks like (e.g. `'No'`, `0`)
   - `COLUMNS_TO_DROP` — any ID or useless columns
   - `MODEL_NAME` — a name for this model (e.g. `'home_credit_2024'`)
4. Run all cells top to bottom
5. At the end, **5 `.pkl` files will download** to your computer

### Step 2 — Upload model files to GitHub

1. Create a folder inside `models/` named exactly the same as your `MODEL_NAME`
2. Upload all 5 `.pkl` files into that folder
3. Commit and push

```
models/
  home_credit_2024/
    home_credit_2024_xgb_model.pkl
    home_credit_2024_scaler.pkl
    home_credit_2024_label_encoders.pkl
    home_credit_2024_X_columns.pkl
    home_credit_2024_config.pkl
```

### Step 3 — Deploy on Streamlit Cloud (free)

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub account
3. Select this repo, branch `main`, file `app.py`
4. Click Deploy — done ✅

### Step 4 — Test a new dataset

1. Open the deployed app
2. Select your model from the sidebar
3. Switch to **Batch Upload** mode
4. Upload a new `.csv` or `.xlsx` file with the same columns
5. The app scores every row and — **if your file includes the actual default labels** — shows accuracy metrics (AUC, F1, confusion matrix)

---

## 🔄 Using Multiple Models

You can train on **different datasets** and keep all models in this repo.
Each model has its own folder in `models/`. The app detects them automatically.

```
models/
  home_credit_2024/          ← Model A: trained on Home Credit data
  lending_club_q3/           ← Model B: trained on LendingClub data
  internal_bank_dataset/     ← Model C: trained on your bank's data
```

Switch between them using the dropdown in the app sidebar.

---

## 📋 Dataset Requirements

Your dataset must:
- Be a `.csv` or `.xlsx` file
- Have a **target column** indicating default (Yes/No, 1/0, True/False, etc.)
- Have **numerical and/or categorical features** (any column names)
- Have **no completely empty columns**

The system automatically handles:
- Missing values (filled with median/mode)
- Categorical encoding (LabelEncoder, saved per model)
- Feature scaling (StandardScaler, saved per model)
- Class imbalance (KMeans-SMOTE or scale_pos_weight)

---

## 🧪 Testing Against a New Dataset

To check if a trained model generalises to a new similar dataset:

1. Prepare a new file with the **same column names** as your training data
2. Include the actual default labels in the target column
3. Upload in Batch mode
4. The app will show:
   - Confusion matrix (TP, TN, FP, FN)
   - AUC-ROC score
   - F1 score
   - Precision and Recall
   - Risk score distribution
   - Decision breakdown (Approve / Review / Reject)

---

## 📊 Risk Score Bands

| Score    | Decision        | Meaning                            |
|----------|-----------------|------------------------------------|
| 700–850  | ✅ Auto Approve | Low risk, recommend approval       |
| 550–699  | ⚠️ Manual Review | Medium risk, human review needed  |
| 300–549  | ❌ Auto Reject  | High risk, recommend rejection     |

---

## 🛠️ Run Locally

```bash
git clone https://github.com/your-username/loan-default-prediction
cd loan-default-prediction
pip install -r requirements.txt
streamlit run app.py
```

---

## 📦 What Each .pkl File Contains

| File | Contents |
|------|----------|
| `*_xgb_model.pkl` | Trained ML model (XGBoost/RandomForest/LogisticRegression) |
| `*_scaler.pkl` | StandardScaler fitted on training data |
| `*_label_encoders.pkl` | LabelEncoders for each categorical column |
| `*_X_columns.pkl` | Ordered list of feature column names |
| `*_config.pkl` | Threshold, column names, model name, performance metrics |

---

## 📌 Notes

- The model folder name and file prefix **must match exactly** (case-sensitive)
- SHAP explanations work in single-applicant mode for individual predictions
- All 5 `.pkl` files must be present for the app to load a model
- Retraining on a new dataset creates a new model — it does not overwrite existing ones

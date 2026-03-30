import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page config
st.set_page_config(page_title="Fintech Credit Risk Detector", layout="wide")

# --- SIDEBAR: FILE UPLOADS ---
st.sidebar.header("📁 Step 1: Upload Model Files")
model_file = st.sidebar.file_uploader("Upload credit_model.pkl", type=['pkl'])
scaler_file = st.sidebar.file_uploader("Upload scaler.pkl", type=['pkl'])
features_file = st.sidebar.file_uploader("Upload features.pkl", type=['pkl'])

st.sidebar.markdown("---")
st.sidebar.header("📊 Step 2: Upload New Data")
data_file = st.sidebar.file_uploader("Upload New Loan Dataset (Excel/CSV)", type=['csv', 'xlsx'])

# --- MAIN INTERFACE ---
st.title("🛡️ AI Credit Risk & Loan Assessment")
st.write("This application uses a trained XGBoost/Random Forest model to detect potential default cases in new loan applications.")

if model_file and scaler_file and features_file and data_file:
    # Load the model files
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    trained_features = joblib.load(features_file)

    # Load the new dataset
    if data_file.name.endswith('.csv'):
        new_data = pd.read_csv(data_file)
    else:
        new_data = pd.read_excel(data_file)

    st.success(f"Successfully loaded {len(new_data)} new applications.")

    # --- PRE-PROCESSING ---
    # We create a copy to process, keeping the original for display
    processing_df = new_data.copy()

    # Handle Categorical Encoding (matches your Colab logic)
    for col in processing_df.select_dtypes(include=['object']).columns:
        # Simple numeric encoding
        processing_df[col] = processing_df[col].astype('category').cat.codes

    # Ensure we only use columns the model was trained on
    try:
        processing_df = processing_df[trained_features]
        
        # Scale the data
        scaled_data = scaler.transform(processing_df)

        # --- PREDICTION ---
        predictions = model.predict(scaled_data)
        probabilities = model.predict_proba(scaled_data)[:, 1]

        # Add results back to the original dataframe
        new_data['Default_Probability (%)'] = (probabilities * 100).round(2)
        new_data['Risk_Prediction'] = ["High Risk (Default)" if p == 1 else "Low Risk (Safe)" for p in predictions]
        
        # Add a FICO-style Risk Score (300-850)
        # Higher probability = Lower Score
        new_data['Risk_Score'] = (850 - (probabilities * 550)).astype(int)

        # --- DISPLAY RESULTS ---
        high_risk_count = (predictions == 1).sum()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Apps", len(new_data))
        col2.metric("High Risk Detected", high_risk_count, delta_color="inverse")
        col3.metric("Avg Risk Score", int(new_data['Risk_Score'].mean()))

        st.subheader("📋 Prediction Results")
        
        # Color coding the dataframe
        def color_risk(val):
            color = '#ffcccc' if val == "High Risk (Default)" else '#ccffcc'
            return f'background-color: {color}'

        st.dataframe(new_data.style.applymap(color_risk, subset=['Risk_Prediction']))

        # --- DOWNLOAD REPORT ---
        st.markdown("### 📥 Download Assessment")
        csv = new_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Full Risk Report as CSV",
            data=csv,
            file_name="Credit_Risk_Assessment_Report.csv",
            mime="text/csv",
        )

    except KeyError as e:
        st.error(f"Error: The new dataset is missing columns that the model needs: {e}")
        st.info(f"The model expects these features: {trained_features}")

else:
    st.info("Please upload all 3 model files (.pkl) and a dataset to begin the analysis.")

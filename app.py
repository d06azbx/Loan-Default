"""
═══════════════════════════════════════════════════════════════
LOAN DEFAULT PREDICTION — STREAMLIT APP
GitHub-hosted inference app. Works with any trained model.
═══════════════════════════════════════════════════════════════

SETUP:
1. Upload your trained .pkl files to: models/<model_name>/
2. Run: streamlit run app.py
3. Or deploy on Streamlit Cloud from this GitHub repo.

FOLDER STRUCTURE REQUIRED:
  app.py                              ← this file
  requirements.txt
  models/
    loan_default_model/               ← one folder per trained model
      loan_default_model_xgb_model.pkl
      loan_default_model_scaler.pkl
      loan_default_model_label_encoders.pkl
      loan_default_model_X_columns.pkl
      loan_default_model_config.pkl
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────

st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 1.2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.3rem 0;
    }
    .metric-card h3 { margin: 0; font-size: 1.8rem; }
    .metric-card p { margin: 0; font-size: 0.8rem; opacity: 0.85; }
    .approve { background: linear-gradient(135deg, #11998e, #38ef7d); }
    .review  { background: linear-gradient(135deg, #f7971e, #ffd200); color: #333; }
    .reject  { background: linear-gradient(135deg, #cb2d3e, #ef473a); }
    .score-bar-bg {
        background: #f0f0f0; border-radius: 10px;
        height: 22px; width: 100%; margin: 8px 0;
    }
    .info-box {
        background: #f0f7ff;
        border-left: 4px solid #3498db;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .section-title {
        font-size: 1.1rem; font-weight: 600;
        color: #2c3e50; margin: 1rem 0 0.5rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.3rem;
    }
    div[data-testid="stMetric"] {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 0.8rem 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

MODELS_DIR = "models"

@st.cache_resource(show_spinner=False)
def load_model_artifacts(model_name):
    """Load all pkl artifacts for a given model name."""
    model_dir = os.path.join(MODELS_DIR, model_name)
    try:
        artifacts = {}
        for key, suffix in [
            ('model',          '_xgb_model.pkl'),
            ('scaler',         '_scaler.pkl'),
            ('label_encoders', '_label_encoders.pkl'),
            ('X_columns',      '_X_columns.pkl'),
            ('config',         '_config.pkl'),
        ]:
            path = os.path.join(model_dir, f"{model_name}{suffix}")
            if not os.path.exists(path):
                return None, f"Missing file: {path}"
            artifacts[key] = joblib.load(path)
        return artifacts, None
    except Exception as e:
        return None, str(e)


def get_available_models():
    """Scan models/ directory for trained models."""
    if not os.path.exists(MODELS_DIR):
        return []
    models = []
    for folder in os.listdir(MODELS_DIR):
        config_path = os.path.join(MODELS_DIR, folder, f"{folder}_config.pkl")
        if os.path.exists(config_path):
            models.append(folder)
    return sorted(models)


def predict_single(row_dict, artifacts):
    """Predict default probability for a single applicant dict."""
    config = artifacts['config']
    X_columns = artifacts['X_columns']
    scaler = artifacts['scaler']
    model = artifacts['model']
    encoders = artifacts['label_encoders']
    threshold = config['best_threshold']

    df_input = pd.DataFrame([row_dict])

    # Encode categoricals
    for col, le in encoders.items():
        if col in df_input.columns:
            val = str(df_input[col].iloc[0])
            if val in le.classes_:
                df_input[col] = le.transform([val])[0]
            else:
                df_input[col] = le.transform([le.classes_[0]])[0]

    # Align columns
    for col in X_columns:
        if col not in df_input.columns:
            df_input[col] = 0

    df_input = df_input[X_columns]
    X_scaled = scaler.transform(df_input)
    prob = model.predict_proba(X_scaled)[0][1]
    score = int(850 - (prob * 550))
    decision = 'Auto Approve' if score >= 700 else ('Manual Review' if score >= 550 else 'Auto Reject')

    return {
        'probability': round(prob * 100, 2),
        'score': score,
        'decision': decision,
        'threshold': round(threshold, 3),
        'predicted_default': prob >= threshold
    }


def predict_batch(df_upload, artifacts):
    """Predict default probability for a full uploaded dataframe."""
    config = artifacts['config']
    X_columns = artifacts['X_columns']
    scaler = artifacts['scaler']
    model = artifacts['model']
    encoders = artifacts['label_encoders']
    threshold = config['best_threshold']
    target_col = config['target_column']

    df = df_upload.copy()

    # Remove target column if present (we're predicting)
    if target_col in df.columns:
        df_true_labels = df[target_col].copy()
        df.drop(columns=[target_col], inplace=True)
    else:
        df_true_labels = None

    # Encode categoricals
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else le.transform([le.classes_[0]])[0]
            )

    # Fill missing columns
    for col in X_columns:
        if col not in df.columns:
            df[col] = 0

    df_feat = df[X_columns]
    X_scaled = scaler.transform(df_feat)
    probs = model.predict_proba(X_scaled)[:, 1]

    df_result = df_upload.copy()
    df_result['Default_Probability_%'] = (probs * 100).round(2)
    df_result['Risk_Score'] = [int(850 - p * 550) for p in probs]
    df_result['Decision'] = df_result['Risk_Score'].apply(
        lambda s: 'Auto Approve' if s >= 700 else ('Manual Review' if s >= 550 else 'Auto Reject')
    )
    df_result['Model_Prediction'] = (probs >= threshold).astype(int).map({1: 'Default', 0: 'No Default'})

    # If ground truth labels exist, compute accuracy metrics
    metrics = None
    if df_true_labels is not None:
        from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
        default_val = config['default_value']
        no_default_val = config['no_default_value']
        y_true = df_true_labels.map({default_val: 1, no_default_val: 0}).fillna(-1)
        y_true = y_true[y_true != -1].astype(int)
        if len(y_true) > 0:
            y_pred = (probs[:len(y_true)] >= threshold).astype(int)
            try:
                cm = confusion_matrix(y_true, y_pred)
                metrics = {
                    'auc': round(roc_auc_score(y_true, probs[:len(y_true)]), 4),
                    'f1': round(f1_score(y_true, y_pred), 4),
                    'cm': cm,
                    'n': len(y_true)
                }
            except Exception:
                pass

    return df_result, metrics


def draw_gauge(prob, score, ax):
    """Draw a semi-circular gauge for the risk score."""
    colors = ['#e74c3c', '#e74c3c', '#f39c12', '#f39c12', '#2ecc71', '#2ecc71']
    angles = np.linspace(np.pi, 0, 7)

    for i in range(6):
        theta = np.linspace(angles[i], angles[i+1], 50)
        x = np.cos(theta)
        y = np.sin(theta)
        ax.fill_between(x * 0.6, y * 0.6, x, y, alpha=0.85, color=colors[i])

    # Needle
    needle_angle = np.pi - (score - 300) / 550 * np.pi
    ax.annotate('', xy=(np.cos(needle_angle)*0.72, np.sin(needle_angle)*0.72),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2.5))

    ax.text(0, 0.2, f'{score}', ha='center', va='center',
            fontsize=22, fontweight='bold', color='#2c3e50')
    ax.text(0, -0.1, f'{prob:.1f}% default risk', ha='center', va='center',
            fontsize=10, color='#7f8c8d')
    ax.text(-0.85, -0.05, '300', ha='center', va='center', fontsize=9, color='#666')
    ax.text(0.85, -0.05, '850', ha='center', va='center', fontsize=9, color='#666')

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.3, 1.1)
    ax.axis('off')


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/bank.png", width=64)
    st.title("🏦 Loan Default Predictor")
    st.markdown("---")

    available_models = get_available_models()

    if not available_models:
        st.error("❌ No trained models found in `models/` folder.")
        st.info("""
**How to add a model:**
1. Run the Colab training notebook
2. Download the 5 `.pkl` files
3. Create folder: `models/<model_name>/`
4. Upload all 5 `.pkl` files there
5. Restart this app
        """)
        st.stop()

    selected_model = st.selectbox(
        "🤖 Select Trained Model",
        available_models,
        help="Each model was trained on a specific dataset"
    )

    artifacts, err = load_model_artifacts(selected_model)
    if err:
        st.error(f"Error loading model: {err}")
        st.stop()

    config = artifacts['config']

    st.markdown("---")
    st.markdown("**📊 Model Info**")
    st.markdown(f"Algorithm: `{config['best_model_name']}`")
    st.markdown(f"Threshold: `{config['best_threshold']}`")

    if 'model_results' in config:
        best = config['best_model_name']
        r = config['model_results'].get(best, {})
        if r:
            st.markdown(f"AUC-ROC: `{r.get('AUC-ROC', 'N/A')}`")
            st.markdown(f"F1 Score: `{r.get('F1 Score', 'N/A')}`")

    st.markdown("---")
    mode = st.radio(
        "🔍 Prediction Mode",
        ["Single Applicant", "Batch Upload (CSV/Excel)"],
        help="Single: fill a form. Batch: upload a file."
    )

    st.markdown("---")
    st.markdown(
        "<small>Built with XGBoost + Streamlit<br>Train new models in Google Colab</small>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.title("🏦 Loan Default Prediction System")
st.markdown(f"**Active model:** `{selected_model}` — {config['best_model_name']}")
st.markdown("---")


# ─────────────────────────────────────────────
# MODE 1: SINGLE APPLICANT
# ─────────────────────────────────────────────

if mode == "Single Applicant":

    X_columns = artifacts['X_columns']
    encoders = artifacts['label_encoders']
    num_cols = config.get('num_cols', [])
    cat_cols = config.get('cat_cols', [])

    st.markdown('<div class="section-title">📋 Applicant Details</div>', unsafe_allow_html=True)

    input_data = {}
    col_groups = [X_columns[i:i+3] for i in range(0, len(X_columns), 3)]

    for group in col_groups:
        cols = st.columns(len(group))
        for i, col_name in enumerate(group):
            with cols[i]:
                if col_name in encoders:
                    le = encoders[col_name]
                    options = list(le.classes_)
                    input_data[col_name] = st.selectbox(col_name, options, key=f"inp_{col_name}")
                else:
                    input_data[col_name] = st.number_input(
                        col_name,
                        value=0.0,
                        format="%.2f",
                        key=f"inp_{col_name}"
                    )

    st.markdown("---")

    if st.button("🔍 Predict Default Risk", type="primary", use_container_width=True):
        with st.spinner("Analysing applicant..."):
            time.sleep(0.3)
            result = predict_single(input_data, artifacts)

        prob = result['probability']
        score = result['score']
        decision = result['decision']

        # Decision banner
        decision_class = {'Auto Approve': 'approve', 'Manual Review': 'review', 'Auto Reject': 'reject'}
        decision_icon = {'Auto Approve': '✅', 'Manual Review': '⚠️', 'Auto Reject': '❌'}
        dc = decision_class.get(decision, 'approve')
        di = decision_icon.get(decision, '?')

        st.markdown(f"""
        <div class="metric-card {dc}" style="margin:1rem 0;padding:1.5rem;">
          <h2 style="margin:0;font-size:2rem;">{di} {decision}</h2>
          <p style="margin:0.3rem 0;font-size:1rem;">
            Default Probability: <strong>{prob}%</strong> &nbsp;|&nbsp;
            Risk Score: <strong>{score}</strong> &nbsp;|&nbsp;
            Threshold: <strong>{result['threshold']}</strong>
          </p>
        </div>
        """, unsafe_allow_html=True)

        # 3 columns: gauge + metrics + explanation
        c1, c2, c3 = st.columns([1.2, 1, 1.2])

        with c1:
            st.markdown("**Risk Score Gauge**")
            fig, ax = plt.subplots(figsize=(4.5, 2.8))
            fig.patch.set_alpha(0)
            draw_gauge(prob, score, ax)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with c2:
            st.markdown("**Score Breakdown**")
            st.metric("Default Probability", f"{prob}%",
                      delta=f"{'↑ Higher risk' if prob > 30 else '↓ Lower risk'}")
            st.metric("Risk Score (300–850)", score)
            st.metric("Decision", decision)

        with c3:
            st.markdown("**Decision Bands**")
            bands = [
                ("✅ Auto Approve", "Score 700–850", "#2ecc71"),
                ("⚠️ Manual Review", "Score 550–699", "#f39c12"),
                ("❌ Auto Reject", "Score 300–549", "#e74c3c"),
            ]
            for band_name, band_range, color in bands:
                highlight = "font-weight:bold;border:2px solid" if band_name.split()[1] in decision else "opacity:0.5"
                st.markdown(
                    f'<div style="background:{color}22;border-left:4px solid {color};padding:0.5rem 0.8rem;'
                    f'border-radius:0 6px 6px 0;margin:0.3rem 0;{highlight}">'
                    f'<strong>{band_name}</strong><br><small>{band_range}</small></div>',
                    unsafe_allow_html=True
                )

        st.markdown("---")

        # Risk factor analysis
        st.markdown('<div class="section-title">🔍 Risk Factor Analysis</div>', unsafe_allow_html=True)

        try:
            import shap
            X_input_df = pd.DataFrame([input_data])
            for col, le in encoders.items():
                if col in X_input_df.columns:
                    val = str(X_input_df[col].iloc[0])
                    X_input_df[col] = le.transform([val if val in le.classes_ else le.classes_[0]])[0]

            for col in X_columns:
                if col not in X_input_df.columns:
                    X_input_df[col] = 0

            X_input_df = X_input_df[X_columns]
            X_input_scaled = artifacts['scaler'].transform(X_input_df)
            X_input_scaled_df = pd.DataFrame(X_input_scaled, columns=X_columns)

            if config['best_model_name'] in ['XGBoost', 'Random Forest']:
                explainer = shap.TreeExplainer(artifacts['model'])
                sv = explainer.shap_values(X_input_scaled_df)
                if isinstance(sv, list):
                    sv = sv[1]
            else:
                explainer = shap.LinearExplainer(artifacts['model'], X_input_scaled_df)
                sv = explainer.shap_values(X_input_scaled_df)

            sv_flat = sv[0] if sv.ndim == 2 else sv
            shap_df = pd.DataFrame({
                'Feature': X_columns,
                'Input Value': [input_data.get(c, 'N/A') for c in X_columns],
                'SHAP Impact': sv_flat
            }).sort_values('SHAP Impact', key=abs, ascending=True).tail(10)

            fig2, ax2 = plt.subplots(figsize=(8, max(3, len(shap_df)*0.4)))
            colors_shap = ['#e74c3c' if v > 0 else '#2ecc71' for v in shap_df['SHAP Impact']]
            ax2.barh(shap_df['Feature'], shap_df['SHAP Impact'], color=colors_shap)
            ax2.axvline(x=0, color='black', linewidth=0.8)
            ax2.set_xlabel('SHAP Value (→ increases default risk)')
            ax2.set_title('Top 10 Factors Affecting This Decision', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close()

            st.markdown("""
            <div class="info-box">
            🔴 <b>Red bars</b> = factors increasing default risk &nbsp;|&nbsp;
            🟢 <b>Green bars</b> = factors reducing default risk
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.info(f"SHAP explanation unavailable for this input: {e}")


# ─────────────────────────────────────────────
# MODE 2: BATCH UPLOAD
# ─────────────────────────────────────────────

else:
    st.markdown('<div class="section-title">📂 Upload New Dataset for Batch Prediction</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    Upload a new dataset (CSV or Excel) that has the same columns as your training data.
    The model will score every row and show you how well it detects defaults
    if the actual labels are included in the file.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("**Required columns:**")
        st.code(", ".join(artifacts['X_columns']))

    with col2:
        st.markdown("**Optional (for accuracy):**")
        st.code(config['target_column'])

    uploaded_file = st.file_uploader(
        "Upload dataset (.csv or .xlsx)",
        type=['csv', 'xlsx'],
        help="Must contain the same feature columns as training data"
    )

    if uploaded_file:
        with st.spinner("Loading dataset..."):
            if uploaded_file.name.endswith('.csv'):
                df_new = pd.read_csv(uploaded_file)
            else:
                df_new = pd.read_excel(uploaded_file)

        st.success(f"✅ Loaded: **{len(df_new):,} rows × {df_new.shape[1]} columns**")

        # Check column alignment
        missing_cols = [c for c in artifacts['X_columns'] if c not in df_new.columns]
        extra_cols = [c for c in df_new.columns
                     if c not in artifacts['X_columns'] and c != config['target_column']]

        if missing_cols:
            st.warning(f"⚠️ Missing columns (will be filled with 0): `{missing_cols}`")
        if extra_cols:
            st.info(f"ℹ️ Extra columns (ignored): `{extra_cols}`")

        if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
            with st.spinner(f"Scoring {len(df_new):,} applicants..."):
                df_results, metrics = predict_batch(df_new, artifacts)

            st.markdown("---")
            st.markdown('<div class="section-title">📊 Batch Results</div>', unsafe_allow_html=True)

            # Summary metrics
            m1, m2, m3, m4 = st.columns(4)
            decision_counts = df_results['Decision'].value_counts()

            with m1:
                st.metric("Total Applicants", f"{len(df_results):,}")
            with m2:
                st.metric("✅ Auto Approve",
                          f"{decision_counts.get('Auto Approve', 0):,}",
                          f"{decision_counts.get('Auto Approve', 0)/len(df_results)*100:.1f}%")
            with m3:
                st.metric("⚠️ Manual Review",
                          f"{decision_counts.get('Manual Review', 0):,}",
                          f"{decision_counts.get('Manual Review', 0)/len(df_results)*100:.1f}%")
            with m4:
                st.metric("❌ Auto Reject",
                          f"{decision_counts.get('Auto Reject', 0):,}",
                          f"{decision_counts.get('Auto Reject', 0)/len(df_results)*100:.1f}%")

            # Accuracy vs ground truth
            if metrics:
                st.markdown("---")
                st.markdown('<div class="section-title">🎯 Model Accuracy vs Ground Truth</div>', unsafe_allow_html=True)

                cm = metrics['cm']
                TP = cm[1,1]; TN = cm[0,0]; FP = cm[0,1]; FN = cm[1,0]

                st.success(f"✅ Ground truth labels found! Evaluated on {metrics['n']:,} rows")

                ca, cb, cc, cd = st.columns(4)
                ca.metric("AUC-ROC", metrics['auc'])
                cb.metric("F1 Score", metrics['f1'])
                cc.metric("Precision", f"{round(TP/max(TP+FP,1),4)}")
                cd.metric("Recall", f"{round(TP/max(TP+FN,1),4)}")

                ce, cf = st.columns(2)
                with ce:
                    st.markdown("**Confusion Matrix**")
                    import seaborn as sns
                    fig3, ax3 = plt.subplots(figsize=(5, 3.5))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                                linewidths=0.5, linecolor='white',
                                xticklabels=['No Default', 'Default'],
                                yticklabels=['No Default', 'Default'])
                    ax3.set_xlabel('Predicted')
                    ax3.set_ylabel('Actual')
                    ax3.set_title(f'TP={TP:,}  TN={TN:,}  FP={FP:,}  FN={FN:,}',
                                  fontsize=9, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig3, use_container_width=True)
                    plt.close()

                with cf:
                    st.markdown("**Risk Score Distribution**")
                    fig4, ax4 = plt.subplots(figsize=(5, 3.5))
                    ax4.hist(df_results['Risk_Score'], bins=40,
                             color='#3498db', edgecolor='white', linewidth=0.5)
                    ax4.axvline(x=700, color='green', linestyle='--', linewidth=2, label='Approve ≥700')
                    ax4.axvline(x=550, color='orange', linestyle='--', linewidth=2, label='Review ≥550')
                    ax4.set_xlabel('Risk Score')
                    ax4.set_ylabel('Count')
                    ax4.set_title('Score Distribution', fontweight='bold')
                    ax4.legend(fontsize=8)
                    plt.tight_layout()
                    st.pyplot(fig4, use_container_width=True)
                    plt.close()

            # Charts
            st.markdown("---")
            st.markdown('<div class="section-title">📈 Visualisations</div>', unsafe_allow_html=True)
            v1, v2 = st.columns(2)

            with v1:
                fig5, ax5 = plt.subplots(figsize=(5, 3.5))
                dc = df_results['Decision'].value_counts()
                colors_pie = [('#2ecc71' if 'Approve' in k else '#f39c12' if 'Review' in k else '#e74c3c')
                              for k in dc.index]
                ax5.pie(dc.values,
                        labels=[f"{k}\n({v:,})" for k, v in dc.items()],
                        colors=colors_pie, autopct='%1.1f%%', startangle=90,
                        wedgeprops={'edgecolor': 'white', 'linewidth': 2})
                ax5.set_title('Decision Breakdown', fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig5, use_container_width=True)
                plt.close()

            with v2:
                fig6, ax6 = plt.subplots(figsize=(5, 3.5))
                ax6.hist(df_results['Default_Probability_%'], bins=40,
                         color='#e74c3c', alpha=0.7, edgecolor='white')
                ax6.axvline(x=config['best_threshold']*100, color='black',
                            linestyle='--', linewidth=2,
                            label=f"Threshold: {config['best_threshold']*100:.1f}%")
                ax6.set_xlabel('Default Probability (%)')
                ax6.set_ylabel('Count')
                ax6.set_title('Default Probability Distribution', fontweight='bold')
                ax6.legend(fontsize=9)
                plt.tight_layout()
                st.pyplot(fig6, use_container_width=True)
                plt.close()

            # Data preview
            st.markdown("---")
            st.markdown('<div class="section-title">📋 Results Preview</div>', unsafe_allow_html=True)
            preview_cols = list(artifacts['X_columns'][:4]) + [
                'Default_Probability_%', 'Risk_Score', 'Decision', 'Model_Prediction'
            ]
            preview_cols = [c for c in preview_cols if c in df_results.columns]
            st.dataframe(df_results[preview_cols].head(50), use_container_width=True)

            # Download
            st.markdown("---")
            csv_out = df_results.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Full Results as CSV",
                data=csv_out,
                file_name=f"predictions_{selected_model}_{uploaded_file.name.split('.')[0]}.csv",
                mime="text/csv",
                use_container_width=True
            )


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<center><small>Loan Default Prediction System · Built with XGBoost + Streamlit · "
    "Train models in Google Colab · Deploy on Streamlit Cloud</small></center>",
    unsafe_allow_html=True
)

import pandas as pd
import numpy as np
import streamlit as st
import pickle
import warnings

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model as keras_load_model

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Prediction Model",
    page_icon="🩺",
    layout="wide",
)

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Montserrat', sans-serif; }

  .hero-title { font-size: 2.8rem; font-weight: 700; color: #1a6b8a; text-align: center; margin-bottom: 0; }
  .hero-sub   { font-size: 1.05rem; color: #888; text-align: center; margin-top: 4px; }
  .section-header {
    font-size: 1.4rem; font-weight: 600; color: #1a6b8a;
    border-bottom: 2px solid #1a6b8a; padding-bottom: 6px; margin-top: 1.8rem;
  }
  .feature-card {
    background: #f0f8ff; border-left: 4px solid #1a6b8a;
    border-radius: 6px; padding: 14px 18px; margin-bottom: 10px;
  }
  .feature-card h4 { color: #1a6b8a; margin: 0 0 4px 0; font-size: 1rem; }
  .feature-card p  { color: #555; margin: 0; font-size: 0.9rem; }
  .stat-card {
    background: #1a6b8a; color: white; border-radius: 10px;
    padding: 18px; text-align: center;
  }
  .stat-card .value { font-size: 1.8rem; font-weight: 700; }
  .stat-card .label { font-size: 0.85rem; opacity: 0.85; margin-top: 2px; }
  .result-high {
    background: linear-gradient(135deg, #c62828, #b71c1c);
    color: white; border-radius: 12px; padding: 28px; text-align: center; margin-top: 1.5rem;
  }
  .result-low {
    background: linear-gradient(135deg, #2e7d32, #1b5e20);
    color: white; border-radius: 12px; padding: 28px; text-align: center; margin-top: 1.5rem;
  }
  .result-verdict { font-size: 2rem; font-weight: 700; }
  .result-label   { font-size: 0.95rem; opacity: 0.85; }
  .disclaimer {
    background: #fff3cd; border-left: 4px solid #ffc107;
    border-radius: 6px; padding: 12px 16px; margin-top: 1rem;
    font-size: 0.88rem; color: #555;
  }
  hr.divider { border: none; border-top: 1px solid #e0e0e0; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ── Data + model ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv('diabetes_prediction_dataset.csv')


@st.cache_resource
def load_model():
    return keras_load_model('cavistadiabetespred.h5')


df   = load_data()
model = load_model()


# ── Feature metadata ───────────────────────────────────────────────────────────
FEATURES = [
    {
        "key": "gender",
        "label": "Gender",
        "icon": "🚻",
        "desc": "Biological sex of the patient. Research shows hormonal and physiological differences between sexes affect diabetes risk and progression.",
    },
    {
        "key": "age",
        "label": "Age",
        "icon": "🎂",
        "desc": "Patient's age in years (0 to 80). Diabetes risk increases significantly with age, particularly after 45.",
    },
    {
        "key": "heart_disease",
        "label": "Heart Disease",
        "icon": "❤️",
        "desc": "Whether the patient has been diagnosed with heart disease (1 = Yes, 0 = No). Cardiovascular conditions and diabetes frequently co-occur.",
    },
    {
        "key": "smoking_history",
        "label": "Smoking History",
        "icon": "🚬",
        "desc": "Patient's smoking status (never, former, current, etc.). Smoking damages blood vessels and impairs insulin sensitivity, raising diabetes risk.",
    },
    {
        "key": "bmi",
        "label": "BMI (Body Mass Index)",
        "icon": "⚖️",
        "desc": "Weight-to-height ratio. BMI ≥ 25 is overweight; ≥ 30 is obese. High BMI is one of the strongest predictors of Type 2 diabetes.",
    },
    {
        "key": "HbA1c_level",
        "label": "HbA1c Level",
        "icon": "🩸",
        "desc": "Hemoglobin A1c — average blood sugar over 2 to 3 months. Values above 6.5% indicate diabetes; 5.7 to 6.4% indicates prediabetes.",
    },
    {
        "key": "blood_glucose_level",
        "label": "Blood Glucose Level",
        "icon": "💉",
        "desc": "Current blood glucose reading (mg/dL). Fasting levels above 126 mg/dL or random readings above 200 mg/dL are diagnostic for diabetes.",
    },
]


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">🩺 Diabetes Prediction Model</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Built by the Orpheus Snipers &nbsp;|&nbsp; Cavista Hackathon</p>', unsafe_allow_html=True)
st.markdown('<br>', unsafe_allow_html=True)

tab_home, tab_predict, tab_data = st.tabs(["🏠 Overview", "🔬 Predict Risk", "📊 Dataset Explorer"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_home:
    col_img, col_desc = st.columns([1, 1.6], gap="large")

    with col_img:
        st.image('glowing-abstract-design-cancer-cells-generated-by-ai.jpg', use_column_width=True)

    with col_desc:
        st.markdown('<p class="section-header">About This App</p>', unsafe_allow_html=True)
        st.markdown("""
**Diabetes** is a chronic metabolic condition affecting over 500 million people worldwide.
Early detection dramatically improves outcomes — yet many cases go undiagnosed for years.

This app uses a **Deep Learning model** (neural network) trained on a clinical dataset to assess
a patient's diabetes risk from 7 key health indicators. It was built at the **Cavista Hackathon**
by the Orpheus Snipers team to demonstrate how machine learning can support early clinical screening.

Navigate to the **Predict Risk** tab to assess a patient, or explore the **Dataset Explorer** for
statistics and visualizations.
        """)

        st.markdown('<div class="disclaimer">⚠️ <strong>Medical Disclaimer:</strong> This tool is for educational and screening purposes only. It does not replace clinical diagnosis. Always consult a qualified healthcare professional for medical advice.</div>', unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        diabetes_rate = df['diabetes'].mean() * 100 if 'diabetes' in df.columns else 0
        with c1:
            st.markdown(f'<div class="stat-card"><div class="value">{len(df):,}</div><div class="label">Patient Records</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-card"><div class="value">{diabetes_rate:.1f}%</div><div class="label">Diabetes Prevalence</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="stat-card"><div class="value">7</div><div class="label">Clinical Features</div></div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-header">Clinical Feature Guide</p>', unsafe_allow_html=True)
    st.markdown("Each feature is a clinically validated risk factor for diabetes. Understanding them helps you enter accurate patient data.")
    st.markdown('<br>', unsafe_allow_html=True)

    left, right = st.columns(2, gap="medium")
    for i, feat in enumerate(FEATURES):
        col = left if i % 2 == 0 else right
        with col:
            st.markdown(f"""
            <div class="feature-card">
              <h4>{feat['icon']} {feat['label']}</h4>
              <p>{feat['desc']}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-header">How the Model Works</p>', unsafe_allow_html=True)
    cols = st.columns(4, gap="medium")
    for col, (step, label) in zip(cols, [
        ("1️⃣", "Categorical inputs (gender, smoking history) are label-encoded"),
        ("2️⃣", "Numeric inputs are standardized to a common scale"),
        ("3️⃣", "A neural network with multiple layers processes the inputs"),
        ("4️⃣", "The output is a probability score — above 0.5 flags high risk"),
    ]):
        with col:
            st.info(f"**{step}** {label}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown('<p class="section-header">Patient Assessment</p>', unsafe_allow_html=True)
    st.markdown("Enter patient details and press **Predict** to assess diabetes risk.")
    st.markdown('<br>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        patient_name         = st.text_input("👤 Patient Name", placeholder="Enter patient name")
        gender               = st.selectbox("🚻 Gender", df['gender'].unique(), help="Biological sex of the patient")
        age                  = st.number_input("🎂 Age", 0, 100, 45, help="Age in years (0 to 100)")
        heart_disease        = st.selectbox("❤️ Heart Disease", [0, 1],
                                            format_func=lambda x: "Yes" if x == 1 else "No",
                                            help="Has the patient been diagnosed with heart disease?")

    with col2:
        smoking_history      = st.selectbox("🚬 Smoking History", df['smoking_history'].unique(),
                                            help="Patient's smoking status")
        bmi                  = st.number_input("⚖️ BMI", 0.0, 100.0, 25.0, format="%.1f",
                                               help="Body Mass Index — weight (kg) divided by height² (m)")
        HbA1c_level          = st.number_input("🩸 HbA1c Level (%)", 3.0, 15.0, 5.5, format="%.1f",
                                               help="Below 5.7 = normal, 5.7-6.4 = prediabetes, ≥6.5 = diabetes")
        blood_glucose_level  = st.number_input("💉 Blood Glucose Level (mg/dL)", 50, 500, 100,
                                               help="Fasting: <100 normal, 100-125 prediabetes, ≥126 diabetes")

    st.markdown('<br>', unsafe_allow_html=True)

    input_df = pd.DataFrame([{
        'gender': gender, 'age': age,
        'heart_disease': heart_disease, 'smoking_history': smoking_history,
        'bmi': bmi, 'HbA1c_level': HbA1c_level,
        'blood_glucose_level': blood_glucose_level,
    }])

    with st.expander("Review input values"):
        st.dataframe(input_df, use_container_width=True)

    if not patient_name:
        st.warning("Enter the patient's name before predicting.")
    elif st.button("🔬 Predict Diabetes Risk", type="primary", use_container_width=True):
        proc = input_df.copy()
        cat_cols = proc.select_dtypes(include=['object', 'category']).columns
        num_cols = proc.select_dtypes(include='number').columns

        for col in num_cols:
            proc[col] = StandardScaler().fit_transform(proc[[col]])
        for col in cat_cols:
            proc[col] = LabelEncoder().fit_transform(proc[col])

        prediction = model.predict(proc)[0][0]

        if prediction >= 0.5:
            st.markdown(f"""
            <div class="result-high">
              <div class="result-label">Assessment for {patient_name}</div>
              <div class="result-verdict">⚠️ High Risk of Diabetes</div>
              <div class="result-label" style="margin-top:10px">Risk score: {prediction*100:.1f}% — recommend follow-up clinical evaluation</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-low">
              <div class="result-label">Assessment for {patient_name}</div>
              <div class="result-verdict">✅ Low Risk of Diabetes</div>
              <div class="result-label" style="margin-top:10px">Risk score: {prediction*100:.1f}% — routine monitoring advised</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="disclaimer">⚠️ This prediction is for screening purposes only and does not constitute a medical diagnosis.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.markdown('<p class="section-header">Dataset Overview</p>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Records",   f"{len(df):,}")
    if 'diabetes' in df.columns:
        m2.metric("Diabetic",    f"{df['diabetes'].sum():,}")
        m3.metric("Non-Diabetic",f"{(df['diabetes'] == 0).sum():,}")
        m4.metric("Prevalence",  f"{df['diabetes'].mean()*100:.1f}%")

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown("**Sample rows from the dataset**")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown("**Descriptive statistics**")
    st.dataframe(df.describe().round(2), use_container_width=True)

    if 'diabetes' in df.columns:
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown("**Distributions**")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        counts = df['diabetes'].value_counts()
        axes[0].pie(counts, labels=['Non-Diabetic', 'Diabetic'], autopct='%1.1f%%',
                    colors=['#2e7d32', '#c62828'], startangle=90)
        axes[0].set_title('Diabetes Distribution')

        sns.boxplot(data=df, x='diabetes', y='blood_glucose_level',
                    palette={0: '#2e7d32', 1: '#c62828'}, ax=axes[1])
        axes[1].set_title('Blood Glucose by Diabetes Status')
        axes[1].set_xlabel('Diabetes (0 = No, 1 = Yes)')
        axes[1].set_ylabel('Blood Glucose Level (mg/dL)')
        fig.tight_layout()
        st.pyplot(fig)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image('pngwing.com (1).png', width=200)
    st.markdown("### Diabetes Prediction Model")
    st.markdown("Use the tabs above to explore data or assess a patient.")
    st.markdown("---")
    st.markdown("**Model:** Deep Neural Network (Keras)")
    st.markdown("**Dataset:** Clinical Diabetes Records")
    st.markdown("**Features:** 7 clinical indicators")
    st.markdown(f"**Records:** {len(df):,}")
    st.markdown("---")
    st.markdown("⚠️ For screening purposes only")
    st.caption("Built by Orpheus Snipers — Cavista Hackathon")

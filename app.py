import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pymongo import MongoClient
from sklearn.ensemble import RandomForestClassifier
import certifi
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="Workforce Turnover Analysis Dashboard", layout="wide")
st.title("Workforce Turnover Analysis Dashboard")

# Influence weights (your defined table)
FEATURE_WEIGHTS = {
    'JobSatisfaction':        0.25,
    'MonthlyIncome':          0.20,
    'WorkLifeBalance':        0.18,
    'EnvironmentSatisfaction':0.15,
    'OverTime':               0.12,
    'YearsAtCompany':         0.07,
    'Age':                    0.03,
}

FEATURES = list(FEATURE_WEIGHTS.keys())
WEIGHTS  = np.array(list(FEATURE_WEIGHTS.values()))   # shape (7,)

# MongoDB
@st.cache_data
def get_data():
    # Use the Environment Variable set in Render
    uri = os.getenv("MONGO_URI") 
    
    # Fallback string if Environment Variable isn't detected yet
    if not uri:
        uri = "mongodb+srv://cluster0:Shashu28@cluster0.nffdoqb.mongodb.net/"
        
    client = MongoClient(
        uri, 
        tlsCAFile=certifi.where(),
        tlsAllowInvalidCertificates=True,  # This fixes the SSL Handshake error
        serverSelectionTimeoutMS=10000     # Stops the infinite loading after 10 seconds
    )
    db = client["hr_database"]
    return pd.DataFrame(list(db["employee_data"].find()))

# Preprocessing
def preprocess(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X['OverTime'] = X['OverTime'].apply(lambda v: 1 if v == 'Yes' else 0)
    # Scale every feature by its influence weight so the RF splits
    # naturally favour the most influential columns
    X[FEATURES] = X[FEATURES].values * WEIGHTS
    return X

# Model training
@st.cache_resource
def train_model(data: pd.DataFrame):
    X = preprocess(data[FEATURES])
    y = data['Attrition'].apply(lambda v: 1 if v == 'Yes' else 0)

    # 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=12,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return model

# Risk label helper
def risk_label(prob: float) -> tuple[str, str]:
    if prob >= 65:
        return "🔴 HIGH RISK", "error"
    elif prob >= 40:
        return "🟡 MODERATE RISK", "warning"
    else:
        return "🟢 LIKELY TO STAY", "success"

# Main app
df = get_data()

if df.empty:
    st.error("No data found in MongoDB. Please load the employee dataset first.")
    st.stop()

model = train_model(df)

tab1, tab2 = st.tabs(["Attrition Dashboard", "Predict Attrition Risk"])

# TAB 1 — DASHBOARD
with tab1:
    st.sidebar.header("Filter Data")
    departments = df["Department"].unique().tolist()
    selected_dept = st.sidebar.multiselect("Department", departments, default=departments)
    filtered_df = df[df["Department"].isin(selected_dept)]

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    total     = len(filtered_df)
    left      = len(filtered_df[filtered_df["Attrition"] == "Yes"])
    rate      = (left / total * 100) if total else 0
    avg_inc   = filtered_df["MonthlyIncome"].mean()

    col1.metric("Employees Analysed",  total)
    col2.metric("Attrition Count",     left)
    col3.metric("Attrition Rate",      f"{rate:.1f}%")
    col4.metric("Avg Monthly Income",  f"${avg_inc:,.0f}")

    st.divider()

    # Charts
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(filtered_df, x="Department", color="Attrition",
                           barmode="group", title="Attrition by Department")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.box(filtered_df, x="Attrition", y="MonthlyIncome",
                     color="Attrition", title="Income Distribution vs Attrition")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.histogram(filtered_df, x="JobSatisfaction", color="Attrition",
                           barmode="group", title="Job Satisfaction vs Attrition")
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = px.histogram(filtered_df, x="WorkLifeBalance", color="Attrition",
                           barmode="group", title="Work-Life Balance vs Attrition")
        st.plotly_chart(fig, use_container_width=True)

    # Feature weight reference table
    st.divider()
    st.subheader("📌 Model Influence Weights")
    weight_df = pd.DataFrame({
        "Factor":        list(FEATURE_WEIGHTS.keys()),
        "Influence (%)": [f"{v*100:.0f}%" for v in FEATURE_WEIGHTS.values()]
    })
    st.table(weight_df)

# TAB 2 — PREDICTOR
with tab2:
    st.subheader("Test Employee Profile")
    st.markdown("Enter details below to predict the employee's flight risk.")

    with st.form("prediction_form"):
        col_a, col_b, col_c = st.columns(3)

        age     = col_a.number_input("Age",                      18, 65,    30)
        income  = col_a.number_input("Monthly Income ($)",     1000, 25000, 5000)
        years   = col_a.number_input("Years at Company",          0, 40,    3)

        job_sat = col_b.slider("Job Satisfaction (1–4)",          1, 4, 3)
        env_sat = col_b.slider("Environment Satisfaction (1–4)",  1, 4, 3)
        wl_bal  = col_b.slider("Work-Life Balance (1–4)",         1, 4, 3)

        overtime = col_c.selectbox("Works Overtime?", ["No", "Yes"])

        submitted = st.form_submit_button("Predict Turnover Risk")

    if submitted:
        raw_input = pd.DataFrame([{
            'JobSatisfaction':         job_sat,
            'MonthlyIncome':           income,
            'WorkLifeBalance':         wl_bal,
            'EnvironmentSatisfaction': env_sat,
            'OverTime':                overtime,
            'YearsAtCompany':          years,
            'Age':                     age,
        }])

        X_input   = preprocess(raw_input)          # apply same weight scaling
        prob      = model.predict_proba(X_input)[0][1] * 100
        label, kind = risk_label(prob)

        st.divider()

        if kind == "error":
            st.error(f"{label} — {prob:.1f}% probability of leaving")
        elif kind == "warning":
            st.warning(f"{label} — {prob:.1f}% probability of leaving")
        else:
            st.success(f"{label} — only {prob:.1f}% probability of leaving")

        # Personalised suggestions based on weakest factors
        st.subheader("💡 Suggestions")
        suggestions = []

        if job_sat <= 2:
            suggestions.append("🔴 **Job Satisfaction is very low (weight: 25%)** — Consider role enrichment, recognition programmes, or a promotion path.")
        if income < 3000:
            suggestions.append("🔴 **Monthly income is below average (weight: 20%)** — Benchmark against market rates and review compensation.")
        if wl_bal <= 2:
            suggestions.append("🟡 **Work-Life Balance is poor (weight: 18%)** — Review workload distribution and offer flexible working options.")
        if env_sat <= 2:
            suggestions.append("🟡 **Environment Satisfaction is low (weight: 15%)** — Address team culture, management style, or workspace conditions.")
        if overtime == "Yes":
            suggestions.append("🟡 **Consistent overtime detected (weight: 12%)** — Monitor for burnout; consider redistributing tasks.")
        if 2 <= years <= 5:
            suggestions.append("ℹ️ **Mid-tenure employee (weight: 7%)** — This band has higher natural attrition; ensure growth opportunities are visible.")

        if suggestions:
            for s in suggestions:
                st.markdown(s)
        else:
            st.markdown("✅ No major risk factors detected. Keep maintaining current conditions.")

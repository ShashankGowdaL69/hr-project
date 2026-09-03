import os
import streamlit as st
import pandas as pd
import plotly.express as px
from pymongo import MongoClient
from catboost import CatBoostClassifier
import certifi


st.set_page_config(
    page_title="Workforce Turnover Analysis Dashboard",
    layout="wide"
)

st.title("Workforce Turnover Analysis Dashboard")


# ============================================================
# MongoDB
# ============================================================

@st.cache_data
def get_data():
    uri = os.getenv("MONGO_URI")

    if not uri:
        uri = "mongodb+srv://cluster0:Shashu28@cluster0.nffdoqb.mongodb.net/"

    client = MongoClient(
        uri,
        tlsCAFile=certifi.where(),
        tlsAllowInvalidCertificates=True,
        serverSelectionTimeoutMS=10000
    )

    db = client["hr_database"]

    return pd.DataFrame(
        list(db["employee_data"].find())
    )


# ============================================================
# Preprocessing
# ============================================================

CATEGORICAL_FEATURES = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime"
]

DROP_COLUMNS = [
    "_id",
    "Attrition",
    "EmployeeNumber",
    "EmployeeCount",
    "StandardHours",
    "Over18"
]


def preprocess(data: pd.DataFrame):
    X = data.copy()

    # Remove columns that should not be used by the model
    columns_to_drop = [
        column for column in DROP_COLUMNS
        if column in X.columns
    ]

    X = X.drop(columns=columns_to_drop)

    # CatBoost requires categorical columns to be strings
    for column in CATEGORICAL_FEATURES:
        if column in X.columns:
            X[column] = X[column].fillna("Missing").astype(str)

    return X


# ============================================================
# Model Training
# ============================================================

@st.cache_resource
def train_model(data: pd.DataFrame):

    X = preprocess(data)

    y = data["Attrition"].apply(
        lambda value: 1 if value == "Yes" else 0
    )

    model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.03,
        depth=5,
        l2_leaf_reg=10,
        loss_function="Logloss",
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=False,
        thread_count=-1
    )

    categorical_indices = [
        X.columns.get_loc(column)
        for column in CATEGORICAL_FEATURES
        if column in X.columns
    ]

    model.fit(
        X,
        y,
        cat_features=categorical_indices
    )

    return model


# ============================================================
# Risk Label
# ============================================================

def risk_label(prob: float) -> tuple[str, str]:

    if prob >= 65:
        return "🔴 HIGH RISK", "error"

    elif prob >= 40:
        return "🟡 MODERATE RISK", "warning"

    else:
        return "🟢 LIKELY TO STAY", "success"


# ============================================================
# Main App
# ============================================================

df = get_data()

if df.empty:
    st.error(
        "No data found in MongoDB. Please load the employee dataset first."
    )
    st.stop()


model = train_model(df)


tab1, tab2 = st.tabs([
    "Attrition Dashboard",
    "Predict Attrition Risk"
])


# ============================================================
# TAB 1 — DASHBOARD
# ============================================================

with tab1:

    st.sidebar.header("Filter Data")

    departments = df["Department"].unique().tolist()

    selected_dept = st.sidebar.multiselect(
        "Department",
        departments,
        default=departments
    )

    filtered_df = df[
        df["Department"].isin(selected_dept)
    ]


    # --------------------------------------------------------
    # KPI Row
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    total = len(filtered_df)

    left = len(
        filtered_df[
            filtered_df["Attrition"] == "Yes"
        ]
    )

    rate = (
        left / total * 100
        if total
        else 0
    )

    avg_inc = (
        filtered_df["MonthlyIncome"].mean()
        if total
        else 0
    )

    col1.metric(
        "Employees Analysed",
        total
    )

    col2.metric(
        "Attrition Count",
        left
    )

    col3.metric(
        "Attrition Rate",
        f"{rate:.1f}%"
    )

    col4.metric(
        "Avg Monthly Income",
        f"${avg_inc:,.0f}"
    )


    st.divider()


    # --------------------------------------------------------
    # Charts
    # --------------------------------------------------------

    c1, c2 = st.columns(2)

    with c1:

        fig = px.histogram(
            filtered_df,
            x="Department",
            color="Attrition",
            barmode="group",
            title="Attrition by Department"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with c2:

        fig = px.box(
            filtered_df,
            x="Attrition",
            y="MonthlyIncome",
            color="Attrition",
            title="Income Distribution vs Attrition"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    c3, c4 = st.columns(2)

    with c3:

        fig = px.histogram(
            filtered_df,
            x="JobSatisfaction",
            color="Attrition",
            barmode="group",
            title="Job Satisfaction vs Attrition"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with c4:

        fig = px.histogram(
            filtered_df,
            x="WorkLifeBalance",
            color="Attrition",
            barmode="group",
            title="Work-Life Balance vs Attrition"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# TAB 2 — PREDICTOR
# ============================================================

with tab2:

    st.subheader("Test Employee Profile")

    st.markdown(
        "Enter details below to predict the employee's flight risk."
    )


    with st.form("prediction_form"):

        # ====================================================
        # COLUMN 1
        # ====================================================

        col_a, col_b, col_c = st.columns(3)


        with col_a:

            st.markdown("### 👤 Personal")

            age = st.number_input(
                "Age",
                min_value=18,
                max_value=65,
                value=30
            )

            gender = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

            marital_status = st.selectbox(
                "Marital Status",
                [
                    "Single",
                    "Married",
                    "Divorced"
                ]
            )

            distance = st.number_input(
                "Distance From Home",
                min_value=1,
                max_value=30,
                value=5
            )

            education = st.selectbox(
                "Education Level",
                [1, 2, 3, 4, 5],
                index=2,
                help="1 = Below College, 5 = Doctorate"
            )

            education_field = st.selectbox(
                "Education Field",
                [
                    "Life Sciences",
                    "Medical",
                    "Marketing",
                    "Technical Degree",
                    "Human Resources",
                    "Other"
                ]
            )


        # ====================================================
        # COLUMN 2
        # ====================================================

        with col_b:

            st.markdown("### 💼 Job")

            department = st.selectbox(
                "Department",
                [
                    "Sales",
                    "Research & Development",
                    "Human Resources"
                ]
            )

            job_role = st.selectbox(
                "Job Role",
                [
                    "Sales Executive",
                    "Research Scientist",
                    "Laboratory Technician",
                    "Manufacturing Director",
                    "Healthcare Representative",
                    "Manager",
                    "Sales Representative",
                    "Research Director",
                    "Human Resources"
                ]
            )

            business_travel = st.selectbox(
                "Business Travel",
                [
                    "Travel_Rarely",
                    "Travel_Frequently",
                    "Non-Travel"
                ]
            )

            job_level = st.slider(
                "Job Level",
                1,
                5,
                2
            )

            job_involvement = st.slider(
                "Job Involvement",
                1,
                4,
                3
            )

            job_sat = st.slider(
                "Job Satisfaction (1–4)",
                1,
                4,
                3
            )

            env_sat = st.slider(
                "Environment Satisfaction (1–4)",
                1,
                4,
                3
            )

            overtime = st.selectbox(
                "Works Overtime?",
                ["No", "Yes"]
            )


        # ====================================================
        # COLUMN 3
        # ====================================================

        with col_c:

            st.markdown("### 📊 Compensation & Experience")

            income = st.number_input(
                "Monthly Income ($)",
                min_value=1000,
                max_value=25000,
                value=5000
            )

            monthly_rate = st.number_input(
                "Monthly Rate",
                min_value=2000,
                max_value=27000,
                value=14000
            )

            daily_rate = st.number_input(
                "Daily Rate",
                min_value=100,
                max_value=1500,
                value=800
            )

            stock_option = st.slider(
                "Stock Option Level",
                0,
                3,
                1
            )

            total_years = st.number_input(
                "Total Working Years",
                min_value=0,
                max_value=40,
                value=8
            )

            years = st.number_input(
                "Years at Company",
                min_value=0,
                max_value=40,
                value=3
            )

            years_role = st.number_input(
                "Years in Current Role",
                min_value=0,
                max_value=18,
                value=2
            )

            years_manager = st.number_input(
                "Years With Current Manager",
                min_value=0,
                max_value=17,
                value=2
            )

            years_promotion = st.number_input(
                "Years Since Last Promotion",
                min_value=0,
                max_value=15,
                value=1
            )

            companies = st.number_input(
                "Number of Companies Worked",
                min_value=0,
                max_value=10,
                value=2
            )

            wl_bal = st.slider(
                "Work-Life Balance (1–4)",
                1,
                4,
                3
            )


        submitted = st.form_submit_button(
            "Predict Turnover Risk"
        )


    # ========================================================
    # Prediction
    # ========================================================

    if submitted:

        raw_input = pd.DataFrame([{

            "Age": age,

            "BusinessTravel": business_travel,

            "DailyRate": daily_rate,

            "Department": department,

            "DistanceFromHome": distance,

            "Education": education,

            "EducationField": education_field,

            "EnvironmentSatisfaction": env_sat,

            "Gender": gender,

            "HourlyRate": 65,

            "JobInvolvement": job_involvement,

            "JobLevel": job_level,

            "JobRole": job_role,

            "JobSatisfaction": job_sat,

            "MaritalStatus": marital_status,

            "MonthlyIncome": income,

            "MonthlyRate": monthly_rate,

            "NumCompaniesWorked": companies,

            "OverTime": overtime,

            "PercentSalaryHike": 15,

            "PerformanceRating": 3,

            "RelationshipSatisfaction": 3,

            "StockOptionLevel": stock_option,

            "TotalWorkingYears": total_years,

            "TrainingTimesLastYear": 3,

            "WorkLifeBalance": wl_bal,

            "YearsAtCompany": years,

            "YearsInCurrentRole": years_role,

            "YearsSinceLastPromotion": years_promotion,

            "YearsWithCurrManager": years_manager

        }])


        X_input = preprocess(raw_input)


        prob = (
            model.predict_proba(X_input)[0][1]
            * 100
        )


        label, kind = risk_label(prob)


        st.divider()


        if kind == "error":

            st.error(
                f"{label} — "
                f"{prob:.1f}% probability of leaving"
            )

        elif kind == "warning":

            st.warning(
                f"{label} — "
                f"{prob:.1f}% probability of leaving"
            )

        else:

            st.success(
                f"{label} — "
                f"only {prob:.1f}% probability of leaving"
            )


        # ====================================================
        # Suggestions
        # ====================================================

        st.subheader("💡 Suggestions")

        suggestions = []


        if job_sat <= 2:

            suggestions.append(
                "🔴 **Job Satisfaction is very low** — "
                "Consider role enrichment, recognition programmes, "
                "or a promotion path."
            )


        if income < 3000:

            suggestions.append(
                "🔴 **Monthly income is relatively low** — "
                "Benchmark compensation against market rates "
                "and review salary progression."
            )


        if wl_bal <= 2:

            suggestions.append(
                "🟡 **Work-Life Balance is poor** — "
                "Review workload distribution and offer flexible "
                "working options."
            )


        if env_sat <= 2:

            suggestions.append(
                "🟡 **Environment Satisfaction is low** — "
                "Address team culture, management style, "
                "or workspace conditions."
            )


        if overtime == "Yes":

            suggestions.append(
                "🟡 **Consistent overtime detected** — "
                "Monitor for burnout and consider redistributing tasks."
            )


        if years_manager <= 1:

            suggestions.append(
                "ℹ️ **Short tenure with current manager** — "
                "Ensure regular feedback, support, and clear expectations."
            )


        if stock_option == 0:

            suggestions.append(
                "ℹ️ **No stock options** — "
                "Consider whether additional long-term incentives "
                "could improve retention."
            )


        if distance > 15:

            suggestions.append(
                "ℹ️ **Long commute distance** — "
                "Consider flexible or hybrid working options where possible."
            )


        if not suggestions:

            st.markdown(
                "✅ No major risk factors detected. "
                "Keep maintaining current conditions."
            )

        else:

            for suggestion in suggestions:
                st.markdown(suggestion)
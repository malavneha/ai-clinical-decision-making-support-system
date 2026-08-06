import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="AI Clinical Decision Support System",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 AI Clinical Decision Support System")
st.write("Predict whether a diabetic patient is at risk of hospital readmission.")

# Load dataset
df = pd.read_csv("diabetic_data.csv")

# Convert target to binary
df["readmitted"] = df["readmitted"].apply(lambda x: 0 if x == "NO" else 1)

# Selected features
features = [
    "age",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "race",
    "gender",
    "A1Cresult",
    "insulin"
]

data = df[features + ["readmitted"]].copy()

# Encode categorical columns
for col in data.select_dtypes(include="object").columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))

X = data.drop("readmitted", axis=1)
y = data["readmitted"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
st.header("🩺 Patient Electronic Health Record")

age = st.selectbox(
    "Age",
    sorted(df["age"].unique())
)

time_in_hospital = st.slider(
    "Days in Hospital",
    1, 14, 5
)

num_lab_procedures = st.slider(
    "Number of Lab Procedures",
    1, 150, 40
)

num_procedures = st.slider(
    "Number of Procedures",
    0, 10, 1
)

num_medications = st.slider(
    "Number of Medications",
    1, 80, 15
)

number_outpatient = st.slider(
    "Outpatient Visits",
    0, 20, 0
)

number_emergency = st.slider(
    "Emergency Visits",
    0, 20, 0
)

number_inpatient = st.slider(
    "Previous Inpatient Visits",
    0, 20, 0
)

number_diagnoses = st.slider(
    "Number of Diagnoses",
    1, 16, 5
)

race = st.selectbox(
    "Race",
    sorted(df["race"].astype(str).unique())
)

gender = st.selectbox(
    "Gender",
    sorted(df["gender"].astype(str).unique())
)

a1c_options = (
    df["A1Cresult"]
    .fillna("Unknown")
    .astype(str)
    .drop_duplicates()
    .tolist()
)

A1Cresult = st.selectbox(
    "HbA1c Result",
    a1c_options
)


insulin_options = (
    df["insulin"]
    .fillna("Unknown")
    .astype(str)
    .drop_duplicates()
    .tolist()
)

insulin = st.selectbox(
    "Insulin",
    insulin_options
)


# =====================================================
# AI PREDICTION
# =====================================================

st.divider()
st.header("🤖 AI Clinical Decision Support")

if st.button("🔍 Predict Readmission Risk", use_container_width=True):

    # Encode categorical inputs using the same mappings
    age_val = LabelEncoder().fit(df["age"]).transform([age])[0]
    race_val = LabelEncoder().fit(df["race"].astype(str)).transform([race])[0]
    gender_val = LabelEncoder().fit(df["gender"].astype(str)).transform([gender])[0]
    a1c_val = LabelEncoder().fit(df["A1Cresult"].astype(str)).transform([A1Cresult])[0]
    insulin_val = LabelEncoder().fit(df["insulin"].astype(str)).transform([insulin])[0]

    patient = pd.DataFrame({
        "age":[age_val],
        "time_in_hospital":[time_in_hospital],
        "num_lab_procedures":[num_lab_procedures],
        "num_procedures":[num_procedures],
        "num_medications":[num_medications],
        "number_outpatient":[number_outpatient],
        "number_emergency":[number_emergency],
        "number_inpatient":[number_inpatient],
        "number_diagnoses":[number_diagnoses],
        "race":[race_val],
        "gender":[gender_val],
        "A1Cresult":[a1c_val],
        "insulin":[insulin_val]
    })

    prediction = model.predict(patient)[0]
    probability = model.predict_proba(patient)[0][1]

    st.subheader("📋 AI Assessment")

    if prediction == 1:
        st.error("🔴 HIGH RISK OF READMISSION")
    else:
        st.success("🟢 LOW RISK OF READMISSION")

    st.metric(
        "Predicted Risk",
        f"{probability*100:.1f}%"
    )

    st.progress(float(probability))

    st.divider()

    st.subheader("👩‍⚕️ Clinical Recommendations")

    if prediction == 1:

        st.warning("""
✔ Schedule follow-up within 7 days

✔ Medication reconciliation

✔ Diabetes education

✔ Review discharge plan

✔ Coordinate outpatient care
""")

    else:

        st.success("""
✔ Standard discharge

✔ Routine follow-up

✔ Continue current management

✔ Encourage medication adherence
""")

# =====================================================
# EXECUTIVE DASHBOARD
# =====================================================

st.divider()

st.header("📊 Executive Dashboard")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "👥 Patients",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "🏥 Avg Stay",
        f"{df['time_in_hospital'].mean():.1f} Days"
    )

with col3:
    st.metric(
        "💊 Avg Medications",
        f"{df['num_medications'].mean():.1f}"
    )

with col4:
    rate = (df["readmitted"] == 1).mean() * 100

    st.metric(
        "📈 Readmission Rate",
        f"{rate:.2f}%"
    )

st.divider()

# =====================================================
# ANALYTICS
# =====================================================

st.header("📈 Hospital Analytics")

tabA, tabB, tabC = st.tabs([
    "Readmission",
    "Hospital Stay",
    "Demographics"
])

with tabA:

    st.subheader("Readmission Distribution")

    st.bar_chart(df["readmitted"].value_counts())

with tabB:

    st.subheader("Hospital Stay")

    st.bar_chart(
        df["time_in_hospital"].value_counts().sort_index()
    )

with tabC:

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Gender")

        st.bar_chart(df["gender"].value_counts())

    with col2:

        st.subheader("Race")

        st.bar_chart(df["race"].value_counts())

# =====================================================
# DOWNLOAD REPORT
# =====================================================

st.divider()

st.header("📄 Clinical Report")

report = f"""
AI CLINICAL DECISION SUPPORT SYSTEM

Total Patients : {len(df)}

Average Hospital Stay : {df['time_in_hospital'].mean():.2f}

Average Medications : {df['num_medications'].mean():.2f}

Readmission Rate : {rate:.2f}%

Generated using AI Clinical Decision Support System
"""

st.download_button(
    "📥 Download Hospital Summary",
    report,
    file_name="Hospital_Report.txt"
)

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "🏥 AI Clinical Decision Support System | Developed by Dr. Neha Malav"
)

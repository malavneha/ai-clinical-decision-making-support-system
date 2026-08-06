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

A1Cresult = st.selectbox(
    "HbA1c Result",
    sorted(df["A1Cresult"].astype(str).unique())
)

insulin = st.selectbox(
    "Insulin",
    sorted(df["insulin"].astype(str).unique())
)

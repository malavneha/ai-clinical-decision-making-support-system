import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------

st.set_page_config(
    page_title="AI Clinical Decision Support System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------------
# HOSPITAL THEME
# --------------------------------------------------------

st.markdown("""
<style>

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

.block-container{
padding-top:1rem;
padding-bottom:1rem;
padding-left:2rem;
padding-right:2rem;
}

body{
background:#F4F8FB;
}

.big-title{
font-size:40px;
font-weight:700;
color:#1565C0;
}

.subtitle{
font-size:18px;
color:gray;
margin-top:-10px;
margin-bottom:20px;
}

.card{

background:white;

padding:18px;

border-radius:15px;

box-shadow:0px 2px 12px rgba(0,0,0,.08);

margin-bottom:15px;

}

.kpi{

background:white;

padding:18px;

border-radius:12px;

text-align:center;

box-shadow:0px 2px 10px rgba(0,0,0,.08);

}

</style>
""",unsafe_allow_html=True)

# --------------------------------------------------------
# LOAD DATA
# --------------------------------------------------------

@st.cache_data
def load_data():

    return pd.read_csv("diabetic_data.csv")

df=load_data()

# --------------------------------------------------------
# PREPROCESSING
# --------------------------------------------------------

data=df.copy()

data.replace("?",np.nan,inplace=True)

drop_cols=[
"encounter_id",
"patient_nbr"
]

for col in drop_cols:

    if col in data.columns:

        data.drop(col,axis=1,inplace=True)

for col in data.columns:

    if data[col].dtype=="object":

        data[col].fillna(
            data[col].mode()[0],
            inplace=True
        )

    else:

        data[col].fillna(
            data[col].median(),
            inplace=True
        )

data["readmitted"]=data["readmitted"].replace({
"NO":"Low",
">30":"Medium",
"<30":"High"
})

# --------------------------------------------------------
# HEADER
# --------------------------------------------------------

st.markdown(
"""
<div class='big-title'>
🏥 AI Clinical Decision Support System
</div>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<div class='subtitle'>
Hospital Readmission Prediction | Electronic Health Record
</div>
""",
unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------------
# TOP MENU
# --------------------------------------------------------

tab1,tab2,tab3,tab4,tab5,tab6=st.tabs([

"🏠 Dashboard",

"👤 Patients",

"📊 Analytics",

"🤖 AI",

"📄 Reports",

"ℹ About"

])

# ========================================================
# DASHBOARD
# ========================================================

with tab1:

    st.subheader("Executive Dashboard")

    c1,c2,c3,c4=st.columns(4)

    c1.metric(
        "Patients",
        f"{len(df):,}"
    )

    c2.metric(
        "High Risk",
        (data["readmitted"]=="High").sum()
    )

    c3.metric(
        "Medium Risk",
        (data["readmitted"]=="Medium").sum()
    )

    c4.metric(
        "Low Risk",
        (data["readmitted"]=="Low").sum()
    )

    st.divider()

    st.subheader("Recent Patients")

    preview=df[[
        "race",
        "gender",
        "age",
        "time_in_hospital",
        "readmitted"
    ]].head(15)

    st.dataframe(
        preview,
        use_container_width=True
    )
    # ========================================================
# PATIENT EXPLORER
# ========================================================

with tab2:

    st.subheader("👤 Patient Explorer")

    patient_no = st.slider(
        "Select Patient",
        0,
        len(df)-1,
        0
    )

    patient = df.iloc[patient_no]

    left,right = st.columns([1,2])

    # -----------------------------
    # Patient Card
    # -----------------------------

    with left:

        st.markdown("## 👤 Patient Profile")

        st.info(f"""
**Patient ID**

{patient_no+1}

**Gender**

{patient['gender']}

**Race**

{patient['race']}

**Age**

{patient['age']}
""")

        if patient["readmitted"]=="<30":

            st.error("🔴 HIGH RISK")

        elif patient["readmitted"]==">30":

            st.warning("🟠 MEDIUM RISK")

        else:

            st.success("🟢 LOW RISK")

    # -----------------------------
    # Clinical Information
    # -----------------------------

    with right:

        st.markdown("## 🏥 Hospital Encounter")

        c1,c2,c3=st.columns(3)

        c1.metric(
            "Hospital Stay",
            patient["time_in_hospital"]
        )

        c2.metric(
            "Lab Procedures",
            patient["num_lab_procedures"]
        )

        c3.metric(
            "Medications",
            patient["num_medications"]
        )

        st.divider()

        c4,c5,c6=st.columns(3)

        c4.metric(
            "Diagnoses",
            patient["number_diagnoses"]
        )

        c5.metric(
            "Outpatient",
            patient["number_outpatient"]
        )

        c6.metric(
            "Emergency",
            patient["number_emergency"]
        )

        st.divider()

        st.subheader("💊 Diabetes Management")

        info=pd.DataFrame({

            "Field":[

                "Diabetes Medication",

                "Insulin",

                "A1C Result",

                "Glucose Serum Test"

            ],

            "Value":[

                patient["diabetesMed"],

                patient["insulin"],

                patient["A1Cresult"],

                patient["max_glu_serum"]

            ]

        })

        st.dataframe(
            info,
            use_container_width=True
        )

        st.subheader("📋 Admission Information")

        admission=pd.DataFrame({

            "Field":[

                "Admission Type",

                "Discharge Disposition",

                "Admission Source"

            ],

            "Value":[

                patient["admission_type_id"],

                patient["discharge_disposition_id"],

                patient["admission_source_id"]

            ]

        })

        st.dataframe(
            admission,
            use_container_width=True
        )

# ========================================================
# AI CLINICAL DECISION SUPPORT
# ========================================================

with tab4:

    st.subheader("🤖 AI Clinical Decision Support")

    st.info(
        "Enter patient clinical information and let the AI estimate the readmission risk."
    )

    left, right = st.columns(2)

    with left:

        age = st.selectbox(
            "Age Group",
            sorted(df["age"].unique())
        )

        gender = st.selectbox(
            "Gender",
            sorted(df["gender"].unique())
        )

        race = st.selectbox(
            "Race",
            sorted(df["race"].dropna().unique())
        )

        time_in_hospital = st.slider(
            "Hospital Stay",
            1,
            14,
            4
        )

        num_lab = st.slider(
            "Lab Procedures",
            1,
            150,
            40
        )

        num_med = st.slider(
            "Medications",
            1,
            80,
            15
        )

    with right:

        diagnoses = st.slider(
            "Number of Diagnoses",
            1,
            16,
            5
        )

        outpatient = st.slider(
            "Outpatient Visits",
            0,
            20,
            1
        )

        emergency = st.slider(
            "Emergency Visits",
            0,
            20,
            0
        )

        inpatient = st.slider(
            "Inpatient Visits",
            0,
            20,
            0
        )

        diabetes_med = st.selectbox(
            "Diabetes Medication",
            sorted(df["diabetesMed"].unique())
        )

        insulin = st.selectbox(
            "Insulin",
            sorted(df["insulin"].unique())
        )

    st.divider()

    if st.button("🧠 Run AI Assessment", use_container_width=True):

        # --------------------------------------------------
        # Temporary demonstration score
        # Replace with model.predict() after model.pkl
        # --------------------------------------------------

        score = (
            time_in_hospital*4
            + diagnoses*3
            + emergency*5
            + inpatient*5
            + num_med*0.6
            + outpatient*2
        )

        probability = min(score/100,1)

        st.subheader("AI Assessment")

        if probability >= 0.70:

            st.error("🔴 HIGH RISK")

        elif probability >= 0.40:

            st.warning("🟠 MODERATE RISK")

        else:

            st.success("🟢 LOW RISK")

        st.metric(
            "Estimated Readmission Risk",
            f"{probability:.1%}"
        )

        st.progress(probability)

        st.divider()

        st.subheader("Clinical Decision Support")

        if probability >= 0.70:

            st.error("""
• Follow-up within 7 days

• Medication reconciliation

• Care coordinator referral

• Diabetes education

• Arrange telehealth monitoring

• Review discharge plan
""")

        elif probability >= 0.40:

            st.warning("""
• Routine follow-up

• Reinforce medication adherence

• Monitor symptoms

• Lifestyle counselling
""")

        else:

            st.success("""
Patient currently appears to be at relatively
low risk for readmission.

Proceed with standard discharge planning.
""")

        st.divider()

        st.subheader("Risk Summary")

        summary = pd.DataFrame({

            "Metric":[

                "Risk Probability",

                "Hospital Stay",

                "Diagnoses",

                "Emergency Visits",

                "Medications"

            ],

            "Value":[

                f"{probability:.1%}",

                time_in_hospital,

                diagnoses,

                emergency,

                num_med

            ]

        })

        st.dataframe(
            summary,
            use_container_width=True
        )
        # ========================================================
# HEALTHCARE ANALYTICS
# ========================================================

with tab3:

    st.subheader("📊 Healthcare Analytics Dashboard")

    st.markdown("### Patient Demographics")

    col1, col2 = st.columns(2)

    with col1:

        gender_count = df["gender"].value_counts()

        st.bar_chart(gender_count)

    with col2:

        race_count = df["race"].value_counts()

        st.bar_chart(race_count)

    st.divider()

    st.subheader("🏥 Hospital Stay Analysis")

    st.bar_chart(df["time_in_hospital"].value_counts().sort_index())

    st.divider()

    st.subheader("💊 Medication Distribution")

    med = df["num_medications"]

    st.line_chart(med)

    st.divider()

    st.subheader("🧪 Laboratory Procedures")

    lab = df["num_lab_procedures"]

    st.line_chart(lab)

    st.divider()

    st.subheader("📋 Readmission Distribution")

    readmission = df["readmitted"].value_counts()

    st.bar_chart(readmission)

    st.divider()

    st.subheader("📈 Summary Statistics")

    summary = pd.DataFrame({

        "Metric":[
            "Total Patients",
            "Average Hospital Stay",
            "Average Medications",
            "Average Lab Procedures",
            "Average Diagnoses"
        ],

        "Value":[
            len(df),
            round(df["time_in_hospital"].mean(),2),
            round(df["num_medications"].mean(),2),
            round(df["num_lab_procedures"].mean(),2),
            round(df["number_diagnoses"].mean(),2)
        ]

    })

    st.dataframe(summary, use_container_width=True)

    st.success("Healthcare analytics generated successfully.")
    # ========================================================
# CLINICAL REPORT
# ========================================================

with tab5:

    st.subheader("📄 Clinical Decision Support Report")

    patient_no = st.slider(
        "Select Patient for Report",
        0,
        len(df)-1,
        0,
        key="report_patient"
    )

    patient = df.iloc[patient_no]

    st.markdown("---")

    col1, col2 = st.columns([1,2])

    with col1:

        st.markdown("### 👤 Patient")

        st.write(f"**Patient ID:** {patient_no+1}")
        st.write(f"**Age:** {patient['age']}")
        st.write(f"**Gender:** {patient['gender']}")
        st.write(f"**Race:** {patient['race']}")

    with col2:

        st.markdown("### 🏥 Clinical Summary")

        st.write(f"Hospital Stay: **{patient['time_in_hospital']} days**")
        st.write(f"Diagnoses: **{patient['number_diagnoses']}**")
        st.write(f"Lab Procedures: **{patient['num_lab_procedures']}**")
        st.write(f"Medications: **{patient['num_medications']}**")

    st.markdown("---")

    st.subheader("🤖 AI Readmission Assessment")

    if patient["readmitted"] == "<30":

        st.error("🔴 HIGH READMISSION RISK")

        recommendation = """
• Arrange follow-up within 7 days

• Medication reconciliation

• Diabetes education

• Care coordinator referral

• Monitor medication adherence
"""

    elif patient["readmitted"] == ">30":

        st.warning("🟠 MODERATE READMISSION RISK")

        recommendation = """
• Routine outpatient follow-up

• Reinforce medication adherence

• Lifestyle counselling

• Monitor glycemic control
"""

    else:

        st.success("🟢 LOW READMISSION RISK")

        recommendation = """
• Continue standard discharge protocol

• Routine follow-up

• Encourage healthy lifestyle
"""

    st.subheader("👩‍⚕️ Clinical Recommendations")

    st.info(recommendation)

    st.subheader("📋 Discharge Checklist")

    discharge = pd.DataFrame({

        "Task":[

            "Medication Reviewed",

            "Patient Education",

            "Follow-up Scheduled",

            "Discharge Summary Prepared",

            "Primary Care Notified"

        ],

        "Status":[

            "☐",

            "☐",

            "☐",

            "☐",

            "☐"

        ]

    })

    st.dataframe(
        discharge,
        use_container_width=True
    )

    st.subheader("📑 Clinical Report")

    report = f"""
AI CLINICAL DECISION SUPPORT REPORT

Patient ID : {patient_no+1}

Age : {patient['age']}

Gender : {patient['gender']}

Race : {patient['race']}

Hospital Stay : {patient['time_in_hospital']} days

Diagnoses : {patient['number_diagnoses']}

Lab Procedures : {patient['num_lab_procedures']}

Medications : {patient['num_medications']}

Readmission Category : {patient['readmitted']}

Recommendations

{recommendation}
"""

    st.download_button(

        "📥 Download Clinical Report",

        report,

        file_name=f"Patient_{patient_no+1}_Clinical_Report.txt"

    )

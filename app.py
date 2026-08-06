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

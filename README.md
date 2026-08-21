# 🏥 AI Clinical Decision Support System

An AI-powered Clinical Decision Support System (CDSS) that predicts whether a diabetic patient is at risk of hospital readmission using Machine Learning. The application provides real-time predictions, interactive dashboards, hospital analytics, and downloadable clinical reports.

[Research Prototype — Not for Clinical Use:-
This system is intended for research and educational demonstration. It has not been clinically validated and should not replace professional clinical judgment.]


---
What does it do?
Patient data → AI risk prediction → decision-support information → analytics

Who is it for?
Research/clinical-informatics demonstration

Current status?
Working research prototype
Try it
Live Streamlit Demo
Code
GitHub
---
Patient Data
⬇️
Data Preprocessing
⬇️
Random Forest Model
⬇️
Readmission Risk
⬇️
Clinical Decision-Support Layer
⬇️
Streamlit Interface
⬇️
Clinician Review

## 🚀 Live Demo
🔗 https://ai-clinical-decision-making-support-system-pjs7nfntqcvqfqkjjby.streamlit.app

🔗 GitHub Repository:
https://github.com/malavneha/ai-clinical-decision-making-support-system

---

## 📌 Project Overview

Hospital readmissions increase healthcare costs and reduce patient outcomes. This project uses a Random Forest Machine Learning model trained on a diabetic patient dataset to estimate the probability of hospital readmission.

The application allows healthcare professionals to enter patient information and instantly receive:

- Readmission Risk Prediction
- Risk Probability
- Clinical Recommendations
- Executive Dashboard
- Hospital Analytics
- Downloadable Clinical Report

---

## ✨ Features

- 🤖 Machine Learning Prediction
- 📊 Interactive Dashboard
- 📈 Hospital Analytics
- 🏥 Patient Risk Assessment
- 📄 Downloadable Clinical Report
- 🌐 Streamlit Web Application
- ☁️ Cloud Deployment

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib

---

## 📂 Dataset

Dataset used:

Diabetes 130-US Hospitals Dataset

Features include:

- Age
- Gender
- Race
- Time in Hospital
- Number of Lab Procedures
- Number of Medications
- Outpatient Visits
- Emergency Visits
- Inpatient Visits
- Number of Diagnoses
- HbA1c Result
- Insulin Status

Target Variable:

- Readmitted (0 = No, 1 = Yes)

---

## ⚙️ Machine Learning Model

Algorithm:

- Random Forest Classifier

Workflow:

1. Data Cleaning
2. Feature Selection
3. Label Encoding
4. Train/Test Split
5. Model Training
6. Prediction
7. Model Serialization (Joblib)

---

## 📊 Dashboard

The application includes:

- Total Patients
- Average Hospital Stay
- Average Medications
- Readmission Rate

Analytics:

- Readmission Distribution
- Hospital Stay Distribution
- Gender Distribution
- Race Distribution

---

## 📄 Clinical Report

The application generates a downloadable report containing:

- Hospital Statistics
- Readmission Rate
- Average Stay
- Average Medications

---

## 📷 Screenshots

### Home Page

 ![home page](home%20page.jpg)



### Dashboard
  ![Project Screenshot](Dashboard.jpg)



### Hospital Analytics

  ![Hospital Analytics](hospital%20analytics.jpg)



---

## ▶️ Installation

Clone the repository
https://github.com/malavneha/ai-clinical-decision-making-support-system

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---
## Responsible AI" section 

●Human oversight
AI output is intended to support—not replace—clinical judgment.
●Privacy
Do not enter identifiable patient information.
●Bias
Model performance may vary across patient populations and should be evaluated across relevant subgroups.
●Validation
The system is a research prototype and requires external/clinical validation before real-world deployment.
---

## 📈 Future Improvements

- SHAP Explainable AI
- Multiple ML Models
- Patient History Tracking
- Authentication
- PDF Clinical Reports
- Database Integration

---

## 👩‍💻 Developer

**Dr. Neha Malav**

GitHub:
https://github.com/malavneha

LinkedIn:
(https://www.linkedin.com/in/dr-neha-malav-743a25332)

## 📜 License

This project is licensed under the MIT License.

---

⭐ If you found this project useful, consider giving it a star!

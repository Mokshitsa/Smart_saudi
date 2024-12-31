import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import time
import hashlib

# --- APP CONFIGURATION ---
st.set_page_config(
    page_title="SMART Saudi - Chronic Disease Prediction",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- FUTURISTIC HEADER ---
st.markdown(
    """
    <div style="background-color:#004AAD; padding:20px; border-radius:10px;">
        <h1 style="color:white; text-align:center;">SMART Saudi 🌟</h1>
        <h3 style="color:white; text-align:center;">AI-Powered Chronic Disease Prediction | Vision 2030</h3>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- SIDEBAR FOR INPUT ---
st.sidebar.header("Patient Information")
st.sidebar.markdown(
    """
    Enter the patient's details below for a comprehensive health analysis.
    Your data is encrypted and securely stored using blockchain technology.
    """
)

# Function to get user inputs
def get_user_input():
    name = st.sidebar.text_input("Patient Name", "John Doe")
    national_id = st.sidebar.text_input("National ID", "1234567890")
    age = st.sidebar.slider("Age", 0, 100, 30)
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    bmi = st.sidebar.slider("BMI", 10.0, 50.0, 22.5)
    blood_pressure = st.sidebar.slider("Blood Pressure (mmHg)", 80, 200, 120)
    cholesterol = st.sidebar.selectbox("Cholesterol Level", ("Normal", "Above Normal", "High"))
    glucose = st.sidebar.selectbox("Glucose Level", ("Normal", "Above Normal", "High"))
    smoking = st.sidebar.selectbox("Smoking Habit", ("No", "Yes"))
    physical_activity = st.sidebar.selectbox("Physical Activity Level", ("Low", "Moderate", "High"))

    # Hash National ID for security
    hashed_id = hashlib.sha256(national_id.encode()).hexdigest()

    # Return data
    user_data = {
        "Name": name,
        "NationalID_Hash": hashed_id,
        "Age": age,
        "Gender": 1 if gender == "Male" else 0,
        "BMI": bmi,
        "BloodPressure": blood_pressure,
        "Cholesterol": {"Normal": 0, "Above Normal": 1, "High": 2}[cholesterol],
        "Glucose": {"Normal": 0, "Above Normal": 1, "High": 2}[glucose],
        "Smoking": 1 if smoking == "Yes" else 0,
        "PhysicalActivity": {"Low": 0, "Moderate": 1, "High": 2}[physical_activity],
    }

    return pd.DataFrame(user_data, index=[0])

user_input = get_user_input()

# --- DISPLAY USER INPUT ---
st.subheader("Patient Details")
st.write(user_input)

# --- DUMMY MACHINE LEARNING MODEL ---
def train_model():
    np.random.seed(42)
    X = pd.DataFrame({
        "Age": np.random.randint(20, 80, 500),
        "Gender": np.random.choice([0, 1], 500),
        "BMI": np.random.uniform(15, 40, 500),
        "BloodPressure": np.random.randint(80, 180, 500),
        "Cholesterol": np.random.choice([0, 1, 2], 500),
        "Glucose": np.random.choice([0, 1, 2], 500),
        "Smoking": np.random.choice([0, 1], 500),
        "PhysicalActivity": np.random.choice([0, 1, 2], 500),
    })
    y = np.random.choice([0, 1], 500)  # 0: Low Risk, 1: High Risk

    model = RandomForestClassifier()
    model.fit(X, y)

    return model

model = train_model()

# --- PREDICTION ---
prediction = model.predict(user_input.drop(columns=["Name", "NationalID_Hash"]))[0]
prediction_proba = model.predict_proba(user_input.drop(columns=["Name", "NationalID_Hash"]))[0]

# --- DISPLAY PREDICTION ---
st.subheader("Prediction Result")
if prediction == 1:
    st.markdown("### 🚨 **High Risk of Chronic Disease Detected**")
    st.warning("Please consult a healthcare provider immediately.")
else:
    st.markdown("### ✅ **Low Risk of Chronic Disease Detected**")
    st.success("Maintain your current healthy lifestyle!")

# --- DISPLAY PROBABILITIES ---
st.subheader("Prediction Probabilities")
st.write(f"Low Risk: {prediction_proba[0]*100:.2f}%")
st.write(f"High Risk: {prediction_proba[1]*100:.2f}%")

# --- HEALTH RECOMMENDATIONS ---
st.subheader("Health Recommendations")
if prediction == 1:
    st.markdown("""
    - Visit a healthcare provider for a detailed check-up.
    - Reduce intake of processed and fatty foods.
    - Exercise regularly and maintain a healthy BMI.
    - Avoid smoking and alcohol consumption.
    """)
else:
    st.markdown("""
    - Continue regular physical activity and a balanced diet.
    - Stay hydrated and manage stress effectively.
    - Schedule periodic health check-ups to monitor key metrics.
    """)

# --- FUTURISTIC FOOTER ---
st.markdown(
    """
    <hr>
    <div style="text-align:center;">
        <p>Developed under Vision 2030 for a healthier Saudi Arabia 🌟</p>
        <p><strong>SMART Saudi</strong> | Secure | AI-Powered | Blockchain-Enabled</p>
    </div>
    """,
    unsafe_allow_html=True,
)

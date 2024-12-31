import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Title and Description
st.title("SMART Saudi: Chronic Disease Prediction")
st.write("""
### A predictive tool for identifying chronic disease risks based on patient data.
This tool helps in early identification of chronic disease risks to enable timely interventions.
""")

# Sidebar for User Input
st.sidebar.header("Patient Input Details")

def get_user_input():
    age = st.sidebar.slider("Age", 0, 100, 30)
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    bmi = st.sidebar.slider("BMI", 10.0, 50.0, 22.5)
    blood_pressure = st.sidebar.slider("Blood Pressure (mmHg)", 80, 200, 120)
    cholesterol = st.sidebar.selectbox("Cholesterol Level", ("Normal", "Above Normal", "High"))
    glucose = st.sidebar.selectbox("Glucose Level", ("Normal", "Above Normal", "High"))
    smoking = st.sidebar.selectbox("Smoking", ("No", "Yes"))
    physical_activity = st.sidebar.selectbox("Physical Activity", ("Low", "Moderate", "High"))

    # Convert inputs into a dataframe
    user_data = {
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

# Get user input
user_input = get_user_input()

# Display user input
st.subheader("Patient Details")
st.write(user_input)

# Dummy data and model training (Replace this with actual trained model later)
def train_model():
    # Generate dummy data for training
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

    # Train Random Forest Classifier
    model = RandomForestClassifier()
    model.fit(X, y)

    return model

# Train model
model = train_model()

# Predict risk based on user input
prediction = model.predict(user_input)[0]
prediction_proba = model.predict_proba(user_input)[0]

# Display Prediction
st.subheader("Prediction")
if prediction == 1:
    st.write("### High Risk of Chronic Disease")
    st.write("The patient is at high risk of developing chronic diseases. Please consult a healthcare provider.")
else:
    st.write("### Low Risk of Chronic Disease")
    st.write("The patient has a low risk of chronic diseases. Maintain a healthy lifestyle.")

# Display Prediction Probabilities
st.subheader("Prediction Probabilities")
st.write(f"Low Risk: {prediction_proba[0]*100:.2f}%")
st.write(f"High Risk: {prediction_proba[1]*100:.2f}%")

# Health Tips Section
st.subheader("Health Tips")
if prediction == 1:
    st.write("""
    - Schedule regular check-ups with your doctor.
    - Adopt a balanced diet rich in fruits and vegetables.
    - Avoid smoking and excessive alcohol consumption.
    - Engage in regular physical activity.
    """)
else:
    st.write("""
    - Maintain your current healthy lifestyle.
    - Continue regular exercise and a balanced diet.
    - Stay hydrated and manage stress effectively.
    """)


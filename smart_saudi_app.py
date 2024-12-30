import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Title and Description
st.title("SMART-Saudi: Chronic Disease Management")
st.write("Welcome to SMART-Saudi, an AI-powered platform for improving medication adherence and patient safety.")

# Sidebar for Input
st.sidebar.header("Patient Information")
age = st.sidebar.number_input("Age", min_value=1, max_value=100, value=30)
education_years = st.sidebar.number_input("Years of Education", min_value=0, max_value=20, value=12)
previous_adherence = st.sidebar.selectbox("Previous Adherence", [0, 1], format_func=lambda x: "Yes" if x else "No")

# AI Model (Simplified)
data = {
    'age': [25, 34, 50, 45, 60],
    'education_years': [12, 16, 10, 8, 14],
    'previous_adherence': [1, 1, 0, 0, 1],
    'risk_label': [0, 0, 1, 1, 0]
}
df = pd.DataFrame(data)
X = df[['age', 'education_years', 'previous_adherence']]
y = df['risk_label']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Prediction
patient_data = [[age, education_years, previous_adherence]]
risk_prediction = model.predict(patient_data)[0]

# Output
st.subheader("Risk Prediction")
if risk_prediction == 1:
    st.error("High Risk: Immediate intervention is recommended.")
else:
    st.success("Low Risk: Keep monitoring and maintaining adherence.")

# Visualization Placeholder
st.subheader("Adherence Insights")
st.write("Feature coming soon: Visualize trends and adherence data for better decision-making.")

# Contact Section
st.sidebar.header("Contact Us")
st.sidebar.info("For queries, email us at info@smarts.com.")

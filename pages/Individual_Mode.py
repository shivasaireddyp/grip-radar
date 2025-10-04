import streamlit as st
from tensorflow.keras.models import load_model
import pickle
import pandas as pd

# ------------------- Load Model and Encoders -------------------
@st.cache_resource
def load_resources():
    model = load_model('grip_model.h5')
    with open('label_encoder_gender.pkl', 'rb') as f:
        label_encoder_gender = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, label_encoder_gender, scaler

model, label_encoder_gender, scaler = load_resources()

st.title("Individual Prediction Mode")

gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.number_input('Age', min_value=18, max_value=100, value=30)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=600)
estimated_salary = st.number_input('Estimated Salary')
tenure = st.number_input('Tenure', min_value=0, max_value=10, value=3)
num_of_products = st.selectbox('Number Of Products', [1, 2, 3, 4])
has_cr_card = st.selectbox('Has Credit Card?', [0, 1])
is_active_member = st.selectbox('Is Active Member?', [0, 1])

if st.button("🚀 Ask Model"):
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    scaled_input = scaler.transform(input_data)
    prob = model.predict(scaled_input)[0][0]

    if prob > 0.5:
        st.error(f"🚨 The customer has a {prob*100:.2f}% chance of leaving.")
    else:
        st.success(f"The customer has a {(1-prob)*100:.2f}% chance of staying.")

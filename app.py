import streamlit as st
import numpy as np
import tensorflow as tf
import pickle as pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the model
model = tf.keras.models.load_model("model.h5")

# Load the scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
    
st.title("Customer Churn Prediction")
st.write("Enter customer details to predict churn.")

# Input fields for customer details
age = st.number_input("Age", min_value=18, max_value=100, value=30)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.selectbox("Gender", ["Male", "Female"])
tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=3)
balance = st.number_input("Balance", min_value=0.0, max_value=250000.0, value=50000.0)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member", ["Yes", "No"])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, max_value=200000.0, value=50000.0)

input_data = {
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_of_products,
    "HasCrCard": 1 if has_cr_card == "Yes" else 0,
    "IsActiveMember": 1 if is_active_member == "Yes" else 0,
    "EstimatedSalary": estimated_salary
}
# st.write(input_data)
input_df = pd.DataFrame([input_data])

# Preprocess the input data
X = scaler.transform(input_df)
print(X)
prediction = model.predict(X)
predicted_class = (prediction > 0.5).astype(int)
st.write(f"According to our model there's a {prediction[0][0]*100}% chance that this customer will churn.")

import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load the pre-trained model and preprocessor
with open('model8.pkl', 'rb') as file:
    model = pickle.load(file)

with open('preprocessor7.pkl', 'rb') as file:
    preprocessor = pickle.load(file)

# App title
st.title("Customer Churn Prediction App")

# Input form
st.sidebar.header("Customer Information")
gender = st.sidebar.selectbox("Gender", ('Male', 'Female'))
InternetService = st.sidebar.selectbox("Internet Service", ('DSL', 'Fiber optic', 'No'))
Contract = st.sidebar.selectbox("Contract", ('Month-to-month', 'One year', 'Two year'))
tenure = st.sidebar.number_input("Tenure (in months)", min_value=0, max_value=100, value=1)
MonthlyCharges = st.sidebar.number_input("Monthly Charges", min_value=0, max_value=200, value=50)
TotalCharges = st.sidebar.number_input("Total Charges", min_value=0, max_value=10000, value=0)

submitted = st.sidebar.button("Submit")

# Data preparation
custom_data_input_dict = {
    "gender": [gender],
    "InternetService": [InternetService],
    "Contract": [Contract],
    "tenure": [tenure],
    "MonthlyCharges": [MonthlyCharges],
    "TotalCharges": [TotalCharges]
}
data = pd.DataFrame(custom_data_input_dict)

if submitted:
    # Transform input data using preprocessor
    transformed_data = preprocessor.transform(data)

    # Make prediction
    prediction = model.predict(transformed_data)[0]
    churn_probability = model.predict_proba(transformed_data)[0, 1]

    # Display prediction result
    st.write('---')
    st.header('Prediction Result')

    if prediction == 0:
        st.success(f"This customer is likely to stay.")
    else:
        st.error(f"This customer is likely to churn.")

    st.write(f"**Churn Probability:** {churn_probability:.2f}")

    # Visualizations
    st.write('---')
    st.header('Insights & Visualizations')


    # Churn likelihood pie chart
    st.subheader('Churn Likelihood')
    fig, ax = plt.subplots()
    labels = ['Stay', 'Leave']
    sizes = [1 - churn_probability, churn_probability]
    ax.pie(sizes, labels=labels, colors=['green', 'red'], startangle=90,autopct='%1.1f%%')
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

# Additional info about the model
st.write('---')
st.info('This app predicts customer churn using a pre-trained machine learning model.')

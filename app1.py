import streamlit as st
import pandas as pd
import joblib

# Load the models and preprocessor
knn_model = joblib.load('KNeighborsClassifier_model.pkl')  # KNN model for predicting depression
rf_model = joblib.load('RandomForestClassifier_best_model (1).pkl')  # Random Forest model for predicting dietary habits
preprocessor = joblib.load('preprocessor (6).pkl')  # Preprocessor for feature transformation

# Load sentiment analysis model and vectorizer
sentiment_model = joblib.load('modeld')  # Load your sentiment analysis model (Scikit-learn)
vectorizer = joblib.load('optimized_tokenizer.pkl')  # Load vectorizer (e.g., CountVectorizer or TfidfVectorizer)
label_encoder = joblib.load('encodeded1')  # Load your label encoder

# Mapping for dietary habits encoding
dietary_habits_encoding = {'Healthy': 0, 'Moderate': 1, 'Normal': 2, 'Unhealthy': 3}
inverse_dietary_habits_encoding = {v: k for k, v in dietary_habits_encoding.items()}

def main():
    st.title("Dietary Habits and Depression Prediction")

    # User input fields based on dataset features
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=0, max_value=100)
    suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", ["Yes", "No"])
    academic_pressure = st.number_input("Academic Pressure (1-5)", min_value=1, max_value=5)
    cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0)
    financial_stress = st.number_input("Financial Stress (1-5)", min_value=1, max_value=5)
    degree = st.selectbox("Degree", ["Class 12", "B.Sc", "B.Tech", "M.Sc", "MBA", "Other"])
    work_study_hours = st.number_input("Work/Study Hours (per week)", min_value=0, max_value=168)
    sleep_duration = st.selectbox("Sleep Duration", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"])

    # Input field for sentiment analysis
    user_text = st.text_area("Enter text for sentiment analysis:")

    # Button to make predictions
    if st.button("Predict"):
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Age': [age],
            'Have you ever had suicidal thoughts ?': [suicidal_thoughts],
            'Academic Pressure': [academic_pressure],
            'CGPA': [cgpa],
            'Financial Stress': [financial_stress],
            'Degree': [degree],
            'Work/Study Hours': [work_study_hours],
            'Sleep Duration': [sleep_duration]
        })

        if input_data.isna().any().any():
            st.error("Please fill all fields correctly.")
        else:
            try:
                # Preprocess input data
                processed_data = preprocessor.transform(input_data)

                # Make predictions for depression and dietary habits
                depression_prediction = knn_model.predict(processed_data)
                dietary_habits_prediction = rf_model.predict(processed_data)

                dietary_habit_label = inverse_dietary_habits_encoding.get(dietary_habits_prediction[0])

                # Display results
                st.write(f"Depression Prediction: {'Depressed' if depression_prediction[0] == 1 else 'Not Depressed'}")
                st.write(f"Dietary Habits Prediction: {dietary_habit_label}")

                # Sentiment Analysis Prediction
                if user_text:
                    vectorized_text = vectorizer.transform([user_text])
                    sentiment_prediction = sentiment_model.predict(vectorized_text)
                    sentiment_label = label_encoder.inverse_transform(sentiment_prediction)[0]

                    st.write(f"Sentiment Analysis Prediction: {sentiment_label}")

            except Exception as e:
                st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()

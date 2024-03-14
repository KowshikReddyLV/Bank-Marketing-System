#################               ASAN INNOVATORS                 ###################


import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the trained XGBoost model
xg = joblib.load("xgboost_model.pkl")

# Define a function to preprocess user input
def preprocess_input(data):
    # Encode categorical variables using LabelEncoder
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
    
    # Scale numerical variables using StandardScaler
    numerical_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    
    return data

# Define the Streamlit app
def main():
    st.title("Bank Direct Marketing Prediction")
    
    # Display image after the title
    st.image("image1.png", use_column_width=True)

    # Gather user input
    age = st.slider("Age", min_value=18, max_value=95, step=1)
    job = st.selectbox("Job Category", ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"])
    marital = st.selectbox("Marital Status", ["divorced", "married", "single"])
    education = st.selectbox("Education Level", ["primary", "secondary", "tertiary", "unknown"])
    default = st.selectbox("Has Default?", ["yes", "no"])
    balance = st.number_input("Balance", value=-6847.0, min_value=-6847.0, max_value=81204.0)
    housing = st.selectbox("Has Housing Loan?", ["yes", "no"])
    loan = st.selectbox("Has Personal Loan?", ["yes", "no"])
    contact = st.selectbox("Contact Type", ["cellular", "telephone", "unknown"])
    day = st.slider("Day of Contact", min_value=1, max_value=31, step=1)
    month = st.selectbox("Month of Contact", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
    duration = st.number_input("Duration of Contact", value=2.0, min_value=2.0, max_value=3881.0)
    campaign = st.slider("Number of Contacts During Campaign", min_value=1, max_value=63, step=1)
    pdays = st.slider("Number of Days Since Last Contact", min_value=-1, max_value=854, step=1)
    previous = st.slider("Number of Contacts Before This Campaign", min_value=0, max_value=58, step=1)
    poutcome = st.selectbox("Previous Campaign Outcome", ["failure", "other", "success", "unknown"])

    # Create a DataFrame from the user input
    user_data = pd.DataFrame({
        'age': [age],
        'job': [job],
        'marital': [marital],
        'education': [education],
        'default': [default],
        'balance': [balance],
        'housing': [housing],
        'loan': [loan],
        'contact': [contact],
        'day': [day],
        'month': [month],
        'duration': [duration],
        'campaign': [campaign],
        'pdays': [pdays],
        'previous': [previous],
        'poutcome': [poutcome]
    })

    # Define a flag to check if prediction has been made
    prediction_made = False

    # Check if the "Predict" button is clicked
    if st.button("Predict"):
        prediction_made = True
        # Preprocess the user input
        preprocessed_user_data = preprocess_input(user_data)

        # Make prediction using the XGBoost model
        prediction = xg.predict(preprocessed_user_data)

        # Display the prediction
        st.subheader("Prediction:")
        if prediction[0] == 1:
            st.success("Customer is likely to subscribe to a term deposit.")
        else:
            st.write("Customer is unlikely to subscribe to a term deposit.")

    # Display a message if prediction has not been made yet
    if not prediction_made:
        st.write("Click the 'Predict' button to see the prediction.")

if __name__ == "__main__":
    main()

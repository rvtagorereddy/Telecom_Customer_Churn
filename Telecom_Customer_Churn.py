import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
@st.cache
def load_data():
    df = pd.read_csv('Telecom_Customer_Churn.csv')
    return df

# Split the dataset into features and target variable
def split_data(df):
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    return X, y

# Train the model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model1 = RandomForestClassifier()
    model2 = RandomForestClassifier()
    voting_model = VotingClassifier(estimators=[('model1', model1), ('model2', model2)], voting='hard')
    voting_model.fit(X_train, y_train)
    y_pred = voting_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return voting_model, accuracy

# Streamlit app
def main():
    # Load the dataset
    df = load_data()

    # Split the dataset
    X, y = split_data(df)

    # Train the model
    model, accuracy = train_model(X, y)

    # Create the app
    st.title('Telecom Churn Customer Prediction')
    st.write('Accuracy:', accuracy)

    # User inputs
    st.sidebar.header('User Inputs')
    customer_id = st.sidebar.text_input('Customer ID')
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    # Include other relevant features as per your dataset
    
    # Predict churn
    if st.sidebar.button('Predict Churn'):
        input_data = pd.DataFrame({
            'CustomerID': [customer_id],
            'Gender': [gender]
            # Include other relevant features as per your dataset
        })
        prediction = model.predict(input_data)
        st.sidebar.write('Churn Prediction:', prediction)

if __name__ == '__main__':
    main()

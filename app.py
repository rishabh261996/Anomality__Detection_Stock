import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler

# Title and description
st.title("Stock Returns Anomaly Detection App")
st.write("Enter a single 'Returns' value, and the app will identify if it is an anomaly using Isolation Forest.")

# Input for single "Returns" value
returns_value = st.number_input("Enter Returns value:", value=0.0)

# Load the model
my_model = pickle.load(open('model.pkl', 'rb'))

# Load the dataset
try:
    train_data = pd.read_excel('Google Dataset.xlsx')
    # Check if the 'Close' column is in the dataset
    if 'Close' not in train_data.columns:
        st.error("The dataset does not contain a 'Close' column.")
    else:
        # Calculate the daily returns
        train_data['Returns'] = train_data['Close'].pct_change()
        train_data['Returns'] = train_data['Returns'].fillna(train_data['Returns'].mean())

        # Scale the 'Returns' column
        scaler = StandardScaler()
        # Scale the 'Returns' column in train_data
        train_data['Returns'] = scaler.fit_transform(train_data[['Returns']].values)

        
        # Predict if the entered Returns value is an anomaly
        data_point_train_data = pd.DataFrame({'Returns': [returns_value]})
        data_point_train_data['Returns'] = scaler.transform(data_point_train_data[['Returns']])
        prediction = my_model.predict(data_point_train_data)[0]
        anomaly_status = "Anomaly" if prediction == -1 else "Normal"

        # Display the result
        st.write("Anomaly Detection Result:")
        st.write(f"The entered 'Returns' value is classified as: **{anomaly_status}**")

        # Visualization
        st.write("Anomaly Visualization (based on historical data)")

        # Predict anomalies in the training data
        train_data['anomaly'] = my_model.predict(train_data[['Returns']])
        train_data['anomaly'] = train_data['anomaly'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')

        # Plotting the historical returns data and the input point
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=train_data, x=train_data.index, y='Returns', hue='anomaly', palette=['green', 'red'])
        plt.scatter(len(train_data), returns_value, color='blue', label='Entered Point', s=100)
        plt.legend()
        plt.title("Anomaly Detection Plot for Returns")
        st.pyplot(plt)

except FileNotFoundError:
    st.error("The file 'Google Dataset.xlsx' was not found. Please make sure it is in the correct directory.")
except Exception as e:
    st.error(f"An error occurred: {e}")
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load Pre-trained Models and Data
churn_model = joblib.load("churn_model.pkl")  # Churn Prediction Model
demand_model = joblib.load("demand_model.pkl")  # Demand Forecasting Model
data = pd.read_csv("final_features_dataset.csv")  # Dataset with customer/book info

# Set Page Configuration
st.set_page_config(
    page_title="Interactive Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title and Description
st.title("📊 Interactive Dashboard for Bookstore Analytics")
st.write("This dashboard provides **book recommendations**, **customer churn predictions**, and **demand forecasts** for strategic insights.")

# Sidebar for Navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a Feature:", ["Recommendations", "Churn Prediction", "Demand Forecasting"])

# Function for Recommendations
def display_recommendations(customer_id):
    customer_data = data[data["customer_id"] == customer_id]
    if customer_data.empty:
        st.warning("Customer ID not found!")
        return
    
    st.subheader(f"📚 Recommendations for Customer ID: {customer_id}")
    # Dummy Recommendations (Replace with a real recommendation model if available)
    recommended_books = data.sample(5)["title"].tolist()
    for i, book in enumerate(recommended_books, 1):
        st.write(f"**{i}. {book}**")

# Function for Churn Prediction
def predict_churn(customer_id):
    customer_data = data[data["customer_id"] == customer_id]
    if customer_data.empty:
        st.warning("Customer ID not found!")
        return

    st.subheader(f"🔍 Churn Prediction for Customer ID: {customer_id}")
    features = customer_data[["total_orders", "title_length", "order_year", "order_month", "is_weekend"]]
    churn_prob = churn_model.predict_proba(features)[0][1]  # Probability of churn
    is_churn = churn_model.predict(features)[0]  # Predicted class (0 or 1)

    st.write(f"**Churn Probability:** {churn_prob:.2%}")
    if is_churn:
        st.error("The customer is likely to churn. Consider engagement strategies.")
    else:
        st.success("The customer is not likely to churn.")

# Function for Demand Forecasting
def forecast_demand():
    st.subheader("📈 Demand Forecasting")
    # Dummy Input Features (Replace with a real input form if needed)
    year = st.slider("Select Year:", 2023, 2025, 2024)
    month = st.slider("Select Month:", 1, 12, datetime.now().month)

    # Predict Demand
    demand_features = np.array([[year, month, data["total_orders"].mean()]])
    predicted_demand = demand_model.predict(demand_features)[0]

    st.write(f"**Predicted Demand for {year}-{month:02d}:** {int(predicted_demand)} orders")

# Main Dashboard Logic
if options == "Recommendations":
    st.sidebar.subheader("Customer Recommendations")
    customer_id = st.sidebar.number_input("Enter Customer ID:", min_value=1, max_value=int(data["customer_id"].max()), step=1)
    if st.sidebar.button("Get Recommendations"):
        display_recommendations(customer_id)

elif options == "Churn Prediction":
    st.sidebar.subheader("Churn Prediction")
    customer_id = st.sidebar.number_input("Enter Customer ID:", min_value=1, max_value=int(data["customer_id"].max()), step=1)
    if st.sidebar.button("Predict Churn"):
        predict_churn(customer_id)

elif options == "Demand Forecasting":
    forecast_demand()


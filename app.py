
import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd

# Load models and scaler
@st.cache_resource
def load_models():
    custom_objects = {'mse': tf.keras.losses.MeanSquaredError()}
    
    churn_model = tf.keras.models.load_model("churn_model_multiclass.h5", custom_objects={'LeakyReLU': tf.keras.layers.LeakyReLU})
    demand_model = tf.keras.models.load_model("lstm_demand_forecast.h5", custom_objects=custom_objects)
    scaler = joblib.load("churn_scaler.pkl")
    
    return churn_model, demand_model, scaler

churn_model, demand_model, scaler = load_models()
df_books = pd.read_csv("preprocessed_book_data.csv")

# Sidebar Navigation
st.sidebar.image("book.png", width=180) 
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📉 Churn Prediction", "📈 Demand Forecasting", "📚 Book Recommendations"])

# Home Page
if page == "🏠 Home":
    st.title("📖 Bookstore Analytics Dashboard")
    st.markdown(
        "Welcome to the **Bookstore Analytics Dashboard**! 📚\n"
        "Use the **sidebar** to navigate through **Churn Prediction**, **Demand Forecasting**, and **Book Recommendations**."
    )
    st.image("books_banner.jpeg", use_column_width=True)

# Churn Prediction Page
elif page == "📉 Churn Prediction":
    st.title("🔍 Customer Churn Prediction")
    
    with st.form("churn_form"):
        st.subheader("Enter Customer Details")
        total_orders = st.number_input("📦 Total Orders", min_value=0, step=1)
        total_spent = st.number_input("💰 Total Amount Spent ($)", min_value=0.0, step=1.0)
        avg_order_value = st.number_input("📊 Average Order Value ($)", min_value=0.0, step=1.0)
        days_since_last_order = st.number_input("📅 Days Since Last Order", min_value=0, step=1)

        submitted = st.form_submit_button("🚀 Predict Churn")

    if submitted:
        input_data = np.array([[total_orders, total_spent, avg_order_value, days_since_last_order]])
        input_data_scaled = scaler.transform(input_data)
        prediction_prob = churn_model.predict(input_data_scaled)[0][0]
        churn_prediction = "⚠️ Likely to Churn" if prediction_prob > 0.5 else "✅ Not Likely to Churn"

        st.subheader("🔹 Prediction Result")
        st.markdown(f"**Churn Probability:** `{prediction_prob:.2f}`")
        st.success(churn_prediction if prediction_prob <= 0.5 else st.warning(churn_prediction))

# Demand Forecasting Page
elif page == "📈 Demand Forecasting":
    st.title("📊 Future Demand Forecasting")

    months_to_forecast = st.slider("📅 Select Months to Predict:", 1, 6, 3)
    forecast_button = st.button("🚀 Forecast Demand")

    if forecast_button:
        expected_features = demand_model.input_shape[2]
        last_known_data = np.random.rand(1, 3, expected_features)
        future_predictions = []

        for _ in range(months_to_forecast):
            pred = demand_model.predict(last_known_data)[0][0]
            future_predictions.append(pred)

            new_input = np.zeros((1, 3, expected_features))
            new_input[0, :-1, :] = last_known_data[0, 1:, :]
            new_input[0, -1, 0] = pred
            last_known_data = new_input

        st.subheader("📈 Forecasted Demand")
        for i, demand in enumerate(future_predictions, 1):
            st.markdown(f"**📅 Month {i}:** `{demand:.2f} units`")
            # st.progress(min(demand / 100, 1.0))  # Visual progress bar

# Book Recommendations Page
elif page == "📚 Book Recommendations":
    st.title("📖 Personalized Book Recommendations")
    recommendation_type = st.radio("📌 Select Recommendation Type:", ["🌟 Popular Books", "📝 Author-Based"])

    if recommendation_type == "🌟 Popular Books":
        popular_books = df_books.sort_values(by="order_count", ascending=False).head(5)
        st.subheader("🔥 Top 5 Popular Books")
        for idx, row in popular_books.iterrows():
            st.markdown(f"📚 **{row['title']}** by *{row['author']}* - 🛒 `{row['order_count']} orders`")

    elif recommendation_type == "📝 Author-Based":
        author = st.text_input("✍️ Enter Author Name:")
        if author:
            author_books = df_books[df_books["author"].str.contains(author, case=False, na=False)].head(5)
            if not author_books.empty:
                st.subheader(f"📖 Books by **{author}**")
                for idx, row in author_books.iterrows():
                    st.markdown(f"📖 **{row['title']}**")
            else:
                st.warning("🚫 No books found for this author.")
**Final Documentation & Report**

# **Publishing Industry Analytics & Prediction System**

## **1️⃣ Introduction**

### **Project Overview**

The Publishing Industry Analytics & Prediction System aims to provide insights into customer behavior, book demand, and personalized recommendations using machine learning models. This project integrates multiple data sources, performs analysis, and builds predictive models for customer churn and book demand forecasting.

### **Objectives & Goals**

- Predict customer churn to improve retention strategies.
- Forecast book demand to optimize inventory management.
- Provide personalized book recommendations based on order history.
- Develop an interactive **Streamlit** dashboard for visualization.
- Deploy models on **AWS EC2** for real-time predictions.

### **Importance of the Project**

Understanding customer behavior and demand trends helps publishers make data-driven decisions, leading to better sales, reduced costs, and improved customer satisfaction.

## **2️⃣ Data Collection & Preprocessing**

### **Datasets Used**

- **Book Data** (book title, author, ISBN, publisher, etc.)
- **Customer Data** (customer demographics and purchase history)
- **Order Data** (order details, order count, revenue, etc.)

### **SQL Queries for Data Merging**

To create a **preprocessed dataset**, SQL queries were used to join relevant tables:

```sql
SELECT
    b.book_id,
    b.title,
    a.author_name AS author,
    COUNT(ol.line_id) AS order_count
FROM book b
LEFT JOIN book_author ba ON b.book_id = ba.book_id
LEFT JOIN author a ON ba.author_id = a.author_id
LEFT JOIN order_line ol ON b.book_id = ol.book_id
GROUP BY b.book_id, b.title, a.author_name
ORDER BY order_count DESC;
```

### **Data Cleaning & Transformation**

- **Handling Missing Values**: Replaced NaNs with suitable values.
- **Feature Engineering**: Created **lag features & moving averages** for forecasting.
- **Normalization**: Applied **MinMaxScaler** for model training.

## **3️⃣ Exploratory Data Analysis (EDA)**

### **Key Insights & Visualizations**

- **Top-selling books** based on order count.
- **Customer segmentation** based on spending patterns.
- **Seasonal demand trends** for books.
- **Churn rate analysis** based on order frequency.

**Sample Visualization:** (Include screenshots of charts and graphs.)

## **4️⃣ Model Development**

### **Customer Churn Prediction (ANN Model)**

- **Features Used**: Total Orders, Total Spent, Average Order Value, Days Since Last Order.
- **Model Architecture**:
  - Input Layer → Dense Layers (with LeakyReLU activation) → Output Layer.
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score.

### **Demand Forecasting (LSTM Model)**

- **Features Used**: Order count (with lag variables and moving averages).
- **Model Architecture**:
  - LSTM Layers → Dense Layers → Output Layer.
- **Evaluation Metrics**: MAE, RMSE.

## **5️⃣ Model Deployment**

### **Steps to Deploy on AWS EC2**

1. **Create EC2 Instance** on AWS.
2. **Transfer trained models** using SCP command:
   ```sh
   scp -i ~/.ssh/bull.pem churn_model_multiclass.h5 ubuntu@<AWS_IP>:/home/ubuntu/
   scp -i ~/.ssh/bull.pem lstm_demand_forecast.h5 ubuntu@<AWS_IP>:/home/ubuntu/
   ```
3. **Install dependencies** on EC2:
   ```sh
   sudo apt update && sudo apt install python3-pip
   pip3 install tensorflow pandas streamlit joblib
   ```
4. **Run Streamlit app on EC2**:
   ```sh
   streamlit run app.py --server.port 8501
   ```

## **6️⃣ Streamlit Application**

### **Features of the Dashboard**

- **Customer Churn Prediction**: Users input customer details to predict churn probability.
- **Demand Forecasting**: Predicts book demand for upcoming months.
- **Personalized Book Recommendations**:
  - **Popular Books** (based on highest orders).
  - **Author-Based Recommendations** (user inputs author name).

## **7️⃣ Challenges & Learnings**

### **Challenges Faced**

- Model input shape mismatches during inference.
- Deployment setup on AWS with permission errors.
- UI enhancements in Streamlit for better interaction.

### **Solutions Implemented**

- Debugged input transformations to match model expectations.
- Ensured proper file transfer using SCP for deployment.
- Improved UI using **Streamlit components & layout optimizations**.

## **8️⃣ Conclusion & Future Scope**

### **Summary of Key Findings**

- Machine learning models provide valuable insights for the publishing industry.
- Churn prediction can help **increase customer retention**.
- Demand forecasting optimizes **inventory management**.

### **Future Improvements**

- **Enhance Book Recommendations** using collaborative filtering.
- **Incorporate NLP** to analyze customer reviews for better insights.
- **Deploy on cloud-based ML platforms** for scalability.

---

📌 **Final Steps**:

- 📂 Ensure all files (datasets, models, scripts) are uploaded to **GitHub**.
- 📜 Review Streamlit UI for further enhancements.
- 🚀 Final Deployment Check on AWS.

📌 **Would you like any additional formatting or sections added?**


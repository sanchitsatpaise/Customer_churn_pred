# 📉 Customer Churn Prediction Model

This project is a **Customer Churn Prediction System** designed to identify customers who are likely to discontinue a service. By analyzing **demographic details, account information, and service usage patterns**, the model enables businesses to take proactive retention actions, improve customer satisfaction, and maximize long-term revenue through data-driven decisions.

**APP LINK:https://custchurnpred.streamlit.app/**
---

## 📌 1. Project Objective

The key objectives of this project are:

* Predict whether a customer is likely to churn
* Identify important factors contributing to churn
* Help businesses design proactive retention strategies
* Build a scalable and deployable machine learning solution

---

## 📌 2. Business Problem

Customer churn directly impacts revenue and growth. Acquiring new customers is often more expensive than retaining existing ones. By predicting churn in advance, businesses can:

* Offer personalized discounts
* Improve customer support
* Enhance service quality
* Increase customer lifetime value (CLV)

---

## 📌 3. Dataset Description

The dataset includes multiple customer-related features such as:

### 🔹 Demographic Data

* Gender
* Senior citizen status
* Partner and dependents

### 🔹 Account Information

* Tenure
* Contract type
* Payment method
* Monthly and total charges

### 🔹 Service Usage Data

* Internet service type
* Phone service
* Add-on services (streaming, security, backup, etc.)

The target variable is:

* **Churn** (Yes / No)

---

## 📌 4. Data Preprocessing (Step-by-Step)

### 🔹 Step 1: Data Cleaning

* Handled missing values
* Converted incorrect data types
* Removed duplicate records

---

### 🔹 Step 2: Encoding Categorical Features

* Categorical variables encoded using **OneHotEncoding / Label Encoding**
* Ensured all features are numerical for model compatibility

---

### 🔹 Step 3: Feature Scaling

* Applied **StandardScaler** to numerical features
* Ensured equal contribution of all variables to the model

---

## 📌 5. Exploratory Data Analysis (EDA)

* Analyzed churn distribution
* Studied churn behavior across tenure, contract type, and services
* Identified high-risk customer segments

EDA helped uncover meaningful patterns influencing churn.

---

## 📌 6. Model Selection

Several machine learning models can be used, such as:

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting

The final model is selected based on:

* Accuracy
* Precision & Recall
* ROC-AUC Score

---

## 📌 7. Model Training

* Split data into training and testing sets
* Trained the selected model on historical customer data
* Tuned hyperparameters for optimal performance

---

## 📌 8. Prediction Logic

### 🔹 How prediction works:

1. Customer data is provided as input
2. Data is preprocessed using the trained pipeline
3. Model predicts churn probability
4. Output is classified as **Churn / No Churn**

Example:

* Churn Probability: **72%**
* Prediction: **High Risk Customer**

---

## 📌 9. Model Evaluation

* Confusion Matrix
* Precision, Recall, F1-score
* ROC-AUC Curve

Special emphasis is placed on **Recall**, as missing a churned customer is costlier than false positives.

---

## 📌 10. Deployment

The model can be deployed using **Streamlit**, enabling:

* User input for customer details
* Instant churn prediction
* Probability-based risk assessment
* Business-friendly UI

---

## 📌 11. Project Workflow Summary

1. Load customer dataset
2. Clean and preprocess data
3. Perform EDA
4. Train churn prediction model
5. Evaluate performance
6. Deploy as a web application

---

## 📌 12. Key Features

✅ Predicts customer churn probability
✅ Helps design proactive retention strategies
✅ Scalable ML pipeline
✅ Business-oriented insights
✅ Easy deployment

---

## 📌 13. Technologies Used

* **Python**
* **Pandas & NumPy**
* **Scikit-learn**
* **Matplotlib & Seaborn**
* **Streamlit**

---

## 📌 14. Future Enhancements

* Use advanced ensemble models (XGBoost, LightGBM)
* Add explainability using SHAP or LIME
* Integrate real-time CRM data
* Automate retention recommendations

---

## 📌 15. Conclusion

This Customer Churn Prediction project demonstrates how **machine learning can drive strategic business decisions**. By identifying customers at risk of leaving, organizations can act early to retain them, improve satisfaction, and boost long-term revenue.

---

⭐ *If you find this project useful, don’t forget to give it a star on GitHub!*

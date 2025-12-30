import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# --- Page Configuration (must be the first Streamlit command) ---
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="👋", # You can use an emoji or a path to an image file
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Decorative Header ---
#st.image('path/to/your/header_image.png', use_column_width=True, caption='Predict Customer Churn') # Replace with your image path
st.title('Customer Churn Prediction App 📊')
st.markdown("--- ✨ Welcome to the Customer Churn Prediction Tool! ✨ ---")
st.write('Fill in the customer details below to predict their likelihood of churning.')

# Load the trained model and encoders
try:
    with open('customer_churn_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('encoders.pkl', 'rb') as encoders_file:
        encoders = pickle.load(encoders_file)
    with open('x_train_smote_columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)
except FileNotFoundError:
    st.error("⚠️ Error: Model, encoder, or feature column files not found. Please ensure 'customer_churn_model.pkl', 'encoders.pkl', and 'x_train_smote_columns.pkl' are in the same directory.")
    st.stop()

# Define the input fields
input_data = {}

# --- Sidebar for numerical inputs ---
st.sidebar.header('Customer Numerical Data 🔢')
input_data['tenure'] = st.sidebar.slider('Tenure (months)', min_value=0, max_value=72, value=12)
input_data['MonthlyCharges'] = st.sidebar.number_input('Monthly Charges ($)', min_value=0.0, max_value=120.0, value=50.0, step=0.01)
input_data['TotalCharges'] = st.sidebar.number_input('Total Charges ($)', min_value=0.0, max_value=9000.0, value=1000.0, step=0.01)

# --- Main area for categorical inputs ---
st.header('Customer Demographic & Service Details 👥')

# Categorical features with their original unique values
categorical_features = {
    'gender': ['Female', 'Male'],
    'SeniorCitizen': [0, 1], # 0 for No, 1 for Yes
    'Partner': ['No', 'Yes'],
    'Dependents': ['No', 'Yes'],
    'PhoneService': ['No', 'Yes'],
    'MultipleLines': ['No phone service', 'No', 'Yes'],
    'InternetService': ['DSL', 'Fiber optic', 'No'],
    'OnlineSecurity': ['No internet service', 'No', 'Yes'],
    'OnlineBackup': ['No internet service', 'No', 'Yes'],
    'DeviceProtection': ['No internet service', 'No', 'Yes'],
    'TechSupport': ['No internet service', 'No', 'Yes'],
    'StreamingTV': ['No internet service', 'No', 'Yes'],
    'StreamingMovies': ['No internet service', 'No', 'Yes'],
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'PaperlessBilling': ['No', 'Yes'],
    'PaymentMethod': ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check']
}

# Arrange categorical inputs in columns for better layout
cols = st.columns(3) # Create 3 columns
col_idx = 0

for feature, options in categorical_features.items():
    with cols[col_idx % 3]: # Cycle through columns
        if feature == 'SeniorCitizen':
            display_options = {0: 'No', 1: 'Yes'}
            selected_option = st.selectbox(f'Is customer a {feature}?', options=list(display_options.keys()), format_func=lambda x: display_options[x], key=feature)
            input_data[feature] = selected_option
        else:
            selected_option = st.selectbox(f'{feature}', options=options, key=feature)
            input_data[feature] = encoders[feature].transform([selected_option])[0]
    col_idx += 1

st.markdown("--- ")

if st.button('🚀 Predict Churn'):
    # Create a DataFrame from the input data
    input_df = pd.DataFrame([input_data])

    # Ensure column order matches the training data
    input_df = input_df[feature_columns] # Reorder columns here
    
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    churn_status = 'Yes' if prediction[0] == 1 else 'No'
    churn_probability = prediction_proba[0][1] * 100

    st.subheader('Prediction Result:')
    if churn_status == 'Yes':
        st.error(f'🚨 Customer will **CHURN** with a probability of **{churn_probability:.2f}%**! 🚨')
        st.warning('Consider implementing retention strategies immediately.')
    else:
        st.success(f'✅ Customer will likely **NOT CHURN** (Probability: **{churn_probability:.2f}%**) ✅')
        st.info('Good news! This customer is likely to remain with your service.')

st.markdown("""
---
Created with ❤️ using Streamlit and Scikit-learn.
""")

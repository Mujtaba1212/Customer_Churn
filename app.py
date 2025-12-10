import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# -------------------------------------
# Load Random Forest model only
# -------------------------------------
rf_model = pickle.load(open("random_forest.pkl", "rb"))

# Feature Columns (exact order used during model training)
feature_columns = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
    'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
    'Balance_to_Salary',
    'Geography_Germany', 'Geography_Spain',
    'Gender_Male',
    'AgeBucket_31–45', 'AgeBucket_46–60', 'AgeBucket_60+'
]

# -------------------------------------
# Streamlit Page Config
# -------------------------------------
st.set_page_config(
    page_title="Churn Prediction App",
    layout="centered",
    page_icon="📊"
)

st.markdown("""
    <h1 style='text-align:center; color:#4CAF50;'>📊 Customer Churn Prediction</h1>
    <p style='text-align:center;'>Predict whether a customer will Stay or Leave using Machine Learning</p>
""", unsafe_allow_html=True)


# -------------------------------------
# INPUT FORM
# -------------------------------------
with st.container():
    st.markdown("### 📝 Enter Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        CreditScore = st.number_input("Credit Score", min_value=300, max_value=900)
        Age = st.number_input("Age", min_value=18, max_value=100)
        Tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10)
        NumOfProducts = st.selectbox("Number of Products", [1, 2, 3, 4])
        HasCrCard = st.selectbox("Has Credit Card?", [0, 1])

    with col2:
        Balance = st.number_input("Balance", min_value=0.0)
        IsActiveMember = st.selectbox("Active Member?", [0, 1])
        EstimatedSalary = st.number_input("Estimated Salary", min_value=0.0)
        Geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        Gender = st.selectbox("Gender", ["Female", "Male"])

    AgeBucket = st.selectbox("Age Bucket", ["18–30", "31–45", "46–60", "60+"])


# -------------------------------------
# PROCESS INPUT FUNCTION
# -------------------------------------
def create_input_dataframe():
    df = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)

    df['CreditScore'] = CreditScore
    df['Age'] = Age
    df['Tenure'] = Tenure
    df['Balance'] = Balance
    df['NumOfProducts'] = NumOfProducts
    df['HasCrCard'] = HasCrCard
    df['IsActiveMember'] = IsActiveMember
    df['EstimatedSalary'] = EstimatedSalary

    df["Balance_to_Salary"] = Balance / (EstimatedSalary + 1e-6)

    if Geography == "Germany":
        df["Geography_Germany"] = 1
    if Geography == "Spain":
        df["Geography_Spain"] = 1

    if Gender == "Male":
        df["Gender_Male"] = 1

    if AgeBucket == "31–45":
        df["AgeBucket_31–45"] = 1
    if AgeBucket == "46–60":
        df["AgeBucket_46–60"] = 1
    if AgeBucket == "60+":
        df["AgeBucket_60+"] = 1

    return df


# -------------------------------------
# AUTO SCALING inside UI (Without scaler.pkl)
# -------------------------------------
def scale_input(df):
    numeric_cols = [
        'CreditScore', 'Age', 'Tenure', 'Balance', 
        'NumOfProducts', 'EstimatedSalary', 'Balance_to_Salary'
    ]

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df


# -------------------------------------
# Prediction Button
# -------------------------------------
if st.button("🔍 Predict"):
    final_df = create_input_dataframe()
    final_df = scale_input(final_df)   # <-- scaling inside UI

    prediction = rf_model.predict(final_df)[0]
    prob = rf_model.predict_proba(final_df)[0][1]

    st.markdown("---")

    if prediction == 1:
        st.markdown(f"""
            <div style='padding:20px; background:#FFCDD2; border-radius:10px;'>
                <h2 style='color:#C62828;'>❌ Customer is likely to LEAVE</h2>
                <h3>Probability: <b>{prob:.2f}</b></h3>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style='padding:20px; background:#C8E6C9; border-radius:10px;'>
                <h2 style='color:#2E7D32;'>✅ Customer will STAY</h2>
                <h3>Probability: <b>{prob:.2f}</b></h3>
            </div>
        """, unsafe_allow_html=True)

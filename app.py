import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model and scaler
model = joblib.load('autism_best_model.pkl')
scaler = joblib.load('scaler.pkl')

# App Title
st.title("Early Prediction of Autism Using ML Models 🚀")
st.markdown("Please fill the following information:")

def user_input_features():
    A1_Score = st.selectbox('A1 Score (0 = No, 1 = Yes)', (0, 1))
    A2_Score = st.selectbox('A2 Score (0 = No, 1 = Yes)', (0, 1))
    A3_Score = st.selectbox('A3 Score (0 = No, 1 = Yes)', (0, 1))
    A4_Score = st.selectbox('A4 Score (0 = No, 1 = Yes)', (0, 1))
    A5_Score = st.selectbox('A5 Score (0 = No, 1 = Yes)', (0, 1))
    A6_Score = st.selectbox('A6 Score (0 = No, 1 = Yes)', (0, 1))
    A7_Score = st.selectbox('A7 Score (0 = No, 1 = Yes)', (0, 1))
    A8_Score = st.selectbox('A8 Score (0 = No, 1 = Yes)', (0, 1))
    A9_Score = st.selectbox('A9 Score (0 = No, 1 = Yes)', (0, 1))
    A10_Score = st.selectbox('A10 Score (0 = No, 1 = Yes)', (0, 1))
    age = st.slider('Age', 2, 60, 25)
    gender = st.selectbox('Gender (0 = Female, 1 = Male)', (0, 1))
    jaundice = st.selectbox('History of Jaundice (0 = No, 1 = Yes)', (0, 1))
    family_mem_with_ASD = st.selectbox('Family Member with ASD? (0 = No, 1 = Yes)', (0, 1))

    data = {
        'A1_Score': A1_Score,
        'A2_Score': A2_Score,
        'A3_Score': A3_Score,
        'A4_Score': A4_Score,
        'A5_Score': A5_Score,
        'A6_Score': A6_Score,
        'A7_Score': A7_Score,
        'A8_Score': A8_Score,
        'A9_Score': A9_Score,
        'A10_Score': A10_Score,
        'age': age,
        'gender': gender,
        'jaundice': jaundice,
        'family_mem_with_ASD': family_mem_with_ASD
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

if st.button('Predict Autism Risk'):
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)

    confidence = np.max(prediction_proba) * 100

    if prediction[0] == 1:
        if confidence >= 85:
            st.error(f'**High Risk of Autism** 🚨 (Confidence: {confidence:.2f}%)')
        else:
            st.warning(f'**Moderate Risk of Autism** ⚠️ (Confidence: {confidence:.2f}%)')
    else:
        if confidence >= 85:
            st.success(f'**Low Risk of Autism** ✅ (Confidence: {confidence:.2f}%)')
        else:
            st.info(f'**Very Low Risk of Autism** 🛡️ (Confidence: {confidence:.2f}%)')

st.markdown("---")
st.subheader("Feature Importance Analysis")

if st.button('Show Feature Importances'):
    feature_importances = np.abs(model.coef_[0])
    feature_names = input_df.columns

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    st.bar_chart(importance_df.set_index('Feature'))

st.markdown("""
---
📝 **Disclaimer:** This is a preliminary screening tool based on questionnaire responses and should not be considered a final diagnosis.  
For professional evaluation, consult a healthcare provider.
""")

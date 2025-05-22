import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model and scaler
model = joblib.load('autism_best_model.pkl')
scaler = joblib.load('scaler.pkl')

# --- WELCOME PAGE ---
st.set_page_config(page_title="Pragati: Autism Risk Screening", page_icon="🧠", layout="centered")

st.title("🧠 Pragati: Early Autism Risk Screening System")
st.markdown("""
Welcome to **Pragati** — a machine learning-powered tool designed for preliminary autism risk screening based on behavioral and demographic information.

🛡️ *Disclaimer:* This tool provides early screening and is **not** a substitute for professional diagnosis.  
Please consult a healthcare professional for any concerns.
""")
st.markdown("---")

# --- INPUT FORM ---
with st.form(key='input_form'):
    st.subheader("📋 Please fill out the screening form:")

    col1, col2 = st.columns(2)
    
    with col1:
        A1_Score = st.selectbox('I often notice small sounds when others do not.', (0, 1))
        A2_Score = st.selectbox('I usually concentrate more on the whole picture rather than details.', (0, 1))
        A3_Score = st.selectbox('I find it easy to do more than one thing at once.', (0, 1))
        A4_Score = st.selectbox('If there is an interruption, I can switch back very quickly.', (0, 1))
        A5_Score = st.selectbox('I find it easy to "read between the lines" when someone is talking to me.', (0, 1))
    
    with col2:
        A6_Score = st.selectbox('I know how to tell if someone listening to me is getting bored.', (0, 1))
        A7_Score = st.selectbox('When reading a story, I find it difficult to work out characters\' intentions.', (0, 1))
        A8_Score = st.selectbox('I like to collect information about categories of things.', (0, 1))
        A9_Score = st.selectbox('I find it easy to work out what someone is thinking or feeling just by looking at their face.', (0, 1))
        A10_Score = st.selectbox('I find it difficult to work out people\'s intentions.', (0, 1))

    st.markdown("### 🧬 Demographic Information")
    age = st.slider('Age', 2, 60, 25)
    gender = st.selectbox('Gender', (0, 1))  # 0 = Female, 1 = Male
    ethnicity = st.selectbox('Ethnicity', ['White-European', 'Latino', 'Others', 'Black', 'Asian', 'Middle Eastern', 'South Asian'])
    jaundice = st.selectbox('Have you had jaundice?', (0, 1))
    autism = st.selectbox('Family member with ASD?', (0, 1))
    country_of_res = st.text_input('Country of Residence', 'United States')
    relation = st.selectbox('Relation to individual', ['Self', 'Parent', 'Sibling', 'Relative', 'Others'])
    used_app_before = st.selectbox('Used this app before?', (0, 1))

    submit_button = st.form_submit_button(label='Predict Autism Risk')

# --- PREDICTION HANDLER ---
if submit_button:
    with st.spinner('🔍 Predicting autism risk... Please wait...'):
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
            'ethnicity': ethnicity,
            'jaundice': jaundice,
            'autism': autism,
            'country_of_res': country_of_res,
            'relation': relation,
            'used_app_before': used_app_before
        }

        input_df = pd.DataFrame(data, index=[0])

        feature_order = ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
                         'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
                         'age', 'gender', 'ethnicity', 'jaundice', 'autism',
                         'country_of_res', 'used_app_before', 'relation']

        input_df = input_df[feature_order]

        # Encoding
        encoding_maps = {
            'ethnicity': {
                'White-European': 0,
                'Latino': 1,
                'Others': 2,
                'Black': 3,
                'Asian': 4,
                'Middle Eastern': 5,
                'South Asian': 6
            },
            'country_of_res': {
                'United States': 0,
                'Others': 1
            },
            'relation': {
                'Self': 0,
                'Parent': 1,
                'Sibling': 2,
                'Relative': 3,
                'Others': 4
            }
        }
        input_df['ethnicity'] = input_df['ethnicity'].map(encoding_maps['ethnicity'])
        input_df['country_of_res'] = input_df['country_of_res'].map(encoding_maps['country_of_res'])
        input_df['relation'] = input_df['relation'].map(encoding_maps['relation'])

        input_df = input_df.fillna(1)
        
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)
        prediction_proba = model.predict_proba(scaled_input)

        confidence = np.max(prediction_proba) * 100

        st.markdown("---")
        if prediction[0] == 1:
            if confidence <= 85:
                st.error(f'🚨 **High Risk of Autism** (Confidence: {confidence:.2f}%)')
            else:
                st.warning(f'⚠️ **Moderate Risk of Autism** (Confidence: {confidence:.2f}%)')
        else:
            if confidence >= 85:
                st.success(f'✅ **Low Risk of Autism** (Confidence: {confidence:.2f}%)')
            else:
                st.info(f'🛡️ **Very Low Risk of Autism** (Confidence: {confidence:.2f}%)')

        # --- FEATURE IMPORTANCE ---
        st.markdown("### 🔍 Feature Importance Analysis")
        feature_importances = np.abs(model.coef_[0])
        feature_names = input_df.columns

        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        st.bar_chart(importance_df.set_index('Feature'))

# --- FOOTER ---
st.markdown("""
---
<center>
    Powered by **Pragati** | *Screening Purpose Only*
</center>
""", unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model and scaler
model = joblib.load('autism_best_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Early Prediction of Autism Using ML Models")
st.markdown("### Please fill the following questionnaire carefully: 0 = No, 1 = Yes")

def user_input_features():
    A1_Score = st.selectbox('I often notice small sounds when others do not.', (0, 1))
    A2_Score = st.selectbox('I usually concentrate more on the whole picture rather than details.', (0, 1))
    A3_Score = st.selectbox('I find it easy to do more than one thing at once.', (0, 1))
    A4_Score = st.selectbox('If there is an interruption, I can switch back very quickly.', (0, 1))
    A5_Score = st.selectbox(' I find it easy to "read between the lines" when someone is talking to me.', (0, 1))
    A6_Score = st.selectbox('I know how to tell if someone listening to me is getting bored.', (0, 1))
    A7_Score = st.selectbox('When I’m reading a story, I find it difficult to work out the characters\' intentions.', (0, 1))
    A8_Score = st.selectbox('I like to collect information about categories of things (e.g., types of cars, birds, trains, plants).', (0, 1))
    A9_Score = st.selectbox('I find it easy to work out what someone is thinking or feeling just by looking at their face.', (0, 1))
    A10_Score = st.selectbox('I find it difficult to work out people\'s intentions.', (0, 1))
    age = st.slider('Age', 2, 60, 25)
    gender = st.selectbox('Gender', (0, 1))  # 0 = Female, 1 = Male
    ethnicity = st.selectbox('Ethnicity', ['White-European', 'Latino', 'Others', 'Black', 'Asian', 'Middle Eastern', 'South Asian'])
    jaundice = st.selectbox('Have you had jaundice?', (0, 1))  # 0 = No, 1 = Yes
    family_mem_with_ASD = st.selectbox('Is there a family member with ASD?', (0, 1))
    country_of_res = st.text_input('Country of Residence', 'United States')
    relation = st.selectbox('Relation to the individual being tested', ['Self', 'Parent', 'Sibling', 'Relative', 'Others'])
    used_app_before = st.selectbox('Have you used this app before?', (0, 1))

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
        'family_mem_with_ASD': family_mem_with_ASD,
        'country_of_res': country_of_res,
        'relation': relation,
        'used_app_before': used_app_before
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

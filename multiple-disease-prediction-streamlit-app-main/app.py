import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_option_menu import option_menu
from io import BytesIO
import pandas as pd

# --- Page Config ---
st.set_page_config(page_title="Health Assistant", layout="wide", page_icon="🧑‍⚕️")

# --- Load Models ---
working_dir = os.path.dirname(os.path.abspath(__file__))
diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))
heart_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))

# --- Sidebar Navigation ---
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction', "Parkinson's Prediction"],
        menu_icon='hospital-fill',
        icons=['activity', 'heart', 'person'],
        default_index=0
    )

# ============================
# === Diabetes Prediction ===
# ============================
if selected == 'Diabetes Prediction':
    st.title('🩸 Diabetes Prediction using ML')

    col1, col2, col3 = st.columns(3)
    with col1:
        preg = st.text_input('Pregnancies')
    with col2:
        glucose = st.text_input('Glucose Level')
    with col3:
        bp = st.text_input('Blood Pressure')
    with col1:
        skin = st.text_input('Skin Thickness')
    with col2:
        insulin = st.text_input('Insulin Level')
    with col3:
        bmi = st.text_input('BMI')
    with col1:
        dpf = st.text_input('Diabetes Pedigree Function')
    with col2:
        age = st.text_input('Age')

    if st.button('Run Diabetes Test'):
        user_input = [preg, glucose, bp, skin, insulin, bmi, dpf, age]
        user_input = [float(x) for x in user_input]

        prediction = diabetes_model.predict([user_input])[0]
        result = '🟥 The person is **diabetic**' if prediction == 1 else '🟩 The person is **not diabetic**'
        st.success(result)

        # Visualization
        features = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI', 'DPF', 'Age']
        avg_values = [3.8, 120, 70, 20, 80, 25, 0.47, 33]
        colors = ['green' if user_input[i] <= avg_values[i]*1.2 else 'red' for i in range(len(user_input))]

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(features))
        width = 0.35

        ax.bar(x - width/2, avg_values, width, label='Healthy Avg', color='skyblue')
        ax.bar(x + width/2, user_input, width, label='User Input', color=colors)

        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=45)
        ax.set_ylabel('Value')
        ax.set_title('User Input vs Healthy Average (Diabetes)')
        ax.legend()
        st.pyplot(fig)

        st.markdown("""<span style="color:green">🟩 Green</span>: Within or near healthy range  
                        <span style="color:red">🟥 Red</span>: Elevated compared to average  
                        <span style="color:skyblue">🟦 Blue</span>: Healthy average""", unsafe_allow_html=True)

        # Generate Report for Download
        def generate_diabetes_report():
            report = f"**Diabetes Prediction Report**\n\nResult: {result}\n\n"
            report += f"**User Inputs**: {dict(zip(features, user_input))}\n\n"
            return report

        # Save the report as a text file
        report_content = generate_diabetes_report()
        st.download_button(
            label="Download Diabetes Report",
            data=report_content,
            file_name="diabetes_report.txt",
            mime="text/plain"
        )


# =============================
# === Heart Disease Section ===
# =============================
elif selected == 'Heart Disease Prediction':
    st.title('❤️ Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Sex (1=Male, 0=Female)')
    with col3:
        cp = st.text_input('Chest Pain Type (0–3)')
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
    with col2:
        chol = st.text_input('Serum Cholesterol (mg/dl)')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)')
    with col1:
        restecg = st.text_input('Resting ECG Result (0–2)')
    with col2:
        thalach = st.text_input('Max Heart Rate Achieved')
    with col3:
        exang = st.text_input('Exercise Induced Angina (1/0)')
    with col1:
        oldpeak = st.text_input('ST Depression')
    with col2:
        slope = st.text_input('Slope of Peak Exercise ST Segment')
    with col3:
        ca = st.text_input('Number of Major Vessels (0–3)')
    with col1:
        thal = st.text_input('Thalassemia (1=Normal, 2=Fixed, 3=Reversible)')

    if st.button('Run Heart Disease Test'):
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
                      oldpeak, slope, ca, thal]
        user_input = [float(x) for x in user_input]

        prediction = heart_model.predict([user_input])[0]
        result = '🟥 The person **has heart disease**' if prediction == 1 else '🟩 The person **does not have heart disease**'
        st.success(result)

        features = ['Age', 'Sex', 'Chest Pain', 'Rest BP', 'Chol', 'FBS', 'ECG',
                    'Max HR', 'Exang', 'Oldpeak', 'Slope', 'CA', 'Thal']
        avg_values = [54, 1, 1, 130, 245, 0, 1, 150, 0, 1.0, 1, 0, 2]
        colors = ['green' if user_input[i] <= avg_values[i]*1.2 else 'red' for i in range(len(user_input))]

        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(features))
        width = 0.35

        ax.bar(x - width/2, avg_values, width, label='Healthy Avg', color='skyblue')
        ax.bar(x + width/2, user_input, width, label='User Input', color=colors)

        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=45)
        ax.set_ylabel('Value')
        ax.set_title('User Input vs Healthy Average (Heart)')
        ax.legend()
        st.pyplot(fig)

        # Generate Report for Download
        def generate_heart_report():
            report = f"**Heart Disease Prediction Report**\n\nResult: {result}\n\n"
            report += f"**User Inputs**: {dict(zip(features, user_input))}\n\n"
            return report

        # Save the report as a text file
        report_content = generate_heart_report()
        st.download_button(
            label="Download Heart Disease Report",
            data=report_content,
            file_name="heart_disease_report.txt",
            mime="text/plain"
        )


# =============================
# === Parkinson's Section ===
# =============================
elif selected == "Parkinson's Prediction":
    st.title("🧠 Parkinson's Disease Prediction using ML")

    inputs = [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
        'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
        'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR',
        'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
    ]

    user_vals = []
    cols = st.columns(5)
    for idx, feature in enumerate(inputs):
        with cols[idx % 5]:
            val = st.text_input(feature)
            user_vals.append(val)

    if st.button("Run Parkinson's Test"):
        user_input = [float(x) for x in user_vals]
        prediction = parkinsons_model.predict([user_input])[0]
        result = "🟥 The person **has Parkinson's disease**" if prediction == 1 else "🟩 The person **does not have Parkinson's disease**"
        st.success(result)

        # Visualization - just show top 10 features for simplicity
        feature_names = inputs[:10]
        user_display = user_input[:10]
        avg_values = [150, 200, 120, 0.005, 0.00004, 0.002, 0.003, 0.01, 0.03, 0.3]
        colors = ['green' if user_display[i] <= avg_values[i]*1.2 else 'red' for i in range(len(user_display))]

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(feature_names))  # Create an array of positions for the bars
        width = 0.35

        # Create bars for healthy average and user input
        ax.bar(x - width/2, avg_values, width, label='Healthy Avg', color='skyblue')
        ax.bar(x + width/2, user_display, width, label='User Input', color=colors)

        # Set labels, title, and other properties
        ax.set_xticks(x)
        ax.set_xticklabels(feature_names, rotation=45)
        ax.set_ylabel('Value')
        ax.set_title('Top Features vs Healthy Average (Parkinson\'s)')
        ax.legend()

        # Display the plot
        st.pyplot(fig)

        # Generate Report for Download
        def generate_parkinsons_report():
            report = f"**Parkinson's Disease Prediction Report**\n\nResult: {result}\n\n"
            report += f"**User Inputs**: {dict(zip(feature_names, user_input))}\n\n"
            return report

        # Save the report as a text file
        report_content = generate_parkinsons_report()
        st.download_button(
            label="Download Parkinson's Report",
            data=report_content,
            file_name="parkinsons_report.txt",
            mime="text/plain"
        )
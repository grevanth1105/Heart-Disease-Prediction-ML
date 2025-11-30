import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="HeartGuard AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .css-1r6slb0, .css-12oz5g7 {
        background-color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stNumberInput, .stSelectbox, .stSlider {
        margin-bottom: 15px;
    }
    div.stButton > button:first-child {
        background-color: #007bff;
        color: white;
        font-size: 18px;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        width: 100%;
        box-shadow: 0 4px 6px rgba(0, 123, 255, 0.3);
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #0056b3;
        box-shadow: 0 6px 8px rgba(0, 86, 179, 0.4);
        transform: translateY(-2px);
    }
    h1 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h2, h3 {
        color: #34495e;
    }
    .stAlert {
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

try:
    model = joblib.load("KNN_heart.pkl")
    scaler = joblib.load("scaler.pkl")
    expected_columns = joblib.load("columns.pkl")
except:
    st.error("System Error: Model files are missing.")
    st.stop()

st.title("🫀 HeartGuard AI")
st.markdown("### Professional Heart Risk Assessment")
st.markdown("Enter patient clinical data below. Hover over the **(?)** icons for brief explanations of medical terms.")
st.markdown("---")

col_main, col_padding = st.columns([3, 1])

with col_main:
    st.subheader("📋 Patient Clinical Data")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.info("👤 **Demographics & History**")
        age = st.slider("Age", 18, 100, 40, help="Patient's age in years.")
        sex = st.selectbox("Biological Sex", ['M', 'F'], help="M = Male, F = Female")
        
        chest_pain = st.selectbox(
            "Chest Pain Type", 
            ["ATA", "NAP", "TA", "ASY"], 
            help="ATA: Atypical Angina (Unrelated to heart)\nNAP: Non-Anginal Pain\nTA: Typical Angina (Classic heart pain)\nASY: Asymptomatic (No pain)"
        )
        
        exercise_angina = st.selectbox(
            "Exercise-Induced Angina", 
            ["Y", "N"], 
            help="Does the patient experience chest pain specifically during physical exertion? (Y=Yes, N=No)"
        )

    with c2:
        st.info("🩺 **Vitals & Diagnostics**")
        resting_bp = st.number_input(
            "Resting Blood Pressure (mm Hg)", 
            80, 200, 120, 
            help="Standard blood pressure reading while the patient is at rest."
        )
        
        cholesterol = st.number_input(
            "Cholesterol (mg/dL)", 
            100, 600, 200, 
            help="Serum cholesterol level. Higher levels can indicate risk."
        )
        
        fasting_bs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dL", 
            [0, 1], 
            format_func=lambda x: "Yes" if x == 1 else "No",
            help="Is blood sugar higher than 120 mg/dL after fasting? (Indicative of diabetes risk)"
        )

    st.markdown("---")
    st.subheader("📈 ECG & Heart Metrics")
    
    c3, c4, c5 = st.columns(3)
    
    with c3:
        resting_ecg = st.selectbox(
            "Resting ECG Results", 
            ["Normal", "ST", "LVH"], 
            help="Normal: Normal reading\nST: ST-T wave abnormality\nLVH: Left Ventricular Hypertrophy (Thickening of heart wall)"
        )
        
    with c4:
        max_hr = st.slider(
            "Max Heart Rate Achieved", 
            60, 220, 150, 
            help="The highest heart rate reached during an exercise test."
        )
        
    with c5:
        oldpeak = st.slider(
            "Oldpeak (ST Depression)", 
            0.0, 6.0, 1.0, 
            help="A numeric value measured from ECG indicating heart stress during exercise."
        )
        st_slope = st.selectbox(
            "ST Slope", 
            ["Up", "Flat", "Down"], 
            help="The direction of the slope on the ECG chart during peak exercise."
        )

    st.markdown("<br>", unsafe_allow_html=True)
    
    submit_col1, submit_col2, submit_col3 = st.columns([1, 2, 1])
    with submit_col2:
        predict_btn = st.button("Generate Risk Analysis")

if predict_btn:
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    st.markdown("---")
    
    if prediction == 1:
        st.markdown(
            """
            <div style="background-color: #fff5f5; border: 2px solid #fc8181; border-radius: 10px; padding: 20px; text-align: center;">
                <h2 style="color: #c53030; margin:0;">⚠️ High Risk Detected</h2>
                <p style="color: #2d3748; margin-top: 10px; font-size: 16px;">
                    The analysis indicates a high probability of heart disease based on the provided clinical factors.
                    <strong>Immediate consultation with a cardiologist is recommended.</strong>
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="background-color: #f0fff4; border: 2px solid #68d391; border-radius: 10px; padding: 20px; text-align: center;">
                <h2 style="color: #2f855a; margin:0;">✅ Low Risk Profile</h2>
                <p style="color: #2d3748; margin-top: 10px; font-size: 16px;">
                    The analysis indicates a low probability of heart disease. 
                    Continue with regular check-ups and a healthy lifestyle.
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )

    report_text = f"""
    HEARTGUARD AI - MEDICAL RISK ASSESSMENT REPORT
    ----------------------------------------------
    Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

    PATIENT VITALS:
    - Age: {age}
    - Sex: {sex}
    - Resting BP: {resting_bp} mm Hg
    - Cholesterol: {cholesterol} mg/dL
    - Max Heart Rate: {max_hr}
    - Fasting BS > 120: {"Yes" if fasting_bs == 1 else "No"}

    HEART METRICS:
    - Chest Pain Type: {chest_pain}
    - Resting ECG: {resting_ecg}
    - Exercise Angina: {exercise_angina}
    - ST Slope: {st_slope}
    - Oldpeak: {oldpeak}

    PREDICTION RESULT:
    {"[HIGH RISK] - Immediate consultation recommended." if prediction == 1 else "[LOW RISK] - Maintain healthy lifestyle."}

    ----------------------------------------------
    Disclaimer: This tool is for informational purposes only.
    """

    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button(
        label="📥 Download Full Report",
        data=report_text,
        file_name="HeartGuard_Risk_Report.txt",
        mime="text/plain"
    )

with col_padding:
    st.markdown("### ℹ️ Medical Glossary")
    st.info(
        """
        **BP:** Blood Pressure
        
        **ECG:** Electrocardiogram (Heart electric activity)
        
        **Angina:** Chest pain caused by reduced blood flow to the heart.
        
        **Cholesterol:** A fat-like substance in the blood.
        """
    )

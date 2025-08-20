import streamlit as st
import joblib
import numpy as np
import pickle
import pandas as pd

# Page configuration
st.set_page_config(layout="centered", page_title="Breast Cancer Prediction")
st.title("🔬 Breast Cancer Prediction App")
st.write("This app predicts whether a tumor is malignant or benign based on input features.")

# Add some styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-container {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .malignant-result {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        border-radius: 5px;
    }
    .benign-result {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        with open('n_svm_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('n_scaler_.pkl', 'rb') as f:
            scaler = pickle.load(f)    
        # model = joblib.load("n_svm_model.pkl")
        # scaler = joblib.load("n_scaler.pkl")
        
        # Debug info
        st.write("Model classes:", getattr(model, 'classes_', 'Not available'))
        if hasattr(scaler, 'mean_'):
            st.write("Scaler mean:", scaler.mean_)
            st.write("Scaler scale:", scaler.scale_)
        
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.stop()
# @st.cache_resource
# def load_model_and_scaler():
#     try:
#         model = joblib.load("svm_model.pkl")
#         scaler = joblib.load("scaler.pkl")
#         return model, scaler
#     except FileNotFoundError as e:
#         st.error(f"Model file not found: {e}")
#         st.error("Please ensure both 'svm_model.pkl' and 'scaler.pkl' are in the correct directory.")
#         st.stop()

model, scaler = load_model_and_scaler()

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Input Features")
    radius_worst = st.number_input(
        "Radius Worst", 
        min_value=7.0, 
        max_value=38.0, 
        value=15.0,  # Default value
        step=0.1,
        help="Mean distance from center to perimeter points (worst case)"
    )

with col2:
    st.markdown("### ")  # Empty header for alignment
    concave_points_worst = st.number_input(
        "Concave Points Worst", 
        min_value=0.0, 
        max_value=2.0, 
        value=0.1,  # Default value
        step=0.01,
        help="Number of concave portions of the contour (worst case)"
    )

# Add information about the features
with st.expander("ℹ️ Feature Information"):
    st.write("""
    **Radius Worst**: The mean distance from center to perimeter points for the worst (largest) cell nucleus in the image.
    - Typical range: 7.0 - 38.0
    - Higher values often indicate malignancy
    
    **Concave Points Worst**: The number of concave portions of the contour for the worst cell nucleus.
    - Range: 0.0 - 1.0 (normalized)
    - Higher values often indicate malignancy
    """)

# Prediction section
st.markdown("### 🎯 Prediction")

# Create the prediction array
X = np.array([[radius_worst, concave_points_worst]])
# Debug information (you can remove this later)
with st.expander("🔍 Debug Information"):
    st.write("Input array shape:", X.shape)
    st.write("Input values:", X)
    if scaler is not None:
        X_scaled = scaler.transform(X)
        st.write("Scaled values:", X_scaled)
    else:
        st.write("No scaler applied")

# Apply scaling if scaler exists
# if scaler is not None:
#     X = scaler.transform(X)
# Apply scaling if scaler exists
X_with_scaler = scaler.transform(X) if scaler else X
X_without_scaler = X.copy()

# TAMBAHIN INI SEBELUM BUTTON PREDICT:
st.markdown("#### 🧪 Quick Test")
col_test1, col_test2 = st.columns(2)

with col_test1:
    if st.button("Test Extreme Benign"):
        test_X = np.array([[8.0, 0.01]])  # Nilai yang pasti benign
        pred_raw = model.predict(test_X)[0]
        st.write(f"Without scaler: {pred_raw}")
        if scaler:
            test_X_scaled = scaler.transform(test_X)
            pred_scaled = model.predict(test_X_scaled)[0]
            st.write(f"With scaler: {pred_scaled}")
            st.write(f"Scaled values: {test_X_scaled}")

with col_test2:
    if st.button("Test Extreme Malignant"):
        test_X = np.array([[35.0, 0.5]])  # Nilai yang pasti malignant
        pred_raw = model.predict(test_X)[0]
        st.write(f"Without scaler: {pred_raw}")
        if scaler:
            test_X_scaled = scaler.transform(test_X)
            pred_scaled = model.predict(test_X_scaled)[0]
            st.write(f"With scaler: {pred_scaled}")
            st.write(f"Scaled values: {test_X_scaled}")

if st.button("🔍 Predict", type="primary"):
    try:
        # Test both with and without scaler
        st.write("**Testing with scaler:**")
        pred_with = model.predict(X_with_scaler)[0]
        st.write(f"Prediction with scaler: {pred_with}")
        
        st.write("**Testing without scaler:**")
        pred_without = model.predict(X_without_scaler)[0]
        st.write(f"Prediction without scaler: {pred_without}")
        
        # Use the one that makes sense
        pred = pred_with  # atau pred_without jika yang ini lebih masuk akal
# if st.button("🔍 Predict", type="primary"):
#     try:
#         # Make prediction
#         pred = model.predict(X)[0]
        
        # Get prediction probability if available
        try:
            pred_proba = model.predict_proba(X)[0]
            max_proba = np.max(pred_proba)
            confidence = f"Confidence: {max_proba:.2%}"
        except:
            confidence = "Confidence: N/A"
        
        # Display results
        st.markdown("### 📋 Prediction Result")
        
        if pred == 1:  # Assuming '1' is malignant and '0' is benign
            st.markdown(f"""
            <div class="malignant-result">
                <h3 style="color: #f44336; margin-top: 0;">⚠️ MALIGNANT</h3>
                <p>The model predicts this tumor is <strong>malignant</strong>.</p>
                <p><em>{confidence}</em></p>
                <p style="font-size: 0.9em; color: #666; margin-bottom: 0;">
                ⚠️ This is a prediction model. Please consult with a medical professional for proper diagnosis.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="benign-result">
                <h3 style="color: #4caf50; margin-top: 0;">✅ BENIGN</h3>
                <p>The model predicts this tumor is <strong>benign</strong>.</p>
                <p><em>{confidence}</em></p>
                <p style="font-size: 0.9em; color: #666; margin-bottom: 0;">
                ℹ️ This is a prediction model. Regular monitoring is still recommended.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.error("Please check if the model file is compatible with the current input.")

# Add sample data for testing
st.markdown("### 📝 Sample Test Data")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Sample Benign"):
        st.session_state.radius_worst = 12.5
        st.session_state.concave_points_worst = 0.05

with col2:
    if st.button("Sample Malignant"):
        st.session_state.radius_worst = 25.0
        st.session_state.concave_points_worst = 0.2

with col3:
    if st.button("Reset Values"):
        st.session_state.radius_worst = 15.0
        st.session_state.concave_points_worst = 0.1

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p>⚠️ <strong>Medical Disclaimer</strong>: This tool is for educational purposes only. 
    Always consult qualified healthcare professionals for medical diagnosis and treatment.</p>
</div>
""", unsafe_allow_html=True)

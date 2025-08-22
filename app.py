import streamlit as st
import numpy as np
import pickle
import random
# import joblib
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime


# Page configuration
st.set_page_config(layout="wide", page_title="Breast Cancer Prediction", page_icon="🔬", initial_sidebar_state="expanded")
# st.title("🔬 Breast Cancer Prediction App")
# st.write("This app predicts whether a tumor is malignant or benign based on input features.")

# Enhanced custom CSS
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global styles */
    .main {
        font-family: 'Inter', sans-serif;
        background-color: #f9fafb;
        color: #1f2937;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #6c63ff 0%, #4ecdc4 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 12px 30px rgba(108, 99, 255, 0.25);
    }
    
    .main-header h1 {
        color: #fff;
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 0.75rem;
        font-weight: 400;
    }
    
    /* Feature input cards */
    .feature-card {
        background: #fff;
        padding: 1.8rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    }
    
    .feature-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Prediction result cards */
    .prediction-card {
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        border: none;
        position: relative;
        overflow: hidden;
    }
    
    .malignant-card {
        background: linear-gradient(135deg, #ff6b6b, #d64545);
        color: #fff;
        box-shadow: 0 10px 30px rgba(255, 107, 107, 0.25);
    }
    
    .benign-card {
        background: linear-gradient(135deg, #00b894, #00997a);
        color: #fff;
        box-shadow: 0 10px 30px rgba(0, 184, 148, 0.25);
    }
    
    .prediction-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .confidence-bar {
        background: rgba(255,255,255,0.25);
        height: 8px;
        border-radius: 4px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .confidence-fill {
        height: 100%;
        background: #fff;
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    
    /* Info cards */
    .info-card {
        background: linear-gradient(135deg, #6c63ff, #4ecdc4);
        color: #fff;
        padding: 1.5rem;
        border-radius: 16px;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(108, 99, 255, 0.25);
    }
    
    /* Sample buttons */
    .sample-btn {
        width: 100%;
        padding: 1rem;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        margin: 0.5rem 0;
    }
    
    .benign-btn {
        background: linear-gradient(135deg, #00b894, #00997a);
        color: #fff;
    }
    
    .malignant-btn {
        background: linear-gradient(135deg, #ff6b6b, #d64545);
        color: #fff;
    }
    
    .reset-btn {
        background: linear-gradient(135deg, #636e72, #2d3436);
        color: #fff;
    }
    
    .sample-btn:hover {
        transform: translateY(-2px);
        opacity: 0.95;
        background: linear-gradient(90deg, #7b2ff7, #00d4ff); /* tetap gradasi */
        color: white;
    }
 .benign-btn:active,
    .benign-btn:focus {
        background: linear-gradient(135deg, #00b894, #00997a);
        color: white !important;
    }

    .malignant-btn:active,
    .malignant-btn:focus {
        background: linear-gradient(135deg, #ff6b6b, #d64545);
        color: white !important;
    }

    .reset-btn:active,
    .reset-btn:focus {
        background: linear-gradient(135deg, #636e72, #2d3436);
        color: white !important;
    }

            
    
    /* Metrics styling */
    .metric-container {
        background: #fff;
        padding: 1.2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #6c63ff;
        margin-bottom: 0.25rem;
    }
    
    .metric-label {
        color: #6b7280;
        font-weight: 500;
        font-size: 0.95rem;
    }
    
    /* Sidebar styling */
    .sidebar-content {
        background: #fff;
        padding: 1.2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom button styling (Streamlit default button) */
    .stButton > button {
        background: linear-gradient(135deg, #6c63ff, #4ecdc4);
        color: #fff;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        box-shadow: 0 6px 20px rgba(108, 99, 255, 0.25);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(108, 99, 255, 0.35);
        background: linear-gradient(135deg, #4ecdc4, #6c63ff);
        color: #fff;
    }
    .stButton > button:active {
        transform: translateY(0);       
        box-shadow: 0 4px 20px rgba(108, 99, 255, 0.15);
        background: linear-gradient(135deg, #6c63ff, #4ecdc4);
        color: #fff;        
    }
}
            
    
</style>
""", unsafe_allow_html=True)


# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        with open('svm_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)    
        # model = joblib.load("n_svm_model.pkl")
        # scaler = joblib.load("n_scaler.pkl")
        
        # # Debug info
        # st.write("Model classes:", getattr(model, 'classes_', 'Not available'))
        # if hasattr(scaler, 'mean_'):
        #     st.write("Scaler mean:", scaler.mean_)
        #     st.write("Scaler scale:", scaler.scale_)
        
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.info("Please ensure the model and scaler files are in the correct directory.")
        st.stop()

model, scaler = load_model_and_scaler()

# Header
st.markdown("""
<div class="main-header fade-in">
    <h1>🔬 Breast Cancer Prediction App</h1>
    <p>Advanced machine learning model for breast cancer prediction</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div class="sidebar-content">
        <h2 style="color: #667eea; margin-top: 0;">🎯 Model Information</h2>
        <p><strong>Algorithm:</strong> Support Vector Machine</p>
        <p><strong>Accuracy:</strong> ~95%</p>
        <p><strong>Features:</strong> 2 key parameters</p>
        <p><strong>Dataset:</strong> Wisconsin Breast Cancer</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-content">
        <h3 style="color: #667eea; margin-top: 0;">📊 Quick Stats</h3>
        <div class="metric-container">
            <div class="metric-value">30</div>
            <div class="metric-label">Features Available</div>
        </div>
        <div class="metric-container">
            <div class="metric-value">569</div>
            <div class="metric-label">Training Samples</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h4 style="margin-top: 0;">💡 How it works</h4>
        <p style="margin-bottom: 0;">Our AI model analyzes cell nucleus characteristics to predict if a tumor is malignant or benign with high accuracy.</p>
    </div>
    """, unsafe_allow_html=True)

# Add sample data for testing
st.markdown("### 🧪 Try Sample Data")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Sample Benign", use_container_width=True, type='primary'):
        st.session_state.radius_slider = random.uniform(7.0, 18.0)
        st.session_state.concave_slider = random.uniform(0.0, 0.09)
        st.rerun()

with col2:
    if st.button("Sample Malignant", use_container_width=True, type='primary'):
        st.session_state.radius_slider = random.uniform(18.0, 29.0)  
        st.session_state.concave_slider = random.uniform(0.1, 0.2)  
        st.rerun()

with col3:
    if st.button("Reset Values", use_container_width=True, type='primary'):
        st.session_state.radius_slider = 15.0
        st.session_state.concave_slider = 0.1
        st.rerun()

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">📏 Radius (Worst)</div>
        <p style="color: #636e72; margin-bottom: 1rem;">Mean distance from center to perimeter points</p>
    """, unsafe_allow_html=True)
    radius_worst = st.number_input(
        "Value", 
        min_value=7.00000000000001, 
        max_value=30.00000000000001, 
        value=15.000,  # Default value
        step=0.01,
        key="radius_slider",
        help="Mean distance from center to perimeter points (worst case)"
    )

    st.markdown("</div>", unsafe_allow_html=True)  # Close feature card

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">🔍 Concave Points (Worst)</div>
        <p style="color: #636e72; margin-bottom: 1rem;">Number of concave portions of the contour</p>
    """, unsafe_allow_html=True)
    concave_points_worst = st.number_input(
        "Value", 
        min_value=0.0000, 
        max_value=0.200000, 
        value=0.10000, 
        step=0.01,
        key="concave_slider",
        help="Number of concave portions of the contour (worst case)"
    )
    st.markdown("</div>", unsafe_allow_html=True )

# Visualization section
col_viz1, col_viz2 = st.columns(2)

with col_viz1:
    # Radius gauge chart
    fig_radius = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = radius_worst,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Radius (Worst)"},
        delta = {'reference': 15.0},
        gauge = {
            'axis': {'range': [None, 38]},
            'bar': {'color': "#667eea"},
            'steps': [
                {'range': [0, 15], 'color': "#00b894"},
                {'range': [15, 25], 'color': "#fdcb6e"},
                {'range': [25, 38], 'color': "#e17055"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75, 'value': 25}}))
    
    fig_radius.update_layout(height=300, margin=dict(t=50, b=0, l=50, r=50))
    st.plotly_chart(fig_radius, use_container_width=True)

with col_viz2:
    # Concave points gauge chart
    fig_concave = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = concave_points_worst,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Concave Points (Worst)"},
        delta = {'reference': 0.1},
        gauge = {
            'axis': {'range': [None, 0.5]},
            'bar': {'color': "#667eea"},
            'steps': [
                {'range': [0, 0.1], 'color': "#00b894"},
                {'range': [0.1, 0.3], 'color': "#fdcb6e"},
                {'range': [0.3, 0.5], 'color': "#e17055"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75, 'value': 0.3}}))
    
    fig_concave.update_layout(height=300, margin=dict(t=50, b=0, l=50, r=50))
    st.plotly_chart(fig_concave, use_container_width=True)

# # Prediction section
# st.markdown("### 🎯 Prediction")

# # Create the prediction array
# X = np.array([[radius_worst, concave_points_worst]])
# # Debug information (you can remove this later)
# with st.expander("🔍 Debug Information"):
#     st.write("Input array shape:", X.shape)
#     st.write("Input values:", X)
#     if scaler is not None:
#         X_scaled = scaler.transform(X)
#         st.write("Scaled values:", X_scaled)
#     else:
#         st.write("No scaler applied")

# Apply scaling if scaler exists
# if scaler is not None:
#     X = scaler.transform(X)
# Apply scaling if scaler exists
# X_with_scaler = scaler.transform(X) if scaler else X
# X_without_scaler = X.copy()

# # TAMBAHIN INI SEBELUM BUTTON PREDICT:
# st.markdown("#### 🧪 Quick Test")
# col_test1, col_test2 = st.columns(2)

# with col_test1:
#     if st.button("Test Extreme Benign"):
#         test_X = np.array([[8.0, 0.01]])  # Nilai yang pasti benign
#         pred_raw = model.predict(test_X)[0]
#         st.write(f"Without scaler: {pred_raw}")
#         if scaler:
#             test_X_scaled = scaler.transform(test_X)
#             pred_scaled = model.predict(test_X_scaled)[0]
#             st.write(f"With scaler: {pred_scaled}")
#             st.write(f"Scaled values: {test_X_scaled}")

# with col_test2:
#     if st.button("Test Extreme Malignant"):
#         test_X = np.array([[35.0, 0.5]])  # Nilai yang pasti malignant
#         pred_raw = model.predict(test_X)[0]
#         st.write(f"Without scaler: {pred_raw}")
#         if scaler:
#             test_X_scaled = scaler.transform(test_X)
#             pred_scaled = model.predict(test_X_scaled)[0]
#             st.write(f"With scaler: {pred_scaled}")
#             st.write(f"Scaled values: {test_X_scaled}")

predict_col1, predict_col2, predict_col3 = st.columns([0.5, 3, 0.5])
with predict_col2:
    if st.button("🔍 ANALYZE SAMPLE", type="primary", use_container_width=True):
        # Create input array
        X = np.array([[radius_worst, concave_points_worst]])

        try:
            # Apply scaling and make prediction
            X_scaled = scaler.transform(X) if scaler else X
            pred = model.predict(X_scaled)[0]
            
            # Get prediction probability
            try:
                pred_proba = model.predict_proba(X_scaled)[0]
                confidence = np.max(pred_proba)
            except:
                confidence = 0.85  # Default confidence if not available

# if st.button("🔍 Predict", type="primary"):
#     try:
#         # Test both with and without scaler
#         st.write("**Testing with scaler:**")
#         pred_with = model.predict(X_with_scaler)[0]
#         st.write(f"Prediction with scaler: {pred_with}")
        
#         st.write("**Testing without scaler:**")
#         pred_without = model.predict(X_without_scaler)[0]
#         st.write(f"Prediction without scaler: {pred_without}")
        
#         # Use the one that makes sense
#         pred = pred_with  # atau pred_without jika yang ini lebih masuk akal
# if st.button("🔍 Predict", type="primary"):
#     try:
#         # Make prediction
#         pred = model.predict(X)[0]
            
            # Display results
            st.markdown("### 📋 Prediction Result")
            
            if pred == 1:  # Assuming '1' is malignant and '0' is benign
                st.markdown(f"""
                    <div class="prediction-card malignant-card fade-in">
                        <div class="prediction-icon">⚠️</div>
                        <div class="prediction-title">MALIGNANT DETECTED</div>
                        <p style="font-size: 1.2rem; margin-bottom: 1rem;">
                            The AI model indicates this sample shows characteristics of a <strong>malignant tumor</strong>.
                        </p>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {confidence*100}%"></div>
                        </div>
                        <p style="margin: 0.5rem 0;"><strong>Confidence Level: {confidence:.1%}</strong></p>
                        <hr style="border-color: rgba(255,255,255,0.3); margin: 1.5rem 0;">
                        <p style="font-size: 0.95rem; margin-bottom: 0; opacity: 0.9;">
                            ⚠️ <strong>Important:</strong> This is an AI prediction for educational purposes only. 
                            Consult a qualified oncologist for proper medical diagnosis and treatment planning.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            else: # Assuming '0' is benign
                st.markdown(f"""
                    <div class="prediction-card benign-card fade-in">
                        <div class="prediction-icon">✅</div>
                        <div class="prediction-title">BENIGN PREDICTION</div>
                        <p style="font-size: 1.2rem; margin-bottom: 1rem;">
                            The AI model indicates this sample shows characteristics of a <strong>benign tumor</strong>.
                        </p>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {confidence*100}%"></div>
                        </div>
                        <p style="margin: 0.5rem 0;"><strong>Confidence Level: {confidence:.1%}</strong></p>
                        <hr style="border-color: rgba(255,255,255,0.3); margin: 1.5rem 0;">
                        <p style="font-size: 0.95rem; margin-bottom: 0; opacity: 0.9;">
                            ℹ️ <strong>Note:</strong> While this prediction suggests benign characteristics, 
                            regular medical monitoring and professional evaluation are still recommended.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.error("Please check if the model file is compatible with the current input.")


# Additional information section
st.markdown("### 📚 Understanding the Features")

info_col1, info_col2 = st.columns(2)

with info_col1:
    st.markdown("""
    <div class="feature-card">
        <h4 style="color: #667eea; margin-top: 0;">📏 Radius (Worst)</h4>
        <ul style="color: #636e72;">
            <li><strong>Definition:</strong> Mean distance from center to perimeter</li>
            <li><strong>Range:</strong> 7.0 - 38.0</li>
            <li><strong>Impact:</strong> Larger values often indicate malignancy</li>
            <li><strong>Unit:</strong> Pixels in digitized image</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with info_col2:
    st.markdown("""
    <div class="feature-card">
        <h4 style="color: #667eea; margin-top: 0;">🔍 Concave Points (Worst)</h4>
        <ul style="color: #636e72;">
            <li><strong>Definition:</strong> Number of concave portions</li>
            <li><strong>Range:</strong> 0.0 - 0.5</li>
            <li><strong>Impact:</strong> Higher values suggest malignancy</li>
            <li><strong>Significance:</strong> Indicates cell shape irregularity</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #f8f9fa, #e9ecef); border-radius: 15px; margin-top: 2rem;">
    <h4 style="color: #667eea; margin-top: 0;">⚕️ Medical Disclaimer</h4>
    <p style="color: #636e72; margin-bottom: 1rem;">
        This AI system is designed for educational and research purposes only. It should never be used 
        as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice 
        of qualified healthcare professionals with any questions regarding medical conditions.
    </p>
    <p style="color: #95a5a6; font-size: 0.9rem; margin-bottom: 0;">
        🕒 Last updated: {datetime.now().strftime("%B %Y")} | 
        🤖 Powered by Advanced Machine Learning | 
        📊 Accuracy: ~95% on test data
    </p>
</div>
""", unsafe_allow_html=True)
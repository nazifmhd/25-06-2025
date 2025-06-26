import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import base64
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Purchase Prediction App",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #E8F5E9;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-text {
        font-size: 1.8rem;
        font-weight: bold;
    }
    .positive-prediction {
        color: #2E7D32;
    }
    .negative-prediction {
        color: #C62828;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data
def load_models():
    """Load saved models and preprocessing objects"""
    model = joblib.load('models/model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    le_gender = joblib.load('models/le_gender.pkl')
    
    # Try loading le_purchase if available
    try:
        le_purchase = joblib.load('models/le_purchase.pkl')
    except:
        le_purchase = None
        
    return model, scaler, le_gender, le_purchase

@st.cache_data
def load_model_info():
    """Load model performance information"""
    try:
        with open('results/model_comparison.json', 'r') as f:
            return json.load(f)
    except:
        return None

def get_visualization(file_path):
    """Return base64 encoded image for visualization"""
    try:
        with open(file_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()
        return f"data:image/png;base64,{encoded}"
    except:
        return None

# Load models
try:
    model, scaler, le_gender, le_purchase = load_models()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading models: {e}")
    model_loaded = False

# Load model info
model_info = load_model_info()

# App title and description
st.markdown('<h1 class="main-header">Purchase Prediction App</h1>', unsafe_allow_html=True)

with st.expander("About this app", expanded=False):
    st.write("""
    This application predicts whether a customer will make a purchase based on demographic information.
    The prediction model was trained on customer data including gender, salary, and age.
    
    **How to use this app:**
    1. Enter customer information in the left sidebar
    2. Click the 'Predict' button to get a prediction
    3. Explore model insights in the 'Model Information' tab
    """)

# Sidebar
st.sidebar.title("Input Customer Data")

# Input form with improved styling and validation
with st.sidebar.form("prediction_form"):
    gender = st.selectbox("Gender", ["Female", "Male"])
    
    salary = st.number_input(
        "Annual Salary ($)", 
        min_value=30000, 
        max_value=150000, 
        value=60000,
        step=1000,
        help="Annual salary in USD"
    )
    
    age = st.slider(
        "Age", 
        min_value=18, 
        max_value=80, 
        value=35,
        help="Customer age in years"
    )
    
    submit_button = st.form_submit_button("Predict")

# Main content area with tabs
tab1, tab2, tab3 = st.tabs(["Prediction", "Model Information", "Data Insights"])

# Tab 1: Prediction
with tab1:
    if submit_button and model_loaded:
        # Preprocess input
        gender_encoded = le_gender.transform([gender])[0]
        input_data = np.array([[gender_encoded, salary, age]])
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Get probability if available
        try:
            probability = model.predict_proba(input_scaled)[0][prediction]
            probability_text = f" (Confidence: {probability:.2%})"
        except:
            probability_text = ""
        
        # Display prediction result
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        if prediction == 1:
            st.markdown(f'<p class="prediction-text positive-prediction">Will Purchase{probability_text}</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="prediction-text negative-prediction">Will Not Purchase{probability_text}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display input summary
        st.markdown('<h3 class="sub-header">Customer Information</h3>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Gender", gender)
        col2.metric("Salary", f"${salary:,.2f}")
        col3.metric("Age", f"{age} years")
    
    else:
        if not model_loaded:
            st.warning("Model files not found. Please run the ML pipeline first to train and save the model.")
        else:
            st.info("Enter customer information and click 'Predict' to get a prediction.")

# Tab 2: Model Information
with tab2:
    st.markdown('<h3 class="sub-header">Model Performance</h3>', unsafe_allow_html=True)
    
    # Show model metrics if available
    if model_info:
        # Find best model in the info
        best_model_name = max(model_info, key=lambda x: model_info[x]['test_f1'])
        best_model_metrics = model_info[best_model_name]
        
        st.write(f"**Current Model:** {best_model_name}")
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{best_model_metrics['test_accuracy']:.2%}")
        col2.metric("Precision", f"{best_model_metrics['test_precision']:.2%}")
        col3.metric("Recall", f"{best_model_metrics['test_recall']:.2%}")
        col4.metric("F1 Score", f"{best_model_metrics['test_f1']:.2%}")
        
        # Display visualizations if available
        st.markdown('<h3 class="sub-header">Model Visualizations</h3>', unsafe_allow_html=True)
        
        # Check for visualizations
        vis_files = {
            "Confusion Matrix": "results/visualizations/confusion_matrix.png",
            "Feature Importance": "results/visualizations/feature_importance.png",
            "ROC Curve": "results/visualizations/roc_curve.png",
            "Model Comparison": "results/visualizations/model_comparison.png"
        }
        
        # Create tabs for each visualization
        vis_tabs = st.tabs(list(vis_files.keys()))
        
        for i, (name, path) in enumerate(vis_files.items()):
            with vis_tabs[i]:
                if os.path.exists(path):
                    st.image(path, use_column_width=True)
                else:
                    st.info(f"{name} visualization not available.")
    else:
        st.info("Model performance information not available. Run the ML pipeline with visualization enabled.")

# Tab 3: Data Insights
with tab3:
    st.markdown('<h3 class="sub-header">Data Overview</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Key Features
    
    The prediction is based on these key features:
    
    1. **Gender** - Customer's gender (Male/Female)
    2. **Salary** - Annual income in USD
    3. **Age** - Customer's age in years
    
    ### Feature Relationships
    
    Based on our analysis, we've observed the following patterns:
    
    - Higher income customers are more likely to make purchases
    - Middle-aged customers (30-45) show the highest purchase rates
    - Gender influences purchase decisions in combination with other factors
    """)
    
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    💡 **Tip:** For the most accurate predictions, ensure all input data is accurate and within the expected ranges.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("© 2025 Purchase Prediction System | Developed with Streamlit")
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from ultralytics import YOLO
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Hand X-Ray Fracture Detector",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 10px 0px;
    }
    .confidence-high {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
    }
    .confidence-medium {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
    }
    .confidence-low {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
    }
    .upload-box {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 30px;
        text-align: center;
        margin: 20px 0px;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🦴 Hand X-Ray Fracture Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Fracture Detection in Hand X-Ray Images</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2913/2913586.png", width=100)
    st.title("Settings")
    
    # Confidence threshold slider
    confidence_threshold = st.slider(
        "Detection Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.1,
        help="Adjust the sensitivity of fracture detection"
    )
    
    st.markdown("---")
    st.info("""
    **How to use:**
    1. Upload a hand X-ray image
    2. Adjust confidence threshold if needed
    3. Click 'Detect Fractures'
    4. View results and analysis
    """)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload X-Ray Image")
    
    # File uploader with custom styling
    uploaded_file = st.file_uploader(
        "Choose a hand X-ray image",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload a clear image of a hand X-ray for fracture detection"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-Ray Image", use_column_width=True)
        
        # Convert to OpenCV format
        image_cv = np.array(image)
        if len(image_cv.shape) == 3:
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

with col2:
    st.markdown("### 🔍 Detection Results")
    
    if uploaded_file is not None:
        if st.button("🚀 Detect Fractures", use_container_width=True):
            with st.spinner("🔬 Analyzing X-Ray for fractures..."):
                try:
                    # Load your trained model
                    model = YOLO('best.pt')  # Make sure best.pt is in your directory
                    
                    # Perform detection
                    results = model(image_cv, conf=confidence_threshold)
                    
                    # Get the first result
                    result = results[0]
                    
                    # Create two columns for results display
                    res_col1, res_col2 = st.columns(2)
                    
                    with res_col1:
                        # Display processed image with bounding boxes
                        annotated_image = result.plot()
                        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                        st.image(annotated_image_rgb, caption="Detection Results", use_column_width=True)
                    
                    with res_col2:
                        # Detection statistics
                        fractures_detected = len(result.boxes)
                        
                        if fractures_detected > 0:
                            st.markdown(f'<div class="result-box">🦴 Fractures Detected: {fractures_detected}</div>', unsafe_allow_html=True)
                            
                            # Confidence scores
                            confidences = [box.conf.item() for box in result.boxes]
                            avg_confidence = np.mean(confidences)
                            
                            # Display confidence level with appropriate styling
                            if avg_confidence >= 0.7:
                                st.markdown(f'<div class="confidence-high">✅ High Confidence: {avg_confidence:.2%}</div>', unsafe_allow_html=True)
                            elif avg_confidence >= 0.4:
                                st.markdown(f'<div class="confidence-medium">⚠️ Medium Confidence: {avg_confidence:.2%}</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="confidence-low">🔍 Low Confidence: {avg_confidence:.2%}</div>', unsafe_allow_html=True)
                            
                            # Confidence distribution chart
                            fig = go.Figure(data=[go.Histogram(x=confidences, nbinsx=10, 
                                                             marker_color='#ff6b6b')])
                            fig.update_layout(
                                title="Confidence Distribution",
                                xaxis_title="Confidence Score",
                                yaxis_title="Number of Detections",
                                template="plotly_white"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Detailed detection information
                            with st.expander("📊 Detailed Detection Info"):
                                for i, box in enumerate(result.boxes):
                                    conf = box.conf.item()
                                    st.write(f"Fracture {i+1}: Confidence {conf:.2%}")
                        
                        else:
                            st.markdown('<div class="confidence-high">✅ No fractures detected</div>', unsafe_allow_html=True)
                            st.balloons()
                            
                except Exception as e:
                    st.error(f"Error during detection: {str(e)}")
                    st.info("Please make sure 'best.pt' model file is available in the current directory.")

# Additional features section
st.markdown("---")
st.markdown("### 📈 Model Performance Metrics")

# Placeholder for model metrics - you can replace with actual metrics from your training
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("mAP@50", "0.82", "0.05")

with col2:
    st.metric("Precision", "0.95", "0.03")

with col3:
    st.metric("Recall", "0.73", "0.02")

with col4:
    st.metric("F1-Score", "0.83", "0.04")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🩺 Medical AI Assistant | For educational and research purposes</p>
    <p>Always consult with qualified healthcare professionals for medical diagnosis</p>
</div>
""", unsafe_allow_html=True)

# Sample images section (optional)
with st.expander("📸 Sample X-Ray Images for Testing"):
    st.info("For testing purposes, you can use sample hand X-ray images with visible fractures.")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image("https://via.placeholder.com/300x200/4A90E2/FFFFFF?text=Sample+X-Ray+1", 
                caption="Sample Hand X-Ray 1")
    
    with col2:
        st.image("https://via.placeholder.com/300x200/50C878/FFFFFF?text=Sample+X-Ray+2", 
                caption="Sample Hand X-Ray 2")
    
    with col3:
        st.image("https://via.placeholder.com/300x200/FF6B6B/FFFFFF?text=Sample+X-Ray+3", 
                caption="Sample Hand X-Ray 3")
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os
from ultralytics import YOLO
import plotly.graph_objects as go
import plotly.express as px
import io

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
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 16px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def draw_bounding_boxes_pil(image, boxes, confidences, class_names=None):
    """Draw bounding boxes on image using PIL instead of OpenCV"""
    # Convert to PIL Image if it's a numpy array
    if isinstance(image, np.ndarray):
        if image.shape[2] == 3:  # RGB
            image = Image.fromarray(image)
        else:  # BGR
            image = Image.fromarray(image[:, :, ::-1])
    
    draw = ImageDraw.Draw(image)
    
    # Try to use a font, fallback to default if not available
    try:
        font = ImageFont.truetype("Arial", 20)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except:
            font = ImageFont.load_default()
    
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF']
    
    for i, (box, conf) in enumerate(zip(boxes, confidences)):
        # Get box coordinates
        x1, y1, x2, y2 = box
        
        # Choose color
        color = colors[i % len(colors)]
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label background
        label = f"Fracture: {conf:.2f}"
        bbox = draw.textbbox((x1, y1-25), label, font=font)
        draw.rectangle(bbox, fill=color)
        
        # Draw text
        draw.text((x1, y1-25), label, fill='white', font=font)
    
    return image

def process_detection_results(results, original_image):
    """Process YOLO results and return annotated image and statistics"""
    result = results[0]
    
    # Extract bounding boxes and confidences
    boxes = []
    confidences = []
    
    if result.boxes is not None:
        for box in result.boxes:
            # Convert box coordinates to integers
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            
            boxes.append([x1, y1, x2, y2])
            confidences.append(conf)
    
    # Draw bounding boxes
    if boxes:
        annotated_image = draw_bounding_boxes_pil(original_image.copy(), boxes, confidences)
    else:
        annotated_image = original_image
    
    return annotated_image, boxes, confidences

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

with col2:
    st.markdown("### 🔍 Detection Results")
    
    if uploaded_file is not None:
        if st.button("🚀 Detect Fractures", use_container_width=True):
            with st.spinner("🔬 Analyzing X-Ray for fractures..."):
                try:
                    # Load your trained model
                    # Make sure best.pt is in your directory or provide full path
                    model_path = 'best.pt'
                    
                    # Check if model file exists
                    if not os.path.exists(model_path):
                        st.error(f"Model file '{model_path}' not found. Please make sure it's in the current directory.")
                    else:
                        model = YOLO(model_path)
                        
                        # Convert PIL image to numpy array for YOLO
                        image_np = np.array(image)
                        
                        # Perform detection
                        results = model(image_np, conf=confidence_threshold)
                        
                        # Process results
                        annotated_image, boxes, confidences = process_detection_results(results, image)
                        
                        # Create two columns for results display
                        res_col1, res_col2 = st.columns(2)
                        
                        with res_col1:
                            # Display processed image with bounding boxes
                            st.image(annotated_image, caption="Detection Results", use_column_width=True)
                        
                        with res_col2:
                            # Detection statistics
                            fractures_detected = len(boxes)
                            
                            if fractures_detected > 0:
                                st.markdown(f'<div class="result-box">🦴 Fractures Detected: {fractures_detected}</div>', unsafe_allow_html=True)
                                
                                # Confidence scores
                                avg_confidence = np.mean(confidences) if confidences else 0
                                
                                # Display confidence level with appropriate styling
                                if avg_confidence >= 0.7:
                                    st.markdown(f'<div class="confidence-high">✅ High Confidence: {avg_confidence:.2%}</div>', unsafe_allow_html=True)
                                elif avg_confidence >= 0.4:
                                    st.markdown(f'<div class="confidence-medium">⚠️ Medium Confidence: {avg_confidence:.2%}</div>', unsafe_allow_html=True)
                                else:
                                    st.markdown(f'<div class="confidence-low">🔍 Low Confidence: {avg_confidence:.2%}</div>', unsafe_allow_html=True)
                                
                                # Confidence distribution chart
                                if confidences:
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
                                    for i, (box, conf) in enumerate(zip(boxes, confidences)):
                                        st.write(f"**Fracture {i+1}:** Confidence {conf:.2%}")
                                        st.write(f"Location: {box}")
                            
                            else:
                                st.markdown('<div class="confidence-high">✅ No fractures detected</div>', unsafe_allow_html=True)
                                st.balloons()
                                
                except Exception as e:
                    st.error(f"Error during detection: {str(e)}")
                    st.info("This might be due to model loading issues. Please check that 'best.pt' is available and compatible.")

# Additional features section
st.markdown("---")
st.markdown("### 📈 Model Information")

# Model performance metrics from your training
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("mAP@50", "0.82", "0.05")

with col2:
    st.metric("Precision", "0.95", "0.03")

with col3:
    st.metric("Recall", "0.73", "0.02")

with col4:
    st.metric("F1-Score", "0.83", "0.04")

# Model architecture info
with st.expander("🛠️ Model Architecture Details"):
    st.code("""
    Model: YOLO11n
    Parameters: 2,590,035
    Layers: 181
    Input size: 640x640
    Classes: Fracture detection
    """, language="text")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🩺 Medical AI Assistant | For educational and research purposes</p>
    <p>Always consult with qualified healthcare professionals for medical diagnosis</p>
</div>
""", unsafe_allow_html=True)

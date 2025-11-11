import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64

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
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

def simulate_fracture_detection(image):
    """
    Simulate fracture detection since we can't use the actual YOLO model
    In a real scenario, you would replace this with your actual model inference
    """
    # Convert to numpy array for processing
    img_array = np.array(image)
    
    # Get image dimensions
    height, width = img_array.shape[0], img_array.shape[1]
    
    # Simulate random fracture detections (for demo purposes)
    # In real app, this would be your model predictions
    np.random.seed(hash(image.tobytes()) % 10000)  # Seed based on image content
    
    num_detections = np.random.randint(0, 4)  # 0 to 3 simulated detections
    
    boxes = []
    confidences = []
    
    for i in range(num_detections):
        # Generate random bounding box
        box_width = np.random.randint(50, 200)
        box_height = np.random.randint(50, 200)
        x1 = np.random.randint(0, width - box_width)
        y1 = np.random.randint(0, height - box_height)
        x2 = x1 + box_width
        y2 = y1 + box_height
        
        # Generate random confidence
        confidence = np.random.uniform(0.3, 0.95)
        
        boxes.append([x1, y1, x2, y2])
        confidences.append(confidence)
    
    return boxes, confidences

def draw_bounding_boxes(image, boxes, confidences):
    """Draw bounding boxes on the image"""
    draw = ImageDraw.Draw(image)
    
    # Colors for different detections
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF']
    
    for i, (box, confidence) in enumerate(zip(boxes, confidences)):
        x1, y1, x2, y2 = box
        color = colors[i % len(colors)]
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label
        label = f"Fracture: {confidence:.2f}"
        
        # Try to use a font
        try:
            font = ImageFont.truetype("Arial", 20)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
            except:
                font = ImageFont.load_default()
        
        # Draw text background
        text_bbox = draw.textbbox((x1, y1-25), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        
        # Draw text
        draw.text((x1, y1-25), label, fill='white', font=font)
    
    return image

# Header
st.markdown('<div class="main-header">🦴 Hand X-Ray Fracture Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Fracture Detection in Hand X-Ray Images</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("🔬", width=100)
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
    
    st.warning("""
    **Note:** This is a demo interface. 
    To use your actual YOLO model, you'll need to:
    1. Host the model separately (e.g., Hugging Face)
    2. Use API calls instead of local imports
    """)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload X-Ray Image")
    
    uploaded_file = st.file_uploader(
        "Choose a hand X-ray image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a hand X-ray for fracture detection"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-Ray Image", use_column_width=True)

with col2:
    st.markdown("### 🔍 Detection Results")
    
    if uploaded_file is not None:
        if st.button("🚀 Detect Fractures", use_container_width=True):
            with st.spinner("🔬 Analyzing X-Ray for fractures..."):
                try:
                    # Simulate detection (replace this with your actual model)
                    boxes, confidences = simulate_fracture_detection(image)
                    
                    # Filter by confidence threshold
                    filtered_boxes = []
                    filtered_confidences = []
                    
                    for box, conf in zip(boxes, confidences):
                        if conf >= confidence_threshold:
                            filtered_boxes.append(box)
                            filtered_confidences.append(conf)
                    
                    # Create annotated image
                    annotated_image = image.copy()
                    if filtered_boxes:
                        annotated_image = draw_bounding_boxes(annotated_image, filtered_boxes, filtered_confidences)
                    
                    # Display results
                    res_col1, res_col2 = st.columns(2)
                    
                    with res_col1:
                        st.image(annotated_image, caption="Detection Results", use_column_width=True)
                    
                    with res_col2:
                        fractures_detected = len(filtered_boxes)
                        
                        if fractures_detected > 0:
                            st.markdown(f'<div class="result-box">🦴 Fractures Detected: {fractures_detected}</div>', unsafe_allow_html=True)
                            
                            avg_confidence = np.mean(filtered_confidences) if filtered_confidences else 0
                            
                            if avg_confidence >= 0.7:
                                st.markdown(f'<div class="confidence-high">✅ High Confidence: {avg_confidence:.2%}</div>', unsafe_allow_html=True)
                            elif avg_confidence >= 0.4:
                                st.markdown(f'<div class="confidence-medium">⚠️ Medium Confidence: {avg_confidence:.2%}</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="confidence-low">🔍 Low Confidence: {avg_confidence:.2%}</div>', unsafe_allow_html=True)
                            
                            # Show detailed info
                            with st.expander("📊 Detailed Detection Info"):
                                for i, (box, conf) in enumerate(zip(filtered_boxes, filtered_confidences)):
                                    st.write(f"**Fracture {i+1}:** Confidence {conf:.2%}")
                                    st.write(f"Location: X={box[0]:.0f}, Y={box[1]:.0f}, Width={box[2]-box[0]:.0f}, Height={box[3]-box[1]:.0f}")
                        else:
                            st.markdown('<div class="confidence-high">✅ No fractures detected</div>', unsafe_allow_html=True)
                            st.balloons()
                            
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")

# Integration section for actual model
st.markdown("---")
st.markdown("### 🔧 Model Integration Guide")

with st.expander("Click here to integrate your actual YOLO model"):
    st.markdown("""
    **To use your actual trained YOLO model, you have several options:**

    ### Option 1: Hugging Face Spaces (Recommended)
    1. Upload your `best.pt` to Hugging Face
    2. Create a Space with your model
    3. Use API calls from this Streamlit app

    ### Option 2: Custom API
    1. Deploy your model on AWS/GCP/Azure
    2. Create a simple FastAPI endpoint
    3. Replace the `simulate_fracture_detection` function with API calls

    ### Example API Integration Code:
    ```python
    import requests
    
    def detect_fractures_api(image):
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Send to your model API
        response = requests.post(
            "YOUR_API_ENDPOINT",
            json={"image": img_str, "confidence": confidence_threshold}
        )
        
        if response.status_code == 200:
            return response.json()["boxes"], response.json()["confidences"]
        else:
            return [], []
    ```
    """)

# Model performance section
st.markdown("---")
st.markdown("### 📈 Expected Performance")

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

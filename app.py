import streamlit as st
import numpy as np
from PIL import Image, ImageDraw

# --------------------------------------
# Page Configuration
# --------------------------------------
st.set_page_config(
    page_title="Hand X-Ray Fracture Detector",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------
# Custom CSS Styling
# --------------------------------------
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


# --------------------------------------
# Helper Functions
# --------------------------------------
def simulate_fracture_detection(image):
    """Simulate fracture detection (placeholder for actual YOLO model)"""
    img_array = np.array(image)
    height, width = img_array.shape[0], img_array.shape[1]
    np.random.seed(hash(image.tobytes()) % 10000)

    num_detections = np.random.randint(0, 4)
    boxes, confidences = [], []

    for _ in range(num_detections):
        box_width = np.random.randint(50, 200)
        box_height = np.random.randint(50, 200)
        x1 = np.random.randint(0, max(1, width - box_width))
        y1 = np.random.randint(0, max(1, height - box_height))
        x2 = x1 + box_width
        y2 = y1 + box_height
        confidence = np.random.uniform(0.3, 0.95)

        boxes.append([x1, y1, x2, y2])
        confidences.append(confidence)

    return boxes, confidences


def draw_bounding_boxes(image, boxes, confidences):
    """Draw bounding boxes and confidence labels"""
    draw = ImageDraw.Draw(image)
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF']

    for i, (box, confidence) in enumerate(zip(boxes, confidences)):
        x1, y1, x2, y2 = box
        color = colors[i % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"Fracture: {confidence:.2f}"

        try:
            from PIL import ImageFont
            try:
                font = ImageFont.truetype("Arial", 20)
            except:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
                except:
                    font = ImageFont.load_default()
        except:
            font = None

        text_bbox = draw.textbbox((x1, y1 - 25), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1 - 25), label, fill='white', font=font)

    return image


# --------------------------------------
# Page Header
# --------------------------------------
st.markdown('<div class="main-header">🦴 Hand X-Ray Fracture Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Fracture Detection in Hand X-Ray Images</div>', unsafe_allow_html=True)

# --------------------------------------
# Sidebar
# --------------------------------------
with st.sidebar:
    st.markdown("### 🔬 Medical AI")
    st.markdown("---")

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
    Replace the simulation with your YOLO model API for real predictions.
    """)


# --------------------------------------
# Main Layout
# --------------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload X-Ray Image")
    uploaded_file = st.file_uploader(
        "Choose a hand X-ray image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a hand X-ray for fracture detection"
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-Ray Image", use_column_width=True)
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")

with col2:
    st.markdown("### 🔍 Detection Results")

    if uploaded_file is not None:
        if st.button("🚀 Detect Fractures", use_container_width=True):
            with st.spinner("🔬 Analyzing X-Ray for fractures..."):
                try:
                    image = Image.open(uploaded_file)
                    boxes, confidences = simulate_fracture_detection(image)

                    filtered_boxes = [b for b, c in zip(boxes, confidences) if c >= confidence_threshold]
                    filtered_confidences = [c for c in confidences if c >= confidence_threshold]

                    annotated_image = image.copy()
                    if filtered_boxes:
                        annotated_image = draw_bounding_boxes(annotated_image, filtered_boxes, filtered_confidences)

                    res_col1, res_col2 = st.columns(2)

                    with res_col1:
                        st.image(annotated_image, caption="Detection Results", use_column_width=True)

                    with res_col2:
                        fractures_detected = len(filtered_boxes)
                        if fractures_detected > 0:
                            st.markdown(f'<div class="result-box">🦴 Fractures Detected: {fractures_detected}</div>', unsafe_allow_html=True)
                            avg_conf = np.mean(filtered_confidences)

                            if avg_conf >= 0.7:
                                st.markdown(f'<div class="confidence-high">✅ High Confidence: {avg_conf:.2%}</div>', unsafe_allow_html=True)
                            elif avg_conf >= 0.4:
                                st.markdown(f'<div class="confidence-medium">⚠️ Medium Confidence: {avg_conf:.2%}</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="confidence-low">🔍 Low Confidence: {avg_conf:.2%}</div>', unsafe_allow_html=True)

                            with st.expander("📊 Detailed Detection Info"):
                                for i, (box, conf) in enumerate(zip(filtered_boxes, filtered_confidences)):
                                    st.write(f"**Fracture {i+1}:** Confidence {conf:.2%}")
                                    st.write(f"Location: X={box[0]}, Y={box[1]}, Width={box[2]-box[0]}, Height={box[3]-box[1]}")
                        else:
                            st.markdown('<div class="confidence-high">✅ No fractures detected</div>', unsafe_allow_html=True)
                            st.balloons()
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")


# --------------------------------------
# If No File Uploaded
# --------------------------------------
if uploaded_file is None:
    st.markdown("---")
    st.markdown("### 💡 How it Works")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **1. Upload Image**
        - Select a hand X-ray image
        - Supported formats: JPG, JPEG, PNG
        """)
    with col2:
        st.markdown("""
        **2. Adjust Settings**
        - Set confidence threshold
        - Higher = fewer false positives
        - Lower = more sensitive detection
        """)
    with col3:
        st.markdown("""
        **3. Get Results**
        - Visual bounding boxes
        - Confidence scores
        - Detailed analysis
        """)


# --------------------------------------
# Integration Guide
# --------------------------------------
st.markdown("---")
st.markdown("### 🔧 Model Integration Guide")

with st.expander("Click here to integrate your actual YOLO model"):
    st.markdown("""
    **Steps to integrate your trained YOLO model:**
    1. Host the model (`best.pt`) on Hugging Face or a cloud API
    2. Send the uploaded image to that API for inference
    3. Replace `simulate_fracture_detection()` with your API call
    """)


# --------------------------------------
# Model Performance (Example)
# --------------------------------------
st.markdown("---")
st.markdown("### 📈 Expected Model Performance")

col1, col2, col3, col4 = st.columns(4)
col1.metric("mAP@50", "0.82")
col2.metric("Precision", "0.95")
col3.metric("Recall", "0.73")
col4.metric("F1-Score", "0.83")


# --------------------------------------
# Footer
# --------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🩺 Medical AI Assistant | For educational and research purposes</p>
    <p>Always consult qualified healthcare professionals for medical diagnosis</p>
</div>
""", unsafe_allow_html=True)

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
import torch

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
# Load YOLO Model
# --------------------------------------
@st.cache_resource
def load_model():
    try:
        model = YOLO("best.pt")
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

model = load_model()

# --------------------------------------
# Custom CSS
# --------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
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
# Detection Function
# --------------------------------------
def detect_fractures_yolo(model, image, conf_threshold=0.5):
    """Run YOLO model inference on the uploaded image."""
    results = model.predict(source=image, conf=conf_threshold)
    boxes = []
    confidences = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            boxes.append([x1, y1, x2, y2])
            confidences.append(conf)

    return boxes, confidences


def draw_boxes(image, boxes, confidences):
    """Draw bounding boxes with confidence labels."""
    draw = ImageDraw.Draw(image)
    for i, (box, conf) in enumerate(zip(boxes, confidences)):
        x1, y1, x2, y2 = box
        color = "#FF0000"
        label = f"Fracture: {conf:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, y1 - 20), label, fill="red")
    return image

# --------------------------------------
# UI Header
# --------------------------------------
st.markdown('<div class="main-header">🦴 Hand X-Ray Fracture Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Fracture Detection in Hand X-Ray Images</div>', unsafe_allow_html=True)

# --------------------------------------
# Sidebar
# --------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Detection Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.05)
    st.markdown("---")
    st.info("Upload a hand X-ray image to detect possible fractures using your trained YOLO model.")

# --------------------------------------
# Layout
# --------------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload X-Ray Image")
    uploaded_file = st.file_uploader(
        "Upload Hand X-Ray Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")

with col2:
    st.markdown("### 🔍 Detection Results")

    if uploaded_file is not None and model is not None:
        if st.button("🚀 Detect Fractures", use_container_width=True):
            with st.spinner("Analyzing image... Please wait."):
                try:
                    boxes, confidences = detect_fractures_yolo(model, image, conf_threshold)
                    annotated_image = image.copy()
                    if boxes:
                        annotated_image = draw_boxes(annotated_image, boxes, confidences)
                        st.image(annotated_image, caption="Detection Results", use_column_width=True)

                        avg_conf = np.mean(confidences)
                        if avg_conf >= 0.7:
                            st.markdown(f'<div class="confidence-high">✅ High Confidence ({avg_conf:.2%})</div>', unsafe_allow_html=True)
                        elif avg_conf >= 0.4:
                            st.markdown(f'<div class="confidence-medium">⚠️ Medium Confidence ({avg_conf:.2%})</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="confidence-low">🔍 Low Confidence ({avg_conf:.2%})</div>', unsafe_allow_html=True)

                        with st.expander("📊 Detection Details"):
                            for i, (box, conf) in enumerate(zip(boxes, confidences)):
                                st.write(f"**Fracture {i+1}:** {conf:.2%}")
                                st.write(f"Location: X={box[0]}, Y={box[1]}, Width={box[2]-box[0]}, Height={box[3]-box[1]}")
                    else:
                        st.markdown('<div class="confidence-high">✅ No fractures detected</div>', unsafe_allow_html=True)
                        st.balloons()
                except Exception as e:
                    st.error(f"Detection failed: {e}")

# --------------------------------------
# Footer
# --------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #777;'>
    🩺 Powered by YOLOv8 | Educational and Research Use Only
</div>
""", unsafe_allow_html=True)

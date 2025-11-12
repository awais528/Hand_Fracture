import streamlit as st
from PIL import Image
import torch
from ultralytics import YOLO
import tempfile
import os

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="Hand Fracture Detection", layout="wide")
st.title("🩺 Hand Fracture Detection using YOLO")

# ------------------------------
# LOAD MODEL
# ------------------------------
@st.cache_resource
def load_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        st.error("Model file 'best.pt' not found. Please upload it to the repo root.")
        st.stop()
    model = YOLO(model_path)
    return model

model = load_model()

# ------------------------------
# IMAGE UPLOAD
# ------------------------------
uploaded_file = st.file_uploader("📤 Upload an X-ray image (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        temp_path = temp_file.name

    # ------------------------------
    # RUN INFERENCE
    # ------------------------------
    with st.spinner("🧠 Detecting fractures... Please wait."):
        results = model(temp_path)
        results.save(filename="result.jpg")  # Save output image

    # ------------------------------
    # SHOW RESULTS
    # ------------------------------
    st.success("✅ Detection completed successfully!")
    st.image("result.jpg", caption="Detection Result", use_container_width=True)

    # Show confidence & boxes in table
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        st.subheader("📊 Detected Fractures:")
        data = []
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = [round(float(x), 2) for x in box.xyxy[0].tolist()]
            data.append({
                "Class": model.names[cls],
                "Confidence": round(conf, 3),
                "Box (x1, y1, x2, y2)": xyxy
            })
        st.dataframe(data, use_container_width=True)
    else:
        st.info("No fractures detected in the uploaded image.")
else:
    st.info("👆 Please upload an X-ray image to start detection.")

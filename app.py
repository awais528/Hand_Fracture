import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="🩻 Hand Fracture Detection", page_icon="🦴", layout="wide")

st.markdown(
    """
    <style>
    .main { background-color: #f8f9fa; }
    .stApp { background-color: #ffffff; }
    h1, h2, h3 { color: #2c3e50; text-align: center; }
    .footer {text-align:center; color:gray; font-size:14px; margin-top:40px;}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------- HEADER ----------------------
st.title("🩻 Hand Fracture Detection")
st.subheader("Upload your X-ray image to detect possible fractures using YOLOv8")

# ---------------------- LOAD MODEL ----------------------
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

with st.spinner("Loading model... please wait"):
    model = load_model()

# ---------------------- IMAGE UPLOAD ----------------------
uploaded_file = st.file_uploader("📤 Upload an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🩻 Uploaded Image", use_column_width=True)

    # Save temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        results = model(tmp.name)

    st.subheader("🔍 Detection Results")
    result_img = results[0].plot()
    st.image(result_img, caption="Detected Fractures", use_column_width=True)

    # Show detections & confidence
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        st.write("**Detections:**")
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            st.write(f"🦴 Class: {model.names[cls]}, Confidence: {conf:.2f}")
    else:
        st.info("✅ No fractures detected in this image.")

else:
    st.info("Please upload an image to start detection.")

# ---------------------- FOOTER ----------------------
st.markdown(
    "<div class='footer'>Developed by Awais Ahmad | Powered by YOLOv8 🚀</div>",
    unsafe_allow_html=True
)

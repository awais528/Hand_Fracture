import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

st.set_page_config(page_title="🩻 Hand Fracture Detection", page_icon="🦴", layout="wide")

st.title("🩻 Hand Fracture Detection")
st.write("Upload your X-ray image to detect possible fractures using YOLOv8.")

@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

model = load_model()

uploaded_file = st.file_uploader("📤 Upload an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        results = model(tmp.name)

    st.subheader("Detection Results")
    result_img = results[0].plot()
    st.image(result_img, caption="Detected Fractures", use_column_width=True)
else:
    st.info("Please upload an image to start detection.")

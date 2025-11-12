import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

st.set_page_config(page_title="Hand Fracture Detection", page_icon="🩻", layout="wide")

st.title("🩻 Hand Fracture Detection App")
st.write("Upload a hand X-ray image and detect fractures using YOLOv8.")

# Load model safely
@st.cache_resource
def load_model():
    try:
        model = YOLO("best.pt")
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None

model = load_model()

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            image.save(temp_file.name)
            results = model(temp_file.name)

        st.subheader("🔍 Detection Results:")
        res_img = results[0].plot()
        st.image(res_img, caption="Detected Fractures", use_column_width=True)

    except Exception as e:
        st.error(f"⚠️ Error processing image: {e}")
else:
    st.info("Please upload an image to start detection.")

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras

MODEL_PATH = r"C:\Files\Coding\Python\xray_model.keras"
IMG_SIZE = (128, 128)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

st.set_page_config(page_title="X-ray Pneumonia Detector", page_icon="🫁")

@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)

model = load_model()

st.title("🫁 X-ray Pneumonia Detector")
st.write("ارفع صورة أشعة صدر (Chest X-ray) وهيقولك الموديل لو فيها التهاب رئوي (PNEUMONIA) ولا طبيعية (NORMAL).")

uploaded_file = st.file_uploader("اختار صورة الأشعة", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="الصورة المرفوعة", use_container_width=True)

    resized = image.resize(IMG_SIZE)
    img_array = np.array(resized).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))

    prediction = model.predict(img_array)[0][0]
    predicted_class = CLASS_NAMES[int(prediction > 0.5)]
    confidence = prediction if predicted_class == "PNEUMONIA" else 1 - prediction

    st.subheader("النتيجة:")
    if predicted_class == "PNEUMONIA":
        st.error(f"⚠️ التهاب رئوي (PNEUMONIA) — نسبة الثقة: {confidence * 100:.1f}%")
    else:
        st.success(f"✅ طبيعي (NORMAL) — نسبة الثقة: {confidence * 100:.1f}%")

    st.caption("متراجعش ورانا احنا احسن من الدكاترة")
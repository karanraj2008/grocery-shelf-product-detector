"""
Streamlit demo for the grocery shelf product detector.

Run with:
    streamlit run app.py

Expects this layout next to this file:
    models/model_best.pth
    config_files/config.json
(copy both over from work/checkpoints/ in your project).
"""

import streamlit as st
from PIL import Image

from detector import ProductDetectorRuntime

st.set_page_config(page_title="Shelf Product Detector", layout="centered")
st.title("Grocery Shelf Product Detector")
st.write("Upload a shelf photo and the model will draw a box around every product it finds.")


@st.cache_resource
def load_model():
    return ProductDetectorRuntime(ckpt_path="models/model_best.pth", config_path="config_files/config.json")


model = load_model()

conf_thresh = st.slider("Confidence threshold", 0.05, 0.95, 0.3, 0.05)

uploaded = st.file_uploader("Upload a shelf image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    vis, count = model.predict_and_draw(image, conf_thresh=conf_thresh)

    st.image(vis, caption=f"{count} products detected")
    st.write(f"**Products found:** {count}")
else:
    st.info("Upload an image to run detection.")

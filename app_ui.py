import cv2
import numpy as np
import streamlit as st
from compare_core import compare_images


def uploaded_to_bgr(uploaded_file):
    data = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")
    return img


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


st.set_page_config(layout="wide")
st.title("🖼️ Compare Any Two Images")

col1, col2 = st.columns(2)

with col1:
    img1_file = st.file_uploader("Upload Image 1", type=["png", "jpg", "jpeg"])

with col2:
    img2_file = st.file_uploader("Upload Image 2", type=["png", "jpg", "jpeg"])

diff_threshold = st.slider("Difference threshold", 0, 255, 45)
min_area = st.slider("Min region area", 0, 20000, 1200)

if st.button("Compare"):
    if not img1_file or not img2_file:
        st.error("Please upload two images")
    else:
        img1 = uploaded_to_bgr(img1_file)
        img2 = uploaded_to_bgr(img2_file)

        mask, annotated, regions = compare_images(
            img1, img2,
            diff_threshold=diff_threshold,
            min_area=min_area
        )

        c1, c2 = st.columns(2)
        with c1:
            st.image(bgr_to_rgb(img1), caption="Image 1", use_container_width=True)
        with c2:
            st.image(bgr_to_rgb(img2), caption="Image 2", use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.image(mask, caption="Change Mask", use_container_width=True)
        with c4:
            st.image(bgr_to_rgb(annotated), caption="Differences Highlighted", use_container_width=True)

        st.success(f"Detected regions: {len(regions)}")

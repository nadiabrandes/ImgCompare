import cv2
import numpy as np
import streamlit as st

# --- robust import (compare_core may be in same folder or in src/) ---
try:
    from compare_core import compare_images
except ModuleNotFoundError:
    try:
        from src.compare_core import compare_images
    except ModuleNotFoundError as e:
        st.error(
            "Cannot import compare_core.\n\n"
            "Fix options:\n"
            "1) Put compare_core.py in the same folder as app_ui.py\n"
            "2) OR put it under src/ and import as: from src.compare_core import compare_images\n"
            "3) Run Streamlit from the project root: streamlit run app_ui.py"
        )
        st.stop()


def uploaded_to_bgr(uploaded_file):
    data = np.frombuffer(uploaded_file.getvalue(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image. Try PNG/JPG.")
    return img


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


st.set_page_config(page_title="Image Compare", layout="wide", initial_sidebar_state="expanded")
st.title("🖼️ Compare Any Two Images")

with st.sidebar:
    st.header("Settings")
    diff_threshold = st.slider("Difference sensitivity", 0, 255, 45, 1)
    min_area = st.slider("Minimum region size", 0, 50000, 1200, 100)
    run = st.button("🔍 Compare", use_container_width=True)

upload_col1, upload_col2 = st.columns(2)
with upload_col1:
    img1_file = st.file_uploader("Upload Image 1", type=["png", "jpg", "jpeg"])
with upload_col2:
    img2_file = st.file_uploader("Upload Image 2", type=["png", "jpg", "jpeg"])

if run:
    if not img1_file or not img2_file:
        st.error("Please upload BOTH images.")
    else:
        try:
            img1 = uploaded_to_bgr(img1_file)
            img2 = uploaded_to_bgr(img2_file)

            mask, annotated, regions = compare_images(
                img1, img2,
                diff_threshold=diff_threshold,
                min_area=min_area
            )

            st.success(f"Detected regions: {len(regions)}")

            st.subheader("Original Images")
            col1, col2 = st.columns(2)
            with col1:
                st.image(bgr_to_rgb(img1), caption="Image 1", use_container_width=True)
            with col2:
                st.image(bgr_to_rgb(img2), caption="Image 2", use_container_width=True)

            st.subheader("Comparison Result")
            col3, col4 = st.columns(2)
            with col3:
                st.image(mask, caption="Change Mask (white = change)", use_container_width=True)
            with col4:
                st.image(bgr_to_rgb(annotated), caption="Differences Highlighted", use_container_width=True)

            if regions:
                st.subheader("Detected Regions")
                st.dataframe(regions, use_container_width=True)

        except Exception as e:
            st.error(f"Error during processing: {e}")

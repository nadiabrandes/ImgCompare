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


# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Image Compare",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🖼️ Compare Any Two Images")

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.header("Settings")

    diff_threshold = st.slider(
        "Difference sensitivity",
        min_value=0,
        max_value=255,
        value=45,
        step=1
    )

    min_area = st.slider(
        "Minimum region size",
        min_value=0,
        max_value=50000,
        value=1200,
        step=100
    )

    run = st.button("🔍 Compare", use_container_width=True)

# ------------------ UPLOAD AREA ------------------
upload_col1, upload_col2 = st.columns(2)

with upload_col1:
    img1_file = st.file_uploader(
        "Upload Image 1",
        type=["png", "jpg", "jpeg"]
    )

with upload_col2:
    img2_file = st.file_uploader(
        "Upload Image 2",
        type=["png", "jpg", "jpeg"]
    )

# ------------------ PROCESS ------------------
if run:
    if not img1_file or not img2_file:
        st.error("Please upload BOTH images.")
    else:
        img1 = uploaded_to_bgr(img1_file)
        img2 = uploaded_to_bgr(img2_file)

        mask, annotated, regions = compare_images(
            img1,
            img2,
            diff_threshold=diff_threshold,
            min_area=min_area
        )

        st.success(f"Detected regions: {len(regions)}")

        # ---------- ORIGINAL IMAGES ----------
        st.subheader("Original Images")
        col1, col2 = st.columns(2)

        with col1:
            st.image(
                bgr_to_rgb(img1),
                caption="Image 1",
                use_container_width=True
            )

        with col2:
            st.image(
                bgr_to_rgb(img2),
                caption="Image 2",
                use_container_width=True
            )

        # ---------- RESULTS ----------
        st.subheader("Comparison Result")
        col3, col4 = st.columns(2)

        with col3:
            st.image(
                mask,
                caption="Change Mask (white = change)",
                use_container_width=True
            )

        with col4:
            st.image(
                bgr_to_rgb(annotated),
                caption="Differences Highlighted",
                use_container_width=True
            )

        # ---------- REGIONS TABLE ----------
        if regions:
            st.subheader("Detected Regions")
            st.dataframe(regions, use_container_width=True)

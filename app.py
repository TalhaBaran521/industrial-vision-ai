"""Local Streamlit image preprocessing interface."""

from io import BytesIO

import streamlit as st
from PIL import Image

from industrial_vision.vision import preprocessing as ops
from industrial_vision.vision.image_loader import load_image

st.set_page_config(page_title="Industrial Vision AI", layout="wide")
st.title("INDUSTRIAL VISION AI")
st.caption("Image preprocessing · CPU · Research prototype")
max_bytes = st.sidebar.number_input("Maximum upload bytes", 1, 50_000_000, 10_000_000)
max_pixels = st.sidebar.number_input("Maximum pixels", 1, 40_000_000, 20_000_000)
operation = st.sidebar.selectbox(
    "Operation",
    [
        "Grayscale",
        "Gaussian blur",
        "Median blur",
        "CLAHE",
        "Canny",
        "Threshold",
        "Morphology",
        "Resize",
    ],
)
kernel = st.sidebar.select_slider("Kernel size", [3, 5, 7, 9], value=5)
threshold = st.sidebar.slider("Threshold", 0, 255, 127)
low, high = st.sidebar.slider("Canny thresholds", 0, 255, (50, 150))
clip = st.sidebar.slider("CLAHE clip limit", 0.1, 10.0, 2.0)
morphology = st.sidebar.selectbox("Morphology", ["open", "close"])
width = st.sidebar.number_input("Resize width", 1, 4096, 512)
height = st.sidebar.number_input("Resize height", 1, 4096, 512)
upload = st.file_uploader("PNG or JPEG image", type=["png", "jpg", "jpeg"])
if upload is not None:
    try:
        image, metadata = load_image(upload.getvalue(), max_bytes, max_pixels)
        operations = {
            "Grayscale": lambda: ops.convert_to_grayscale(image),
            "Gaussian blur": lambda: ops.apply_gaussian_blur(image, kernel),
            "Median blur": lambda: ops.apply_median_blur(image, kernel),
            "CLAHE": lambda: ops.apply_clahe(image, clip),
            "Canny": lambda: ops.detect_canny_edges(image, low, high),
            "Threshold": lambda: ops.apply_threshold(image, threshold),
            "Morphology": lambda: ops.apply_morphology(
                ops.apply_threshold(image, threshold), morphology, kernel
            ),
            "Resize": lambda: ops.resize_image(image, width, height),
        }
        result = operations[operation]()
        st.caption(
            f"Source dimensions {metadata.width} × {metadata.height} · {metadata.format}"
        )
        left, right = st.columns(2)
        left.image(image, caption="Original (EXIF oriented)")
        right.image(result, caption=operation)
        buffer = BytesIO()
        Image.fromarray(result).save(buffer, format="PNG")
        st.download_button(
            "Download result PNG", buffer.getvalue(), "processed.png", "image/png"
        )
    except ValueError as error:
        st.error(str(error))

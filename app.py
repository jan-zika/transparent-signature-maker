# Transparent Signature Maker

import streamlit as st
import cv2
import numpy as np
from io import BytesIO
from PIL import Image as PILImage

# Core function: convert a scanned signature to transparent PNG
def create_signature_png(image_bgr, threshold_value=150, tint_color=None):
    """
    Convert an uploaded signature image to PNG with transparency.

    Parameters
    ----------
    image_bgr : np.ndarray
        Input image in BGR format (as read by OpenCV).
    threshold_value : int
        Value used to separate background and signature.
    tint_color : tuple or None
        Optional BGR tuple (e.g. (255, 0, 0)) to tint the signature color.

    Returns
    -------
    np.ndarray
        RGBA image with alpha channel.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)

    color = image_bgr.copy()
    if tint_color:
        tint = np.full_like(color, tint_color, dtype=np.uint8)
        color = cv2.addWeighted(color, 1, tint, 0.5, 0)

    b, g, r = cv2.split(color)
    rgba = cv2.merge([b, g, r, alpha])
    return rgba


# Streamlit interface
st.set_page_config(page_title="Transparent Signature Maker", layout="centered")
st.title("Transparent Signature Maker")

st.write(
    "Upload a scanned signature image (JPG or PNG). "
    "The application will remove the background and create a transparent PNG "
    "suitable for inserting into electronic documents."
)

uploaded_file = st.file_uploader("Upload a signature image", type=["jpg", "jpeg", "png"])

threshold_value = st.slider(
    "Threshold value (adjust for best background removal)", 100, 250, 150
)

tint_choice = st.selectbox(
    "Optional color tint", ["None", "Blue", "Black", "Red"]
)
tints = {"None": None, "Blue": (255, 0, 0), "Black": (0, 0, 0), "Red": (0, 0, 255)}

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.subheader("Original Image")
    st.image(
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
        caption="Uploaded signature",
        use_container_width=True,
    )

    result_rgba = create_signature_png(image_bgr, threshold_value, tints[tint_choice])

    st.subheader("Resulting Transparent PNG")
    st.image(
        cv2.cvtColor(result_rgba, cv2.COLOR_BGRA2RGBA),
        caption="Processed transparent signature",
        use_container_width=True,
    )

    result_pil = PILImage.fromarray(cv2.cvtColor(result_rgba, cv2.COLOR_BGRA2RGBA))
    buf = BytesIO()
    result_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="Download Transparent Signature (PNG)",
        data=byte_im,
        file_name="transparent_signature.png",
        mime="image/png",
    )

else:
    st.info("Please upload a scanned signature image to begin.")

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="Transparent Signature Maker",
    layout="wide"
)

# Initialize session state
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'cropped_image' not in st.session_state:
    st.session_state.cropped_image = None

# ============================================================
# Helper Functions
# ============================================================

def validate_image(uploaded_file):
    """Validate uploaded file is an image."""
    try:
        img = Image.open(uploaded_file)
        img.verify()
        return True
    except Exception:
        return False


def load_image(uploaded_file):
    """Load image from uploaded file as numpy array."""
    try:
        image = Image.open(uploaded_file)
        return np.array(image)
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None


def auto_crop_signature(image, padding=10):
    """Automatically crop signature region from image."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image.copy()
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            x_min, y_min = image.shape[1], image.shape[0]
            x_max, y_max = 0, 0
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x + w), max(y_max, y + h)
            x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
            x_max, y_max = min(image.shape[1], x_max + padding), min(image.shape[0], y_max + padding)
            return image[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)

        return image, (0, 0, image.shape[1], image.shape[0])
    except Exception as e:
        st.error(f"Error in auto-cropping: {str(e)}")
        return image, (0, 0, image.shape[1], image.shape[0])


def manual_crop(image, x_start, y_start, x_end, y_end):
    """Crop image manually based on slider values."""
    try:
        return image[y_start:y_end, x_start:x_end]
    except Exception as e:
        st.error(f"Error in manual cropping: {str(e)}")
        return image


def apply_threshold(image, threshold_value=150):
    """Simple and robust threshold for scanned signatures."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image.copy()
        _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
        return mask
    except Exception as e:
        st.error(f"Error in thresholding: {str(e)}")
        return None


def refine_edges(mask, kernel_size=2):
    """Apply morphological operations to refine edges."""
    try:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        if kernel_size > 1:
            mask = cv2.dilate(mask, kernel, iterations=1)
        return mask
    except Exception as e:
        st.error(f"Error in edge refinement: {str(e)}")
        return mask


def create_transparent_signature(image, mask, ink_color):
    """Create transparent PNG with chosen ink color."""
    try:
        hex_color = ink_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        h, w = mask.shape
        transparent = np.zeros((h, w, 4), dtype=np.uint8)
        transparent[mask > 0] = (*rgb, 255)
        return transparent
    except Exception as e:
        st.error(f"Error creating transparent image: {str(e)}")
        return None


def process_signature(image, crop_bounds, threshold_value, ink_color, edge_refinement, kernel_size):
    """Complete processing pipeline."""
    try:
        if crop_bounds:
            image = manual_crop(image, *crop_bounds)
        mask = apply_threshold(image, threshold_value)
        if mask is None:
            return None
        if edge_refinement:
            mask = refine_edges(mask, kernel_size)
        transparent = create_transparent_signature(image, mask, ink_color)
        return transparent
    except Exception as e:
        st.error(f"Error processing signature: {str(e)}")
        return None


def image_to_bytes(image_array):
    """Convert numpy array to bytes for download."""
    try:
        img = Image.fromarray(image_array)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()
    except Exception as e:
        st.error(f"Error converting image: {str(e)}")
        return None


# ============================================================
# Main App
# ============================================================

def main():
    st.title("Transparent Signature Maker")
    st.markdown("""
    Convert a scanned handwritten signature into a clean, transparent PNG.
    Automatically removes the paper background and allows custom ink color.

    **Usage:**
    1. Upload a signature image (JPG or PNG)
    2. Adjust cropping if needed
    3. Fine-tune background removal sensitivity
    4. Choose ink color
    5. Download the transparent PNG
    """)

    with st.sidebar:
        st.header("Upload and Adjust")
        uploaded_file = st.file_uploader("Upload a signature image", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            if not validate_image(uploaded_file):
                st.error("Invalid file type. Please upload a valid image.")
                return
            image = load_image(uploaded_file)
            if image is None:
                return

            st.session_state.original_image = image
            st.success("Image uploaded successfully!")

            st.header("Cropping")
            auto_crop = st.checkbox("Auto-crop signature", value=True)
            if auto_crop:
                cropped, bounds = auto_crop_signature(image, padding=10)
                st.session_state.cropped_image = cropped
                crop_bounds = None
            else:
                st.subheader("Manual crop")
                col1, col2 = st.columns(2)
                with col1:
                    x_start = st.slider("Left", 0, image.shape[1], 0)
                    y_start = st.slider("Top", 0, image.shape[0], 0)
                with col2:
                    x_end = st.slider("Right", 0, image.shape[1], image.shape[1])
                    y_end = st.slider("Bottom", 0, image.shape[0], image.shape[0])
                crop_bounds = (x_start, y_start, x_end, y_end)
                st.session_state.cropped_image = manual_crop(image, *crop_bounds)

            st.header("Background Removal")
            threshold_value = st.slider(
                "Background removal sensitivity",
                80, 220, 150,
                help="Move right to remove more of the paper background."
            )

            st.header("Ink Color")
            ink_color = st.color_picker("Select ink color", "#000000")

            st.header("Edge Refinement")
            edge_refinement = st.checkbox("Smooth edges", value=True)
            kernel_size = st.slider("Smoothing strength", 1, 5, 2) if edge_refinement else 2

            # Automatically process image
            processed = process_signature(
                st.session_state.cropped_image if auto_crop else image,
                crop_bounds if not auto_crop else None,
                threshold_value, ink_color, edge_refinement, kernel_size
            )
            if processed is not None:
                st.session_state.processed_image = processed

    # Main content area
    if st.session_state.original_image is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(st.session_state.original_image, width="stretch")

        with col2:
            st.subheader("Processed Transparent Signature")
            if st.session_state.processed_image is not None:
                # True transparency preview (checkerboard + alpha compositing)
                rgba = st.session_state.processed_image
                h, w, _ = rgba.shape

                # Create checkerboard background
                tile = 20
                pattern = np.array([[240, 200], [200, 240]], dtype=np.uint8)
                tiles_y = int(np.ceil(h / (2 * tile)))
                tiles_x = int(np.ceil(w / (2 * tile)))
                cb = np.tile(pattern, (tiles_y * tile, tiles_x * tile))
                cb = cb[:h, :w]
                checker_bg = cv2.merge([cb, cb, cb]).astype(np.uint8)

                # Extract alpha and normalize
                alpha = rgba[:, :, 3].astype(float) / 255.0
                rgb = rgba[:, :, :3].astype(float)

                # Proper alpha compositing over checkerboard
                composite = (rgb * alpha[..., None] + checker_bg * (1 - alpha[..., None])).astype(np.uint8)

                st.image(composite, width="stretch", caption="Transparency preview")

                # Download
                img_bytes = image_to_bytes(rgba)
                if img_bytes:
                    filename = uploaded_file.name.rsplit('.', 1)[0] + "_transparent.png"
                    st.download_button(
                        "Download Transparent PNG",
                        data=img_bytes,
                        file_name=filename,
                        mime="image/png",
                        use_container_width=True
                    )
            else:
                st.info("Adjust parameters to generate the transparent signature.")


if __name__ == "__main__":
    main()

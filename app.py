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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None

# ============================================================
# Helper Functions
# ============================================================

def validate_image(uploaded_file):
    """Validate uploaded file is an image."""
    try:
        img = Image.open(uploaded_file)
        img.verify()
        uploaded_file.seek(0)  # Reset file pointer after verify
        return True
    except Exception:
        return False


def load_image(uploaded_file):
    """Load image from uploaded file as numpy array."""
    try:
        image = Image.open(uploaded_file).convert('RGB')
        return np.array(image)
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None


def auto_crop_signature(image, padding=20):
    """
    Reliably auto-crop signature region from image.
    Uses adaptive thresholding and contour analysis to find the main signature area.
    """
    try:
        # Ensure we're working with RGB
        if len(image.shape) == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply adaptive thresholding for better results on varying backgrounds
        binary = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV,
            blockSize=21,
            C=10
        )
        
        # Apply slight blur to connect nearby components
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image
        
        # Filter out tiny noise (less than 0.1% of image area)
        h, w = gray.shape
        min_area = h * w * 0.001
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        if not valid_contours:
            return image
        
        # Find the bounding box of all valid contours
        all_points = np.vstack(valid_contours)
        x, y, w, h = cv2.boundingRect(all_points)
        
        # Add padding
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        # Crop the image
        cropped = image[y:y+h, x:x+w]
        
        # Ensure we return something valid
        if cropped.size == 0:
            return image
            
        return cropped
        
    except Exception as e:
        st.warning(f"Auto-crop failed, using original image: {str(e)}")
        return image


def manual_crop(image, left, top, right, bottom):
    """Crop image manually based on pixel margins."""
    try:
        h, w = image.shape[:2]
        
        # Convert margins to coordinates
        x_start = max(0, left)
        y_start = max(0, top)
        x_end = max(x_start + 1, w - right)
        y_end = max(y_start + 1, h - bottom)
        
        return image[y_start:y_end, x_start:x_end]
    except Exception as e:
        st.error(f"Error in manual cropping: {str(e)}")
        return image


def process_signature(image, threshold_value=150, ink_color="#000000", smooth_edges=True, smooth_strength=2):
    """
    Convert image to transparent signature with specified ink color.
    """
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply threshold to create mask
        _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
        
        # Apply edge smoothing if requested
        if smooth_edges and smooth_strength > 1:
            kernel = np.ones((smooth_strength, smooth_strength), np.uint8)
            # Clean up small noise
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            # Close small gaps
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            # Slight dilation to restore stroke thickness
            if smooth_strength > 2:
                mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Parse ink color
        hex_color = ink_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        # Create RGBA image
        h, w = mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Set ink color where mask is active
        rgba[mask > 0] = [r, g, b, 255]
        
        return rgba
        
    except Exception as e:
        st.error(f"Error processing signature: {str(e)}")
        return None


def create_preview(rgba_image, bg_type="white"):
    """
    Create a preview of the transparent signature.
    Dynamically scales the checkerboard size so it appears consistent
    regardless of the image's pixel resolution or Streamlit canvas size.
    """
    try:
        h, w = rgba_image.shape[:2]

        if bg_type == "checkerboard":
            # === Dynamic checkerboard scaling ===
            base_tile = 20             # logical tile size (visible pixel size)
            display_width_px = 800     # approximate display width in Streamlit
            scale_factor = max(1, int((w / display_width_px) * base_tile))
            tile_size = scale_factor

            # Build checkerboard background
            tiles_x = (w + tile_size - 1) // tile_size
            tiles_y = (h + tile_size - 1) // tile_size
            checkerboard = np.zeros((h, w, 3), dtype=np.uint8)

            light_color = [240, 240, 240]
            dark_color = [200, 200, 200]

            for i in range(tiles_y):
                for j in range(tiles_x):
                    color = light_color if (i + j) % 2 == 0 else dark_color
                    y1, x1 = i * tile_size, j * tile_size
                    y2, x2 = min(y1 + tile_size, h), min(x1 + tile_size, w)
                    checkerboard[y1:y2, x1:x2] = color

            background = checkerboard
        else:
            # Plain white background
            background = np.ones((h, w, 3), dtype=np.uint8) * 255

        # === Alpha compositing ===
        alpha = rgba_image[:, :, 3:4].astype(float) / 255.0
        rgb = rgba_image[:, :, :3].astype(float)

        composite = rgb * alpha + background.astype(float) * (1 - alpha)
        composite = composite.astype(np.uint8)

        return composite

    except Exception as e:
        st.error(f"Error creating preview: {str(e)}")
        return None


def image_to_bytes(image_array):
    """Convert numpy array to bytes for download."""
    try:
        img = Image.fromarray(image_array, mode='RGBA')
        buf = io.BytesIO()
        img.save(buf, format='PNG', optimize=True)
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
    Convert your scanned signature into a clean, transparent PNG ready for digital documents.
    
    **How to use:**
    1. Upload a photo or scan of your signature
    2. Adjust cropping to focus on the signature
    3. Fine-tune the background removal
    4. Choose your preferred ink color
    5. Download the transparent PNG
    """)
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### ⬆︎ Upload Signature")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear photo or scan of your signature"
        )

        
        if uploaded_file is not None:
            if not validate_image(uploaded_file):
                st.error("❌ Invalid file. Please upload a valid image.")
                return
            
            image = load_image(uploaded_file)
            if image is None:
                return
            
            st.session_state.original_image = image
            st.success("✅ Image uploaded successfully!")
            
            # Cropping section
            st.markdown("### ✂︎ Cropping")
            crop_mode = st.radio(
                "Crop mode",
                ["Auto-crop", "Manual crop", "No crop"],
                help="Auto-crop tries to detect the signature area automatically"
            )
            
            if crop_mode == "Auto-crop":
                cropped = auto_crop_signature(image, padding=20)
            elif crop_mode == "Manual crop":
                st.markdown("##### Adjust margins (pixels)")
                col1, col2 = st.columns(2)
                with col1:
                    left = st.number_input("Left", 0, image.shape[1]//2, 0, 10)
                    top = st.number_input("Top", 0, image.shape[0]//2, 0, 10)
                with col2:
                    right = st.number_input("Right", 0, image.shape[1]//2, 0, 10)
                    bottom = st.number_input("Bottom", 0, image.shape[0]//2, 0, 10)
                cropped = manual_crop(image, left, top, right, bottom)
            else:
                cropped = image
            
            # Processing settings
            st.markdown("### ⚙︎ Signature Settings")
            
            threshold = st.slider(
                "Background removal",
                min_value=100,
                max_value=200,
                value=150,
                step=5,
                help="Lower values keep more detail, higher values remove more background"
            )
            
            ink_color = st.color_picker(
                "Ink color",
                value="#000000",
                help="Choose the color for your signature"
            )
            
            st.markdown("### ✧ Enhancement")
            smooth_edges = st.checkbox("Smooth edges", value=True)
            smooth_strength = 2
            if smooth_edges:
                smooth_strength = st.slider(
                    "Smoothing strength",
                    min_value=2,
                    max_value=5,
                    value=2,
                    help="Higher values create smoother edges"
                )
            
            # Preview background
            st.markdown("### 👁 Preview Options")
            preview_bg = st.radio(
                "Preview background",
                ["White", "Checkerboard"],
                help="Choose how to preview transparency"
            )
            
            # Process the signature
            processed = process_signature(
                cropped,
                threshold_value=threshold,
                ink_color=ink_color,
                smooth_edges=smooth_edges,
                smooth_strength=smooth_strength
            )
            
            if processed is not None:
                st.session_state.processed_image = processed
    
    # Main content area
    if st.session_state.original_image is not None:
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("### Original Image")
            st.image(st.session_state.original_image, use_container_width=True)
        
        with col2:
            st.markdown("### Processed Signature")
            
            if st.session_state.processed_image is not None:
                # Create preview
                preview_bg_type = "checkerboard" if "Checkerboard" in preview_bg else "white"
                preview = create_preview(st.session_state.processed_image, preview_bg_type)
                
                if preview is not None:
                    st.image(preview, use_container_width=True)
                    
                    # Download section
                    st.markdown("---")
                    img_bytes = image_to_bytes(st.session_state.processed_image)
                    if img_bytes:
                        filename = uploaded_file.name.rsplit('.', 1)[0] + "_signature.png"
                        col_dl1, col_dl2 = st.columns(2)
                        with col_dl1:
                            st.download_button(
                                label="📥 Download PNG",
                                data=img_bytes,
                                file_name=filename,
                                mime="image/png",
                                use_container_width=True,
                                type="primary"
                            )
                        with col_dl2:
                            file_size = len(img_bytes) / 1024
                            st.metric("File size", f"{file_size:.1f} KB")
                else:
                    st.error("Failed to create preview")
            else:
                st.info("Adjust the settings in the sidebar to process your signature")
    else:
        # Welcome screen
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info("👈 Upload a signature image in the sidebar to get started")
            
            with st.expander("ℹ️ Tips for best results"):
                st.markdown("""
                - Use a white or light-colored paper
                - Sign with a dark pen (black or blue works best)
                - Ensure good lighting without shadows
                - Keep the signature flat and straight
                - Higher resolution images produce better results
                """)


if __name__ == "__main__":
    main()

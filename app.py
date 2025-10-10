# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Transparent Signature Maker",
    page_icon="✍️",
    layout="wide"
)

# Initialize session state
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'cropped_image' not in st.session_state:
    st.session_state.cropped_image = None

# ==================== Helper Functions ====================

def validate_image(uploaded_file):
    """Validate uploaded file is an image."""
    try:
        img = Image.open(uploaded_file)
        img.verify()
        return True
    except:
        return False

def load_image(uploaded_file):
    """Load image from uploaded file."""
    try:
        image = Image.open(uploaded_file)
        return np.array(image)
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

def auto_crop_signature(image, padding=10):
    """Automatically crop signature region from image."""
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply threshold to find signature region
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get bounding box of all contours
            x_min, y_min = image.shape[1], image.shape[0]
            x_max, y_max = 0, 0
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)
            
            # Add padding
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(image.shape[1], x_max + padding)
            y_max = min(image.shape[0], y_max + padding)
            
            return image[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)
        
        return image, (0, 0, image.shape[1], image.shape[0])
    except Exception as e:
        st.error(f"Error in auto-cropping: {str(e)}")
        return image, (0, 0, image.shape[1], image.shape[0])

def manual_crop(image, x_start, y_start, x_end, y_end):
    """Manually crop image based on slider values."""
    try:
        return image[y_start:y_end, x_start:x_end]
    except Exception as e:
        st.error(f"Error in manual cropping: {str(e)}")
        return image

def apply_threshold(image, method='adaptive_gaussian', threshold_value=127, block_size=11, c=2):
    """Apply different thresholding methods to remove background."""
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        if method == 'adaptive_gaussian':
            # Ensure block_size is odd
            block_size = block_size if block_size % 2 == 1 else block_size + 1
            mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, block_size, c)
        elif method == 'adaptive_mean':
            block_size = block_size if block_size % 2 == 1 else block_size + 1
            mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                        cv2.THRESH_BINARY_INV, block_size, c)
        elif method == 'otsu':
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:  # global
            _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
        
        return mask
    except Exception as e:
        st.error(f"Error in thresholding: {str(e)}")
        return None

def refine_edges(mask, kernel_size=2):
    """Apply morphological operations to refine edges."""
    try:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Close gaps
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Optional: Slight dilation for smoother edges
        if kernel_size > 1:
            mask = cv2.dilate(mask, kernel, iterations=1)
        
        return mask
    except Exception as e:
        st.error(f"Error in edge refinement: {str(e)}")
        return mask

def create_transparent_signature(image, mask, ink_color):
    """Create transparent PNG with custom ink color."""
    try:
        # Parse hex color
        hex_color = ink_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        # Create RGBA image
        h, w = mask.shape
        transparent = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Set ink color where mask is white (signature pixels)
        transparent[mask > 0] = (*rgb, 255)
        
        return transparent
    except Exception as e:
        st.error(f"Error creating transparent image: {str(e)}")
        return None

def process_signature(image, crop_bounds, method, threshold_value, block_size, c, 
                      ink_color, edge_refinement, kernel_size):
    """Complete signature processing pipeline."""
    try:
        # Manual crop if bounds provided
        if crop_bounds:
            image = manual_crop(image, *crop_bounds)
        
        # Apply threshold
        mask = apply_threshold(image, method, threshold_value, block_size, c)
        
        if mask is None:
            return None
        
        # Refine edges if enabled
        if edge_refinement:
            mask = refine_edges(mask, kernel_size)
        
        # Create transparent signature
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

# ==================== Main App ====================

def main():
    # Title and description
    st.title("✍️ Transparent Signature Maker")
    st.markdown("""
    Transform your scanned handwritten signature into a professional transparent PNG with customizable ink color.
    Perfect for electronic documents, PDFs, contracts, and digital forms.
    
    **Instructions:**
    1. Upload your signature image (JPG or PNG)
    2. Adjust cropping if needed
    3. Select background removal method and fine-tune settings
    4. Choose your desired ink color
    5. Download your transparent signature
    """)
    
    # Sidebar controls
    with st.sidebar:
        st.header("📤 Upload Signature")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a scanned or photographed signature"
        )
        
        if uploaded_file:
            if not validate_image(uploaded_file):
                st.error("Invalid image file. Please upload a valid JPG or PNG.")
                return
            
            # Load image
            image = load_image(uploaded_file)
            if image is None:
                return
            
            st.session_state.original_image = image
            
            st.success("✅ Image uploaded successfully!")
            
            # Cropping controls
            st.header("✂️ Cropping")
            
            auto_crop = st.checkbox("Auto-crop signature", value=True)
            
            if auto_crop:
                cropped, bounds = auto_crop_signature(image, padding=10)
                st.session_state.cropped_image = cropped
                crop_bounds = None
            else:
                st.subheader("Manual Crop Adjustments")
                col1, col2 = st.columns(2)
                with col1:
                    x_start = st.slider("Left", 0, image.shape[1], 0)
                    y_start = st.slider("Top", 0, image.shape[0], 0)
                with col2:
                    x_end = st.slider("Right", 0, image.shape[1], image.shape[1])
                    y_end = st.slider("Bottom", 0, image.shape[0], image.shape[0])
                
                crop_bounds = (x_start, y_start, x_end, y_end)
                st.session_state.cropped_image = manual_crop(image, *crop_bounds)
            
            # Thresholding controls
            st.header("🎯 Background Removal")
            
            method = st.selectbox(
                "Thresholding Method",
                ["adaptive_gaussian", "adaptive_mean", "otsu", "global"],
                format_func=lambda x: {
                    "adaptive_gaussian": "Adaptive Gaussian (Recommended)",
                    "adaptive_mean": "Adaptive Mean",
                    "otsu": "Otsu's Method",
                    "global": "Global Fixed Threshold"
                }[x]
            )
            
            # Method-specific parameters
            threshold_value = 127
            block_size = 11
            c = 2
            
            if method in ["adaptive_gaussian", "adaptive_mean"]:
                block_size = st.slider("Block Size", 3, 51, 11, step=2,
                                      help="Size of pixel neighborhood for adaptive threshold")
                c = st.slider("Constant C", -10, 10, 2,
                            help="Constant subtracted from weighted mean")
            elif method == "global":
                threshold_value = st.slider("Threshold Value", 0, 255, 127,
                                           help="Fixed threshold value")
            
            # Color customization
            st.header("🎨 Ink Color")
            ink_color = st.color_picker("Choose ink color", "#000000")
            
            # Edge refinement
            st.header("✨ Edge Refinement")
            edge_refinement = st.checkbox("Enable edge smoothing", value=True)
            kernel_size = 2
            if edge_refinement:
                kernel_size = st.slider("Smoothing strength", 1, 5, 2,
                                       help="Higher values = smoother edges")
            
            # Process button
            if st.button("🔄 Process Signature", type="primary", use_container_width=True):
                with st.spinner("Processing signature..."):
                    processed = process_signature(
                        st.session_state.cropped_image if auto_crop else image,
                        crop_bounds if not auto_crop else None,
                        method, threshold_value, block_size, c,
                        ink_color, edge_refinement, kernel_size
                    )
                    
                    if processed is not None:
                        st.session_state.processed_image = processed
                        st.success("✅ Signature processed successfully!")
    
    # Main area - Preview
    if st.session_state.original_image is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(st.session_state.original_image, use_container_width=True)
        
        with col2:
            st.subheader("✨ Processed Signature")
            if st.session_state.processed_image is not None:
                # Display with checkerboard background to show transparency
                st.image(st.session_state.processed_image, use_container_width=True)
                
                # Download button
                img_bytes = image_to_bytes(st.session_state.processed_image)
                if img_bytes:
                    filename = uploaded_file.name.rsplit('.', 1)[0] + "_transparent.png"
                    st.download_button(
                        label="📥 Download Transparent PNG",
                        data=img_bytes,
                        file_name=filename,
                        mime="image/png",
                        use_container_width=True
                    )
            else:
                st.info("👆 Click 'Process Signature' to see the result")
    else:
        # Welcome message when no image is uploaded
        st.info("👈 Please upload a signature image to get started")

if __name__ == "__main__":
    main()

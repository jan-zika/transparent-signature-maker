# Transparent Signature Maker

A professional Streamlit application that converts scanned handwritten signatures into transparent PNGs with customizable ink colors. Perfect for creating digital signatures for electronic documents, PDFs, contracts, and forms.

## Features

- **Smart Background Removal**: Multiple thresholding algorithms to cleanly extract signatures from paper backgrounds
- **Automatic Cropping**: Intelligently detects and crops signature regions with manual fine-tuning options
- **Custom Ink Colors**: Change your signature color to match any document requirement
- **Edge Refinement**: Smooth jagged edges and reduce noise for professional results
- **Real-time Preview**: See changes instantly as you adjust parameters
- **High-Quality Export**: Download signatures as transparent PNGs preserving alpha channels

## Example

Transform your scanned signature from a paper background to a clean, transparent digital signature:

- **Original**: Standard scanned signature with paper background
- **Transparent**: Clean signature with transparent background and custom ink color

## Installation

```bash
# Clone the repository
git clone https://github.com/janzika/transparent-signature-maker.git
cd transparent-signature-maker

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the application locally:

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

1. Upload your signature image (JPG or PNG)
2. Choose automatic or manual cropping
3. Select a thresholding method for background removal
4. Customize the ink color
5. Enable edge refinement if needed
6. Click "Process Signature"
7. Download your transparent signature

## Deployment on Streamlit Cloud

1. Fork this repository to your GitHub account
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Select your forked repository
5. Set the main file path to `app.py`
6. Click "Deploy"

The app will be live at `https://[your-username]-transparent-signature-maker.streamlit.app`

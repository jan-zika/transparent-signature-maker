# Transparent Signature Maker

A lightweight Streamlit application that converts scanned handwritten signatures into transparent PNG files suitable for use in electronic documents such as PDFs, forms, or contracts.  
The tool removes paper backgrounds, preserves the ink strokes, and optionally applies a color tint.

---

## Features

- Upload scanned signature images (JPG or PNG)
- Adjust threshold interactively to remove the background
- Optional color tint for ink appearance (black, blue, or red)
- Live preview of the processed result
- Download a transparent PNG ready for document insertion

---

## Example

| Original Image | Transparent Result |
|----------------|--------------------|
| ![original](docs/sample_original.jpg) | ![transparent](docs/sample_result.png) |

*(Images above are illustrative examples.)*

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/transparent-signature-maker.git
cd transparent-signature-maker

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate        # On macOS or Linux
venv\Scripts\activate           # On Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app_signature.py

# Open the URL displayed in the terminal (usually http://localhost:8501)

# Transparent Signature Maker

A professional Streamlit application that converts scanned handwritten signatures into transparent PNGs with customizable ink colors.  
Perfect for creating clean digital signatures for electronic documents, PDFs, contracts, and forms.

---

## Features

- **Automatic Background Removal** – Removes paper backgrounds with an intuitive sensitivity slider  
- **Smart Cropping** – Automatically detects and crops the signature region, with manual adjustment available  
- **Custom Ink Colors** – Choose any ink color using a built-in color picker  
- **Edge Refinement** – Smooth jagged edges and reduce background noise  
- **Instant Preview** – Updates the transparent preview automatically as parameters change  
- **Checkerboard Transparency View** – Displays results on a professional transparency grid  
- **One-Click Export** – Download a high-quality PNG file with a preserved alpha channel

---

## Example

Transform your scanned signature from a photographed or scanned paper into a clean transparent overlay:

| Original | Transparent Result |
|-----------|--------------------|
| ![original](docs/sample_original.jpg) | ![transparent](docs/sample_result.png) |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/transparent-signature-maker.git
cd transparent-signature-maker

# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

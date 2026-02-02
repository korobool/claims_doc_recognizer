# Document Recognition System

A local web application for document OCR (Optical Character Recognition) with image deskewing, text recognition, and document classification.

## Features

- **Image Upload**: Support for JPEG, PNG, TIFF, BMP formats
- **Deskewing**: Automatic document straightening using Projection Profile Analysis (OpenCV)
- **OCR Recognition**: Surya OCR with 90+ language support
- **Document Classification**: Hybrid CLIP + keyword-based classification
- **Interactive Editing**: Overlay for editing recognized text
- **Export**: Save results to JSON

## Installation

```bash
# Clone the repository
cd document_recognition_local

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install PyTorch with GPU support (auto-detects platform)
chmod +x install_pytorch.sh
./install_pytorch.sh

# Install other dependencies
pip install -r requirements.txt
```

### GPU Acceleration

The `install_pytorch.sh` script automatically detects your platform and installs the appropriate PyTorch version:

| Platform | Acceleration |
|----------|--------------|
| NVIDIA GPU (x86_64) | CUDA |
| NVIDIA GPU (ARM64/DGX) | CUDA via NVIDIA PyPI |
| Apple Silicon (M1/M2/M3) | MPS |
| CPU only | CPU |

**Manual PyTorch installation** (if script fails):

```bash
# NVIDIA GPU (x86_64):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# NVIDIA GPU (ARM64/DGX/Jetson):
pip install torch torchvision --index-url https://pypi.nvidia.com

# Apple Silicon or CPU:
pip install torch torchvision
```

**Note**: On first run, Surya OCR models will be downloaded automatically (~1.5GB). CLIP model (~150MB) downloads on first classification.

## Running

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open in browser: http://localhost:8000

## Usage

1. **Upload**: Click "Upload" or drag and drop images to the left panel
2. **Select**: Click on an image in the list
3. **Normalize**: Straighten a skewed document (optional)
4. **Recognize**: Run OCR recognition
5. **Edit**: Click on text in the overlay to edit
6. **Save**: Save results to a JSON file

## Project Structure

```
document_recognition_local/
├── app/
│   ├── main.py              # FastAPI application
│   ├── routers/api.py       # API endpoints
│   ├── services/
│   │   ├── ocr_service.py   # Surya OCR + CLIP classification
│   │   └── image_service.py # Deskew with OpenCV
│   ├── static/              # CSS, JS
│   └── templates/           # HTML
├── uploads/                 # Uploaded files
├── requirements.txt
├── TECHNICAL_SPEC.md        # Technical specification
├── DEVELOPER_GUIDE.md       # Developer guide with code examples
└── LICENSE_AND_PRICING.md   # Licensing and pricing analysis
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/upload` | POST | Upload images |
| `/api/normalize` | POST | Deskew image |
| `/api/recognize` | POST | OCR recognition + classification |
| `/api/recognize-region` | POST | OCR on selected region |
| `/api/image/{id}` | GET | Retrieve image |

## Requirements

- Python 3.10+
- PyTorch (automatically installed with surya-ocr)
- ~4GB RAM recommended for all models

## Documentation

- **[TECHNICAL_SPEC.md](TECHNICAL_SPEC.md)** - Complete technical specification
- **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)** - Developer guide with code examples
- **[LICENSE_AND_PRICING.md](LICENSE_AND_PRICING.md)** - Licensing and pricing analysis

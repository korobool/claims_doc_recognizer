# Technical Specification: Document Recognition System

## 1. Overview

A local web application for document OCR (Optical Character Recognition) with image deskewing, text recognition, document classification, and interactive result editing.

**Technology Stack:**
- **Backend**: Python 3.12+, FastAPI
- **OCR Engine**: Surya OCR (VikParuchuri/surya) - transformer-based multilingual OCR
- **Image Processing**: OpenCV, SciPy, NumPy
- **Document Classification**: OpenAI CLIP (clip-vit-base-patch32) + keyword-based hybrid
- **Frontend**: Vanilla JavaScript, HTML5, CSS3 (served via Jinja2 templates)

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Frontend (Web UI)                            │
├──────────────────┬──────────────────────────┬───────────────────────┤
│   Left Panel     │      Center Panel        │     Right Panel       │
│   - File Upload  │   - Image Viewer         │   - Document Class    │
│   - Image List   │   - Normalize Button     │   - JSON Output       │
│   - Thumbnails   │   - Recognize Button     │   - Inline Editing    │
│                  │   - Add BBox Button      │   - Save Button       │
│                  │   - Zoom Controls        │                       │
│                  │   - Text Overlay         │                       │
└──────────────────┴──────────────────────────┴───────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Backend (FastAPI)                               │
├─────────────────────────────────────────────────────────────────────┤
│  POST /api/upload          - Upload images                          │
│  POST /api/normalize       - Deskew document                        │
│  POST /api/recognize       - Full OCR + classification              │
│  POST /api/recognize-region - OCR on selected region                │
│  GET  /api/image/{id}      - Retrieve image                         │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Services Layer                               │
├─────────────────────────────────────────────────────────────────────┤
│  image_service.py:                                                  │
│    - deskew_image() - Projection Profile Analysis                   │
│    - rotate_image_no_crop() - Rotation without content loss         │
│                                                                     │
│  ocr_service.py:                                                    │
│    - recognize_text() - Surya OCR                                   │
│    - recognize_region() - Partial OCR                               │
│    - classify_document_hybrid() - CLIP + keyword classification     │
└─────────────────────────────────────────────────────────────────────┘
```

## 3. Core Algorithms

### 3.1 Document Deskew (Projection Profile Analysis)

The deskew algorithm corrects rotated document scans by finding the angle that maximizes horizontal text line separation.

**Algorithm Steps:**
1. Convert image to grayscale
2. Binarize using Otsu's threshold (inverted: text = white)
3. Test rotation angles from -15° to +15° (0.5° steps)
4. For each angle, compute horizontal projection (sum of pixels per row)
5. Calculate variance of projection profile
6. Best angle = maximum variance (clearest line separation)
7. Refine with 0.1° precision around best angle
8. Rotate original image using the detected angle

**Why Projection Profile Works:**
When text lines are perfectly horizontal, the projection profile shows distinct peaks (text rows) and valleys (inter-line gaps). This creates maximum variance. When text is skewed, peaks blur together, reducing variance.

### 3.2 OCR with Surya

Surya OCR is a transformer-based OCR engine that provides:
- Text detection (bounding boxes)
- Text recognition (content)
- Word-level segmentation
- Confidence scores
- Multi-language support (90+ languages)

**Output Structure:**
```json
{
  "text_lines": [
    {
      "text": "Recognized text content",
      "confidence": 0.95,
      "bbox": [x1, y1, x2, y2],
      "polygon": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
      "words": [
        {"text": "Recognized", "bbox": [...], "confidence": 0.96}
      ]
    }
  ],
  "image_bbox": [0, 0, width, height],
  "document_class": {
    "class": "Receipt",
    "confidence": 0.85,
    "method": "hybrid"
  }
}
```

### 3.3 Hybrid Document Classification

Classification combines two methods for robust results:

**Method 1: CLIP Image Classification**
- Model: `openai/clip-vit-base-patch32` (~150MB)
- Zero-shot classification using text descriptions
- Class descriptions:
  - "a photo of a receipt or invoice with prices and totals"
  - "a photo of a medical prescription or medication document"
  - "a photo of a form or application document with fields to fill"
  - "a photo of a contract or legal agreement document"
  - "a photo of an unknown or unclassified document"

**Method 2: Keyword-Based Text Classification**
- Analyzes recognized text for class-specific keywords
- Supports English and Russian keywords
- Weighted scoring based on keyword matches

**Hybrid Decision Logic:**
1. If both methods agree → high confidence (combined score)
2. If text-based has high confidence (≥0.5) → prefer text result
3. If CLIP has high confidence (≥0.3) → use image result
4. Fallback to any available result with reduced confidence

**Document Classes:**
- Receipt
- Medication Prescription
- Form
- Contract
- Undetected

## 4. Project Structure

```
document_recognition_local/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   ├── routers/
│   │   ├── __init__.py
│   │   └── api.py              # REST API endpoints
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ocr_service.py      # Surya OCR + CLIP classification
│   │   └── image_service.py    # Deskew and image processing
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css       # Dark theme UI styles
│   │   └── js/
│   │       └── app.js          # Frontend application logic
│   └── templates/
│       └── index.html          # Main HTML template
├── uploads/                     # Temporary image storage
├── requirements.txt
├── README.md
└── TECHNICAL_SPEC.md
```

## 5. API Reference

### POST /api/upload
Upload one or more images for processing.

**Request:** `multipart/form-data` with `files` field
**Response:**
```json
{
  "images": [
    {"id": "uuid-string", "filename": "document.jpg"}
  ]
}
```

### POST /api/normalize
Deskew a document image to make text horizontal.

**Request:**
```json
{"image_id": "uuid-string"}
```
**Response:**
```json
{
  "image_id": "uuid-string",
  "normalized": true,
  "angle": 2.5
}
```

### POST /api/recognize
Perform full OCR and document classification.

**Request:**
```json
{"image_id": "uuid-string"}
```
**Response:** Full OCR result with text_lines, image_bbox, and document_class

### POST /api/recognize-region
Perform OCR on a specific region (user-drawn bounding box).

**Request:**
```json
{
  "image_id": "uuid-string",
  "bbox": [x1, y1, x2, y2]
}
```
**Response:** OCR result for the specified region only

### GET /api/image/{image_id}
Retrieve the current version of an image (original or normalized).

**Response:** PNG image binary

## 6. Frontend Features

### 6.1 Image Management
- Drag-and-drop file upload
- Multiple file selection
- Thumbnail preview list
- Click to select active image

### 6.2 Image Viewer
- Zoom controls (25% - 300%)
- Keyboard shortcuts (Ctrl++, Ctrl+-, Ctrl+0)
- Slider and button controls

### 6.3 Text Overlay
- Bounding boxes displayed over recognized text
- Color-coded by confidence:
  - **Red**: confidence < 0.8
  - **Orange**: 0.8 ≤ confidence < 0.93
  - **Green**: confidence ≥ 0.93
- Click to select and highlight in JSON
- Double-click to edit text inline

### 6.4 Manual Bounding Box
- "Add BBox" button enters drawing mode
- Draw rectangle over image region
- Automatically removes overlapping boxes (>30% overlap)
- Triggers OCR on new region
- Results merged into existing data

### 6.5 Document Classification Display
- Shows detected document type with colored badge
- Displays confidence percentage
- Updates after each recognition

### 6.6 JSON Output
- Real-time JSON display
- Selected line highlighting
- Save to file functionality

## 7. Dependencies

```
# Web Framework
fastapi>=0.104.0
uvicorn>=0.24.0
python-multipart>=0.0.6
jinja2>=3.1.0
aiofiles>=23.0.0

# OCR Engine
surya-ocr>=0.4.0

# Image Processing
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
scipy>=1.10.0

# Classification
transformers>=4.36.0
torch>=2.0.0

# Utilities
requests>=2.31.0
```

## 8. Installation & Running

```bash
# Clone repository
git clone <repository-url>
cd document_recognition_local

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**First Run Notes:**
- Surya OCR models (~1.5GB) download automatically on first use
- CLIP model (~150MB) downloads on first classification
- Models are cached in `~/.cache/huggingface/`

## 9. Model Information

### Surya OCR
- **Repository**: https://github.com/VikParuchuri/surya
- **Architecture**: Transformer-based detection + recognition
- **Languages**: 90+ languages supported
- **License**: GPL-3.0

### OpenAI CLIP
- **Model**: `openai/clip-vit-base-patch32`
- **Documentation**: https://huggingface.co/openai/clip-vit-base-patch32
- **Architecture**: Vision Transformer (ViT-B/32)
- **Parameters**: ~150M
- **License**: MIT

## 10. Performance Considerations

- **Deskew**: ~0.5-1s per image (depends on resolution)
- **OCR**: ~2-10s per image (depends on text density)
- **Classification**: ~0.5s per image (after model loading)
- **First request**: Additional 5-30s for model loading
- **Memory**: ~2-4GB RAM recommended for all models

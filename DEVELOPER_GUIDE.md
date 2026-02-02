# Building a Local Document Recognition System with Python

A comprehensive guide to building a production-ready document OCR application with deskewing, text recognition, and AI-powered classification.

## Table of Contents

1. [Introduction](#introduction)
2. [Technology Choices](#technology-choices)
3. [Document Deskewing with Projection Profile Analysis](#document-deskewing-with-projection-profile-analysis)
4. [OCR with Surya](#ocr-with-surya)
5. [Hybrid Document Classification with CLIP](#hybrid-document-classification-with-clip)
6. [Building the REST API](#building-the-rest-api)
7. [Frontend Implementation](#frontend-implementation)
8. [Complete Code Reference](#complete-code-reference)
9. [Deployment and Performance](#deployment-and-performance)

---

## Introduction

This guide walks through building a complete document recognition system that runs entirely locally. The system provides:

- **Document deskewing** - Automatically straighten rotated scans
- **OCR** - Extract text with bounding boxes and confidence scores
- **Classification** - Identify document types (receipts, prescriptions, forms, contracts)
- **Interactive editing** - Manual corrections via web UI

All processing happens on-device with no cloud dependencies after initial model downloads.

---

## Technology Choices

### Why These Libraries?

| Component | Library | Rationale |
|-----------|---------|-----------|
| Web Framework | FastAPI | Async support, automatic OpenAPI docs, Pydantic validation |
| OCR Engine | Surya OCR | State-of-the-art accuracy, 90+ languages, open source |
| Image Processing | OpenCV + SciPy | Industry standard, comprehensive algorithms |
| Classification | CLIP | Zero-shot capability, no training required |
| Frontend | Vanilla JS | No build step, simple deployment |

### Dependencies

```python
# requirements.txt
fastapi>=0.104.0
uvicorn>=0.24.0
python-multipart>=0.0.6
surya-ocr>=0.4.0
opencv-python>=4.8.0
Pillow>=10.0.0
jinja2>=3.1.0
aiofiles>=23.0.0
numpy>=1.24.0
scipy>=1.10.0
transformers>=4.36.0
torch>=2.0.0
requests>=2.31.0
```

**Documentation Links:**
- FastAPI: https://fastapi.tiangolo.com/
- Surya OCR: https://github.com/VikParuchuri/surya
- OpenCV: https://docs.opencv.org/4.x/
- CLIP: https://huggingface.co/openai/clip-vit-base-patch32
- SciPy ndimage: https://docs.scipy.org/doc/scipy/reference/ndimage.html

---

## Document Deskewing with Projection Profile Analysis

### The Problem

Scanned or photographed documents are often slightly rotated. Even a 2-3° skew can significantly degrade OCR accuracy because text detection models expect horizontal baselines.

### Failed Approaches

Before arriving at the working solution, we tried:

1. **minAreaRect on all foreground pixels** - Computes the minimum bounding rectangle of all text, but this gives the angle of the text *block*, not the text *lines*. Fails for multi-line documents.

2. **Hough Line Transform** - Detects lines in edge-detected images. Too noisy for documents - picks up letter strokes, table borders, and other non-baseline features.

### The Solution: Projection Profile Analysis

This classical document analysis technique works reliably because it directly measures what we care about: horizontal text line alignment.

**Core Insight:** When text lines are perfectly horizontal, summing pixel values row-by-row creates a profile with sharp peaks (text rows) and valleys (gaps between lines). The variance of this profile is maximized when lines are horizontal.

```python
"""
Document deskew using Projection Profile Analysis.
"""
import cv2
import numpy as np
from scipy import ndimage


def deskew_image(image: np.ndarray, debug: bool = False) -> tuple[np.ndarray, float, dict]:
    """
    Deskew a document image to make text baseline horizontal.
    
    Args:
        image: Input BGR image as numpy array
        debug: If True, include debug information in output
        
    Returns:
        tuple: (corrected_image, angle_degrees, debug_info)
    """
    debug_info = {}
    
    # Step 1: Preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Otsu's binarization with inversion (text becomes white/foreground)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Step 2: Coarse angle search (-15° to +15°, 0.5° steps)
    angles = np.arange(-15, 15.5, 0.5)
    best_angle = 0.0
    best_variance = 0.0
    
    for angle in angles:
        # Rotate binary image (fast, nearest-neighbor interpolation)
        rotated = ndimage.rotate(binary, angle, reshape=False, order=0)
        
        # Horizontal projection: sum pixels in each row
        projection = np.sum(rotated, axis=1)
        
        # Variance measures "peakiness" of the profile
        variance = np.var(projection)
        
        if variance > best_variance:
            best_variance = variance
            best_angle = angle
    
    # Step 3: Fine refinement (0.1° precision)
    fine_angles = np.arange(best_angle - 0.5, best_angle + 0.55, 0.1)
    for angle in fine_angles:
        rotated = ndimage.rotate(binary, angle, reshape=False, order=0)
        projection = np.sum(rotated, axis=1)
        variance = np.var(projection)
        
        if variance > best_variance:
            best_variance = variance
            best_angle = angle
    
    # Step 4: Apply rotation if significant
    if abs(best_angle) < 0.2:
        return image.copy(), 0.0, debug_info
    
    rotated = rotate_image_no_crop(image, best_angle)
    return rotated, best_angle, debug_info
```

### Rotation Without Cropping

Standard rotation crops corners. For documents, we need to expand the canvas:

```python
def rotate_image_no_crop(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate image without cropping any content.
    
    Args:
        image: Input image
        angle: Rotation angle in degrees (positive = counter-clockwise)
        
    Returns:
        Rotated image with expanded canvas
    """
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new bounding box dimensions
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust translation to keep image centered
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # Apply rotation with border replication
    rotated = cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return rotated
```

**Key Parameters:**
- `INTER_CUBIC` - High-quality interpolation for the final rotation
- `BORDER_REPLICATE` - Extends edge pixels to fill corners (better than black borders)
- `order=0` in ndimage.rotate - Fast nearest-neighbor for angle search (quality doesn't matter)

---

## OCR with Surya

### Why Surya?

Surya OCR is a modern, transformer-based OCR engine that outperforms traditional approaches:

- **Architecture**: Uses separate detection and recognition models
- **Languages**: Supports 90+ languages out of the box
- **Output**: Provides line-level and word-level bounding boxes with confidence scores
- **Local**: Runs entirely on-device after model download

### Integration

```python
from PIL import Image
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from surya.foundation import FoundationPredictor
import io

# Lazy initialization (models are large, load once)
_foundation_predictor = None
_recognition_predictor = None
_detection_predictor = None


def get_predictors():
    """Lazy initialization of Surya OCR predictors."""
    global _foundation_predictor, _recognition_predictor, _detection_predictor
    
    if _recognition_predictor is None:
        _foundation_predictor = FoundationPredictor()
        _detection_predictor = DetectionPredictor()
        _recognition_predictor = RecognitionPredictor(_foundation_predictor)
    
    return _recognition_predictor, _detection_predictor


def recognize_text(image_bytes: bytes) -> dict:
    """
    Perform OCR on an image.
    
    Args:
        image_bytes: Image as bytes
        
    Returns:
        dict: OCR results with text_lines, bboxes, confidence scores
    """
    image = Image.open(io.BytesIO(image_bytes))
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    recognition_predictor, detection_predictor = get_predictors()
    
    # Run OCR pipeline
    predictions = recognition_predictor([image], det_predictor=detection_predictor)
    
    if not predictions or len(predictions) == 0:
        return {"text_lines": [], "image_bbox": [0, 0, image.width, image.height]}
    
    result = predictions[0]
    
    # Format output
    text_lines = []
    for line in result.text_lines:
        line_data = {
            "text": line.text,
            "confidence": line.confidence,
            "bbox": line.bbox,  # [x1, y1, x2, y2]
            "polygon": line.polygon if hasattr(line, 'polygon') else None,
        }
        
        # Include word-level data if available
        if hasattr(line, 'words') and line.words:
            line_data["words"] = [
                {
                    "text": word.text,
                    "bbox": word.bbox,
                    "confidence": word.confidence
                }
                for word in line.words
            ]
        
        text_lines.append(line_data)
    
    return {
        "text_lines": text_lines,
        "image_bbox": [0, 0, image.width, image.height]
    }
```

### Region-Based OCR

For user-drawn bounding boxes, we crop and OCR just that region:

```python
def recognize_region(image_bytes: bytes, bbox: list) -> dict:
    """
    Perform OCR on a specific region of an image.
    
    Args:
        image_bytes: Full image as bytes
        bbox: Region coordinates [x1, y1, x2, y2]
        
    Returns:
        dict: OCR results with coordinates adjusted to full image
    """
    image = Image.open(io.BytesIO(image_bytes))
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Crop region (with bounds checking)
    x1, y1, x2, y2 = [int(c) for c in bbox]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.width, x2)
    y2 = min(image.height, y2)
    
    cropped = image.crop((x1, y1, x2, y2))
    
    recognition_predictor, detection_predictor = get_predictors()
    predictions = recognition_predictor([cropped], det_predictor=detection_predictor)
    
    if not predictions or len(predictions) == 0:
        return {"text_lines": [], "region_bbox": bbox}
    
    result = predictions[0]
    
    # Adjust coordinates back to full image space
    text_lines = []
    for line in result.text_lines:
        line_bbox = [
            line.bbox[0] + x1,
            line.bbox[1] + y1,
            line.bbox[2] + x1,
            line.bbox[3] + y1
        ]
        
        line_data = {
            "text": line.text,
            "confidence": line.confidence,
            "bbox": line_bbox,
            "polygon": None,
        }
        
        # Adjust polygon coordinates
        if hasattr(line, 'polygon') and line.polygon:
            line_data["polygon"] = [
                [p[0] + x1, p[1] + y1] for p in line.polygon
            ]
        
        # Adjust word coordinates
        if hasattr(line, 'words') and line.words:
            line_data["words"] = [
                {
                    "text": word.text,
                    "bbox": [
                        word.bbox[0] + x1,
                        word.bbox[1] + y1,
                        word.bbox[2] + x1,
                        word.bbox[3] + y1
                    ],
                    "confidence": word.confidence
                }
                for word in line.words
            ]
        
        text_lines.append(line_data)
    
    return {"text_lines": text_lines, "region_bbox": bbox}
```

---

## Hybrid Document Classification with CLIP

### The Challenge

Document classification typically requires:
1. Large labeled datasets
2. Model training/fine-tuning
3. Retraining when adding new classes

### The Solution: Zero-Shot Classification with CLIP

CLIP (Contrastive Language-Image Pre-training) can classify images using natural language descriptions without any training on your specific classes.

**How CLIP Works:**
1. Encode the image into a feature vector
2. Encode text descriptions of each class into feature vectors
3. Compare image features to text features using cosine similarity
4. Highest similarity = predicted class

```python
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Lazy loading (model is ~150MB)
_clip_model = None
_clip_processor = None

# Natural language descriptions for each class
CLIP_CLASS_DESCRIPTIONS = [
    "a photo of a receipt or invoice with prices and totals",
    "a photo of a medical prescription or medication document",
    "a photo of a form or application document with fields to fill",
    "a photo of a contract or legal agreement document",
    "a photo of an unknown or unclassified document"
]

CLIP_CLASS_NAMES = ["Receipt", "Medication Prescription", "Form", "Contract", "Undetected"]


def get_clip_model():
    """Lazy initialization of CLIP model."""
    global _clip_model, _clip_processor
    
    if _clip_model is None:
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.eval()  # Inference mode
    
    return _clip_model, _clip_processor


def classify_image_with_clip(image: Image.Image) -> dict:
    """
    Classify document image using CLIP zero-shot classification.
    
    Args:
        image: PIL Image
        
    Returns:
        dict: {"class": "Receipt", "confidence": 0.85}
    """
    try:
        model, processor = get_clip_model()
        
        # Prepare inputs
        inputs = processor(
            text=CLIP_CLASS_DESCRIPTIONS,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        # Find best class
        best_idx = probs.argmax().item()
        best_prob = probs[0][best_idx].item()
        
        return {
            "class": CLIP_CLASS_NAMES[best_idx],
            "confidence": round(best_prob, 2)
        }
    except Exception as e:
        print(f"CLIP classification error: {e}")
        return {"class": "Undetected", "confidence": 0.0}
```

### Keyword-Based Text Classification

CLIP alone may miss document-specific terminology. We complement it with keyword matching:

```python
# Keywords for each document class (English + Russian)
DOCUMENT_KEYWORDS = {
    "Receipt": [
        "receipt", "total", "subtotal", "tax", "payment", "cash", "change",
        "qty", "price", "amount", "item", "чек", "итого", "сумма", "оплата"
    ],
    "Medication Prescription": [
        "prescription", "rx", "medication", "dose", "dosage", "tablet",
        "mg", "ml", "daily", "doctor", "patient", "pharmacy",
        "рецепт", "препарат", "доза", "таблетка", "врач", "аптека"
    ],
    "Form": [
        "form", "application", "name", "date", "signature", "address",
        "please fill", "required", "checkbox", "заявление", "форма", "анкета"
    ],
    "Contract": [
        "contract", "agreement", "party", "parties", "terms", "conditions",
        "hereby", "whereas", "witness", "signed", "договор", "контракт"
    ]
}


def classify_document(text_lines: list) -> dict:
    """
    Classify document based on recognized text keywords.
    
    Args:
        text_lines: List of OCR text line results
        
    Returns:
        dict: {"class": "Receipt", "confidence": 0.75}
    """
    if not text_lines:
        return {"class": "Undetected", "confidence": 0.0}
    
    # Combine all text
    full_text = " ".join([line.get("text", "") for line in text_lines]).lower()
    
    # Score each class by keyword matches
    scores = {}
    for doc_class, keywords in DOCUMENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in full_text)
        scores[doc_class] = score
    
    # Find best class
    max_class = max(scores.keys(), key=lambda k: scores[k])
    max_score = scores[max_class]
    
    if max_score == 0:
        return {"class": "Undetected", "confidence": 0.0}
    
    # Normalize confidence (30% keyword match = 100% confidence)
    total_keywords = len(DOCUMENT_KEYWORDS[max_class])
    confidence = min(max_score / (total_keywords * 0.3), 1.0)
    
    return {
        "class": max_class,
        "confidence": round(confidence, 2)
    }
```

### Hybrid Classification

Combine both methods for robust results:

```python
def classify_document_hybrid(image: Image.Image, text_lines: list) -> dict:
    """
    Hybrid classification combining CLIP (image) and keyword (text) methods.
    
    Args:
        image: PIL Image
        text_lines: OCR results
        
    Returns:
        dict: {"class": "Receipt", "confidence": 0.85, "method": "hybrid"}
    """
    # Get both classifications
    image_result = classify_image_with_clip(image)
    text_result = classify_document(text_lines)
    
    # Decision logic
    
    # Case 1: Both methods agree - high confidence
    if (image_result["class"] == text_result["class"] and 
        image_result["class"] != "Undetected"):
        combined_confidence = min(1.0, 
            (image_result["confidence"] + text_result["confidence"]) / 1.5)
        return {
            "class": image_result["class"],
            "confidence": round(combined_confidence, 2),
            "method": "hybrid"
        }
    
    # Case 2: Text-based has high confidence - prefer it
    if text_result["class"] != "Undetected" and text_result["confidence"] >= 0.5:
        return {
            "class": text_result["class"],
            "confidence": round(text_result["confidence"] * 0.9, 2),
            "method": "text"
        }
    
    # Case 3: CLIP has high confidence - use it
    if image_result["class"] != "Undetected" and image_result["confidence"] >= 0.3:
        return {
            "class": image_result["class"],
            "confidence": round(image_result["confidence"], 2),
            "method": "image"
        }
    
    # Case 4: Fallback to any result
    if text_result["class"] != "Undetected":
        return {
            "class": text_result["class"],
            "confidence": round(text_result["confidence"] * 0.7, 2),
            "method": "text"
        }
    
    if image_result["class"] != "Undetected":
        return {
            "class": image_result["class"],
            "confidence": round(image_result["confidence"] * 0.7, 2),
            "method": "image"
        }
    
    return {"class": "Undetected", "confidence": 0.0, "method": "none"}
```

---

## Building the REST API

### FastAPI Application Structure

```python
# app/main.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import os

from app.routers import api

app = FastAPI(
    title="Document Recognition",
    description="Local OCR and document classification system",
    version="1.0.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

app.include_router(api.router)


@app.get("/")
async def root(request: Request):
    """Serve the main web UI."""
    return templates.TemplateResponse("index.html", {"request": request})
```

### API Endpoints

```python
# app/routers/api.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List
import uuid
import os

from app.services.ocr_service import recognize_text, recognize_region
from app.services.image_service import normalize_image

router = APIRouter(prefix="/api", tags=["api"])

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory image store (use Redis/DB for production)
images_store: dict[str, dict] = {}


class ImageIdRequest(BaseModel):
    image_id: str


class RecognizeRegionRequest(BaseModel):
    image_id: str
    bbox: List[float]


@router.post("/upload")
async def upload_images(files: List[UploadFile] = File(...)):
    """Upload one or more images."""
    uploaded = []
    
    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            continue
        
        image_id = str(uuid.uuid4())
        content = await file.read()
        
        file_path = os.path.join(UPLOAD_DIR, f"{image_id}.png")
        with open(file_path, "wb") as f:
            f.write(content)
        
        images_store[image_id] = {
            "filename": file.filename,
            "path": file_path,
            "original_path": file_path
        }
        
        uploaded.append({"id": image_id, "filename": file.filename})
    
    if not uploaded:
        raise HTTPException(status_code=400, detail="No valid images uploaded")
    
    return {"images": uploaded}


@router.post("/normalize")
async def normalize(request: ImageIdRequest):
    """Deskew a document image."""
    image_id = request.image_id
    
    if image_id not in images_store:
        raise HTTPException(status_code=404, detail="Image not found")
    
    image_info = images_store[image_id]
    
    with open(image_info["path"], "rb") as f:
        image_bytes = f.read()
    
    normalized_bytes, angle = normalize_image(image_bytes)
    
    normalized_path = os.path.join(UPLOAD_DIR, f"{image_id}_normalized.png")
    with open(normalized_path, "wb") as f:
        f.write(normalized_bytes)
    
    images_store[image_id]["path"] = normalized_path
    
    return {"image_id": image_id, "normalized": True, "angle": angle}


@router.post("/recognize")
async def recognize(request: ImageIdRequest):
    """Perform OCR and classification on an image."""
    image_id = request.image_id
    
    if image_id not in images_store:
        raise HTTPException(status_code=404, detail="Image not found")
    
    image_info = images_store[image_id]
    
    with open(image_info["path"], "rb") as f:
        image_bytes = f.read()
    
    result = recognize_text(image_bytes)
    return result


@router.post("/recognize-region")
async def recognize_region_endpoint(request: RecognizeRegionRequest):
    """Perform OCR on a specific region."""
    image_id = request.image_id
    
    if image_id not in images_store:
        raise HTTPException(status_code=404, detail="Image not found")
    
    image_info = images_store[image_id]
    
    with open(image_info["path"], "rb") as f:
        image_bytes = f.read()
    
    result = recognize_region(image_bytes, request.bbox)
    return result


@router.get("/image/{image_id}")
async def get_image(image_id: str):
    """Retrieve an image by ID."""
    if image_id not in images_store:
        raise HTTPException(status_code=404, detail="Image not found")
    
    image_info = images_store[image_id]
    
    with open(image_info["path"], "rb") as f:
        image_bytes = f.read()
    
    return Response(content=image_bytes, media_type="image/png")
```

---

## Frontend Implementation

### Key Features

1. **Confidence-Based Coloring** - Bounding boxes colored by OCR confidence
2. **Interactive Drawing** - User can draw regions for partial OCR
3. **Inline Editing** - Double-click to edit recognized text
4. **Zoom Controls** - Examine document details

### Confidence Coloring

```javascript
function getConfidenceClass(confidence) {
    if (confidence < 0.8) {
        return 'confidence-low';      // Red
    } else if (confidence < 0.93) {
        return 'confidence-medium';   // Orange
    } else {
        return 'confidence-high';     // Green
    }
}

function renderTextOverlay() {
    const overlay = document.getElementById('textOverlay');
    overlay.innerHTML = '';
    
    if (!state.ocrResult || !state.ocrResult.text_lines) return;
    
    state.ocrResult.text_lines.forEach((line, index) => {
        const [x1, y1, x2, y2] = line.bbox;
        
        const block = document.createElement('div');
        block.className = 'bbox-block ' + getConfidenceClass(line.confidence);
        
        block.style.left = `${x1 * scale}px`;
        block.style.top = `${y1 * scale}px`;
        block.style.width = `${(x2 - x1) * scale}px`;
        block.style.height = `${(y2 - y1) * scale}px`;
        
        overlay.appendChild(block);
    });
}
```

### CSS for Confidence Levels

```css
.bbox-block.confidence-low {
    background: rgba(255, 82, 82, 0.15);
    border: 1px solid rgba(255, 82, 82, 0.5);
}

.bbox-block.confidence-medium {
    background: rgba(255, 165, 0, 0.15);
    border: 1px solid rgba(255, 165, 0, 0.5);
}

.bbox-block.confidence-high {
    background: rgba(76, 175, 80, 0.15);
    border: 1px solid rgba(76, 175, 80, 0.5);
}
```

### Drawing Mode for Manual Regions

```javascript
function startDrawing(e) {
    state.isDrawing = true;
    const rect = e.target.getBoundingClientRect();
    
    state.drawStart = {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
    };
    
    // Create visual feedback rectangle
    const drawRect = document.createElement('div');
    drawRect.className = 'drawing-rect';
    drawRect.id = 'drawingRect';
    document.getElementById('imageContainer').appendChild(drawRect);
}

async function finishDrawing(e) {
    // Calculate bbox in image coordinates
    const imgWidth = mainImage.naturalWidth;
    const displayWidth = mainImage.width;
    const scaleX = imgWidth / displayWidth;
    
    const bbox = [
        state.drawStart.x * scaleX,
        state.drawStart.y * scaleY,
        endX * scaleX,
        endY * scaleY
    ];
    
    // Remove overlapping existing boxes
    state.ocrResult.text_lines = state.ocrResult.text_lines.filter(
        line => !bboxOverlaps(line.bbox, bbox)
    );
    
    // OCR the new region
    const response = await fetch('/api/recognize-region', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            image_id: state.selectedImageId,
            bbox: bbox
        })
    });
    
    const result = await response.json();
    
    // Merge new results
    state.ocrResult.text_lines.push(...result.text_lines);
    renderTextOverlay();
}
```

---

## Complete Code Reference

### File: `app/services/image_service.py`

Complete deskew implementation with all helper functions.

**Key Functions:**
- `deskew_image(image, debug)` - Main deskew algorithm
- `rotate_image_no_crop(image, angle)` - Rotation preserving content
- `normalize_image(image_bytes)` - API wrapper

### File: `app/services/ocr_service.py`

Complete OCR and classification implementation.

**Key Functions:**
- `get_predictors()` - Lazy Surya model loading
- `get_clip_model()` - Lazy CLIP model loading
- `recognize_text(image_bytes)` - Full OCR pipeline
- `recognize_region(image_bytes, bbox)` - Partial OCR
- `classify_document_hybrid(image, text_lines)` - Combined classification

### File: `app/routers/api.py`

REST API endpoints with request/response models.

### File: `app/static/js/app.js`

Frontend application with state management, event handlers, and UI updates.

---

## Deployment and Performance

### Performance Benchmarks

| Operation | Time (typical) | Notes |
|-----------|---------------|-------|
| Deskew | 0.5-1s | Depends on image resolution |
| OCR | 2-10s | Depends on text density |
| Classification | 0.5s | After model loading |
| First request | +5-30s | Model loading overhead |

### Memory Requirements

- **Surya OCR models**: ~1.5GB disk, ~2GB RAM
- **CLIP model**: ~150MB disk, ~500MB RAM
- **Total recommended**: 4GB RAM minimum

### Production Considerations

1. **Model Preloading** - Load models at startup, not first request
2. **GPU Acceleration** - Both Surya and CLIP support CUDA
3. **Image Storage** - Use object storage (S3, GCS) instead of local disk
4. **Session Management** - Replace in-memory dict with Redis
5. **Rate Limiting** - OCR is CPU-intensive, limit concurrent requests

### Running with GPU

```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Models will automatically use GPU if available
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## References

- **Surya OCR**: https://github.com/VikParuchuri/surya
- **CLIP Paper**: https://arxiv.org/abs/2103.00020
- **CLIP Model**: https://huggingface.co/openai/clip-vit-base-patch32
- **FastAPI**: https://fastapi.tiangolo.com/
- **OpenCV Documentation**: https://docs.opencv.org/4.x/
- **SciPy ndimage**: https://docs.scipy.org/doc/scipy/reference/ndimage.html
- **Projection Profile Analysis**: https://en.wikipedia.org/wiki/Projection_profile_method

---

## License

This project uses the following licenses:
- Surya OCR: GPL-3.0
- CLIP: MIT
- FastAPI: MIT
- OpenCV: Apache 2.0

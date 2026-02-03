# Document Recognition System — Technical Architecture

## Executive Summary

This document provides a comprehensive technical overview of the Document Recognition System, designed for insurance claims processing and document understanding workflows. The system combines state-of-the-art OCR technology with intelligent document classification, all running locally on-device for privacy and performance.

---

## 1. The Problem

### Insurance Claims Document Processing Challenges

Insurance companies process millions of documents annually:
- **Receipts** for expense reimbursement
- **Medical prescriptions** for health claims
- **Forms and applications** for policy changes
- **Contracts and legal documents** for underwriting

Traditional approaches face several challenges:

| Challenge | Impact |
|-----------|--------|
| Manual data entry | High labor costs, slow processing |
| Cloud-based OCR | Privacy concerns, latency, ongoing costs |
| Generic OCR tools | Poor accuracy on specialized documents |
| No classification | Documents require manual sorting |

### The Solution

A **local, GPU-accelerated document recognition system** that:
1. Runs entirely on-device (no cloud dependency)
2. Leverages modern AI models for high accuracy
3. Automatically classifies document types
4. Provides human-in-the-loop correction
5. Exports structured data for downstream systems

---

## 2. System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           USER INTERFACE                                 │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  Upload  │  │  Preview  │  │   LLM    │  │ Schemas  │  │ Settings │  │
│  │          │  │  & Edit   │  │ Results  │  │Templates │  │          │  │
│  └──────────┘  └───────────┘  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         REST API (FastAPI)                               │
│  /api/upload  /api/recognize  /api/llm/*  /api/schemas/*  /api/device   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          SERVICE LAYER                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │  Image Service  │  │   OCR Service   │  │      LLM Service        │  │
│  │  - Deskewing    │  │  - Surya OCR    │  │  - Ollama Client        │  │
│  │  - OpenCV       │  │  - CLIP Class.  │  │  - Gemini Client        │  │
│  │                 │  │  - Hybrid Class │  │  - Schema-based Extract │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────┘  │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                    Document Schema System                           │ │
│  │  YAML Schemas → Field Definitions → LLM Prompts → Structured Data  │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      HARDWARE ACCELERATION                               │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌──────────────────────┐  │
│  │ Apple MPS │  │ NVIDIA    │  │ DGX Spark │  │ CPU Fallback         │  │
│  │ (Metal)   │  │ CUDA 12.x │  │ CUDA 13.0 │  │                      │  │
│  └───────────┘  └───────────┘  └───────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      LLM BACKENDS (with Vision Support)                  │
│  ┌─────────────────────────────┐  ┌─────────────────────────────────┐   │
│  │  Ollama (Local)             │  │  Google Gemini (Cloud)          │   │
│  │  Vision: Gemma 3, LLaVA,    │  │  - Gemini 2.5 Pro [Vision]      │   │
│  │    Llama Vision, MiniCPM-V  │  │  - API Key Required             │   │
│  │  Text: Devstral, Qwen       │  │  - Full multimodal support      │   │
│  │  Medical: Meditron, MedGemma│  │                                 │   │
│  └─────────────────────────────┘  └─────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Breakdown

#### Frontend (Browser-Based UI)
- **Technology**: Vanilla JavaScript, HTML5, CSS3
- **Features**:
  - Drag-and-drop image upload
  - Real-time image preview with zoom/pan
  - Interactive bounding box overlay
  - Inline text editing for corrections
  - Region selection for partial OCR
  - JSON result visualization
  - **LLM Results tab**: Structured field extraction display
  - **Schema Templates tab**: View/edit/create document schemas
  - **Settings tab**: LLM model selection and configuration

#### Backend (Python FastAPI)
- **Framework**: FastAPI with async support
- **Responsibilities**:
  - Serve static assets and HTML templates
  - Handle file uploads and storage
  - Route API requests to services
  - Manage model lifecycle

#### Service Layer
- **Image Service**: OpenCV-based image processing
- **OCR Service**: AI model inference and hardware detection
- **LLM Service**: Local and cloud LLM integration for post-processing

---

## 3. Core Technologies

### 3.1 Surya OCR

**What it is**: A state-of-the-art OCR engine supporting 90+ languages with superior accuracy on diverse document types.

**Why Surya over alternatives**:

| Feature | Surya | Tesseract | Cloud OCR |
|---------|-------|-----------|-----------|
| Accuracy | ★★★★★ | ★★★☆☆ | ★★★★★ |
| Speed (GPU) | ★★★★★ | ★★☆☆☆ | ★★★☆☆ |
| Privacy | ★★★★★ | ★★★★★ | ★☆☆☆☆ |
| Languages | 90+ | 100+ | Varies |
| Cost | Free | Free | Per-page |
| Offline | ✓ | ✓ | ✗ |

**How it works**:
1. Image is preprocessed (resize, normalize)
2. Vision transformer encodes the image
3. Text decoder generates recognized text
4. Bounding boxes are computed for each text line

**Output format**:
```json
{
  "text_lines": [
    {
      "text": "Invoice #12345",
      "bbox": [100, 50, 300, 80],
      "confidence": 0.98
    }
  ]
}
```

### 3.2 CLIP (Contrastive Language-Image Pre-training)

**What it is**: OpenAI's vision-language model that understands both images and text, enabling zero-shot classification.

**How classification works**:

```
┌─────────────┐     ┌─────────────────────────────────────┐
│   Document  │     │         Text Descriptions           │
│    Image    │     │  "a photo of a receipt"             │
│             │     │  "a photo of a medical prescription"│
│             │     │  "a photo of a form"                │
│             │     │  "a photo of a contract"            │
└──────┬──────┘     └──────────────────┬──────────────────┘
       │                               │
       ▼                               ▼
┌──────────────┐               ┌──────────────┐
│ CLIP Image   │               │ CLIP Text    │
│ Encoder      │               │ Encoder      │
└──────┬───────┘               └──────┬───────┘
       │                               │
       ▼                               ▼
┌──────────────┐               ┌──────────────┐
│ Image        │               │ Text         │
│ Embedding    │◄─────────────►│ Embeddings   │
│ (512-dim)    │  Similarity   │ (512-dim)    │
└──────────────┘               └──────────────┘
                    │
                    ▼
            ┌──────────────┐
            │ Classification│
            │ "receipt"     │
            │ conf: 0.87    │
            └──────────────┘
```

**Hybrid Classification Approach**:
1. **CLIP embedding similarity** (primary)
2. **Keyword matching** (fallback/boost)
3. **Confidence thresholding** (quality control)

### 3.3 Image Normalization (Deskewing)

**The problem**: Scanned or photographed documents are often rotated or skewed, reducing OCR accuracy.

**Solution**: Projection Profile Analysis with OpenCV

```
Original Image          Projection Profile       Corrected Image
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│    ╱╱╱╱╱    │         │ ▁▂▃▅▇█▇▅▃▂▁│         │ ═══════════ │
│   ╱╱╱╱╱     │   ──►   │             │   ──►   │ ═══════════ │
│  ╱╱╱╱╱      │         │   Analyze   │         │ ═══════════ │
│ ╱╱╱╱╱       │         │   Rotation  │         │ ═══════════ │
└─────────────┘         └─────────────┘         └─────────────┘
   Skewed                Find Angle              Straightened
```

**Algorithm**:
1. Convert to grayscale
2. Apply edge detection
3. Compute Hough transform for line detection
4. Calculate dominant angle
5. Rotate image to correct orientation

---

## 4. Hardware Acceleration

### The Challenge

Deep learning models require significant computational resources. Without GPU acceleration, inference can take 10-30 seconds per image. With GPU acceleration, this drops to 1-3 seconds.

### Multi-Platform Support

The system automatically detects and utilizes available hardware:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Platform Detection Flow                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Check nvidia-smi│
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
        ┌──────────┐                 ┌──────────────┐
        │ NVIDIA   │                 │ No NVIDIA    │
        │ Detected │                 │ GPU          │
        └────┬─────┘                 └──────┬───────┘
             │                              │
    ┌────────┴────────┐                     ▼
    │                 │              ┌──────────────┐
    ▼                 ▼              │ Check macOS  │
┌────────┐      ┌──────────┐         │ + arm64      │
│ x86_64 │      │ aarch64  │         └──────┬───────┘
│ CUDA   │      │ (DGX)    │                │
│ 12.x   │      │ CUDA 13  │         ┌──────┴───────┐
└────────┘      └──────────┘         │              │
    │                │               ▼              ▼
    ▼                ▼         ┌──────────┐   ┌──────────┐
┌────────┐      ┌──────────┐   │ Apple    │   │ CPU      │
│ cu121  │      │ cu130    │   │ Silicon  │   │ Fallback │
│ wheels │      │ wheels   │   │ MPS      │   │          │
└────────┘      └──────────┘   └──────────┘   └──────────┘
```

### Platform Comparison

| Platform | Device | Acceleration | Typical Inference Time |
|----------|--------|--------------|------------------------|
| MacBook Pro M3 | Apple Silicon | MPS (Metal) | 2-4 seconds |
| Desktop RTX 4090 | NVIDIA GPU | CUDA 12.1 | 0.5-1 second |
| DGX Spark | Blackwell GB10 | CUDA 13.0 | 0.3-0.8 seconds |
| Intel/AMD CPU | None | CPU | 15-30 seconds |

### PyTorch Installation Strategy

Standard `pip install torch` installs CPU-only wheels. For GPU acceleration, platform-specific wheels are required:

```bash
# Apple Silicon (MPS built-in)
pip install torch torchvision

# NVIDIA GPU (x86_64)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# DGX Spark (aarch64 + CUDA 13)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

The `install_pytorch.sh` script automates this detection and installation.

---

## 5. Data Flow

### Complete Request Lifecycle

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         DOCUMENT PROCESSING FLOW                         │
└──────────────────────────────────────────────────────────────────────────┘

User uploads image
        │
        ▼
┌───────────────┐
│ 1. UPLOAD     │  POST /api/upload
│    - Validate │  - Check file type (JPEG, PNG, TIFF, BMP)
│    - Store    │  - Generate unique ID
│    - Return ID│  - Save to uploads/ directory
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ 2. NORMALIZE  │  POST /api/normalize (optional)
│    - Deskew   │  - Detect rotation angle
│    - Enhance  │  - Apply affine transformation
│    - Save     │  - Store normalized version
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ 3. RECOGNIZE  │  POST /api/recognize
│               │
│  ┌──────────┐ │  a) Load image
│  │ Surya    │ │  b) Run OCR model
│  │ OCR      │ │  c) Extract text + bounding boxes
│  └──────────┘ │
│       │       │
│       ▼       │
│  ┌──────────┐ │  d) Encode image with CLIP
│  │ CLIP     │ │  e) Compare to class descriptions
│  │ Classify │ │  f) Return classification + confidence
│  └──────────┘ │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ 4. DISPLAY    │  Frontend renders results
│    - Overlay  │  - Draw bounding boxes on image
│    - Edit     │  - Enable inline text editing
│    - JSON     │  - Show structured output
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ 5. EXPORT     │  User saves corrected results
│    - JSON     │  - Download as JSON file
│    - Integrate│  - Ready for downstream systems
└───────────────┘
```

### JSON Output Structure (OCR)

```json
{
  "image_id": "abc123",
  "document_class": {
    "class": "Receipt",
    "type_id": "receipt",
    "confidence": 0.92,
    "method": "hybrid"
  },
  "text_lines": [
    {
      "text": "ACME STORE",
      "bbox": [120, 45, 380, 85],
      "confidence": 0.99
    },
    {
      "text": "Date: 2024-01-15",
      "bbox": [120, 100, 320, 130],
      "confidence": 0.97
    },
    {
      "text": "Total: $45.99",
      "bbox": [120, 450, 280, 480],
      "confidence": 0.98
    }
  ],
  "image_bbox": [0, 0, 800, 600]
}
```

### JSON Output Structure (LLM Extraction)

```json
{
  "success": true,
  "original_text": "ACME STORE\nDate: 2024-01-15\n...",
  "corrected_text": "ACME STORE\nDate: 2024-01-15\n...",
  "extracted_fields": {
    "store_name": "ACME STORE",
    "date": "2024-01-15",
    "total": "$45.99",
    "items": [
      {"name": "Widget", "quantity": 2, "price": "$15.00"},
      {"name": "Gadget", "quantity": 1, "price": "$15.99"}
    ]
  },
  "document_type": "receipt",
  "document_type_name": "Receipt / Invoice",
  "model": "Gemma 3 (27B) [Vision]",
  "vision_used": true
}
```

---

## 6. Model Lifecycle

### Lazy Loading Strategy

Models are loaded on-demand to optimize startup time:

```
Server Start                First Recognition Request
     │                              │
     ▼                              ▼
┌──────────────┐            ┌──────────────────────┐
│ FastAPI Init │            │ Check if models      │
│ (< 1 second) │            │ are loaded           │
└──────────────┘            └──────────┬───────────┘
                                       │
                            ┌──────────┴───────────┐
                            │                      │
                            ▼                      ▼
                    ┌──────────────┐       ┌──────────────┐
                    │ Not Loaded   │       │ Already      │
                    │              │       │ Loaded       │
                    └──────┬───────┘       └──────┬───────┘
                           │                      │
                           ▼                      │
                    ┌──────────────┐              │
                    │ Download &   │              │
                    │ Load Models  │              │
                    │ (~30 sec)    │              │
                    └──────┬───────┘              │
                           │                      │
                           └──────────┬───────────┘
                                      │
                                      ▼
                              ┌──────────────┐
                              │ Run Inference│
                              │ (1-3 sec)    │
                              └──────────────┘
```

### Model Sizes

| Model | Size | Purpose |
|-------|------|---------|
| Surya OCR | ~1.5 GB | Text recognition |
| CLIP ViT-B/32 | ~150 MB | Document classification |
| **Total** | **~1.7 GB** | First-time download |

---

## 7. Security & Privacy

### Local-First Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     PRIVACY ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        LOCAL DEVICE                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Documents   │  │ AI Models   │  │ Processing              │  │
│  │ (uploads/)  │  │ (cached)    │  │ (inference)             │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│                                                                 │
│  ✓ All data stays on device                                    │
│  ✓ No cloud API calls during inference                         │
│  ✓ No telemetry or tracking                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ One-time model download
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     INTERNET (Optional)                         │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Hugging Face Hub - Model weights download (first run)   │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Key Privacy Features

1. **No cloud dependency** - All OCR and classification runs locally
2. **No data transmission** - Documents never leave the device
3. **Offline capable** - Works without internet after initial setup
4. **Self-hosted** - Full control over deployment
5. **Optional cloud LLM** - Gemini API is opt-in only (requires explicit API key)

---

## 8. Performance Optimization

### Inference Pipeline Optimization

```
┌─────────────────────────────────────────────────────────────────┐
│                   OPTIMIZATION TECHNIQUES                       │
└─────────────────────────────────────────────────────────────────┘

1. GPU Memory Management
   ┌────────────────────────────────────────────────────────────┐
   │ • Models loaded to GPU VRAM once, reused across requests   │
   │ • Batch processing for multiple images                     │
   │ • Automatic memory cleanup after inference                 │
   └────────────────────────────────────────────────────────────┘

2. Image Preprocessing
   ┌────────────────────────────────────────────────────────────┐
   │ • Resize large images to optimal input size                │
   │ • Convert to RGB if necessary                              │
   │ • Normalize pixel values for model input                   │
   └────────────────────────────────────────────────────────────┘

3. Async Processing
   ┌────────────────────────────────────────────────────────────┐
   │ • FastAPI async endpoints for non-blocking I/O             │
   │ • File operations use aiofiles                             │
   │ • Multiple requests can be queued                          │
   └────────────────────────────────────────────────────────────┘
```

### Benchmarks

| Operation | CPU | Apple M3 (MPS) | RTX 4090 | DGX Spark |
|-----------|-----|----------------|----------|-----------|
| OCR (single page) | 15-25s | 2-4s | 0.5-1s | 0.3-0.8s |
| Classification | 2-5s | 0.3-0.5s | 0.1-0.2s | 0.05-0.1s |
| Deskew | 0.5-1s | 0.2-0.3s | 0.1-0.2s | 0.1s |
| **Total** | **18-31s** | **2.5-5s** | **0.7-1.4s** | **0.45-1s** |

---

## 9. Extensibility

### Adding New Document Classes

The CLIP-based classification uses natural language descriptions, making it easy to add new classes:

```python
# In ocr_service.py
CLIP_CLASS_DESCRIPTIONS = [
    "a photo of a receipt or invoice with prices and totals",
    "a photo of a medical prescription or medication document",
    "a photo of a form or application document with fields to fill",
    "a photo of a contract or legal agreement document",
    "a photo of an insurance claim form",           # New class
    "a photo of a driver's license or ID card",    # New class
    "a photo of an unknown or unclassified document"
]
```

No retraining required — CLIP's zero-shot capability handles new classes automatically.

### API Integration Points

```
┌─────────────────────────────────────────────────────────────────┐
│                    INTEGRATION OPTIONS                          │
└─────────────────────────────────────────────────────────────────┘

1. REST API Integration
   ┌────────────────────────────────────────────────────────────┐
   │ POST /api/recognize                                        │
   │ → Returns JSON with OCR results and classification         │
   │ → Easy integration with any backend system                 │
   └────────────────────────────────────────────────────────────┘

2. Batch Processing
   ┌────────────────────────────────────────────────────────────┐
   │ Upload multiple images → Process sequentially              │
   │ → Export all results as JSON                               │
   └────────────────────────────────────────────────────────────┘

3. Downstream Systems
   ┌────────────────────────────────────────────────────────────┐
   │ JSON output → Claims Management System                     │
   │ JSON output → Document Management System                   │
   │ JSON output → Data Warehouse / Analytics                   │
   └────────────────────────────────────────────────────────────┘
```

---

## 10. Deployment Options

### Local Development

```bash
# Quick start
python -m venv venv
source venv/bin/activate
./install_pytorch.sh
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Production Deployment

```bash
# With Gunicorn for production
pip install gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### DGX Spark / NGC Container

```dockerfile
FROM nvcr.io/nvidia/pytorch:25.09-py3

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 11. Future Enhancements

### Roadmap

| Feature | Description | Priority |
|---------|-------------|----------|
| Table extraction | Detect and parse tables in documents | High |
| Handwriting OCR | Support for handwritten text | Medium |
| Multi-page PDF | Process multi-page documents | High |
| Custom model training | Fine-tune on domain-specific documents | Medium |
| Batch API | Process multiple documents in one request | Medium |
| Webhook notifications | Notify external systems on completion | Low |

---

## 12. Summary

The Document Recognition System provides a complete, privacy-preserving solution for document understanding:

### Key Differentiators

1. **Local-first**: All processing on-device, no cloud dependency
2. **Multi-platform GPU acceleration**: Mac, NVIDIA, DGX Spark
3. **State-of-the-art accuracy**: Surya OCR + CLIP classification
4. **Multimodal Vision LLMs**: Image + text processed together for superior extraction
5. **LLM-powered extraction**: Structured data extraction with local or cloud LLMs
6. **Schema-based processing**: YAML document schemas for customizable extraction
7. **Human-in-the-loop**: Interactive correction of OCR results
8. **Zero-shot classification**: Add new document types without retraining
9. **Production-ready**: FastAPI backend, modern frontend

### Technology Stack Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                      TECHNOLOGY STACK                           │
├─────────────────────────────────────────────────────────────────┤
│ Frontend    │ HTML5, CSS3, Vanilla JavaScript                   │
│ Backend     │ Python 3.10+, FastAPI, Uvicorn                    │
│ OCR         │ Surya OCR (Vision Transformer)                    │
│ Classify    │ OpenAI CLIP (ViT-B/32)                            │
│ LLM Local   │ Ollama - Vision: Gemma 3, LLaVA; Text: Devstral   │
│ LLM Cloud   │ Google Gemini 2.5 Pro with Vision (optional)      │
│ Image Proc  │ OpenCV, Pillow                                    │
│ ML Runtime  │ PyTorch 2.0+                                      │
│ Acceleration│ CUDA (NVIDIA), MPS (Apple), CPU                   │
└─────────────────────────────────────────────────────────────────┘
```

---

*This document is intended for technical audiences, infographic designers, and article writers seeking to understand the system architecture and capabilities.*

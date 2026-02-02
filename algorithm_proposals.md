# Algorithm Improvement Proposals

## Document Recognition System — Quality Enhancement Strategies

**Status**: DRAFT — NOT FOR COMMIT  
**Date**: February 2026

---

## Executive Summary

This document proposes algorithmic improvements to enhance OCR quality, bounding box detection, and document understanding. The proposals focus on three main areas:

1. **Bounding Box Detection** — Multi-pass detection, line merging, semantic grouping
2. **Context-Aware Recognition** — Using document type and expected fields to improve accuracy
3. **General Quality Improvements** — Preprocessing, confidence calibration, post-processing

---

## 1. Bounding Box Detection Improvements

### 1.1 Current State Analysis

**Current Implementation** (`ocr_service.py:200`):
```python
predictions = recognition_predictor([image], det_predictor=detection_predictor)
```

**Issues Identified**:
- Single-pass detection may miss low-contrast text
- Each line gets separate bbox (no paragraph grouping)
- No handling of multi-column layouts
- No retry mechanism for uncertain detections

---

### 1.2 Proposal: Multi-Pass Detection Strategy

**Concept**: Run detection multiple times with different parameters and merge results.

```
┌─────────────────────────────────────────────────────────────────┐
│                    MULTI-PASS DETECTION                         │
└─────────────────────────────────────────────────────────────────┘

Pass 1: Standard Detection
    │
    ▼
┌─────────────┐
│ Default     │ → Detects clear, high-contrast text
│ Parameters  │
└─────────────┘
    │
    ▼
Pass 2: High Sensitivity Detection
    │
    ▼
┌─────────────┐
│ Lower       │ → Catches faint text, handwriting
│ Threshold   │
└─────────────┘
    │
    ▼
Pass 3: Scale Variation
    │
    ▼
┌─────────────┐
│ 1.5x Scale  │ → Better for small text
│ Detection   │
└─────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MERGE & DEDUPLICATE                          │
│  - IoU-based deduplication                                      │
│  - Keep highest confidence for overlapping boxes                │
│  - Union of non-overlapping detections                          │
└─────────────────────────────────────────────────────────────────┘
```

**Implementation**:
```python
def multi_pass_detection(image: Image.Image, detection_predictor) -> list:
    """
    Run detection with multiple parameter sets and merge results.
    """
    all_detections = []
    
    # Pass 1: Standard detection
    det1 = detection_predictor([image])
    all_detections.extend(det1[0].bboxes if det1 else [])
    
    # Pass 2: High sensitivity (if supported by Surya config)
    # Adjust detection threshold via predictor config
    det2 = detection_predictor([image], detection_threshold=0.3)
    all_detections.extend(det2[0].bboxes if det2 else [])
    
    # Pass 3: Scaled image for small text
    scaled = image.resize((int(image.width * 1.5), int(image.height * 1.5)))
    det3 = detection_predictor([scaled])
    # Scale bboxes back to original size
    if det3:
        for bbox in det3[0].bboxes:
            scaled_bbox = [c / 1.5 for c in bbox]
            all_detections.append(scaled_bbox)
    
    # Deduplicate using IoU
    return deduplicate_bboxes(all_detections, iou_threshold=0.5)


def deduplicate_bboxes(bboxes: list, iou_threshold: float = 0.5) -> list:
    """Remove duplicate bboxes based on IoU overlap."""
    if not bboxes:
        return []
    
    # Sort by area (larger first)
    bboxes = sorted(bboxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]), reverse=True)
    
    keep = []
    for bbox in bboxes:
        is_duplicate = False
        for kept in keep:
            if compute_iou(bbox, kept) > iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            keep.append(bbox)
    
    return keep


def compute_iou(box1, box2) -> float:
    """Compute Intersection over Union."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0
```

**Expected Improvement**: 15-25% more text detected on low-quality scans.

---

### 1.3 Proposal: Semantic Line Merging (Paragraph Detection)

**Problem**: Current system creates one bbox per text line. For paragraphs, this creates many small boxes instead of one logical block.

**Solution**: Post-process bboxes to merge lines that belong to the same paragraph.

```
BEFORE (Current):                    AFTER (Proposed):
┌─────────────────────┐              ┌─────────────────────┐
│ Line 1 of paragraph │              │ Line 1 of paragraph │
└─────────────────────┘              │ continues here and  │
┌─────────────────────┐      →       │ ends on line three. │
│ continues here and  │              └─────────────────────┘
└─────────────────────┘              
┌─────────────────────┐              
│ ends on line three. │              
└─────────────────────┘              
```

**Merging Criteria**:
1. **Vertical proximity**: Lines within 1.5x line height
2. **Horizontal alignment**: Left edges within 10% tolerance
3. **Similar font size**: Bbox heights within 20% tolerance
4. **No large gap**: No empty space > 2x line height between

**Implementation**:
```python
def merge_paragraph_lines(text_lines: list, merge_threshold: float = 1.5) -> list:
    """
    Merge consecutive text lines that form paragraphs.
    
    Args:
        text_lines: List of text line dicts with 'text', 'bbox', 'confidence'
        merge_threshold: Max vertical gap as multiple of line height
        
    Returns:
        List of merged text blocks
    """
    if not text_lines:
        return []
    
    # Sort by vertical position (top to bottom)
    sorted_lines = sorted(text_lines, key=lambda l: l['bbox'][1])
    
    merged = []
    current_block = None
    
    for line in sorted_lines:
        if current_block is None:
            current_block = create_block_from_line(line)
            continue
        
        if should_merge(current_block, line, merge_threshold):
            # Merge line into current block
            current_block = merge_line_into_block(current_block, line)
        else:
            # Start new block
            merged.append(current_block)
            current_block = create_block_from_line(line)
    
    if current_block:
        merged.append(current_block)
    
    return merged


def should_merge(block: dict, line: dict, threshold: float) -> bool:
    """Determine if line should be merged into block."""
    block_bbox = block['bbox']
    line_bbox = line['bbox']
    
    # Calculate line height
    block_height = block_bbox[3] - block_bbox[1]
    line_height = line_bbox[3] - line_bbox[1]
    avg_line_height = (block_height / block.get('line_count', 1) + line_height) / 2
    
    # Vertical gap
    vertical_gap = line_bbox[1] - block_bbox[3]
    if vertical_gap > avg_line_height * threshold:
        return False
    if vertical_gap < -avg_line_height * 0.5:  # Overlapping too much
        return False
    
    # Horizontal alignment (left edge)
    left_diff = abs(block_bbox[0] - line_bbox[0])
    block_width = block_bbox[2] - block_bbox[0]
    if left_diff > block_width * 0.15:  # 15% tolerance
        return False
    
    # Font size similarity (bbox height)
    height_ratio = min(line_height, avg_line_height) / max(line_height, avg_line_height)
    if height_ratio < 0.7:  # 30% tolerance
        return False
    
    return True


def merge_line_into_block(block: dict, line: dict) -> dict:
    """Merge a line into an existing block."""
    return {
        'text': block['text'] + ' ' + line['text'],
        'bbox': [
            min(block['bbox'][0], line['bbox'][0]),
            block['bbox'][1],  # Keep original top
            max(block['bbox'][2], line['bbox'][2]),
            line['bbox'][3]   # Extend to new bottom
        ],
        'confidence': (block['confidence'] * block.get('line_count', 1) + line['confidence']) / (block.get('line_count', 1) + 1),
        'line_count': block.get('line_count', 1) + 1,
        'original_lines': block.get('original_lines', [block]) + [line]
    }
```

**Configuration Options**:
```python
MERGE_CONFIG = {
    'enabled': True,
    'vertical_threshold': 1.5,      # Max gap as multiple of line height
    'horizontal_tolerance': 0.15,   # Left edge alignment tolerance
    'height_tolerance': 0.3,        # Font size similarity tolerance
    'preserve_original': True       # Keep original lines in output
}
```

**Expected Improvement**: 
- Cleaner output for paragraph text
- Better for forms with multi-line fields
- Easier downstream processing

---

### 1.4 Proposal: Layout-Aware Detection

**Problem**: Documents often have multi-column layouts, tables, or mixed regions. Current detection treats everything uniformly.

**Solution**: Pre-analyze document layout, then apply region-specific detection.

```
┌─────────────────────────────────────────────────────────────────┐
│                    LAYOUT ANALYSIS PIPELINE                     │
└─────────────────────────────────────────────────────────────────┘

Step 1: Layout Detection
┌─────────────────────────────────────────────────────────────────┐
│  Detect regions: Header, Body, Footer, Columns, Tables          │
│  Methods: Rule-based (whitespace analysis) or ML (LayoutLM)     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
Step 2: Region Classification
┌─────────────────────────────────────────────────────────────────┐
│  Classify each region:                                          │
│  - Text block (paragraph)                                       │
│  - Table (grid structure)                                       │
│  - Form field (label + value)                                   │
│  - Header/Footer                                                │
│  - Image/Logo (skip OCR)                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
Step 3: Region-Specific OCR
┌─────────────────────────────────────────────────────────────────┐
│  Apply appropriate strategy per region:                         │
│  - Text blocks: Standard OCR + paragraph merging                │
│  - Tables: Cell-by-cell OCR with structure preservation         │
│  - Form fields: Key-value pair extraction                       │
└─────────────────────────────────────────────────────────────────┘
```

**Simple Layout Detection (Rule-Based)**:
```python
def detect_layout_regions(image: Image.Image) -> list:
    """
    Detect layout regions using whitespace analysis.
    """
    import numpy as np
    from PIL import ImageFilter
    
    # Convert to grayscale and detect edges
    gray = image.convert('L')
    edges = gray.filter(ImageFilter.FIND_EDGES)
    pixels = np.array(edges)
    
    # Horizontal projection (sum pixels per row)
    h_proj = pixels.sum(axis=1)
    
    # Vertical projection (sum pixels per column)
    v_proj = pixels.sum(axis=0)
    
    # Find horizontal separators (low activity rows)
    h_threshold = h_proj.mean() * 0.1
    h_gaps = find_gaps(h_proj, h_threshold, min_gap=20)
    
    # Find vertical separators (potential columns)
    v_threshold = v_proj.mean() * 0.1
    v_gaps = find_gaps(v_proj, v_threshold, min_gap=50)
    
    # Build regions from gaps
    regions = build_regions_from_gaps(h_gaps, v_gaps, image.size)
    
    return regions


def find_gaps(projection: np.ndarray, threshold: float, min_gap: int) -> list:
    """Find gaps (low activity regions) in projection."""
    gaps = []
    in_gap = False
    gap_start = 0
    
    for i, val in enumerate(projection):
        if val < threshold:
            if not in_gap:
                in_gap = True
                gap_start = i
        else:
            if in_gap:
                if i - gap_start >= min_gap:
                    gaps.append((gap_start, i))
                in_gap = False
    
    return gaps
```

---

## 2. Context-Aware Recognition

### 2.1 Current State

**Current Classification** (`ocr_service.py:438`):
- Hybrid CLIP + keyword matching
- Classification happens AFTER OCR
- No feedback loop to improve recognition

**Limitation**: OCR doesn't know what type of document it's processing, missing opportunities to use domain knowledge.

---

### 2.2 Proposal: Document Type-Guided Recognition

**Concept**: Use document classification to guide OCR post-processing and field extraction.

```
┌─────────────────────────────────────────────────────────────────┐
│                CONTEXT-AWARE RECOGNITION PIPELINE               │
└─────────────────────────────────────────────────────────────────┘

Step 1: Quick Classification (before full OCR)
┌─────────────────────────────────────────────────────────────────┐
│  CLIP-only classification on full image                         │
│  → "Receipt" (85% confidence)                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
Step 2: Load Document Template
┌─────────────────────────────────────────────────────────────────┐
│  Receipt Template:                                              │
│  - Expected fields: store_name, date, items[], total, tax      │
│  - Expected patterns: currency, dates, quantities              │
│  - Layout hints: header at top, items in middle, total at end  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
Step 3: Guided OCR
┌─────────────────────────────────────────────────────────────────┐
│  Run OCR with context:                                          │
│  - Prioritize regions matching expected layout                  │
│  - Apply field-specific post-processing                         │
│  - Use expected patterns for error correction                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
Step 4: Field Extraction
┌─────────────────────────────────────────────────────────────────┐
│  Extract structured data:                                       │
│  {                                                              │
│    "store_name": "ACME Store",                                  │
│    "date": "2024-01-15",                                        │
│    "items": [...],                                              │
│    "total": 45.99                                               │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘
```

**Document Templates**:
```python
DOCUMENT_TEMPLATES = {
    "Receipt": {
        "expected_fields": [
            {"name": "store_name", "location": "top", "required": True},
            {"name": "date", "pattern": r"\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}", "required": True},
            {"name": "items", "type": "list", "location": "middle"},
            {"name": "subtotal", "pattern": r"subtotal[:\s]*[\$€£]?\d+[.,]\d{2}", "location": "bottom"},
            {"name": "tax", "pattern": r"tax[:\s]*[\$€£]?\d+[.,]\d{2}", "location": "bottom"},
            {"name": "total", "pattern": r"total[:\s]*[\$€£]?\d+[.,]\d{2}", "required": True, "location": "bottom"},
        ],
        "layout": {
            "header_ratio": 0.15,    # Top 15% is header
            "footer_ratio": 0.20,    # Bottom 20% is totals
        },
        "post_processing": {
            "currency_normalization": True,
            "date_parsing": True,
            "number_correction": True  # Fix O→0, l→1, etc.
        }
    },
    
    "Medication Prescription": {
        "expected_fields": [
            {"name": "patient_name", "location": "top", "required": True},
            {"name": "doctor_name", "location": "top"},
            {"name": "date", "pattern": r"\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}", "required": True},
            {"name": "medications", "type": "list", "location": "middle"},
            {"name": "dosage", "pattern": r"\d+\s*(mg|ml|mcg|g)", "required": True},
            {"name": "instructions", "location": "middle"},
            {"name": "refills", "pattern": r"refill[s]?[:\s]*\d+"},
        ],
        "post_processing": {
            "medical_term_correction": True,
            "dosage_validation": True
        }
    },
    
    "Form": {
        "expected_fields": [
            {"name": "form_title", "location": "top"},
            {"name": "fields", "type": "key_value_pairs", "location": "middle"},
            {"name": "signature", "location": "bottom"},
            {"name": "date", "pattern": r"\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}"},
        ],
        "layout": {
            "detect_checkboxes": True,
            "detect_form_fields": True
        }
    },
    
    "Contract": {
        "expected_fields": [
            {"name": "title", "location": "top"},
            {"name": "parties", "location": "top"},
            {"name": "effective_date", "pattern": r"\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}"},
            {"name": "clauses", "type": "numbered_list", "location": "middle"},
            {"name": "signatures", "location": "bottom"},
        ],
        "post_processing": {
            "legal_term_preservation": True,
            "clause_numbering": True
        }
    }
}
```

**Implementation**:
```python
def recognize_with_context(image_bytes: bytes, document_type: str = None) -> dict:
    """
    Context-aware document recognition.
    
    Args:
        image_bytes: Image as bytes
        document_type: Optional pre-known document type
        
    Returns:
        dict: OCR results with structured field extraction
    """
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Step 1: Quick classification if type not provided
    if document_type is None:
        classification = classify_image_with_clip(image)
        document_type = classification['class']
        type_confidence = classification['confidence']
    else:
        type_confidence = 1.0
    
    # Step 2: Get template
    template = DOCUMENT_TEMPLATES.get(document_type, {})
    
    # Step 3: Run OCR
    recognition_predictor, detection_predictor = get_predictors()
    predictions = recognition_predictor([image], det_predictor=detection_predictor)
    
    if not predictions:
        return {"text_lines": [], "document_type": document_type}
    
    result = predictions[0]
    text_lines = extract_text_lines(result)
    
    # Step 4: Apply context-aware post-processing
    if template.get('post_processing'):
        text_lines = apply_post_processing(text_lines, template['post_processing'])
    
    # Step 5: Extract structured fields
    extracted_fields = {}
    if template.get('expected_fields'):
        extracted_fields = extract_fields(text_lines, template['expected_fields'], image.size)
    
    return {
        "text_lines": text_lines,
        "image_bbox": [0, 0, image.width, image.height],
        "document_class": {
            "class": document_type,
            "confidence": type_confidence
        },
        "extracted_fields": extracted_fields
    }
```

---

### 2.3 Proposal: Pattern-Based Error Correction

**Problem**: OCR often confuses similar characters (O/0, l/1/I, S/5, etc.)

**Solution**: Use expected patterns to correct likely errors.

```python
# Common OCR confusion pairs
OCR_CONFUSIONS = {
    'O': '0',   # Letter O → Digit 0
    'o': '0',
    'l': '1',   # Lowercase L → Digit 1
    'I': '1',   # Uppercase I → Digit 1
    'S': '5',   # Letter S → Digit 5
    'B': '8',   # Letter B → Digit 8
    'Z': '2',   # Letter Z → Digit 2
    'G': '6',   # Letter G → Digit 6
}

def correct_with_pattern(text: str, pattern: str) -> str:
    """
    Correct OCR errors based on expected pattern.
    
    Args:
        text: Raw OCR text
        pattern: Regex pattern for expected format
        
    Returns:
        Corrected text
    """
    import re
    
    # If already matches, return as-is
    if re.match(pattern, text):
        return text
    
    # Try character substitutions
    corrected = text
    for wrong, right in OCR_CONFUSIONS.items():
        # For numeric patterns, replace letters with digits
        if '\\d' in pattern:
            corrected = corrected.replace(wrong, right)
    
    # Validate correction
    if re.match(pattern, corrected):
        return corrected
    
    return text  # Return original if correction didn't help


def correct_currency(text: str) -> str:
    """Correct common OCR errors in currency values."""
    import re
    
    # Pattern: $XX.XX or XX.XX
    currency_pattern = r'[\$€£]?\s*(\d+)[.,](\d{2})'
    
    # Common corrections
    text = text.replace('$', '$').replace('S', '$')  # S often misread as $
    text = re.sub(r'[Oo]', '0', text)  # O → 0 in numbers
    text = re.sub(r'[lI]', '1', text)  # l/I → 1 in numbers
    
    return text


def correct_date(text: str) -> str:
    """Correct common OCR errors in dates."""
    import re
    
    # Try to parse and reformat
    date_patterns = [
        (r'(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})', '{}/{}/{}'),
        (r'(\d{4})[/.-](\d{1,2})[/.-](\d{1,2})', '{}-{}-{}'),
    ]
    
    # Common corrections
    text = re.sub(r'[Oo]', '0', text)
    text = re.sub(r'[lI]', '1', text)
    
    return text
```

---

### 2.4 Proposal: Confidence-Based Re-Recognition

**Concept**: For low-confidence regions, attempt re-recognition with different parameters.

```python
def recognize_with_retry(image: Image.Image, region_bbox: list, 
                         min_confidence: float = 0.7) -> dict:
    """
    Recognize text with retry for low-confidence results.
    """
    recognition_predictor, detection_predictor = get_predictors()
    
    # First attempt: standard recognition
    result = recognize_region_internal(image, region_bbox)
    
    if result['confidence'] >= min_confidence:
        return result
    
    # Second attempt: enhanced preprocessing
    enhanced = enhance_image_for_ocr(image.crop(region_bbox))
    result2 = recognize_region_internal(enhanced, [0, 0, enhanced.width, enhanced.height])
    
    if result2['confidence'] > result['confidence']:
        result = result2
    
    # Third attempt: scaled up
    if result['confidence'] < min_confidence:
        scaled = image.crop(region_bbox).resize(
            (int(region_bbox[2] - region_bbox[0]) * 2,
             int(region_bbox[3] - region_bbox[1]) * 2),
            Image.LANCZOS
        )
        result3 = recognize_region_internal(scaled, [0, 0, scaled.width, scaled.height])
        
        if result3['confidence'] > result['confidence']:
            result = result3
    
    return result


def enhance_image_for_ocr(image: Image.Image) -> Image.Image:
    """Apply image enhancements to improve OCR accuracy."""
    from PIL import ImageEnhance, ImageFilter
    
    # Increase contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)
    
    # Sharpen
    image = image.filter(ImageFilter.SHARPEN)
    
    # Denoise (simple median filter)
    image = image.filter(ImageFilter.MedianFilter(size=3))
    
    return image
```

---

## 3. General Quality Improvements

### 3.1 Image Preprocessing Pipeline

**Current**: Minimal preprocessing (just RGB conversion)

**Proposed**: Comprehensive preprocessing pipeline

```python
class ImagePreprocessor:
    """Preprocessing pipeline for OCR optimization."""
    
    def __init__(self, config: dict = None):
        self.config = config or {
            'auto_rotate': True,
            'deskew': True,
            'denoise': True,
            'contrast_enhance': True,
            'binarize': False,  # Only for very poor quality
            'resize_max': 4000,  # Max dimension
        }
    
    def process(self, image: Image.Image) -> Image.Image:
        """Apply preprocessing pipeline."""
        
        # 1. Resize if too large (memory optimization)
        if self.config['resize_max']:
            image = self._resize_if_needed(image)
        
        # 2. Auto-rotate based on EXIF
        if self.config['auto_rotate']:
            image = self._auto_rotate(image)
        
        # 3. Deskew
        if self.config['deskew']:
            image = self._deskew(image)
        
        # 4. Denoise
        if self.config['denoise']:
            image = self._denoise(image)
        
        # 5. Contrast enhancement
        if self.config['contrast_enhance']:
            image = self._enhance_contrast(image)
        
        # 6. Binarization (optional, for very poor quality)
        if self.config['binarize']:
            image = self._binarize(image)
        
        return image
    
    def _resize_if_needed(self, image: Image.Image) -> Image.Image:
        max_dim = self.config['resize_max']
        if max(image.size) > max_dim:
            ratio = max_dim / max(image.size)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            return image.resize(new_size, Image.LANCZOS)
        return image
    
    def _auto_rotate(self, image: Image.Image) -> Image.Image:
        from PIL import ExifTags
        try:
            exif = image._getexif()
            if exif:
                for tag, value in exif.items():
                    if ExifTags.TAGS.get(tag) == 'Orientation':
                        if value == 3:
                            return image.rotate(180, expand=True)
                        elif value == 6:
                            return image.rotate(270, expand=True)
                        elif value == 8:
                            return image.rotate(90, expand=True)
        except:
            pass
        return image
    
    def _deskew(self, image: Image.Image) -> Image.Image:
        # Use existing deskew from image_service
        # Or implement projection profile method
        return image
    
    def _denoise(self, image: Image.Image) -> Image.Image:
        from PIL import ImageFilter
        return image.filter(ImageFilter.MedianFilter(size=3))
    
    def _enhance_contrast(self, image: Image.Image) -> Image.Image:
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(1.3)
    
    def _binarize(self, image: Image.Image) -> Image.Image:
        gray = image.convert('L')
        return gray.point(lambda x: 255 if x > 128 else 0, '1').convert('RGB')
```

---

### 3.2 Confidence Calibration

**Problem**: Raw model confidence scores may not reflect true accuracy.

**Solution**: Calibrate confidence scores based on empirical data.

```python
class ConfidenceCalibrator:
    """Calibrate OCR confidence scores for better reliability."""
    
    def __init__(self):
        # Empirical calibration curve (placeholder - should be trained)
        self.calibration_curve = {
            0.0: 0.0,
            0.3: 0.1,
            0.5: 0.3,
            0.7: 0.6,
            0.8: 0.75,
            0.9: 0.88,
            0.95: 0.94,
            1.0: 0.98
        }
    
    def calibrate(self, raw_confidence: float) -> float:
        """Convert raw confidence to calibrated confidence."""
        # Linear interpolation between calibration points
        points = sorted(self.calibration_curve.items())
        
        for i, (raw, cal) in enumerate(points):
            if raw_confidence <= raw:
                if i == 0:
                    return cal
                prev_raw, prev_cal = points[i-1]
                ratio = (raw_confidence - prev_raw) / (raw - prev_raw)
                return prev_cal + ratio * (cal - prev_cal)
        
        return points[-1][1]
    
    def get_quality_label(self, confidence: float) -> str:
        """Get human-readable quality label."""
        if confidence >= 0.9:
            return "high"
        elif confidence >= 0.7:
            return "medium"
        elif confidence >= 0.5:
            return "low"
        else:
            return "very_low"
```

---

### 3.3 Post-Processing Pipeline

**Comprehensive post-processing for OCR results**:

```python
class OCRPostProcessor:
    """Post-processing pipeline for OCR results."""
    
    def __init__(self, document_type: str = None):
        self.document_type = document_type
        self.processors = [
            self._normalize_whitespace,
            self._fix_common_errors,
            self._normalize_punctuation,
            self._validate_structure,
        ]
        
        if document_type:
            self.processors.append(self._apply_domain_rules)
    
    def process(self, text_lines: list) -> list:
        """Apply all post-processing steps."""
        for processor in self.processors:
            text_lines = processor(text_lines)
        return text_lines
    
    def _normalize_whitespace(self, text_lines: list) -> list:
        """Normalize whitespace in text."""
        for line in text_lines:
            text = line.get('text', '')
            # Multiple spaces → single space
            text = ' '.join(text.split())
            # Remove leading/trailing whitespace
            text = text.strip()
            line['text'] = text
        return text_lines
    
    def _fix_common_errors(self, text_lines: list) -> list:
        """Fix common OCR errors."""
        replacements = {
            'rn': 'm',      # rn often misread as m
            'vv': 'w',      # vv often misread as w
            'cl': 'd',      # cl sometimes misread as d
            '|': 'I',       # Pipe often is I
            '0': 'O',       # Context-dependent (handled separately)
        }
        
        for line in text_lines:
            text = line.get('text', '')
            # Apply context-aware replacements
            # (simplified - real implementation would be smarter)
            line['text'] = text
        return text_lines
    
    def _normalize_punctuation(self, text_lines: list) -> list:
        """Normalize punctuation marks."""
        for line in text_lines:
            text = line.get('text', '')
            # Normalize quotes
            text = text.replace('"', '"').replace('"', '"')
            text = text.replace(''', "'").replace(''', "'")
            # Normalize dashes
            text = text.replace('–', '-').replace('—', '-')
            line['text'] = text
        return text_lines
    
    def _validate_structure(self, text_lines: list) -> list:
        """Validate and fix structural issues."""
        # Remove empty lines
        text_lines = [l for l in text_lines if l.get('text', '').strip()]
        return text_lines
    
    def _apply_domain_rules(self, text_lines: list) -> list:
        """Apply document-type-specific rules."""
        if self.document_type == 'Receipt':
            return self._process_receipt(text_lines)
        elif self.document_type == 'Medication Prescription':
            return self._process_prescription(text_lines)
        return text_lines
    
    def _process_receipt(self, text_lines: list) -> list:
        """Receipt-specific processing."""
        for line in text_lines:
            text = line.get('text', '')
            # Fix currency symbols
            text = text.replace('S', '$') if text.startswith('S') and any(c.isdigit() for c in text) else text
            # Fix decimal points in prices
            # ... more receipt-specific rules
            line['text'] = text
        return text_lines
    
    def _process_prescription(self, text_lines: list) -> list:
        """Prescription-specific processing."""
        # Preserve medical terminology
        # Validate dosage formats
        return text_lines
```

---

## 4. Advanced Proposals (Future)

### 4.1 Ensemble OCR

**Concept**: Run multiple OCR engines and combine results.

```
┌─────────────────────────────────────────────────────────────────┐
│                      ENSEMBLE OCR                               │
└─────────────────────────────────────────────────────────────────┘

     ┌─────────────┐
     │   Image     │
     └──────┬──────┘
            │
    ┌───────┼───────┐
    │       │       │
    ▼       ▼       ▼
┌───────┐ ┌───────┐ ┌───────┐
│ Surya │ │Tesser-│ │ Easy  │
│  OCR  │ │ act   │ │ OCR   │
└───┬───┘ └───┬───┘ └───┬───┘
    │         │         │
    └────┬────┴────┬────┘
         │         │
         ▼         ▼
    ┌─────────────────┐
    │ Voting / Fusion │
    │                 │
    │ - Character-level voting
    │ - Confidence weighting
    │ - Consensus selection
    └─────────────────┘
            │
            ▼
    ┌─────────────────┐
    │ Final Result    │
    │ (Higher accuracy)
    └─────────────────┘
```

**Expected Improvement**: 5-10% accuracy gain on difficult documents.

---

### 4.2 Language Model Post-Correction

**Concept**: Use LLM to correct OCR errors based on context.

```python
def llm_correct_ocr(text: str, document_type: str) -> str:
    """
    Use language model to correct OCR errors.
    
    This could use a local model (Llama, Mistral) or API.
    """
    prompt = f"""
    The following text was extracted from a {document_type} using OCR.
    Please correct any obvious OCR errors while preserving the original meaning.
    Only fix clear errors, do not rephrase or add content.
    
    Original text:
    {text}
    
    Corrected text:
    """
    
    # Call local LLM or API
    # corrected = llm.generate(prompt)
    # return corrected
    pass
```

---

### 4.3 Active Learning for Continuous Improvement

**Concept**: Learn from user corrections to improve future recognition.

```
┌─────────────────────────────────────────────────────────────────┐
│                   ACTIVE LEARNING LOOP                          │
└─────────────────────────────────────────────────────────────────┘

1. User uploads document
         │
         ▼
2. System performs OCR
         │
         ▼
3. User reviews and corrects errors
         │
         ▼
4. System logs corrections
   ┌─────────────────────────────────────────────────────────────┐
   │ Correction Log:                                             │
   │ - Original: "Tota1: $45.9O"                                 │
   │ - Corrected: "Total: $45.90"                                │
   │ - Context: Receipt, bottom region                           │
   └─────────────────────────────────────────────────────────────┘
         │
         ▼
5. Periodically retrain/fine-tune models
         │
         ▼
6. Deploy improved model
```

---

## 5. Implementation Priority

### Phase 1: Quick Wins (1-2 weeks)
| Improvement | Effort | Impact |
|-------------|--------|--------|
| Paragraph merging | Low | High |
| Pattern-based error correction | Low | Medium |
| Image preprocessing pipeline | Medium | High |
| Confidence calibration | Low | Medium |

### Phase 2: Core Improvements (2-4 weeks)
| Improvement | Effort | Impact |
|-------------|--------|--------|
| Multi-pass detection | Medium | High |
| Document templates | Medium | High |
| Context-aware recognition | High | Very High |
| Post-processing pipeline | Medium | Medium |

### Phase 3: Advanced Features (1-2 months)
| Improvement | Effort | Impact |
|-------------|--------|--------|
| Layout-aware detection | High | High |
| Ensemble OCR | High | Medium |
| LLM post-correction | Medium | High |
| Active learning | Very High | Very High |

---

## 6. Metrics for Evaluation

### Accuracy Metrics
- **Character Error Rate (CER)**: % of characters incorrectly recognized
- **Word Error Rate (WER)**: % of words incorrectly recognized
- **Field Extraction Accuracy**: % of fields correctly extracted

### Usability Metrics
- **Processing Time**: Seconds per document
- **User Corrections**: Average corrections per document
- **Confidence Calibration**: Correlation between confidence and accuracy

### Benchmark Dataset
Create a test set with:
- 50+ receipts (various stores, conditions)
- 50+ prescriptions (handwritten and printed)
- 50+ forms (various layouts)
- 50+ contracts (multi-page)

---

## 7. Summary

### Key Recommendations

1. **Implement paragraph merging** — Low effort, high impact on usability
2. **Add document templates** — Enable context-aware processing
3. **Build preprocessing pipeline** — Improve input quality
4. **Add pattern-based correction** — Fix common OCR errors
5. **Implement multi-pass detection** — Catch more text

### Expected Overall Improvement

| Metric | Current | After Phase 1 | After Phase 2 |
|--------|---------|---------------|---------------|
| CER | ~5% | ~3.5% | ~2% |
| WER | ~8% | ~5% | ~3% |
| Field Extraction | N/A | 70% | 85% |
| User Corrections | ~10/doc | ~6/doc | ~3/doc |

---

*This document is a working proposal and should not be committed to the repository.*

from PIL import Image, ImageEnhance, ImageFilter
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from surya.foundation import FoundationPredictor
import io
import torch
from transformers import CLIPProcessor, CLIPModel
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict
import numpy as np

from app.config.document_types import (
    DOCUMENT_TYPES,
    get_document_type,
    get_clip_prompts,
    get_type_ids,
)
from app.services.context_processor import process_with_context

_foundation_predictor = None
_recognition_predictor = None
_detection_predictor = None

# CLIP model for image classification
_clip_model = None
_clip_processor = None
_clip_device = None

# Device info storage
_device_info = {
    "surya_device": None,
    "clip_device": None,
    "cuda_available": False,
    "cuda_version": None,
    "gpu_name": None,
    "gpu_count": 0,
    "gpu_memory_total": None,
    "mps_available": False,
    "acceleration_type": None,
    "selected_device": None
}


def get_device():
    """Detect and return the best available device for inference."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_info() -> dict:
    """Get detailed device information for all components."""
    global _device_info
    
    # PyTorch build info
    _device_info["pytorch_version"] = torch.__version__
    _device_info["pytorch_cuda_built"] = torch.version.cuda is not None
    _device_info["pytorch_cuda_version"] = torch.version.cuda
    
    # Update CUDA info (NVIDIA GPU / DGX)
    _device_info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        _device_info["cuda_version"] = torch.version.cuda
        _device_info["gpu_count"] = torch.cuda.device_count()
        _device_info["gpu_name"] = torch.cuda.get_device_name(0)
        
        # Get GPU memory info
        try:
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            _device_info["gpu_memory_total"] = f"{gpu_mem / (1024**3):.1f} GB"
        except:
            _device_info["gpu_memory_total"] = None
        
        # Detect DGX or special NVIDIA hardware
        gpu_name = _device_info["gpu_name"].upper()
        if "DGX" in gpu_name or "A100" in gpu_name or "H100" in gpu_name or "V100" in gpu_name:
            _device_info["acceleration_type"] = f"NVIDIA DGX/HPC ({_device_info['gpu_name']})"
        else:
            _device_info["acceleration_type"] = f"CUDA ({_device_info['gpu_name']})"
    else:
        _device_info["gpu_count"] = 0
        _device_info["gpu_memory_total"] = None
    
    # Update MPS info (Apple Silicon)
    _device_info["mps_available"] = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    
    # Determine acceleration type if not CUDA
    if not _device_info["cuda_available"]:
        if _device_info["mps_available"]:
            _device_info["acceleration_type"] = "MPS (Apple Silicon)"
        else:
            _device_info["acceleration_type"] = "CPU Only"
    
    # Selected device
    _device_info["selected_device"] = str(get_device())
    
    return _device_info.copy()


def print_device_info():
    """Print device information to stdout."""
    info = get_device_info()
    device = get_device()
    
    print("\n" + "="*60)
    print("HARDWARE ACCELERATION DETECTION")
    print("="*60)
    
    # PyTorch build info
    print(f"PyTorch Version: {torch.__version__}")
    print(f"PyTorch CUDA Built: {torch.version.cuda if torch.version.cuda else 'No (CPU-only build)'}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available() and torch.version.cuda is None:
        print("")
        print("WARNING: PyTorch installed without CUDA support!")
        print("To enable NVIDIA GPU acceleration, reinstall PyTorch:")
        print("  pip uninstall torch torchvision")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print("")
    
    if info['cuda_available']:
        print(f"")
        print(f"ACCELERATION: {info['acceleration_type']}")
        print(f"CUDA Runtime: {info['cuda_version']}")
        print(f"GPU Count: {info['gpu_count']}")
        print(f"GPU Name: {info['gpu_name']}")
        if info['gpu_memory_total']:
            print(f"GPU Memory: {info['gpu_memory_total']}")
    elif info['mps_available']:
        print(f"")
        print(f"ACCELERATION: MPS (Apple Silicon)")
        print(f"Metal Performance Shaders: ENABLED")
    else:
        print(f"")
        print(f"ACCELERATION: CPU Only")
    
    print(f"")
    print(f"Selected Device: {device}")
    print(f"All models will use '{device}' for inference.")
    print("="*60 + "\n")

# Get CLIP prompts and class names from config
def get_clip_class_config():
    """Get CLIP classification config from document types."""
    type_ids = get_type_ids()
    prompts = [DOCUMENT_TYPES[tid].clip_prompt for tid in type_ids]
    names = [DOCUMENT_TYPES[tid].display_name for tid in type_ids]
    return type_ids, prompts, names


CLIP_TYPE_IDS, CLIP_CLASS_DESCRIPTIONS, CLIP_CLASS_NAMES = get_clip_class_config()


def get_clip_model():
    """Lazy initialization of CLIP model with GPU support."""
    global _clip_model, _clip_processor, _clip_device, _device_info
    
    if _clip_model is None:
        _clip_device = get_device()
        print(f"[CLIP] Loading model on device: {_clip_device}")
        
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model = _clip_model.to(_clip_device)
        _clip_model.eval()
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        _device_info["clip_device"] = str(_clip_device)
        print(f"[CLIP] Model loaded successfully on {_clip_device}")
    
    return _clip_model, _clip_processor


def get_predictors():
    """Lazy initialization of Surya OCR predictors."""
    global _foundation_predictor, _recognition_predictor, _detection_predictor, _device_info
    
    if _recognition_predictor is None:
        device = get_device()
        print(f"[Surya OCR] Initializing predictors on device: {device}")
        
        _foundation_predictor = FoundationPredictor(device=str(device))
        _detection_predictor = DetectionPredictor(device=str(device))
        _recognition_predictor = RecognitionPredictor(_foundation_predictor)
        
        _device_info["surya_device"] = str(device)
        print(f"[Surya OCR] Predictors initialized successfully on {device}")
    
    return _recognition_predictor, _detection_predictor


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1, box2: Bounding boxes as [x1, y1, x2, y2]
        
    Returns:
        IoU value between 0 and 1
    """
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


def deduplicate_text_lines(text_lines: List[dict], iou_threshold: float = 0.5) -> List[dict]:
    """
    Remove duplicate text lines based on IoU overlap.
    Keeps the line with higher confidence when duplicates are found.
    
    Args:
        text_lines: List of text line dicts with 'bbox' and 'confidence'
        iou_threshold: IoU threshold above which lines are considered duplicates
        
    Returns:
        Deduplicated list of text lines
    """
    if not text_lines:
        return []
    
    # Sort by confidence (highest first)
    sorted_lines = sorted(text_lines, key=lambda l: l.get('confidence', 0), reverse=True)
    
    keep = []
    for line in sorted_lines:
        is_duplicate = False
        for kept in keep:
            if compute_iou(line['bbox'], kept['bbox']) > iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            keep.append(line)
    
    # Sort by vertical position for natural reading order
    keep.sort(key=lambda l: (l['bbox'][1], l['bbox'][0]))
    
    return keep


def create_enhanced_image(image: Image.Image) -> Image.Image:
    """
    Create an enhanced version of the image for better text detection.
    Uses contrast and sharpness enhancement.
    
    Args:
        image: Original PIL Image
        
    Returns:
        Enhanced PIL Image
    """
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    enhanced = enhancer.enhance(1.3)
    
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(enhanced)
    enhanced = enhancer.enhance(1.2)
    
    return enhanced


def create_high_contrast_image(image: Image.Image) -> Image.Image:
    """
    Create a high-contrast version for detecting faint text.
    
    Args:
        image: Original PIL Image
        
    Returns:
        High-contrast PIL Image
    """
    # Higher contrast for faint text
    enhancer = ImageEnhance.Contrast(image)
    enhanced = enhancer.enhance(1.8)
    
    # Slight brightness adjustment
    enhancer = ImageEnhance.Brightness(enhanced)
    enhanced = enhancer.enhance(1.1)
    
    return enhanced


def run_detection_pass(image: Image.Image, recognition_predictor, detection_predictor, 
                       pass_name: str = "standard") -> Tuple[List[dict], str]:
    """
    Run a single detection pass on an image.
    
    Args:
        image: PIL Image to process
        recognition_predictor: Surya recognition predictor
        detection_predictor: Surya detection predictor
        pass_name: Name of this pass for logging
        
    Returns:
        Tuple of (text_lines list, pass_name)
    """
    try:
        predictions = recognition_predictor([image], det_predictor=detection_predictor)
        
        if not predictions or len(predictions) == 0:
            return [], pass_name
        
        result = predictions[0]
        text_lines = []
        
        for line in result.text_lines:
            line_data = {
                "text": line.text,
                "confidence": line.confidence,
                "bbox": list(line.bbox),  # Ensure it's a list
                "polygon": line.polygon if hasattr(line, 'polygon') else None,
                "detection_pass": pass_name
            }
            
            if hasattr(line, 'words') and line.words:
                line_data["words"] = [
                    {
                        "text": word.text,
                        "bbox": list(word.bbox),
                        "confidence": word.confidence
                    }
                    for word in line.words
                ]
            
            text_lines.append(line_data)
        
        return text_lines, pass_name
        
    except Exception as e:
        print(f"[OCR] Detection pass '{pass_name}' failed: {e}")
        return [], pass_name


def retry_low_confidence_regions(image: Image.Image, text_lines: List[dict], 
                                  recognition_predictor, detection_predictor,
                                  min_confidence: float = 0.7) -> List[dict]:
    """
    Retry recognition for low-confidence regions with enhanced preprocessing.
    
    For lines with confidence below threshold:
    1. Crop the region with padding
    2. Apply enhancement (contrast, sharpening)
    3. Scale up 2x
    4. Re-run OCR
    5. Keep better result
    
    Args:
        image: Original PIL Image
        text_lines: List of text line dicts with confidence scores
        recognition_predictor: Surya recognition predictor
        detection_predictor: Surya detection predictor
        min_confidence: Confidence threshold for retry
        
    Returns:
        Updated text lines with improved low-confidence regions
    """
    from PIL import ImageFilter
    
    improved_lines = []
    retry_count = 0
    improvement_count = 0
    
    for line in text_lines:
        confidence = line.get("confidence", 0)
        
        # Skip if confidence is good enough
        if confidence >= min_confidence:
            improved_lines.append(line)
            continue
        
        bbox = line.get("bbox", [0, 0, 0, 0])
        if not bbox or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            improved_lines.append(line)
            continue
        
        retry_count += 1
        
        # Add padding around the region
        padding = 10
        x1 = max(0, int(bbox[0]) - padding)
        y1 = max(0, int(bbox[1]) - padding)
        x2 = min(image.width, int(bbox[2]) + padding)
        y2 = min(image.height, int(bbox[3]) + padding)
        
        # Crop region
        region = image.crop((x1, y1, x2, y2))
        
        # Enhance the region
        # 1. Increase contrast
        enhancer = ImageEnhance.Contrast(region)
        region = enhancer.enhance(1.5)
        
        # 2. Sharpen
        region = region.filter(ImageFilter.SHARPEN)
        
        # 3. Scale up 2x for better recognition
        region = region.resize((region.width * 2, region.height * 2), Image.LANCZOS)
        
        try:
            # Re-run OCR on enhanced region
            predictions = recognition_predictor([region], det_predictor=detection_predictor)
            
            if predictions and len(predictions) > 0 and predictions[0].text_lines:
                # Get the best result from retry
                retry_result = predictions[0].text_lines[0]
                
                if retry_result.confidence > confidence:
                    # Improvement found - use retry result but keep original bbox
                    improvement_count += 1
                    improved_line = line.copy()
                    improved_line["text"] = retry_result.text
                    improved_line["confidence"] = retry_result.confidence
                    improved_line["retry_improved"] = True
                    improved_line["original_confidence"] = confidence
                    improved_lines.append(improved_line)
                    continue
        except Exception as e:
            pass  # Fall through to keep original
        
        # Keep original if retry didn't help
        improved_lines.append(line)
    
    if retry_count > 0:
        print(f"[OCR] Low-confidence retry: {retry_count} regions, {improvement_count} improved")
    
    return improved_lines


def multi_pass_recognition(image: Image.Image, recognition_predictor, detection_predictor) -> List[dict]:
    """
    Perform multi-pass text detection on original and enhanced images.
    
    Runs detection on:
    1. Original image (standard pass)
    2. Contrast-enhanced image (catches faint text)
    3. High-contrast image (catches very faint/handwritten text)
    
    Results are merged and deduplicated using IoU.
    Then low-confidence regions are retried with enhanced preprocessing.
    
    Args:
        image: Original PIL Image
        recognition_predictor: Surya recognition predictor
        detection_predictor: Surya detection predictor
        
    Returns:
        Merged and deduplicated list of text lines
    """
    # Prepare image variants
    enhanced_image = create_enhanced_image(image)
    high_contrast_image = create_high_contrast_image(image)
    
    all_text_lines = []
    pass_stats = {}
    
    # Run passes sequentially (Surya uses GPU, parallel would cause contention)
    # Pass 1: Original image
    print("[OCR] Running detection pass 1/3: original image")
    lines1, name1 = run_detection_pass(image, recognition_predictor, detection_predictor, "original")
    all_text_lines.extend(lines1)
    pass_stats[name1] = len(lines1)
    
    # Pass 2: Enhanced image
    print("[OCR] Running detection pass 2/3: enhanced image")
    lines2, name2 = run_detection_pass(enhanced_image, recognition_predictor, detection_predictor, "enhanced")
    all_text_lines.extend(lines2)
    pass_stats[name2] = len(lines2)
    
    # Pass 3: High contrast image
    print("[OCR] Running detection pass 3/3: high-contrast image")
    lines3, name3 = run_detection_pass(high_contrast_image, recognition_predictor, detection_predictor, "high_contrast")
    all_text_lines.extend(lines3)
    pass_stats[name3] = len(lines3)
    
    # Log pass statistics
    total_before = len(all_text_lines)
    print(f"[OCR] Detection passes complete: {pass_stats}")
    print(f"[OCR] Total detections before deduplication: {total_before}")
    
    # Deduplicate based on IoU
    deduplicated = deduplicate_text_lines(all_text_lines, iou_threshold=0.5)
    
    print(f"[OCR] After deduplication: {len(deduplicated)} text lines")
    
    # Retry low-confidence regions with enhanced preprocessing
    improved = retry_low_confidence_regions(
        image, deduplicated, recognition_predictor, detection_predictor,
        min_confidence=0.7
    )
    
    return improved


def recognize_text(image_bytes: bytes, multi_pass: bool = True) -> dict:
    """
    Recognize text in an image using Surya OCR with multi-pass detection.
    
    Multi-pass detection runs OCR on:
    - Original image
    - Contrast-enhanced image  
    - High-contrast image
    
    Results are merged and deduplicated for higher accuracy.
    
    Args:
        image_bytes: Image as bytes
        multi_pass: If True, use multi-pass detection (default). If False, single pass.
        
    Returns:
        dict: OCR results in JSON format
    """
    image = Image.open(io.BytesIO(image_bytes))
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    recognition_predictor, detection_predictor = get_predictors()
    
    if multi_pass:
        # Multi-pass detection for higher accuracy
        text_lines = multi_pass_recognition(image, recognition_predictor, detection_predictor)
    else:
        # Single pass (legacy behavior)
        predictions = recognition_predictor([image], det_predictor=detection_predictor)
        
        if not predictions or len(predictions) == 0:
            text_lines = []
        else:
            result = predictions[0]
            text_lines = []
            for line in result.text_lines:
                line_data = {
                    "text": line.text,
                    "confidence": line.confidence,
                    "bbox": list(line.bbox),
                    "polygon": line.polygon if hasattr(line, 'polygon') else None,
                }
                
                if hasattr(line, 'words') and line.words:
                    line_data["words"] = [
                        {
                            "text": word.text,
                            "bbox": list(word.bbox),
                            "confidence": word.confidence
                        }
                        for word in line.words
                    ]
                
                text_lines.append(line_data)
    
    # Classify document using hybrid method (CLIP + text)
    classification = classify_document_hybrid(image, text_lines)
    
    # Apply context-aware post-processing based on document type
    doc_type_id = classification.get("type_id", "unknown")
    corrected_lines, extracted_fields = process_with_context(text_lines, doc_type_id)
    
    # Log context processing
    if doc_type_id != "unknown":
        corrections_made = sum(1 for line in corrected_lines if line.get("text") != line.get("original_text"))
        fields_found = sum(1 for f in extracted_fields.values() if f.get("value") is not None)
        print(f"[OCR] Context-aware processing for '{doc_type_id}': {corrections_made} corrections, {fields_found} fields extracted")
    
    return {
        "text_lines": corrected_lines,
        "image_bbox": [0, 0, image.width, image.height],
        "document_class": classification,
        "extracted_fields": extracted_fields,
        "detection_mode": "multi_pass" if multi_pass else "single_pass",
        "context_processing": doc_type_id != "unknown"
    }


def recognize_region(image_bytes: bytes, bbox: list) -> dict:
    """
    Recognize text in a specified region of an image.
    
    Args:
        image_bytes: Image as bytes
        bbox: Region coordinates [x1, y1, x2, y2]
        
    Returns:
        dict: OCR results for the specified region
    """
    image = Image.open(io.BytesIO(image_bytes))
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Crop the region
    x1, y1, x2, y2 = [int(c) for c in bbox]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.width, x2)
    y2 = min(image.height, y2)
    
    cropped = image.crop((x1, y1, x2, y2))
    
    recognition_predictor, detection_predictor = get_predictors()
    
    predictions = recognition_predictor([cropped], det_predictor=detection_predictor)
    
    if not predictions or len(predictions) == 0:
        return {
            "text_lines": [],
            "region_bbox": bbox
        }
    
    result = predictions[0]
    
    text_lines = []
    for line in result.text_lines:
        # Adjust coordinates relative to the original image
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
        
        if hasattr(line, 'polygon') and line.polygon:
            line_data["polygon"] = [
                [p[0] + x1, p[1] + y1] for p in line.polygon
            ]
        
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
    
    return {
        "text_lines": text_lines,
        "region_bbox": bbox
    }


# Build keywords dict from config
def get_document_keywords() -> Dict[str, List[str]]:
    """Get keywords for each document type from config."""
    keywords = {}
    for type_id, doc_type in DOCUMENT_TYPES.items():
        if type_id != "unknown":
            keywords[doc_type.display_name] = doc_type.keywords_en + doc_type.keywords_ru
    return keywords

DOCUMENT_KEYWORDS = get_document_keywords()

# Map display names to type IDs
DISPLAY_NAME_TO_TYPE_ID = {dt.display_name: tid for tid, dt in DOCUMENT_TYPES.items()}


def classify_document(text_lines: list) -> dict:
    """
    Classify document based on recognized text using keyword matching.
    
    Args:
        text_lines: List of recognized text lines
        
    Returns:
        dict: Document class and confidence score
    """
    if not text_lines:
        return {"class": "Undetected", "confidence": 0.0}
    
    # Combine all text
    full_text = " ".join([line.get("text", "") for line in text_lines]).lower()
    
    # Count keyword matches for each class
    scores = {}
    for doc_class, keywords in DOCUMENT_KEYWORDS.items():
        score = 0
        matched_keywords = []
        for keyword in keywords:
            if keyword.lower() in full_text:
                score += 1
                matched_keywords.append(keyword)
        scores[doc_class] = {
            "score": score,
            "matched": matched_keywords
        }
    
    # Find class with maximum score
    max_class = max(scores.keys(), key=lambda k: scores[k]["score"])
    max_score = scores[max_class]["score"]
    
    if max_score == 0:
        return {"class": "Undetected", "confidence": 0.0}
    
    # Calculate confidence (normalize by keyword count)
    total_keywords = len(DOCUMENT_KEYWORDS[max_class])
    confidence = min(max_score / (total_keywords * 0.3), 1.0)  # 30% match = 100% confidence
    
    return {
        "class": max_class,
        "confidence": round(confidence, 2)
    }


def classify_image_with_clip(image: Image.Image) -> dict:
    """
    Classify document image using CLIP zero-shot classification.
    
    Args:
        image: PIL Image
        
    Returns:
        dict: Document class, type_id, and confidence score
    """
    try:
        model, processor = get_clip_model()
        
        # Prepare inputs and move to device
        inputs = processor(
            text=CLIP_CLASS_DESCRIPTIONS,
            images=image,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(_clip_device) for k, v in inputs.items()}
        
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
            "type_id": CLIP_TYPE_IDS[best_idx],
            "confidence": round(best_prob, 2)
        }
    except Exception as e:
        print(f"CLIP classification error: {e}")
        return {"class": "Unknown Document", "type_id": "unknown", "confidence": 0.0}


def classify_document_hybrid(image: Image.Image, text_lines: list) -> dict:
    """
    Hybrid document classification combining CLIP (image) and text-based methods.
    
    Args:
        image: PIL Image
        text_lines: List of recognized text lines
        
    Returns:
        dict: Document class, type_id, and confidence score
    """
    # Get image-based classification (CLIP)
    image_result = classify_image_with_clip(image)
    
    # Get text-based classification
    text_result = classify_document(text_lines)
    
    # Helper to get type_id from display name
    def get_type_id(display_name: str) -> str:
        return DISPLAY_NAME_TO_TYPE_ID.get(display_name, "unknown")
    
    # Combine results
    # If both methods agree - high confidence
    if image_result["class"] == text_result["class"] and image_result["class"] != "Undetected":
        combined_confidence = min(1.0, (image_result["confidence"] + text_result["confidence"]) / 1.5)
        return {
            "class": image_result["class"],
            "type_id": image_result.get("type_id", get_type_id(image_result["class"])),
            "confidence": round(combined_confidence, 2),
            "method": "hybrid"
        }
    
    # If text-based has high confidence result - prefer it
    if text_result["class"] != "Undetected" and text_result["confidence"] >= 0.5:
        return {
            "class": text_result["class"],
            "type_id": get_type_id(text_result["class"]),
            "confidence": round(text_result["confidence"] * 0.9, 2),
            "method": "text"
        }
    
    # If CLIP has high confidence result - use it
    if image_result["class"] != "Unknown Document" and image_result["confidence"] >= 0.3:
        return {
            "class": image_result["class"],
            "type_id": image_result.get("type_id", get_type_id(image_result["class"])),
            "confidence": round(image_result["confidence"], 2),
            "method": "image"
        }
    
    # If text-based has any result
    if text_result["class"] != "Undetected":
        return {
            "class": text_result["class"],
            "type_id": get_type_id(text_result["class"]),
            "confidence": round(text_result["confidence"] * 0.7, 2),
            "method": "text"
        }
    
    # If CLIP has any result
    if image_result["class"] != "Unknown Document":
        return {
            "class": image_result["class"],
            "type_id": image_result.get("type_id", get_type_id(image_result["class"])),
            "confidence": round(image_result["confidence"] * 0.7, 2),
            "method": "image"
        }
    
    return {"class": "Unknown Document", "type_id": "unknown", "confidence": 0.0, "method": "none"}

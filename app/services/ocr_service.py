from PIL import Image
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from surya.foundation import FoundationPredictor
import io
import torch
from transformers import CLIPProcessor, CLIPModel

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
    "mps_available": False
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
    
    # Update CUDA info
    _device_info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        _device_info["cuda_version"] = torch.version.cuda
        _device_info["gpu_name"] = torch.cuda.get_device_name(0)
    
    # Update MPS info (Apple Silicon)
    _device_info["mps_available"] = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    
    return _device_info.copy()


def print_device_info():
    """Print device information to stdout."""
    info = get_device_info()
    print("\n" + "="*60)
    print("DEVICE DETECTION RESULTS")
    print("="*60)
    print(f"CUDA Available: {info['cuda_available']}")
    if info['cuda_available']:
        print(f"CUDA Version: {info['cuda_version']}")
        print(f"GPU Name: {info['gpu_name']}")
    print(f"MPS Available (Apple Silicon): {info['mps_available']}")
    print(f"Surya OCR Device: {info['surya_device'] or 'Not initialized'}")
    print(f"CLIP Device: {info['clip_device'] or 'Not initialized'}")
    print("="*60 + "\n")

# Class descriptions for CLIP zero-shot classification
CLIP_CLASS_DESCRIPTIONS = [
    "a photo of a receipt or invoice with prices and totals",
    "a photo of a medical prescription or medication document",
    "a photo of a form or application document with fields to fill",
    "a photo of a contract or legal agreement document",
    "a photo of an unknown or unclassified document"
]

CLIP_CLASS_NAMES = ["Receipt", "Medication Prescription", "Form", "Contract", "Undetected"]


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


def recognize_text(image_bytes: bytes) -> dict:
    """
    Recognize text in an image using Surya OCR.
    
    Args:
        image_bytes: Image as bytes
        
    Returns:
        dict: OCR results in JSON format
    """
    image = Image.open(io.BytesIO(image_bytes))
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    recognition_predictor, detection_predictor = get_predictors()
    
    predictions = recognition_predictor([image], det_predictor=detection_predictor)
    
    if not predictions or len(predictions) == 0:
        return {
            "text_lines": [],
            "image_bbox": [0, 0, image.width, image.height]
        }
    
    result = predictions[0]
    
    text_lines = []
    for line in result.text_lines:
        line_data = {
            "text": line.text,
            "confidence": line.confidence,
            "bbox": line.bbox,
            "polygon": line.polygon if hasattr(line, 'polygon') else None,
        }
        
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
    
    # Classify document using hybrid method (CLIP + text)
    classification = classify_document_hybrid(image, text_lines)
    
    return {
        "text_lines": text_lines,
        "image_bbox": [0, 0, image.width, image.height],
        "document_class": classification
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


# Keywords for document classification (English and Russian)
DOCUMENT_KEYWORDS = {
    "Receipt": [
        "receipt", "total", "subtotal", "tax", "payment", "cash", "change",
        "qty", "price", "amount", "item", "чек", "итого", "сумма", "оплата",
        "касса", "товар", "цена", "ндс", "скидка"
    ],
    "Medication Prescription": [
        "prescription", "rx", "medication", "dose", "dosage", "tablet", "capsule",
        "mg", "ml", "take", "daily", "doctor", "patient", "pharmacy", "refill",
        "рецепт", "препарат", "доза", "таблетка", "капсула", "принимать",
        "врач", "пациент", "аптека", "лекарство"
    ],
    "Form": [
        "form", "application", "name", "date", "signature", "address", "phone",
        "email", "please fill", "required", "checkbox", "заявление", "форма",
        "анкета", "фио", "дата", "подпись", "адрес", "телефон", "заполните"
    ],
    "Contract": [
        "contract", "agreement", "party", "parties", "terms", "conditions",
        "hereby", "whereas", "witness", "signed", "effective date", "obligations",
        "договор", "контракт", "соглашение", "сторона", "стороны", "условия",
        "обязательства", "подписано", "дата вступления"
    ]
}


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
        dict: Document class and confidence score
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
            "confidence": round(best_prob, 2)
        }
    except Exception as e:
        print(f"CLIP classification error: {e}")
        return {"class": "Undetected", "confidence": 0.0}


def classify_document_hybrid(image: Image.Image, text_lines: list) -> dict:
    """
    Hybrid document classification combining CLIP (image) and text-based methods.
    
    Args:
        image: PIL Image
        text_lines: List of recognized text lines
        
    Returns:
        dict: Document class and confidence score
    """
    # Get image-based classification (CLIP)
    image_result = classify_image_with_clip(image)
    
    # Get text-based classification
    text_result = classify_document(text_lines)
    
    # Combine results
    # If both methods agree - high confidence
    if image_result["class"] == text_result["class"] and image_result["class"] != "Undetected":
        combined_confidence = min(1.0, (image_result["confidence"] + text_result["confidence"]) / 1.5)
        return {
            "class": image_result["class"],
            "confidence": round(combined_confidence, 2),
            "method": "hybrid"
        }
    
    # If text-based has high confidence result - prefer it
    if text_result["class"] != "Undetected" and text_result["confidence"] >= 0.5:
        return {
            "class": text_result["class"],
            "confidence": round(text_result["confidence"] * 0.9, 2),
            "method": "text"
        }
    
    # If CLIP has high confidence result - use it
    if image_result["class"] != "Undetected" and image_result["confidence"] >= 0.3:
        return {
            "class": image_result["class"],
            "confidence": round(image_result["confidence"], 2),
            "method": "image"
        }
    
    # If text-based has any result
    if text_result["class"] != "Undetected":
        return {
            "class": text_result["class"],
            "confidence": round(text_result["confidence"] * 0.7, 2),
            "method": "text"
        }
    
    # If CLIP has any result
    if image_result["class"] != "Undetected":
        return {
            "class": image_result["class"],
            "confidence": round(image_result["confidence"] * 0.7, 2),
            "method": "image"
        }
    
    return {"class": "Undetected", "confidence": 0.0, "method": "none"}

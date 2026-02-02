from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List
import uuid
import os

from app.services.ocr_service import recognize_text, recognize_region, get_device_info
from app.services.image_service import normalize_image, deskew_only, enhance_only

router = APIRouter(prefix="/api", tags=["api"])

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

images_store: dict[str, dict] = {}


class ImageInfo(BaseModel):
    id: str
    filename: str


class UploadResponse(BaseModel):
    images: List[ImageInfo]


class ImageIdRequest(BaseModel):
    image_id: str


class RecognizeRegionRequest(BaseModel):
    image_id: str
    bbox: List[float]


class NormalizeRequest(BaseModel):
    image_id: str
    enhance: bool = True  # Apply full enhancement (contrast, denoise, sharpen)


class NormalizeResponse(BaseModel):
    image_id: str
    normalized: bool
    angle: float
    operations: list = []
    quality_metrics: dict = {}


class DeskewResponse(BaseModel):
    image_id: str
    deskewed: bool
    angle: float


class EnhanceResponse(BaseModel):
    image_id: str
    enhanced: bool
    operations: list = []
    quality_metrics: dict = {}
    can_revert: bool = False


class RevertResponse(BaseModel):
    image_id: str
    reverted: bool


@router.post("/upload", response_model=UploadResponse)
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
        
        uploaded.append(ImageInfo(id=image_id, filename=file.filename))
    
    if not uploaded:
        raise HTTPException(status_code=400, detail="No valid images uploaded")
    
    return UploadResponse(images=uploaded)


@router.post("/normalize", response_model=NormalizeResponse)
async def normalize(request: NormalizeRequest):
    """
    Normalize and enhance image for optimal OCR recognition.
    
    Applies preprocessing pipeline:
    - Auto-rotate based on EXIF orientation
    - Deskew (straighten text lines)
    - Denoise (reduce scanner/camera noise) [if enhance=True]
    - Contrast enhancement (CLAHE) [if enhance=True]
    - Sharpen (crisp text edges) [if enhance=True]
    """
    image_id = request.image_id
    
    if image_id not in images_store:
        raise HTTPException(status_code=404, detail="Image not found")
    
    image_info = images_store[image_id]
    
    with open(image_info["path"], "rb") as f:
        image_bytes = f.read()
    
    normalized_bytes, angle, preprocessing_info = normalize_image(image_bytes, enhance=request.enhance)
    
    normalized_path = os.path.join(UPLOAD_DIR, f"{image_id}_normalized.png")
    with open(normalized_path, "wb") as f:
        f.write(normalized_bytes)
    
    images_store[image_id]["path"] = normalized_path
    
    return NormalizeResponse(
        image_id=image_id, 
        normalized=True, 
        angle=angle,
        operations=preprocessing_info.get("operations", []),
        quality_metrics=preprocessing_info.get("quality_metrics", {})
    )


@router.post("/deskew", response_model=DeskewResponse)
async def deskew(request: ImageIdRequest):
    """
    Deskew image only (straighten text lines).
    
    Applies:
    - Auto-rotate based on EXIF orientation
    - Deskew using projection profile analysis
    
    Does NOT apply quality enhancements (contrast, denoise, sharpen).
    """
    image_id = request.image_id
    
    if image_id not in images_store:
        raise HTTPException(status_code=404, detail="Image not found")
    
    image_info = images_store[image_id]
    
    with open(image_info["path"], "rb") as f:
        image_bytes = f.read()
    
    deskewed_bytes, angle = deskew_only(image_bytes)
    
    deskewed_path = os.path.join(UPLOAD_DIR, f"{image_id}_deskewed.png")
    with open(deskewed_path, "wb") as f:
        f.write(deskewed_bytes)
    
    images_store[image_id]["path"] = deskewed_path
    
    return DeskewResponse(
        image_id=image_id,
        deskewed=True,
        angle=angle
    )


@router.post("/enhance", response_model=EnhanceResponse)
async def enhance(request: ImageIdRequest):
    """
    Enhance image quality only (no rotation/deskew).
    
    Applies:
    - Denoise (reduce scanner/camera noise)
    - Contrast enhancement (CLAHE, blended 50% with original)
    - Sharpen (crisp text edges)
    
    Does NOT apply deskew/rotation.
    Stores pre-enhance path for revert functionality.
    """
    image_id = request.image_id
    
    if image_id not in images_store:
        raise HTTPException(status_code=404, detail="Image not found")
    
    image_info = images_store[image_id]
    
    # Store current path for revert (before enhancement)
    images_store[image_id]["pre_enhance_path"] = image_info["path"]
    
    with open(image_info["path"], "rb") as f:
        image_bytes = f.read()
    
    enhanced_bytes, preprocessing_info = enhance_only(image_bytes)
    
    enhanced_path = os.path.join(UPLOAD_DIR, f"{image_id}_enhanced.png")
    with open(enhanced_path, "wb") as f:
        f.write(enhanced_bytes)
    
    images_store[image_id]["path"] = enhanced_path
    
    return EnhanceResponse(
        image_id=image_id,
        enhanced=True,
        operations=preprocessing_info.get("operations", []),
        quality_metrics=preprocessing_info.get("quality_metrics", {}),
        can_revert=True
    )


@router.post("/revert-enhance", response_model=RevertResponse)
async def revert_enhance(request: ImageIdRequest):
    """
    Revert to pre-enhancement image.
    
    Restores the image to its state before the last enhance operation.
    """
    image_id = request.image_id
    
    if image_id not in images_store:
        raise HTTPException(status_code=404, detail="Image not found")
    
    image_info = images_store[image_id]
    
    if "pre_enhance_path" not in image_info:
        raise HTTPException(status_code=400, detail="No enhancement to revert")
    
    # Restore pre-enhance path
    images_store[image_id]["path"] = image_info["pre_enhance_path"]
    del images_store[image_id]["pre_enhance_path"]
    
    return RevertResponse(
        image_id=image_id,
        reverted=True
    )


@router.post("/recognize")
async def recognize(request: ImageIdRequest):
    """Recognize text in image using Surya OCR."""
    image_id = request.image_id
    
    if image_id not in images_store:
        raise HTTPException(status_code=404, detail="Image not found")
    
    image_info = images_store[image_id]
    
    with open(image_info["path"], "rb") as f:
        image_bytes = f.read()
    
    result = recognize_text(image_bytes)
    
    return result


@router.get("/image/{image_id}")
async def get_image(image_id: str):
    """Retrieve image by ID."""
    if image_id not in images_store:
        raise HTTPException(status_code=404, detail="Image not found")
    
    image_info = images_store[image_id]
    
    with open(image_info["path"], "rb") as f:
        image_bytes = f.read()
    
    return Response(content=image_bytes, media_type="image/png")


@router.post("/recognize-region")
async def recognize_region_endpoint(request: RecognizeRegionRequest):
    """Recognize text in a specified region of the image."""
    image_id = request.image_id
    
    if image_id not in images_store:
        raise HTTPException(status_code=404, detail="Image not found")
    
    image_info = images_store[image_id]
    
    with open(image_info["path"], "rb") as f:
        image_bytes = f.read()
    
    result = recognize_region(image_bytes, request.bbox)
    
    return result


@router.get("/device-info")
async def device_info():
    """Get GPU/device information for OCR and CLIP models."""
    return get_device_info()

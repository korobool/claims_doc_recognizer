from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Optional
import uuid
import os

from app.services.ocr_service import recognize_text, recognize_region, get_device_info
from app.services.image_service import normalize_image
from app.services.llm_service import (
    get_ollama_client, 
    get_llm_processor, 
    LLMModel,
    OllamaClient
)

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


class NormalizeResponse(BaseModel):
    image_id: str
    normalized: bool
    angle: float


class LLMProcessRequest(BaseModel):
    text: str
    model: Optional[str] = None
    document_type: Optional[str] = None


class LLMModelInfo(BaseModel):
    id: str
    name: str
    available: bool


class LLMStatusResponse(BaseModel):
    ollama_available: bool
    models: List[LLMModelInfo]


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
async def normalize(request: ImageIdRequest):
    """Normalize image orientation (deskew text)."""
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
    
    return NormalizeResponse(image_id=image_id, normalized=True, angle=angle)


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


# =============================================================================
# LLM Post-Processing Endpoints
# =============================================================================

@router.get("/llm/status", response_model=LLMStatusResponse)
async def llm_status():
    """Check LLM service status and available models."""
    client = get_ollama_client()
    ollama_available = await client.is_available()
    
    models = []
    available_models = await client.list_models() if ollama_available else []
    
    for model in LLMModel:
        model_name = model.value.split(":")[0]
        is_available = any(model_name in m for m in available_models)
        models.append(LLMModelInfo(
            id=model.value,
            name=model.display_name,
            available=is_available
        ))
    
    return LLMStatusResponse(
        ollama_available=ollama_available,
        models=models
    )


@router.post("/llm/pull/{model_id}")
async def llm_pull_model(model_id: str):
    """Pull a model from Ollama registry if not available."""
    client = get_ollama_client()
    
    if not await client.is_available():
        raise HTTPException(status_code=503, detail="Ollama service not available")
    
    # Find the model
    model = LLMModel.from_string(model_id)
    
    # Check if already available
    if await client.is_model_available(model):
        return {"status": "already_available", "model": model.display_name}
    
    # Pull the model (this can take a while)
    success = await client.pull_model(model)
    
    if success:
        return {"status": "pulled", "model": model.display_name}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to pull model {model.display_name}")


@router.post("/llm/process")
async def llm_process_text(request: LLMProcessRequest):
    """
    Process OCR text using LLM for context-aware structured extraction.
    
    Performs:
    - Context-aware text correction based on document type
    - Structured field extraction using document schemas
    - Medicine name recognition (for medical documents)
    - OCR artifact removal
    
    Returns structured data with extracted fields matching the document schema.
    """
    client = get_ollama_client()
    
    if not await client.is_available():
        raise HTTPException(status_code=503, detail="Ollama service not available. Please start Ollama.")
    
    # Get model
    model = LLMModel.from_string(request.model) if request.model else None
    
    # Check if model is available
    if model and not await client.is_model_available(model):
        raise HTTPException(
            status_code=400, 
            detail=f"Model {model.display_name} not available. Please pull it first."
        )
    
    processor = get_llm_processor()
    
    # Use context-aware structured extraction for all document types
    result = await processor.process_text(request.text, model, request.document_type)
    
    return result


@router.post("/llm/denoise")
async def llm_denoise_text(request: LLMProcessRequest):
    """Quick denoising of OCR text - removes artifacts and fixes obvious errors."""
    client = get_ollama_client()
    
    if not await client.is_available():
        raise HTTPException(status_code=503, detail="Ollama service not available")
    
    model = LLMModel.from_string(request.model) if request.model else None
    processor = get_llm_processor()
    
    denoised = await processor.denoise_text(request.text, model)
    
    return {
        "original_text": request.text,
        "denoised_text": denoised
    }

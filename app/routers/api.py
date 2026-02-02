from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, AsyncGenerator
import uuid
import os
import json
import httpx

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
    acceleration: str = "unknown"
    acceleration_details: Optional[str] = None
    version: Optional[str] = None


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
    """Check LLM service status, available models, and acceleration info."""
    client = get_ollama_client()
    
    # Get system info including acceleration
    system_info = await client.get_system_info()
    ollama_available = system_info.get("available", False)
    
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
        models=models,
        acceleration=system_info.get("acceleration", "unknown"),
        acceleration_details=system_info.get("acceleration_details"),
        version=system_info.get("version")
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


@router.get("/llm/pull/{model_id}/stream")
async def llm_pull_model_stream(model_id: str):
    """Pull a model with streaming progress updates."""
    print(f"[LLM Pull] Request to pull model: {model_id}")
    client = get_ollama_client()
    
    if not await client.is_available():
        print("[LLM Pull] ERROR: Ollama service not available")
        raise HTTPException(status_code=503, detail="Ollama service not available")
    
    model = LLMModel.from_string(model_id)
    print(f"[LLM Pull] Resolved model: {model.value} ({model.display_name})")
    
    # Check if already available
    if await client.is_model_available(model):
        print(f"[LLM Pull] Model {model.value} is already available")
        async def already_available():
            yield f"data: {json.dumps({'status': 'already_available', 'model': model.display_name})}\n\n"
        return StreamingResponse(already_available(), media_type="text/event-stream")
    
    print(f"[LLM Pull] Starting download of {model.value}...")
    
    async def pull_stream() -> AsyncGenerator[str, None]:
        """Stream pull progress as Server-Sent Events."""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as http_client:
                async with http_client.stream(
                    "POST",
                    f"{client.base_url}/api/pull",
                    json={"name": model.value}
                ) as response:
                    if response.status_code != 200:
                        print(f"[LLM Pull] ERROR: Failed to start pull, status {response.status_code}")
                        yield f"data: {json.dumps({'status': 'error', 'error': 'Failed to start pull'})}\n\n"
                        return
                    
                    last_percent = -1
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                # Add model info to the progress data
                                data["model"] = model.display_name
                                data["model_id"] = model.value
                                
                                # Log progress to stdout
                                if data.get("total") and data.get("completed"):
                                    percent = int((data["completed"] / data["total"]) * 100)
                                    if percent != last_percent and percent % 5 == 0:
                                        completed_mb = data["completed"] / 1024 / 1024
                                        total_mb = data["total"] / 1024 / 1024
                                        print(f"[LLM Pull] {model.value}: {completed_mb:.1f} MB / {total_mb:.1f} MB ({percent}%)")
                                        last_percent = percent
                                elif data.get("status"):
                                    print(f"[LLM Pull] {model.value}: {data['status']}")
                                
                                yield f"data: {json.dumps(data)}\n\n"
                            except json.JSONDecodeError:
                                pass
                    
                    print(f"[LLM Pull] Download complete: {model.value}")
                    yield f"data: {json.dumps({'status': 'complete', 'model': model.display_name})}\n\n"
        except Exception as e:
            print(f"[LLM Pull] ERROR: {e}")
            yield f"data: {json.dumps({'status': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(pull_stream(), media_type="text/event-stream")


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

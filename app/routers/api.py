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
    OllamaClient,
    is_vision_model,
    select_optimal_model
)
from app.config.document_schemas import (
    get_all_schemas,
    get_schema,
    save_schema,
    reload_schemas,
    list_schema_files,
    DocumentSchema,
    FieldSchema,
    FieldType,
    SCHEMAS_DIR
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
    image_id: Optional[str] = None  # For multimodal/vision processing
    use_vision: Optional[bool] = True  # Whether to use vision if available


class LLMModelInfo(BaseModel):
    id: str
    name: str
    available: bool
    supports_vision: bool = False  # Whether model supports multimodal/vision input


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
    
    # Include image_id in result for vision LLM processing
    result["image_id"] = image_id
    
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
    
    # Include image_id in result for vision LLM processing
    result["image_id"] = image_id
    
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
    from app.services.llm_service import GEMINI_AVAILABLE, get_gemini_client
    
    client = get_ollama_client()
    
    # Get system info including acceleration
    system_info = await client.get_system_info()
    ollama_available = system_info.get("available", False)
    
    models = []
    available_models = await client.list_models() if ollama_available else []
    
    # Track which pulled models are matched to predefined ones
    matched_models = set()
    
    # First add predefined models
    for model in LLMModel:
        model_name = model.value.split(":")[0]
        is_available = any(model_name in m for m in available_models)
        if is_available:
            # Track which available models match this predefined model
            for m in available_models:
                if model_name in m:
                    matched_models.add(m)
        models.append(LLMModelInfo(
            id=model.value,
            name=model.display_name,
            available=is_available,
            supports_vision=model.supports_vision  # Add vision capability
        ))
    
    # Add any pulled models that aren't predefined
    for pulled_model in available_models:
        if pulled_model not in matched_models:
            # This is a model pulled by user but not in our predefined list
            # Use the model name as both id and display name
            display_name = pulled_model.replace(":", " ").title()
            # Check if custom model supports vision
            has_vision = is_vision_model(pulled_model)
            vision_tag = " [Vision]" if has_vision else ""
            models.append(LLMModelInfo(
                id=pulled_model,
                name=f"{display_name}{vision_tag} (custom)",
                available=True,
                supports_vision=has_vision
            ))
    
    # Add Gemini if API key is available (Gemini is always vision-capable)
    if GEMINI_AVAILABLE:
        models.append(LLMModelInfo(
            id="gemini-2.5-pro",
            name="Gemini 2.5 Pro (Google) [Vision]",
            available=True,
            supports_vision=True
        ))
    
    return LLMStatusResponse(
        ollama_available=ollama_available,
        models=models,
        acceleration=system_info.get("acceleration", "unknown"),
        acceleration_details=system_info.get("acceleration_details"),
        version=system_info.get("version")
    )


@router.post("/llm/start-ollama")
async def start_ollama():
    """Start the Ollama service."""
    import subprocess
    import platform
    import shutil
    
    # Check if Ollama is already running
    client = get_ollama_client()
    if await client.is_available():
        return {"status": "already_running", "message": "Ollama is already running"}
    
    # Find ollama executable
    ollama_path = shutil.which("ollama")
    if not ollama_path:
        # Check common locations
        common_paths = [
            "/usr/local/bin/ollama",
            "/opt/homebrew/bin/ollama",
            os.path.expanduser("~/.ollama/bin/ollama"),
        ]
        for path in common_paths:
            if os.path.exists(path):
                ollama_path = path
                break
    
    if not ollama_path:
        raise HTTPException(
            status_code=404, 
            detail="Ollama not found. Please install Ollama first: https://ollama.com"
        )
    
    try:
        # Start Ollama serve in background
        system = platform.system()
        if system == "Darwin":  # macOS
            # Use open command to start Ollama app if installed, otherwise serve
            subprocess.Popen(
                [ollama_path, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
        elif system == "Linux":
            subprocess.Popen(
                [ollama_path, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
        elif system == "Windows":
            subprocess.Popen(
                [ollama_path, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            raise HTTPException(status_code=500, detail=f"Unsupported platform: {system}")
        
        # Wait a moment for Ollama to start
        import asyncio
        await asyncio.sleep(2)
        
        # Check if it started successfully
        if await client.is_available():
            return {"status": "started", "message": "Ollama started successfully"}
        else:
            return {"status": "starting", "message": "Ollama is starting, please wait..."}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start Ollama: {str(e)}")


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
    
    Supports multimodal/vision processing when image_id is provided and the
    selected model supports vision input.
    
    Performs:
    - Vision-enabled document understanding (with image)
    - Context-aware text correction based on document type
    - Structured field extraction using document schemas
    - Medicine name recognition (for medical documents)
    - OCR artifact removal
    
    Returns structured data with extracted fields matching the document schema.
    """
    from app.services.llm_service import GEMINI_AVAILABLE, get_gemini_processor
    
    # === VISION FIX v2 === Debug: Log incoming request
    print(f"[LLM API v2] Received request - image_id: '{request.image_id}', use_vision: {request.use_vision}, model: {request.model}")
    
    # Load image bytes if image_id is provided
    image_bytes = None
    if request.image_id and request.use_vision:
        image_path = None
        
        # First try the in-memory store
        if request.image_id in images_store:
            image_path = images_store[request.image_id]["path"]
        else:
            # Fallback: check if file exists directly on disk (handles server restarts)
            fallback_path = os.path.join(UPLOAD_DIR, f"{request.image_id}.png")
            if os.path.exists(fallback_path):
                image_path = fallback_path
                # Re-add to store for future requests
                images_store[request.image_id] = {"path": fallback_path, "filename": f"{request.image_id}.png"}
                print(f"[LLM] Recovered image from disk: {request.image_id}")
        
        if image_path:
            try:
                with open(image_path, "rb") as f:
                    image_bytes = f.read()
                print(f"[LLM] Loaded image for vision processing: {len(image_bytes)} bytes")
            except Exception as e:
                print(f"[LLM] Warning: Could not load image: {e}")
        else:
            print(f"[LLM] Warning: Image ID {request.image_id} not found in store or on disk")
    
    # Check if Gemini is requested
    if request.model and request.model.lower().startswith("gemini"):
        if not GEMINI_AVAILABLE:
            raise HTTPException(
                status_code=400,
                detail="Gemini API key not configured. Set GEMINI_API_KEY environment variable."
            )
        
        processor = get_gemini_processor()
        result = await processor.process_text(
            request.text, 
            request.document_type,
            image_bytes  # Pass image for Gemini vision
        )
        return result
    
    # Otherwise use Ollama
    client = get_ollama_client()
    
    if not await client.is_available():
        raise HTTPException(status_code=503, detail="Ollama service not available. Please start Ollama.")
    
    # Get available models for potential auto-selection
    available_models = await client.list_models()
    
    # Get model - can be predefined LLMModel or custom string
    model = None
    model_id = request.model
    
    if request.model:
        # Try to match to predefined model first
        for m in LLMModel:
            if m.value.lower() == request.model.lower():
                model = m
                break
        
        # If not a predefined model, check if it's available as custom model
        if model is None:
            if request.model not in available_models:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Model {request.model} not available. Please pull it first."
                )
            # Use the model ID string directly for custom models
            model_id = request.model
        else:
            # Check if predefined model is available
            if not await client.is_model_available(model):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Model {model.display_name} not available. Please pull it first."
                )
    else:
        # Auto-select optimal model based on context
        has_image = image_bytes is not None
        selected = select_optimal_model(
            available_models,
            has_image=has_image,
            document_type=request.document_type,
            prefer_vision=request.use_vision
        )
        if selected:
            model_id = selected
            print(f"[LLM] Auto-selected model: {model_id} (image={'yes' if has_image else 'no'})")
    
    processor = get_llm_processor()
    
    # Use context-aware structured extraction with optional vision
    result = await processor.process_text(
        request.text, 
        model if model else model_id,
        request.document_type,
        image_bytes  # Pass image for vision models
    )
    
    return result


@router.post("/llm/denoise")
async def llm_denoise_text(request: LLMProcessRequest):
    """Quick denoising of OCR text - removes artifacts and fixes obvious errors.
    
    Supports vision-enabled denoising when image_id is provided.
    """
    client = get_ollama_client()
    
    if not await client.is_available():
        raise HTTPException(status_code=503, detail="Ollama service not available")
    
    # Load image bytes if provided
    image_bytes = None
    if request.image_id and request.use_vision:
        if request.image_id in images_store:
            image_path = images_store[request.image_id]["path"]
            try:
                with open(image_path, "rb") as f:
                    image_bytes = f.read()
            except Exception as e:
                print(f"[LLM] Warning: Could not load image for denoising: {e}")
    
    model = LLMModel.from_string(request.model) if request.model else None
    processor = get_llm_processor()
    
    denoised = await processor.denoise_text(request.text, model, image_bytes)
    
    return {
        "original_text": request.text,
        "denoised_text": denoised,
        "vision_used": image_bytes is not None
    }


# =============================================================================
# Schema Management Endpoints
# =============================================================================

class SchemaFieldRequest(BaseModel):
    name: str
    type: str = "text"
    description: str
    required: bool = False


class SchemaRequest(BaseModel):
    type_id: str
    display_name: str
    clip_prompts: List[str] = []
    keywords: List[str] = []
    llm_context: str = ""
    fields: List[SchemaFieldRequest] = []


class GenerateSchemaRequest(BaseModel):
    description: str
    model: Optional[str] = None


@router.get("/schemas")
async def list_schemas():
    """List all available document schemas."""
    schemas = get_all_schemas()
    result = []
    for type_id, schema in schemas.items():
        result.append({
            "type_id": schema.type_id,
            "display_name": schema.display_name,
            "field_count": len(schema.fields),
            "keywords": schema.keywords[:5],  # First 5 keywords
            "source_file": schema.source_file
        })
    return {"schemas": result}


@router.get("/schemas/{type_id}")
async def get_schema_detail(type_id: str):
    """Get detailed schema by type ID."""
    schema = get_schema(type_id)
    if schema.type_id == "unknown" and type_id != "unknown":
        raise HTTPException(status_code=404, detail=f"Schema '{type_id}' not found")
    
    return {
        "type_id": schema.type_id,
        "display_name": schema.display_name,
        "clip_prompts": schema.clip_prompts,
        "keywords": schema.keywords,
        "llm_context": schema.llm_context,
        "fields": [
            {
                "name": f.name,
                "type": f.field_type.value,
                "description": f.description,
                "required": f.required
            }
            for f in schema.fields
        ],
        "source_file": schema.source_file
    }


@router.get("/schemas/{type_id}/yaml")
async def get_schema_yaml(type_id: str):
    """Get raw YAML content of a schema file."""
    import yaml
    schema = get_schema(type_id)
    if schema.type_id == "unknown" and type_id != "unknown":
        raise HTTPException(status_code=404, detail=f"Schema '{type_id}' not found")
    
    if schema.source_file and os.path.exists(schema.source_file):
        with open(schema.source_file, 'r') as f:
            return {"yaml": f.read(), "filename": os.path.basename(schema.source_file)}
    
    # Generate YAML from schema object
    yaml_content = yaml.dump(schema.to_dict(), default_flow_style=False, allow_unicode=True, sort_keys=False)
    return {"yaml": yaml_content, "filename": f"{type_id}.yaml"}


@router.put("/schemas/{type_id}")
async def update_schema(type_id: str, request: SchemaRequest):
    """Update or create a schema."""
    import yaml
    
    # Build schema dict
    schema_dict = {
        "type_id": request.type_id,
        "display_name": request.display_name,
        "clip_prompts": request.clip_prompts,
        "keywords": request.keywords,
        "llm_context": request.llm_context,
        "fields": [
            {
                "name": f.name,
                "type": f.type,
                "description": f.description,
                "required": f.required
            }
            for f in request.fields
        ]
    }
    
    # Save to YAML file
    filepath = SCHEMAS_DIR / f"{request.type_id}.yaml"
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(schema_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    # Reload schemas
    reload_schemas()
    
    print(f"[Schema] Saved schema: {request.type_id} to {filepath}")
    
    return {"status": "saved", "type_id": request.type_id, "filepath": str(filepath)}


@router.put("/schemas/{type_id}/yaml")
async def update_schema_yaml(type_id: str, yaml_content: str):
    """Update a schema from raw YAML content."""
    import yaml
    
    try:
        # Validate YAML
        data = yaml.safe_load(yaml_content)
        if not data or not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="Invalid YAML content")
        
        # Use type_id from YAML or URL
        actual_type_id = data.get("type_id", type_id)
        
        # Save to file
        filepath = SCHEMAS_DIR / f"{actual_type_id}.yaml"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        # Reload schemas
        reload_schemas()
        
        print(f"[Schema] Saved YAML schema: {actual_type_id} to {filepath}")
        
        return {"status": "saved", "type_id": actual_type_id, "filepath": str(filepath)}
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {str(e)}")


@router.delete("/schemas/{type_id}")
async def delete_schema(type_id: str):
    """Delete a schema file."""
    if type_id == "unknown":
        raise HTTPException(status_code=400, detail="Cannot delete the 'unknown' fallback schema")
    
    filepath = SCHEMAS_DIR / f"{type_id}.yaml"
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"Schema file not found: {type_id}.yaml")
    
    os.remove(filepath)
    reload_schemas()
    
    print(f"[Schema] Deleted schema: {type_id}")
    
    return {"status": "deleted", "type_id": type_id}


@router.post("/schemas/generate/stream")
async def generate_schema_with_llm_stream(request: GenerateSchemaRequest):
    """Generate a new schema using LLM with streaming output."""
    from app.services.llm_service import GEMINI_AVAILABLE, get_gemini_client
    
    # Check if Gemini is requested
    use_gemini = request.model and request.model.lower().startswith("gemini")
    
    if use_gemini:
        if not GEMINI_AVAILABLE:
            raise HTTPException(
                status_code=400,
                detail="Gemini API key not configured. Set GEMINI_API_KEY environment variable."
            )
    else:
        client = get_ollama_client()
        if not await client.is_available():
            raise HTTPException(status_code=503, detail="Ollama service not available")
    
    # Get model ID - handle both predefined and custom models
    model_id = None
    if request.model and not use_gemini:
        # Try to match to predefined model first
        for m in LLMModel:
            if m.value.lower() == request.model.lower():
                model_id = m.value
                break
        if model_id is None:
            model_id = request.model  # Use as custom model ID
    elif not use_gemini:
        model_id = LLMModel.DEVSTRAL.value  # Default model
    
    prompt = f"""Generate a document schema YAML for the following document type:

USER DESCRIPTION:
{request.description}

Generate a YAML schema with:
1. type_id: a short lowercase identifier (e.g., "invoice", "medical_report")
2. display_name: human-readable name
3. clip_prompts: 2-3 prompts for image classification
4. keywords: 5-10 keywords that appear in this document type
5. llm_context: brief instructions for extracting data from this document type, including common OCR errors to fix
6. fields: list of fields to extract, each with name, type (text/date/currency/list/number), description, required (true/false)

For list-type fields, describe what each item should contain in the description.

Return ONLY valid YAML, no explanations:"""

    print(f"[Schema] Streaming schema generation for: {request.description[:50]}...")
    
    async def stream_generate() -> AsyncGenerator[str, None]:
        """Stream LLM generation as Server-Sent Events."""
        full_response = ""
        try:
            if use_gemini:
                # Use Gemini API (non-streaming, send result at once)
                gemini_client = get_gemini_client()
                result = await gemini_client.generate(
                    prompt=prompt,
                    temperature=0.3,
                    max_tokens=2048
                )
                if result:
                    full_response = result
                    # Send the full response as tokens for UI consistency
                    yield f"data: {json.dumps({'token': result})}\n\n"
                else:
                    yield f"data: {json.dumps({'error': 'Gemini generation failed'})}\n\n"
                    return
            else:
                # Use Ollama with streaming
                async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as http_client:
                    async with http_client.stream(
                        "POST",
                        f"{client.base_url}/api/generate",
                        json={
                            "model": model_id,
                            "prompt": prompt,
                            "stream": True,
                            "options": {
                                "temperature": 0.3,
                                "num_predict": 2048,
                            }
                        }
                    ) as response:
                        if response.status_code != 200:
                            yield f"data: {json.dumps({'error': 'Failed to start generation'})}\n\n"
                            return
                        
                        async for line in response.aiter_lines():
                            if line:
                                try:
                                    data = json.loads(line)
                                    token = data.get("response", "")
                                    full_response += token
                                    
                                    # Send token to client
                                    yield f"data: {json.dumps({'token': token})}\n\n"
                                    
                                    if data.get("done"):
                                        break
                                except json.JSONDecodeError:
                                    pass
            
            # Process final result
            yaml_content = full_response.strip()
            if yaml_content.startswith("```"):
                lines = yaml_content.split("\n")
                yaml_lines = []
                in_yaml = False
                for line in lines:
                    if line.startswith("```") and not in_yaml:
                        in_yaml = True
                        continue
                    elif line.startswith("```") and in_yaml:
                        break
                    elif in_yaml:
                        yaml_lines.append(line)
                yaml_content = "\n".join(yaml_lines)
            
            # Validate and send final result
            import yaml
            try:
                data = yaml.safe_load(yaml_content)
                if data and isinstance(data, dict):
                    yield f"data: {json.dumps({'done': True, 'yaml': yaml_content, 'type_id': data.get('type_id', 'new_schema'), 'display_name': data.get('display_name', 'New Schema')})}\n\n"
                else:
                    yield f"data: {json.dumps({'done': True, 'yaml': yaml_content, 'error': 'Invalid YAML structure'})}\n\n"
            except yaml.YAMLError as e:
                yield f"data: {json.dumps({'done': True, 'yaml': full_response, 'error': f'YAML parse error: {str(e)}'})}\n\n"
                
        except Exception as e:
            print(f"[Schema] Stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(stream_generate(), media_type="text/event-stream")


@router.post("/schemas/generate")
async def generate_schema_with_llm(request: GenerateSchemaRequest):
    """Generate a new schema using LLM based on user description (non-streaming)."""
    client = get_ollama_client()
    
    if not await client.is_available():
        raise HTTPException(status_code=503, detail="Ollama service not available")
    
    model = LLMModel.from_string(request.model) if request.model else None
    
    prompt = f"""Generate a document schema YAML for the following document type:

USER DESCRIPTION:
{request.description}

Generate a YAML schema with:
1. type_id: a short lowercase identifier (e.g., "invoice", "medical_report")
2. display_name: human-readable name
3. clip_prompts: 2-3 prompts for image classification
4. keywords: 5-10 keywords that appear in this document type
5. llm_context: brief instructions for extracting data from this document type, including common OCR errors to fix
6. fields: list of fields to extract, each with name, type (text/date/currency/list/number), description, required (true/false)

For list-type fields, describe what each item should contain in the description.

Return ONLY valid YAML, no explanations:"""

    print(f"[Schema] Generating schema with LLM for: {request.description[:50]}...")
    
    result = await client.generate(
        prompt=prompt,
        model=model,
        temperature=0.3,
        max_tokens=2048
    )
    
    if not result:
        raise HTTPException(status_code=500, detail="LLM generation failed")
    
    # Clean up the result - extract YAML if wrapped in code blocks
    yaml_content = result.strip()
    if yaml_content.startswith("```"):
        lines = yaml_content.split("\n")
        yaml_lines = []
        in_yaml = False
        for line in lines:
            if line.startswith("```") and not in_yaml:
                in_yaml = True
                continue
            elif line.startswith("```") and in_yaml:
                break
            elif in_yaml:
                yaml_lines.append(line)
        yaml_content = "\n".join(yaml_lines)
    
    # Validate YAML
    import yaml
    try:
        data = yaml.safe_load(yaml_content)
        if not data or not isinstance(data, dict):
            raise HTTPException(status_code=500, detail="LLM generated invalid YAML structure")
    except yaml.YAMLError as e:
        raise HTTPException(status_code=500, detail=f"LLM generated invalid YAML: {str(e)}")
    
    return {
        "yaml": yaml_content,
        "parsed": data,
        "type_id": data.get("type_id", "new_schema"),
        "display_name": data.get("display_name", "New Schema")
    }


@router.post("/schemas/reload")
async def reload_all_schemas():
    """Force reload all schemas from YAML files."""
    schemas = reload_schemas()
    return {
        "status": "reloaded",
        "count": len(schemas),
        "schemas": list(schemas.keys())
    }

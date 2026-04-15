from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, AsyncGenerator
import uuid
import os
import json
import httpx

from app.services.ocr_service import (
    recognize_text,
    recognize_region,
    get_device_info,
    classify_image_with_siglip,
    classify_document_hybrid,
)
from app.services.image_service import normalize_image
from app.services import document_service as doc_service
from PIL import Image as PILImage
from app.services.llm_service import (
    get_ollama_client,
    get_llm_processor,
    LLMModel,
    OllamaClient,
    is_vision_model,
    is_thinking_model,
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

# Legacy flat image registry retained for backwards compatibility with old
# endpoints that pre-date the Document/Page model. New uploads route through
# `document_service` instead; `_resolve_image_path` below understands both.
images_store: dict[str, dict] = {}


def _resolve_image_path(identifier: str) -> Optional[str]:
    """Resolve either a new page_id or a legacy image_id to a disk path."""
    page = doc_service.get_page(identifier)
    if page:
        return page.image_path
    if identifier in images_store:
        return images_store[identifier].get("path")
    # Last-resort disk fallback for the old flat layout.
    fallback = os.path.join(UPLOAD_DIR, f"{identifier}.png")
    if os.path.exists(fallback):
        return fallback
    return None


def _resolve_page(identifier: str):
    """Return the Page object for an identifier, or None if not found."""
    return doc_service.get_page(identifier)


class ImageInfo(BaseModel):
    """Legacy flat-image shape preserved so the frontend keeps working during
    the document-model transition. Each page is exposed as one ImageInfo with
    doc_id set so the UI can group pages under their parent document."""
    id: str
    filename: str
    doc_id: Optional[str] = None
    source_kind: Optional[str] = None
    page_index: Optional[int] = None
    page_count: Optional[int] = None
    has_text_layer: Optional[bool] = None


class PageSummary(BaseModel):
    page_id: str
    doc_id: str
    index: int
    has_text_layer: bool
    width: int
    height: int


class DocumentSummary(BaseModel):
    doc_id: str
    filename: str
    source_kind: str  # "image" | "pdf" | "docx" | "multipage"
    page_count: int
    has_text_layer: bool
    pages: List[PageSummary]


class UploadResponse(BaseModel):
    documents: List[DocumentSummary]
    # Legacy field: one entry per page, flat across all uploaded docs. Keeps
    # the old frontend upload handler working while it migrates to `documents`.
    images: List[ImageInfo] = []


class ImageIdRequest(BaseModel):
    image_id: str


class RecognizeRegionRequest(BaseModel):
    image_id: str
    bbox: List[float]
    enforce_boxes: bool = False  # Skip bbox detection, recognize text directly in the region


class NormalizeResponse(BaseModel):
    image_id: str
    normalized: bool
    angle: float


class LLMProcessRequest(BaseModel):
    text: str
    model: Optional[str] = None
    document_type: Optional[str] = None
    image_id: Optional[str] = None  # For single-page vision processing (legacy)
    doc_id: Optional[str] = None    # For multi-page document processing
    use_vision: Optional[bool] = True


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


def _document_to_summary(doc: doc_service.Document) -> DocumentSummary:
    return DocumentSummary(
        doc_id=doc.doc_id,
        filename=doc.filename,
        source_kind=doc.source_kind,
        page_count=doc.page_count,
        has_text_layer=doc.has_any_text_layer,
        pages=[
            PageSummary(
                page_id=p.page_id,
                doc_id=p.doc_id,
                index=p.index,
                has_text_layer=p.has_text_layer,
                width=p.width,
                height=p.height,
            )
            for p in doc.pages
        ],
    )


def _document_to_legacy_images(doc: doc_service.Document) -> List[ImageInfo]:
    """Flatten a Document's pages into the legacy ImageInfo list the old UI
    expects. Each page becomes one ImageInfo; doc_id/page_index/page_count
    let a modern UI group pages by their parent document."""
    return [
        ImageInfo(
            id=p.page_id,
            filename=(
                f"{doc.filename} [p.{p.index}/{doc.page_count}]"
                if doc.page_count > 1
                else doc.filename
            ),
            doc_id=doc.doc_id,
            source_kind=doc.source_kind,
            page_index=p.index,
            page_count=doc.page_count,
            has_text_layer=p.has_text_layer,
        )
        for p in doc.pages
    ]


@router.post("/upload", response_model=UploadResponse)
async def upload_images(files: List[UploadFile] = File(...)):
    """Upload images, PDFs, and/or DOCX files.

    Each uploaded file becomes its own Document:
      - image -> 1-page Document
      - pdf   -> N-page Document (one page per PDF page); per-page text layer
                 is captured and downstream OCR is skipped for pages that
                 already have embedded text.
      - docx  -> 1-page Document; text is extracted natively.

    To *group* multiple files into a single multi-page document, use the
    sibling endpoint /api/upload/multipage instead.
    """
    documents: List[doc_service.Document] = []

    for file in files:
        content = await file.read()
        filename = file.filename or "upload"
        kind = doc_service.classify_upload(filename, content)
        try:
            if kind == "pdf":
                doc = doc_service.ingest_pdf(filename, content)
            elif kind == "docx":
                doc = doc_service.ingest_docx(filename, content)
            elif kind == "image":
                doc = doc_service.ingest_image(filename, content)
            else:
                print(f"[Upload] skipping unsupported file: {filename} ({file.content_type})")
                continue
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"{filename}: {e}")
        except Exception as e:
            print(f"[Upload] ingest failed for {filename}: {e}")
            raise HTTPException(status_code=500, detail=f"{filename}: {e}")
        documents.append(doc)
        # Mirror every page into the legacy store so old lookups resolve.
        for p in doc.pages:
            images_store[p.page_id] = {"path": p.image_path, "filename": file.filename}
        print(f"[Upload] {kind}: {filename} -> doc={doc.doc_id} pages={doc.page_count}")

    if not documents:
        raise HTTPException(status_code=400, detail="No supported files uploaded")

    summaries = [_document_to_summary(d) for d in documents]
    legacy_images: List[ImageInfo] = []
    for d in documents:
        legacy_images.extend(_document_to_legacy_images(d))
    return UploadResponse(documents=summaries, images=legacy_images)


@router.post("/upload/multipage", response_model=UploadResponse)
async def upload_multipage(files: List[UploadFile] = File(...), name: Optional[str] = None):
    """Upload multiple files and staple them into a single multi-page document.

    Useful when screenshots are sequential pages of the same original document,
    or when a cover page PDF needs to be combined with a scan.
    """
    file_tuples: List[tuple] = []
    for file in files:
        content = await file.read()
        filename = file.filename or "page"
        kind = doc_service.classify_upload(filename, content)
        if kind == "unsupported":
            print(f"[Upload multipage] skipping unsupported: {filename}")
            continue
        file_tuples.append((filename, content))

    if not file_tuples:
        raise HTTPException(status_code=400, detail="No supported files in multipage upload")

    display_name = name or f"Multipage document ({len(file_tuples)} files)"
    try:
        doc = doc_service.ingest_multipage(display_name, file_tuples)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[Upload multipage] failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    for p in doc.pages:
        images_store[p.page_id] = {"path": p.image_path, "filename": display_name}
    print(f"[Upload multipage] {display_name} -> doc={doc.doc_id} pages={doc.page_count}")
    return UploadResponse(
        documents=[_document_to_summary(doc)],
        images=_document_to_legacy_images(doc),
    )


@router.get("/document/{doc_id}")
async def get_document_info(doc_id: str):
    doc = doc_service.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    return _document_to_summary(doc).model_dump()


@router.delete("/document/{doc_id}")
async def delete_document_endpoint(doc_id: str):
    doc = doc_service.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    # Also drop legacy mirror entries.
    for p in doc.pages:
        images_store.pop(p.page_id, None)
    doc_service.delete_document(doc_id)
    return {"status": "deleted", "doc_id": doc_id}


@router.post("/normalize", response_model=NormalizeResponse)
async def normalize(request: ImageIdRequest):
    """Normalize image orientation (deskew text)."""
    image_id = request.image_id

    path = _resolve_image_path(image_id)
    if not path:
        raise HTTPException(status_code=404, detail="Image not found")

    with open(path, "rb") as f:
        image_bytes = f.read()

    normalized_bytes, angle = normalize_image(image_bytes)

    # Write normalized output alongside the original so re-renders pick it up.
    normalized_path = os.path.join(os.path.dirname(path), f"{image_id}_normalized.png")
    with open(normalized_path, "wb") as f:
        f.write(normalized_bytes)

    # Update both registries so either resolution path sees the new file.
    page = _resolve_page(image_id)
    if page:
        page.image_path = normalized_path
    if image_id in images_store:
        images_store[image_id]["path"] = normalized_path
    else:
        images_store[image_id] = {"path": normalized_path, "filename": f"{image_id}.png"}

    return NormalizeResponse(image_id=image_id, normalized=True, angle=angle)


def _synth_text_layer_result(page: doc_service.Page) -> dict:
    """Build a Surya-shaped response from a page's native text layer, so the
    downstream UI can render it identically to OCR output."""
    text = (page.text_layer or "").strip()
    lines = [line for line in text.splitlines() if line.strip()]
    # Synthesize proportional bboxes so the overlay stays roughly aligned.
    W = page.width or 1000
    H = page.height or 1400
    if lines:
        line_h = max(20.0, H / max(len(lines), 1))
        text_lines = []
        for i, ln in enumerate(lines):
            y0 = i * line_h
            y1 = y0 + line_h * 0.9
            text_lines.append(
                {
                    "text": ln,
                    "confidence": 1.0,
                    "bbox": [float(W * 0.05), float(y0), float(W * 0.95), float(y1)],
                }
            )
    else:
        text_lines = []

    # Run the hybrid classifier using the rendered page + text layer so domain
    # routing still works for PDF/DOCX uploads.
    try:
        with PILImage.open(page.image_path) as im:
            im_rgb = im.convert("RGB") if im.mode != "RGB" else im.copy()
        classification = classify_document_hybrid(im_rgb, text_lines)
    except Exception as e:
        print(f"[Recognize] classification fallback failed for {page.page_id}: {e}")
        classification = {"class": "Undetected", "confidence": 0.0, "type_id": "unknown"}

    return {
        "text_lines": text_lines,
        "image_bbox": [0.0, 0.0, float(W), float(H)],
        "document_class": classification,
        "image_id": page.page_id,
        "used_text_layer": True,
        "page_id": page.page_id,
        "doc_id": page.doc_id,
        "text": text,
    }


@router.post("/recognize")
async def recognize(request: ImageIdRequest):
    """Recognize text in a page.

    Fast path: if the identifier is a Page with a native text layer (PDF
    with embedded OCR, DOCX), return a synthesized result built from that
    text layer instead of running Surya. Otherwise fall back to Surya OCR.
    """
    image_id = request.image_id

    # Fast path: native text layer available → skip Surya entirely.
    page = _resolve_page(image_id)
    if page and page.has_text_layer:
        print(f"[Recognize] page {image_id} has text layer, skipping OCR")
        result = _synth_text_layer_result(page)
        return result

    path = _resolve_image_path(image_id)
    if not path:
        raise HTTPException(status_code=404, detail="Image not found")
    with open(path, "rb") as f:
        image_bytes = f.read()

    result = recognize_text(image_bytes)
    result["image_id"] = image_id
    result["used_text_layer"] = False
    if page:
        result["page_id"] = page.page_id
        result["doc_id"] = page.doc_id
    return result


@router.get("/image/{image_id}")
async def get_image(image_id: str):
    """Retrieve page/image bytes by ID (page_id or legacy image_id)."""
    path = _resolve_image_path(image_id)
    if not path:
        raise HTTPException(status_code=404, detail="Image not found")
    with open(path, "rb") as f:
        image_bytes = f.read()
    return Response(content=image_bytes, media_type="image/png")


@router.post("/recognize-region")
async def recognize_region_endpoint(request: RecognizeRegionRequest):
    """Recognize text in a specified region of the image."""
    image_id = request.image_id
    path = _resolve_image_path(image_id)
    if not path:
        raise HTTPException(status_code=404, detail="Image not found")

    with open(path, "rb") as f:
        image_bytes = f.read()

    result = recognize_region(image_bytes, request.bbox, enforce_boxes=request.enforce_boxes)
    result["image_id"] = image_id
    return result


@router.get("/device-info")
async def device_info():
    """Get GPU/device information for Surya OCR and the SigLIP 2 classifier."""
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


@router.post("/llm/process/stream")
async def llm_process_text_stream(request: LLMProcessRequest):
    """
    Process OCR text using LLM with streaming progress updates.
    
    Returns Server-Sent Events with:
    - status: "started", "generating", "processing", "complete", "error"
    - mode: "vision" or "text"
    - model: Model name being used
    - token_count: Running count of generated tokens
    - tokens: Individual tokens as they arrive
    - result: Final structured result (when complete)
    """
    from app.services.llm_service import GEMINI_AVAILABLE, get_gemini_processor, encode_image_to_base64
    
    async def generate_stream() -> AsyncGenerator[str, None]:
        nonlocal request
        
        print(
            f"[LLM Stream] Starting stream for model: {request.model}, "
            f"image_id: {request.image_id}, doc_id: {request.doc_id}"
        )

        # Resolve target document/pages. When doc_id is supplied we operate on
        # the full multi-page document: aggregated text layers across all pages
        # go into the prompt and each page image is sent to vision models that
        # accept multiple images. Otherwise we fall back to the legacy
        # single-page path (image_id).
        target_doc = None
        target_pages: List[doc_service.Page] = []
        if request.doc_id:
            target_doc = doc_service.get_document(request.doc_id)
            if not target_doc:
                yield f"data: {json.dumps({'status': 'error', 'error': f'Document {request.doc_id} not found'})}\n\n"
                return
            target_pages = target_doc.pages
        elif request.image_id:
            page = _resolve_page(request.image_id)
            if page:
                # Expand to the full document so multi-page context isn't lost
                # just because the UI selected one of its pages.
                target_doc = doc_service.get_document(page.doc_id)
                if target_doc:
                    target_pages = target_doc.pages

        # Aggregated text (text layers + the request.text the client already
        # collected from Recognize runs) is what the LLM sees.
        aggregated_text = request.text or ""
        if target_pages:
            page_texts: List[str] = []
            for p in target_pages:
                if p.has_text_layer:
                    page_texts.append(f"=== Page {p.index} (native text) ===\n{p.text_layer.strip()}")
            if page_texts:
                aggregated = "\n\n".join(page_texts)
                aggregated_text = (aggregated_text + "\n\n" if aggregated_text else "") + aggregated
            print(
                f"[LLM Stream] Document mode: doc={target_doc.doc_id if target_doc else None}, "
                f"pages={len(target_pages)}, text_len={len(aggregated_text)}"
            )

        # Collect vision image bytes. For multi-page docs we send every page
        # to vision-capable models (Ollama accepts a list of base64 images).
        image_bytes = None  # back-compat: the first image still populated for old callsites
        image_bytes_list: List[bytes] = []
        if request.use_vision:
            if target_pages:
                for p in target_pages:
                    try:
                        with open(p.image_path, "rb") as f:
                            image_bytes_list.append(f.read())
                    except Exception as e:
                        print(f"[LLM Stream] failed to read page {p.page_id}: {e}")
            elif request.image_id:
                path = _resolve_image_path(request.image_id)
                if path:
                    try:
                        with open(path, "rb") as f:
                            image_bytes_list.append(f.read())
                    except Exception as e:
                        print(f"[LLM Stream] ERROR loading image: {e}")
                        yield f"data: {json.dumps({'status': 'error', 'error': f'Could not load image: {e}'})}\n\n"
                        return

            if image_bytes_list:
                image_bytes = image_bytes_list[0]
                print(
                    f"[LLM Stream] Loaded {len(image_bytes_list)} image(s), "
                    f"first={len(image_bytes)} bytes"
                )
        # The rest of this function uses `request.text`; swap in aggregated.
        request.text = aggregated_text
        
        # Determine model and mode
        model_id = request.model
        model_name = request.model or "auto"
        
        # Check for Gemini
        if request.model and request.model.lower().startswith("gemini"):
            mode = "vision" if image_bytes else "text"
            print(f"[LLM Stream] Using Gemini in {mode} mode")
            yield f"data: {json.dumps({'status': 'started', 'mode': mode, 'model': 'Gemini 2.5 Pro'})}\n\n"
            
            if not GEMINI_AVAILABLE:
                yield f"data: {json.dumps({'status': 'error', 'error': 'Gemini API key not configured'})}\n\n"
                return
            
            processor = get_gemini_processor()
            result = await processor.process_text(request.text, request.document_type, image_bytes)
            yield f"data: {json.dumps({'status': 'complete', 'result': result})}\n\n"
            return
        
        # Ollama processing with streaming
        client = get_ollama_client()
        
        if not await client.is_available():
            print("[LLM Stream] ERROR: Ollama not available")
            yield f"data: {json.dumps({'status': 'error', 'error': 'Ollama service not available'})}\n\n"
            return
        
        # Resolve model
        model = None
        available_models = await client.list_models()
        
        if request.model:
            for m in LLMModel:
                if m.value.lower() == request.model.lower():
                    model = m
                    model_id = m.value
                    model_name = m.display_name
                    break
            if model is None:
                model_id = request.model
                model_name = request.model
        else:
            # Auto-select
            has_image = image_bytes is not None
            selected = select_optimal_model(
                available_models,
                has_image=has_image,
                document_type=request.document_type,
                prefer_vision=request.use_vision
            )
            if selected:
                model_id = selected
                for m in LLMModel:
                    if m.value == selected:
                        model_name = m.display_name
                        model = m
                        break
                if model is None:
                    model_name = selected
        
        # Determine mode
        is_vision = image_bytes is not None and is_vision_model(model_id)
        mode = "vision" if is_vision else "text"
        
        print(f"[LLM Stream] Mode: {mode.upper()}, Model: {model_name}")
        yield f"data: {json.dumps({'status': 'started', 'mode': mode, 'model': model_name, 'model_id': model_id})}\n\n"
        
        # Build prompt using the processor logic
        from app.services.llm_service import get_llm_processor
        from app.config.document_schemas import get_schema
        
        processor = get_llm_processor()
        
        # Get schema
        doc_type = request.document_type or "unknown"
        schema = get_schema(doc_type)
        print(f"[LLM Stream] Document type: {doc_type}, Schema: {schema.display_name if schema else 'None'}")
        
        # Build extraction prompt. The system prompt picks up the active
        # domain description so swapping domains takes effect on the next call.
        from app.services.llm_service import build_system_prompt
        from app.services.domain_service import get_active_domain
        active_domain = get_active_domain()
        if is_vision:
            # For multi-page docs, send every page image. Ollama's /api/generate
            # accepts a list of base64 images and routes them all to the vision
            # model; gemma4:31b handles this fine at reasonable page counts.
            images = [encode_image_to_base64(b) for b in image_bytes_list] if image_bytes_list else [encode_image_to_base64(image_bytes)]
            prompt = processor._build_vision_prompt(request.text, schema)
            system_prompt = build_system_prompt(processor.BASE_VISION_SYSTEM_PROMPT)
        else:
            images = None
            prompt = processor._build_text_prompt(request.text, schema)
            system_prompt = build_system_prompt(processor.BASE_TEXT_SYSTEM_PROMPT)
        print(
            f"[LLM Stream] Active domain: {active_domain.domain_id} "
            f"({active_domain.display_name}), system_prompt={len(system_prompt)} chars, "
            f"images={len(images) if images else 0}"
        )
        
        print(f"[LLM Stream] Prompt length: {len(prompt)} chars")
        
        # Stream generation
        token_count = 0
        full_response = ""
        
        try:
            payload = {
                "model": model_id,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": client.config.temperature,
                    "num_predict": client.config.max_tokens,
                }
            }
            if is_thinking_model(model_id):
                payload["think"] = False
            if system_prompt:
                payload["system"] = system_prompt
            if images:
                payload["images"] = images

            print(f"[LLM Stream] Starting generation...")

            # Streaming: no read deadline so slow cold-loads of large vision models
            # (e.g. first call to gemma4:31b) don't abort before the first token arrives.
            stream_timeout = httpx.Timeout(
                connect=10.0, write=30.0, pool=10.0, read=None
            )
            async with httpx.AsyncClient(timeout=stream_timeout) as http_client:
                async with http_client.stream(
                    "POST",
                    f"{client.base_url}/api/generate",
                    json=payload
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        print(f"[LLM Stream] ERROR: {response.status_code} - {error_text[:200]}")
                        yield f"data: {json.dumps({'status': 'error', 'error': f'LLM error: {response.status_code}'})}\n\n"
                        return
                    
                    print("[LLM Stream] Response: ", end="", flush=True)
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                token = data.get("response", "")
                                if token:
                                    token_count += 1
                                    full_response += token
                                    print(token, end="", flush=True)
                                    
                                    # Send token update every 5 tokens or special chars
                                    if token_count % 5 == 0 or token in ['\n', '.', ',', ':', '{', '}', '[', ']']:
                                        yield f"data: {json.dumps({'status': 'generating', 'token_count': token_count, 'tokens': token})}\n\n"
                                
                                if data.get("done"):
                                    break
                            except json.JSONDecodeError:
                                pass
            
            print()  # Newline after streaming
            print(f"[LLM Stream] Generation complete ({len(full_response)} chars, {token_count} tokens)")
            
            # Parse the response
            yield f"data: {json.dumps({'status': 'processing', 'token_count': token_count, 'message': 'Parsing response...'})}\n\n"
            
            # Parse JSON from response using correct method name
            parsed_result = processor._parse_json_response(full_response, schema)
            
            if parsed_result is None:
                parsed_result = {"extracted_fields": {}, "corrected_text": request.text}
            
            # Build final result
            result = {
                "success": True,
                "document_type": doc_type,
                "document_type_name": schema.display_name if schema else doc_type,
                "extracted_fields": parsed_result.get("extracted_fields", {}),
                "corrected_text": parsed_result.get("corrected_text", request.text),
                "confidence_notes": parsed_result.get("confidence_notes", ""),
                "model": model_name,
                "vision_used": is_vision,
                "token_count": token_count
            }
            
            print(f"[LLM Stream] Sending complete result")
            yield f"data: {json.dumps({'status': 'complete', 'result': result, 'token_count': token_count})}\n\n"
            
        except Exception as e:
            print(f"[LLM Stream] ERROR: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'status': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(generate_stream(), media_type="text/event-stream")


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
async def update_schema_yaml(type_id: str, request: Request):
    """Update or create a schema from a raw YAML body.

    Body: text/plain YAML document. If the YAML's `type_id` differs from the
    URL path, the YAML's value wins (so "save new template" can rename itself
    via the editor).
    """
    import yaml

    yaml_content = (await request.body()).decode("utf-8")
    if not yaml_content.strip():
        raise HTTPException(status_code=400, detail="Empty request body")

    try:
        data = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {str(e)}")

    if not data or not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Invalid YAML: root must be a mapping")

    actual_type_id = data.get("type_id", type_id)
    if not actual_type_id or not isinstance(actual_type_id, str):
        raise HTTPException(status_code=400, detail="Missing or invalid 'type_id' in YAML")

    filepath = SCHEMAS_DIR / f"{actual_type_id}.yaml"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    reload_schemas()
    print(f"[Schema] Saved YAML schema: {actual_type_id} to {filepath}")

    return {"status": "saved", "type_id": actual_type_id, "filepath": str(filepath)}


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
        # Match the rest of the app: use the configured default LLM, not a
        # hard-coded devstral that may not be pulled on this deployment.
        model_id = get_ollama_client().config.default_model.value
    
    from app.services.domain_service import get_active_domain
    active_domain = get_active_domain()

    prompt = f"""Generate a document schema YAML for the following document type.

=== DOMAIN CONTEXT: {active_domain.display_name} ===
{active_domain.description}
=== END DOMAIN CONTEXT ===

USER DESCRIPTION:
{request.description}

Using the domain context above, generate a YAML schema with:
1. type_id: a short lowercase identifier (e.g., "invoice", "medical_report")
2. display_name: human-readable name
3. clip_prompts: 2-3 prompts for image classification
4. keywords: 5-10 keywords that appear in this document type
5. llm_context: brief instructions for extracting data from this document type, including domain-specific OCR errors to fix
6. fields: list of fields to extract, each with name, type (text/date/currency/list/number), description, required (true/false)

For list-type fields, describe what each item should contain in the description.
Tailor field choices to the domain context above (e.g. VIN/plate for motor insurance, drug/dosage for health insurance).

Return ONLY valid YAML, no explanations:"""

    print(f"[Schema] Streaming schema generation for: {request.description[:50]}... (domain: {active_domain.domain_id})")
    
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
                ollama_payload = {
                    "model": model_id,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 2048,
                    },
                }
                # Gemma 4 defaults to reasoning mode; without think=false all
                # output tokens go into the hidden thinking channel and the
                # visible YAML gets truncated to a short tail.
                if is_thinking_model(model_id):
                    ollama_payload["think"] = False
                async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as http_client:
                    async with http_client.stream(
                        "POST",
                        f"{client.base_url}/api/generate",
                        json=ollama_payload,
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
    
    from app.services.domain_service import get_active_domain
    active_domain = get_active_domain()

    prompt = f"""Generate a document schema YAML for the following document type.

=== DOMAIN CONTEXT: {active_domain.display_name} ===
{active_domain.description}
=== END DOMAIN CONTEXT ===

USER DESCRIPTION:
{request.description}

Using the domain context above, generate a YAML schema with:
1. type_id: a short lowercase identifier (e.g., "invoice", "medical_report")
2. display_name: human-readable name
3. clip_prompts: 2-3 prompts for image classification
4. keywords: 5-10 keywords that appear in this document type
5. llm_context: brief instructions for extracting data from this document type, including domain-specific OCR errors to fix
6. fields: list of fields to extract, each with name, type (text/date/currency/list/number), description, required (true/false)

For list-type fields, describe what each item should contain in the description.
Tailor field choices to the domain context above.

Return ONLY valid YAML, no explanations:"""

    print(f"[Schema] Generating schema with LLM for: {request.description[:50]}... (domain: {active_domain.domain_id})")
    
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


# =============================================================================
# Domain description endpoints
# =============================================================================
# A "domain" is the free-form knowledge block injected into LLM prompts so the
# same codebase can serve different verticals (health insurance, motor
# insurance, etc.). Multiple domains can coexist; one is marked active.

class SetActiveDomainRequest(BaseModel):
    domain_id: str


@router.get("/domains")
async def list_all_domains():
    from app.services.domain_service import list_domains, get_active_domain_id
    return {
        "domains": list_domains(),
        "active_domain_id": get_active_domain_id(),
    }


@router.get("/domains/{domain_id}")
async def get_domain_details(domain_id: str):
    from app.services.domain_service import get_domain
    domain = get_domain(domain_id)
    if not domain:
        raise HTTPException(status_code=404, detail=f"Domain '{domain_id}' not found")
    return domain.to_dict()


@router.get("/domains/{domain_id}/yaml")
async def get_domain_yaml(domain_id: str):
    from app.services.domain_service import get_domain
    domain = get_domain(domain_id)
    if not domain or not domain.source_file:
        raise HTTPException(status_code=404, detail=f"Domain '{domain_id}' not found")
    try:
        with open(domain.source_file, "r", encoding="utf-8") as f:
            yaml_content = f.read()
    except IOError as e:
        raise HTTPException(status_code=500, detail=f"Failed to read domain file: {e}")
    return Response(content=yaml_content, media_type="text/plain")


@router.put("/domains/{domain_id}/yaml")
async def update_domain_yaml(domain_id: str, request: Request):
    """Create or update a domain from raw YAML. Body is text/plain."""
    from app.services.domain_service import save_domain_yaml
    yaml_content = (await request.body()).decode("utf-8")
    try:
        saved = save_domain_yaml(domain_id, yaml_content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    print(f"[Domain] Saved: {saved.domain_id} ({saved.display_name})")
    return {
        "status": "saved",
        "domain_id": saved.domain_id,
        "display_name": saved.display_name,
    }


@router.delete("/domains/{domain_id}")
async def delete_domain_endpoint(domain_id: str):
    from app.services.domain_service import delete_domain, get_domains
    if len(get_domains()) <= 1:
        raise HTTPException(status_code=400, detail="Cannot delete the last remaining domain")
    if not delete_domain(domain_id):
        raise HTTPException(status_code=404, detail=f"Domain '{domain_id}' not found")
    print(f"[Domain] Deleted: {domain_id}")
    return {"status": "deleted", "domain_id": domain_id}


@router.get("/domain/active")
async def get_active_domain_info():
    from app.services.domain_service import get_active_domain, get_active_domain_id
    domain = get_active_domain()
    return {
        "domain_id": get_active_domain_id(),
        "resolved_domain_id": domain.domain_id,
        "display_name": domain.display_name,
        "description": domain.description,
    }


@router.put("/domain/active")
async def set_active_domain(payload: SetActiveDomainRequest):
    from app.services.domain_service import get_domain, set_active_domain_id
    domain = get_domain(payload.domain_id)
    if not domain:
        raise HTTPException(status_code=404, detail=f"Domain '{payload.domain_id}' not found")
    set_active_domain_id(payload.domain_id)
    print(f"[Domain] Active domain set to: {payload.domain_id}")
    return {
        "status": "ok",
        "domain_id": domain.domain_id,
        "display_name": domain.display_name,
    }


@router.post("/domains/reload")
async def reload_all_domains():
    from app.services.domain_service import reload_domains
    domains = reload_domains()
    return {
        "status": "reloaded",
        "count": len(domains),
        "domains": list(domains.keys()),
    }

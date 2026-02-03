"""
LLM Service for OCR Post-Processing with Multimodal Support.

Provides integration with local LLMs via Ollama and Google Gemini for:
- Vision-enabled document understanding (multimodal)
- Text normalization and denoising
- Medicine name recognition and correction
- Improved text grouping and structure
- OCR artifact removal

Supports both text-only and vision-enabled models for optimal accuracy.
"""

import asyncio
import base64
import httpx
import json
import os
import re
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from app.config.document_schemas import get_schema, DocumentSchema, FieldSchema

# Check for Gemini API key at module load
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_AVAILABLE = bool(GEMINI_API_KEY)


# =============================================================================
# IMAGE ENCODING UTILITIES
# =============================================================================

def encode_image_to_base64(image_bytes: bytes) -> str:
    """Encode image bytes to base64 string for LLM input."""
    return base64.b64encode(image_bytes).decode('utf-8')


def get_image_mime_type(image_bytes: bytes) -> str:
    """Detect MIME type from image bytes using magic numbers."""
    if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        return "image/png"
    elif image_bytes[:2] == b'\xff\xd8':
        return "image/jpeg"
    elif image_bytes[:6] in (b'GIF87a', b'GIF89a'):
        return "image/gif"
    elif image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
        return "image/webp"
    elif image_bytes[:4] == b'%PDF':
        return "application/pdf"
    else:
        # Default to PNG for unknown formats
        return "image/png"


# =============================================================================
# MODEL REGISTRY WITH VISION CAPABILITIES
# =============================================================================

# Models that support vision/multimodal input
VISION_CAPABLE_MODELS = {
    # Ollama vision models
    "gemma3:27b",
    "gemma3:12b", 
    "gemma3:4b",
    "llava:34b",
    "llava:13b",
    "llava:7b",
    "llama3.2-vision:11b",
    "llama3.2-vision:90b",
    "minicpm-v:8b",
    "moondream:1.8b",
    "bakllava:7b",
    # Gemini models (all vision-capable)
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
}


def is_vision_model(model_id: str) -> bool:
    """Check if a model supports vision/image input."""
    model_lower = model_id.lower()
    # Check exact match or prefix match
    for vision_model in VISION_CAPABLE_MODELS:
        if model_lower == vision_model or model_lower.startswith(vision_model.split(":")[0]):
            return True
    # Also check for common vision model patterns
    vision_patterns = ["vision", "llava", "minicpm-v", "moondream", "bakllava", "gemma3"]
    return any(pattern in model_lower for pattern in vision_patterns)


class LLMModel(Enum):
    """Available LLM models for post-processing."""
    # Vision-enabled models (multimodal) - listed first as preferred
    GEMMA3_27B = "gemma3:27b"           # Google Gemma 3 27B with vision - RECOMMENDED DEFAULT
    GEMMA3_12B = "gemma3:12b"           # Google Gemma 3 12B with vision
    LLAVA_34B = "llava:34b"             # LLaVA 34B - high accuracy
    LLAVA_13B = "llava:13b"             # LLaVA 13B - balanced
    LLAMA_VISION = "llama3.2-vision:11b" # Llama 3.2 Vision
    MINICPM_V = "minicpm-v:8b"          # MiniCPM-V - efficient
    
    # Text-only models
    DEVSTRAL = "devstral:24b"
    QWEN_2_5 = "qwen2.5:7b"
    
    # Medical specialized (text-only)
    LLAMA_MEDITRON = "meditron:7b"
    MEDGEMMA = "medgemma:latest"
    
    # Legacy
    GPT_OSS = "gpt-oss-20b:latest"
    
    @classmethod
    def from_string(cls, model_name: str) -> "LLMModel":
        """Get model enum from string name."""
        model_lower = model_name.lower()
        
        # First try exact match with enum values
        for model in cls:
            if model.value.lower() == model_lower:
                return model
        
        # Then try partial matches
        mapping = {
            "devstral": cls.DEVSTRAL,
            "qwen2.5": cls.QWEN_2_5,
            "qwen": cls.QWEN_2_5,
            "gemma3:27b": cls.GEMMA3_27B,
            "gemma3:12b": cls.GEMMA3_12B,
            "gemma3": cls.GEMMA3_27B,  # Default to 27B
            "llava:34b": cls.LLAVA_34B,
            "llava:13b": cls.LLAVA_13B,
            "llava": cls.LLAVA_13B,  # Default to 13B
            "llama3.2-vision": cls.LLAMA_VISION,
            "llama-vision": cls.LLAMA_VISION,
            "minicpm-v": cls.MINICPM_V,
            "minicpm": cls.MINICPM_V,
            "meditron": cls.LLAMA_MEDITRON,
            "llama-meditron": cls.LLAMA_MEDITRON,
            "medgemma": cls.MEDGEMMA,
            "gpt-oss": cls.GPT_OSS,
            "gpt-oss-20b": cls.GPT_OSS,
        }
        return mapping.get(model_lower, cls.GEMMA3_27B)  # Default to vision-enabled model
    
    @property
    def display_name(self) -> str:
        """Human-readable model name."""
        names = {
            LLMModel.DEVSTRAL: "Devstral (24B)",
            LLMModel.QWEN_2_5: "Qwen 2.5 (7B)",
            LLMModel.GEMMA3_27B: "Gemma 3 (27B) [Vision]",
            LLMModel.GEMMA3_12B: "Gemma 3 (12B) [Vision]",
            LLMModel.LLAVA_34B: "LLaVA (34B) [Vision]",
            LLMModel.LLAVA_13B: "LLaVA (13B) [Vision]",
            LLMModel.LLAMA_VISION: "Llama 3.2 Vision (11B)",
            LLMModel.MINICPM_V: "MiniCPM-V (8B) [Vision]",
            LLMModel.LLAMA_MEDITRON: "Meditron (7B) [Medical]",
            LLMModel.MEDGEMMA: "MedGemma [Medical]",
            LLMModel.GPT_OSS: "GPT-OSS (20B)",
        }
        return names.get(self, self.value)
    
    @property
    def supports_vision(self) -> bool:
        """Check if this model supports vision/image input."""
        return is_vision_model(self.value)
    
    @property
    def is_medical_specialized(self) -> bool:
        """Check if this model is specialized for medical documents."""
        return self in (LLMModel.LLAMA_MEDITRON, LLMModel.MEDGEMMA)


@dataclass
class LLMConfig:
    """Configuration for LLM service."""
    ollama_base_url: str = "http://localhost:11434"
    default_model: LLMModel = LLMModel.GEMMA3_27B  # Vision-enabled model as default
    default_vision_model: LLMModel = LLMModel.GEMMA3_27B  # Default for multimodal
    timeout: float = 180.0  # LLM inference can be slow, especially for structured extraction
    max_tokens: int = 4096
    temperature: float = 0.3  # Lower for more deterministic output
    prefer_vision_models: bool = True  # Prefer vision models when image is available


def select_optimal_model(
    available_models: List[str],
    has_image: bool = False,
    document_type: str = None,
    prefer_vision: bool = True
) -> Optional[str]:
    """
    Select the best model based on context and available models.
    
    Args:
        available_models: List of model IDs available in Ollama
        has_image: Whether an image is available for processing
        document_type: Type of document being processed
        prefer_vision: Whether to prefer vision models when image is available
        
    Returns:
        Selected model ID or None if no suitable model found
    """
    if not available_models:
        return None
    
    # Priority order for vision models (when image is available)
    vision_priority = [
        "gemma3:27b", "gemma3:12b", "llava:34b", "llava:13b", 
        "llama3.2-vision:11b", "minicpm-v:8b", "llava:7b"
    ]
    
    # Priority for medical documents
    medical_priority = ["medgemma:latest", "meditron:7b", "gemma3:27b"]
    
    # Priority for general text processing
    text_priority = ["devstral:24b", "qwen2.5:7b", "gemma3:27b"]
    
    # Check for available models in each category
    available_set = set(m.lower() for m in available_models)
    
    def find_available(priority_list):
        for model in priority_list:
            # Check exact match or prefix match
            if model in available_set:
                return model
            for avail in available_models:
                if avail.lower().startswith(model.split(":")[0]):
                    return avail
        return None
    
    # Select based on context
    if has_image and prefer_vision:
        selected = find_available(vision_priority)
        if selected:
            return selected
    
    if document_type in ["prescription", "medical_form"]:
        selected = find_available(medical_priority)
        if selected:
            return selected
    
    # Fallback to text models
    selected = find_available(text_priority)
    if selected:
        return selected
    
    # Last resort: return first available
    return available_models[0] if available_models else None


class OllamaClient:
    """Client for interacting with Ollama API with multimodal support."""
    
    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig()
        self.base_url = self.config.ollama_base_url
    
    async def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False
    
    async def get_system_info(self) -> Dict[str, Any]:
        """Get Ollama system info including acceleration status."""
        info = {
            "available": False,
            "acceleration": "unknown",
            "acceleration_details": None,
            "version": None
        }
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Check if available
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    info["available"] = True
                
                # Try to get version
                try:
                    ver_response = await client.get(f"{self.base_url}/api/version")
                    if ver_response.status_code == 200:
                        info["version"] = ver_response.json().get("version")
                except:
                    pass
                
                # Detect acceleration by checking a lightweight generate call
                # or by checking system capabilities
                import platform
                system = platform.system()
                
                if system == "Darwin":
                    # macOS - check for Apple Silicon
                    import subprocess
                    try:
                        chip = subprocess.check_output(
                            ["sysctl", "-n", "machdep.cpu.brand_string"],
                            stderr=subprocess.DEVNULL
                        ).decode().strip()
                        if "Apple" in chip:
                            info["acceleration"] = "metal"
                            info["acceleration_details"] = f"Apple Silicon ({chip})"
                        else:
                            info["acceleration"] = "cpu"
                            info["acceleration_details"] = f"Intel Mac ({chip})"
                    except:
                        info["acceleration"] = "unknown"
                        
                elif system == "Linux":
                    # Linux - check for NVIDIA GPU
                    import subprocess
                    try:
                        nvidia_info = subprocess.check_output(
                            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                            stderr=subprocess.DEVNULL
                        ).decode().strip()
                        if nvidia_info:
                            info["acceleration"] = "cuda"
                            info["acceleration_details"] = nvidia_info
                    except:
                        # Check for ROCm (AMD)
                        try:
                            subprocess.check_output(["rocm-smi"], stderr=subprocess.DEVNULL)
                            info["acceleration"] = "rocm"
                            info["acceleration_details"] = "AMD GPU (ROCm)"
                        except:
                            info["acceleration"] = "cpu"
                            info["acceleration_details"] = "No GPU detected"
                else:
                    info["acceleration"] = "cpu"
                    
        except Exception as e:
            print(f"[LLM] Error getting system info: {e}")
        
        return info
    
    async def list_models(self) -> List[str]:
        """List available models in Ollama."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        return []
    
    async def is_model_available(self, model: LLMModel) -> bool:
        """Check if a specific model is available."""
        models = await self.list_models()
        model_name = model.value.split(":")[0]
        return any(model_name in m for m in models)
    
    async def pull_model(self, model: LLMModel, progress_callback=None) -> bool:
        """Pull a model from Ollama registry."""
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:  # 10 min timeout for pulling
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/pull",
                    json={"name": model.value}
                ) as response:
                    if response.status_code != 200:
                        return False
                    
                    async for line in response.aiter_lines():
                        if progress_callback and line:
                            try:
                                import json
                                data = json.loads(line)
                                progress_callback(data)
                            except:
                                pass
                    return True
        except Exception as e:
            print(f"[LLM] Error pulling model: {e}")
            return False
    
    async def generate(
        self, 
        prompt: str, 
        model: LLMModel | str = None,
        system_prompt: str = None,
        images: List[str] = None,  # Base64-encoded images for multimodal models
        temperature: float = None,
        max_tokens: int = None,
        stream_to_stdout: bool = True
    ) -> Optional[str]:
        """Generate text using the specified model with optional streaming output.
        
        Args:
            prompt: The text prompt
            model: Can be LLMModel enum or string model ID for custom models
            system_prompt: Optional system prompt
            images: List of base64-encoded images for vision models
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream_to_stdout: Whether to stream output to stdout
            
        Returns:
            Generated text or None if failed
        """
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens
        
        # Handle both LLMModel enum and string model IDs
        if model is None:
            model = self.config.default_model
            model_id = model.value
            model_name = model.display_name
        elif isinstance(model, str):
            model_id = model
            model_name = model
        else:
            model_id = model.value
            model_name = model.display_name
        
        # Check if using vision with non-vision model
        using_vision = images is not None and len(images) > 0
        model_supports_vision = is_vision_model(model_id)
        
        if using_vision:
            if model_supports_vision:
                print(f"[LLM] Generating with {model_name} (VISION MODE - {len(images)} image(s))...")
            else:
                print(f"[LLM] WARNING: Model {model_name} does not support vision. Images will be ignored.")
                images = None  # Clear images for non-vision models
        else:
            print(f"[LLM] Generating with {model_name} (text-only)...")
        
        print(f"[LLM] Prompt length: {len(prompt)} chars")
        
        payload = {
            "model": model_id,
            "prompt": prompt,
            "stream": stream_to_stdout,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        # Add images for multimodal models (Ollama format)
        if images and model_supports_vision:
            payload["images"] = images
        
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                if stream_to_stdout:
                    # Stream response and print tokens as they arrive
                    full_response = ""
                    print("[LLM] Response: ", end="", flush=True)
                    
                    async with client.stream(
                        "POST",
                        f"{self.base_url}/api/generate",
                        json=payload
                    ) as response:
                        if response.status_code != 200:
                            error_text = await response.aread()
                            print(f"\n[LLM] Error: {response.status_code} - {error_text[:200]}")
                            return None
                        
                        async for line in response.aiter_lines():
                            if line:
                                try:
                                    data = json.loads(line)
                                    token = data.get("response", "")
                                    full_response += token
                                    print(token, end="", flush=True)
                                    
                                    if data.get("done"):
                                        break
                                except json.JSONDecodeError:
                                    pass
                    
                    print()  # Newline after streaming
                    print(f"[LLM] Generation complete ({len(full_response)} chars)")
                    return full_response
                else:
                    # Non-streaming mode
                    response = await client.post(
                        f"{self.base_url}/api/generate",
                        json=payload
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        result = data.get("response", "")
                        print(f"[LLM] Generation complete ({len(result)} chars)")
                        return result
                    else:
                        print(f"[LLM] Error: {response.status_code} - {response.text[:200]}")
                        return None
        except Exception as e:
            print(f"[LLM] Generation error: {e}")
            return None
    
    def generate_sync(
        self,
        prompt: str,
        model: LLMModel = None,
        system_prompt: str = None,
        temperature: float = None,
        max_tokens: int = None
    ) -> Optional[str]:
        """Synchronous wrapper for generate."""
        return asyncio.run(self.generate(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        ))


class LLMPostProcessor:
    """
    LLM-based post-processor for OCR results with multimodal support.
    
    Capabilities:
    - Vision-enabled document understanding (when image is provided)
    - Context-aware text normalization based on document type
    - Structured field extraction using document schemas
    - Medical terminology recognition and correction
    - OCR artifact removal
    """
    
    # System prompt for vision-enabled models (concise, focused on image)
    VISION_SYSTEM_PROMPT = """You are a medical document processor with vision capabilities.

You receive TWO sources for the SAME document:
1. Document IMAGE - you can see it directly
2. OCR TEXT - machine-extracted text (may have errors OR may catch things hard to see)

CRITICAL EXTRACTION RULES:
- Include ALL items from BOTH sources (union, not intersection)
- If OCR found something, verify it in the image and include it (with corrections if needed)
- If you see something in the image that OCR missed, include it too
- OCR might catch faint/small text that's hard to see - don't ignore it
- Your job is to MAXIMIZE recall - capture everything, miss nothing
- Only exclude items if they are clearly OCR artifacts (garbage characters)

Medical terms, drug names, and dosages must be accurate.
Always respond with valid JSON only."""

    # System prompt for text-only models (more detailed guidance)
    TEXT_SYSTEM_PROMPT = """You are a medical document processor specialized in OCR correction.

MEDICAL DOMAIN: These documents contain medical terminology, drug names, dosages, and clinical information.

Common medical OCR errors:
- Drug names: Amoxici11in→Amoxicillin, Metr0nidazole→Metronidazole
- Dosages: 5OOmg→500mg, 1OO→100, 2Omg→20mg
- Units: rng→mg, rnl→ml, lU→IU
- Character confusion: 0/O, 1/l/I, 5/S, 8/B, rn/m

Latin medical abbreviations:
- bid/b.i.d. = twice daily
- tid/t.i.d. = three times daily
- qid/q.i.d. = four times daily
- PRN/prn = as needed
- PO = by mouth
- SIG = instructions

Always respond with valid JSON only."""

    def __init__(self, client: OllamaClient = None):
        self.client = client or OllamaClient()
    
    def _build_field_list(self, schema: DocumentSchema) -> str:
        """Build a compact field list for prompts."""
        fields = []
        for field in schema.fields:
            req = "*" if field.required else ""
            fields.append(f"{field.name}{req} ({field.field_type.value})")
        return ", ".join(fields)
    
    def _build_example_json(self, schema: DocumentSchema) -> str:
        """Build example JSON structure based on schema fields."""
        example_fields = {}
        for field in schema.fields:
            if field.field_type.value == "list":
                example_fields[field.name] = [{"name": "...", "details": "..."}]
            elif field.field_type.value == "date":
                example_fields[field.name] = "YYYY-MM-DD or null"
            elif field.field_type.value == "currency":
                example_fields[field.name] = "amount or null"
            elif field.field_type.value == "number":
                example_fields[field.name] = "number or null"
            else:
                example_fields[field.name] = "value or null"
        return json.dumps(example_fields, indent=2)
    
    def _build_vision_prompt(self, text: str, schema: DocumentSchema) -> str:
        """Build a concise prompt for vision-enabled models."""
        field_list = self._build_field_list(schema)
        example_json = self._build_example_json(schema)
        
        return f"""DOCUMENT TYPE: {schema.display_name}

EXTRACT THESE FIELDS: {field_list}

=== OCR-EXTRACTED TEXT ===
{text}
=== END OCR TEXT ===

TASK: Extract ALL information by combining BOTH the image AND the OCR text above.

RULES:
1. INCLUDE everything the OCR found - correct spelling errors but keep all items
2. ALSO include anything you see in the image that OCR might have missed
3. For each medication/item in OCR text: verify against image, fix errors, but KEEP IT
4. Do NOT drop items just because they're hard to read - OCR often catches faint text
5. Merge duplicates: if same item appears twice, include it once with best spelling
6. Goal: MAXIMIZE extraction - capture ALL medications, names, dates, etc.

Return JSON only:
{{"corrected_text": "all text from document with corrections", "extracted_fields": {example_json}}}"""
    
    def _build_text_prompt(self, text: str, schema: DocumentSchema) -> str:
        """Build a detailed prompt for text-only models."""
        field_list = self._build_field_list(schema)
        example_json = self._build_example_json(schema)
        
        # Build field descriptions
        field_descriptions = []
        for field in schema.fields:
            req = " (REQUIRED)" if field.required else ""
            field_descriptions.append(f"  - {field.name}{req}: {field.description}")
        fields_text = "\n".join(field_descriptions)
        
        return f"""DOCUMENT TYPE: {schema.display_name.upper()}

FIELDS TO EXTRACT:
{fields_text}

{schema.llm_context}

OCR TEXT (may contain errors):
---
{text}
---

TASK:
1. Fix OCR errors using medical domain knowledge
2. Extract all fields listed above
3. For medications: ensure drug names, dosages, and instructions are accurate

Return JSON only:
{{"corrected_text": "full corrected text", "extracted_fields": {example_json}}}"""
    
    def _parse_json_response(self, response: str, schema: DocumentSchema) -> Dict[str, Any]:
        """Parse LLM JSON response, handling common issues."""
        if not response:
            return None
        
        # Try to extract JSON from response
        response = response.strip()
        
        # Handle markdown code blocks
        if response.startswith("```"):
            lines = response.split("\n")
            json_lines = []
            in_json = False
            for line in lines:
                if line.startswith("```") and not in_json:
                    in_json = True
                    continue
                elif line.startswith("```") and in_json:
                    break
                elif in_json:
                    json_lines.append(line)
            response = "\n".join(json_lines)
        
        # Try to find JSON object in response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            response = json_match.group()
        
        # Fix common JSON escape issues from LLMs
        response = re.sub(r'\\([^"\\\/bfnrtu])', r'\\\\\1', response)
        
        try:
            parsed = json.loads(response)
            return parsed
        except json.JSONDecodeError as e:
            print(f"[LLM] JSON parse error: {e}")
            print(f"[LLM] Response was: {response[:500]}...")
            
            # Try a more aggressive fix
            try:
                fixed = re.sub(r'\\(?!["\\/bfnrtu])', '', response)
                parsed = json.loads(fixed)
                print("[LLM] Fixed JSON by removing invalid escapes")
                return parsed
            except json.JSONDecodeError:
                pass
            
            return None
    
    async def extract_structured(
        self,
        text: str,
        document_type: str,
        model: LLMModel | str = None,
        image_bytes: bytes = None
    ) -> Dict[str, Any]:
        """
        Extract structured data from OCR text with optional image for vision models.
        
        Args:
            text: Raw OCR text to process
            document_type: Type ID of the document (e.g., "prescription", "receipt")
            model: LLM model to use
            image_bytes: Optional document image for vision-enabled models
            
        Returns:
            Dict with corrected text, extracted fields, and metadata
        """
        if model is None:
            model = self.client.config.default_model
        
        # Get model ID and display name
        if isinstance(model, LLMModel):
            model_id = model.value
            model_display_name = model.display_name
        else:
            model_id = model
            model_display_name = model
        
        # Check if we can use vision
        use_vision = image_bytes is not None and is_vision_model(model_id)
        
        schema = get_schema(document_type)
        
        print(f"[LLM] Extracting structured data for document type: {document_type}")
        print(f"[LLM] Using schema: {schema.display_name} ({len(schema.fields)} fields)")
        print(f"[LLM] Mode: {'VISION' if use_vision else 'TEXT-ONLY'}")
        
        # Build appropriate prompt based on mode
        if use_vision:
            prompt = self._build_vision_prompt(text, schema)
            system_prompt = self.VISION_SYSTEM_PROMPT
        else:
            prompt = self._build_text_prompt(text, schema)
            system_prompt = self.TEXT_SYSTEM_PROMPT
        
        # Prepare image for vision models
        images = None
        if use_vision and image_bytes:
            images = [encode_image_to_base64(image_bytes)]
        
        # Generate structured extraction
        result = await self.client.generate(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
            images=images,
            temperature=0.1,  # Low temperature for consistent structured output
            max_tokens=4096
        )
        
        if not result:
            return {
                "success": False,
                "original_text": text,
                "error": "LLM generation failed",
                "model": model_display_name,
                "document_type": document_type,
                "vision_used": use_vision,
                "extracted_fields": schema.get_default_extraction()
            }
        
        # Parse the JSON response
        parsed = self._parse_json_response(result, schema)
        
        if parsed:
            # Ensure all schema fields are present
            extracted = parsed.get("extracted_fields", {})
            for field in schema.fields:
                if field.name not in extracted:
                    extracted[field.name] = field.default
            
            return {
                "success": True,
                "original_text": text,
                "corrected_text": parsed.get("corrected_text", text),
                "extracted_fields": extracted,
                "confidence_notes": parsed.get("confidence_notes", ""),
                "model": model_display_name,
                "document_type": document_type,
                "document_type_name": schema.display_name,
                "schema_fields": [f.name for f in schema.fields],
                "required_fields": schema.get_required_fields(),
                "vision_used": use_vision
            }
        else:
            # Fallback: return raw result as corrected text
            return {
                "success": True,
                "original_text": text,
                "corrected_text": result.strip(),
                "extracted_fields": schema.get_default_extraction(),
                "confidence_notes": "JSON parsing failed, returning raw corrected text",
                "model": model_display_name,
                "document_type": document_type,
                "document_type_name": schema.display_name,
                "parse_error": True,
                "vision_used": use_vision
            }
    
    async def process_text(
        self,
        text: str,
        model: LLMModel | str = None,
        document_type: str = None,
        image_bytes: bytes = None
    ) -> Dict[str, Any]:
        """
        Process OCR text using LLM with context-aware extraction.
        
        This is the main entry point that uses document schemas for structured extraction.
        Supports both text-only and vision-enabled processing.
        
        Args:
            text: Raw OCR text to process
            model: LLM model to use (optional)
            document_type: Type ID of the document (optional)
            image_bytes: Document image for vision-enabled models (optional)
        """
        return await self.extract_structured(
            text, 
            document_type or "unknown", 
            model,
            image_bytes
        )
    
    async def denoise_text(
        self,
        text: str,
        model: LLMModel = None,
        image_bytes: bytes = None
    ) -> str:
        """
        Quick denoising of OCR text without structured extraction.
        Can use vision if image is provided and model supports it.
        """
        model = model or self.client.config.default_model
        model_id = model.value if isinstance(model, LLMModel) else model
        
        use_vision = image_bytes is not None and is_vision_model(model_id)
        
        if use_vision:
            prompt = """Look at this document image and the OCR text below.
Fix any OCR errors you can see by comparing to the actual image.
Return only the corrected text, nothing else.

OCR TEXT:
""" + text
            images = [encode_image_to_base64(image_bytes)]
        else:
            prompt = f"""Fix OCR errors in this medical document text.
Common errors: 0/O confusion, 1/l/I confusion, rn/m confusion in drug names.
Return only the corrected text, nothing else.

{text}"""
            images = None

        result = await self.client.generate(
            prompt=prompt,
            model=model,
            images=images,
            temperature=0.1
        )
        
        return result.strip() if result else text
    
    def process_text_sync(
        self,
        text: str,
        model: LLMModel = None,
        document_type: str = None,
        image_bytes: bytes = None
    ) -> Dict[str, Any]:
        """Synchronous wrapper for process_text."""
        return asyncio.run(self.process_text(text, model, document_type, image_bytes))


# =============================================================================
# GEMINI CLIENT WITH MULTIMODAL SUPPORT
# =============================================================================

class GeminiClient:
    """Client for Google Gemini API with multimodal (vision) support."""
    
    GEMINI_MODEL = "gemini-2.5-pro"
    GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or GEMINI_API_KEY
        self.available = bool(self.api_key)
    
    def is_available(self) -> bool:
        """Check if Gemini API is available (API key is set)."""
        return self.available
    
    async def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        images: List[Tuple[str, str]] = None,  # [(base64_data, mime_type), ...]
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Optional[str]:
        """Generate text using Gemini API with optional image input.
        
        Args:
            prompt: The text prompt
            system_prompt: Optional system prompt
            images: List of tuples (base64_data, mime_type) for vision input
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text or None if failed
        """
        if not self.available:
            print("[Gemini] API key not configured")
            return None
        
        using_vision = images is not None and len(images) > 0
        
        if using_vision:
            print(f"[Gemini] Generating with {self.GEMINI_MODEL} (VISION MODE - {len(images)} image(s))...")
        else:
            print(f"[Gemini] Generating with {self.GEMINI_MODEL} (text-only)...")
        print(f"[Gemini] Prompt length: {len(prompt)} chars")
        
        # Build the request
        url = f"{self.GEMINI_API_URL}/{self.GEMINI_MODEL}:generateContent?key={self.api_key}"
        
        # Combine system prompt and user prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        # Build parts array with text and optional images
        parts = []
        
        # Add images first (better for vision models)
        if images:
            for img_base64, mime_type in images:
                parts.append({
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": img_base64
                    }
                })
        
        # Add text prompt
        parts.append({"text": full_prompt})
        
        payload = {
            "contents": [{
                "parts": parts
            }],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.post(url, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    # Extract text from Gemini response
                    candidates = data.get("candidates", [])
                    if candidates:
                        content = candidates[0].get("content", {})
                        parts = content.get("parts", [])
                        if parts:
                            result = parts[0].get("text", "")
                            print(f"[Gemini] Generation complete ({len(result)} chars)")
                            return result
                    print("[Gemini] No content in response")
                    return None
                else:
                    print(f"[Gemini] Error: {response.status_code} - {response.text[:500]}")
                    return None
        except Exception as e:
            print(f"[Gemini] Generation error: {e}")
            return None


class GeminiPostProcessor:
    """LLM post-processor using Gemini API with multimodal (vision) support."""
    
    # Gemini is always vision-capable, use concise prompts
    VISION_SYSTEM_PROMPT = LLMPostProcessor.VISION_SYSTEM_PROMPT
    
    def __init__(self, client: GeminiClient = None):
        self.client = client or GeminiClient()
    
    def _build_field_list(self, schema: DocumentSchema) -> str:
        """Build a compact field list for prompts."""
        fields = []
        for field in schema.fields:
            req = "*" if field.required else ""
            fields.append(f"{field.name}{req} ({field.field_type.value})")
        return ", ".join(fields)
    
    def _build_example_json(self, schema: DocumentSchema) -> str:
        """Build example JSON structure based on schema fields."""
        example_fields = {}
        for field in schema.fields:
            if field.field_type.value == "list":
                example_fields[field.name] = [{"name": "...", "details": "..."}]
            elif field.field_type.value == "date":
                example_fields[field.name] = "YYYY-MM-DD or null"
            elif field.field_type.value == "currency":
                example_fields[field.name] = "amount or null"
            elif field.field_type.value == "number":
                example_fields[field.name] = "number or null"
            else:
                example_fields[field.name] = "value or null"
        return json.dumps(example_fields, indent=2)
    
    def _build_extraction_prompt(self, text: str, schema: DocumentSchema, use_vision: bool = False) -> str:
        """Build extraction prompt optimized for Gemini."""
        field_list = self._build_field_list(schema)
        example_json = self._build_example_json(schema)
        
        if use_vision:
            # Concise prompt when image is provided
            return f"""DOCUMENT TYPE: {schema.display_name}

EXTRACT THESE FIELDS: {field_list}

=== OCR-EXTRACTED TEXT ===
{text}
=== END OCR TEXT ===

TASK: Extract ALL information by combining BOTH the image AND the OCR text above.

RULES:
1. INCLUDE everything the OCR found - correct spelling errors but keep all items
2. ALSO include anything you see in the image that OCR might have missed
3. For each medication/item in OCR text: verify against image, fix errors, but KEEP IT
4. Do NOT drop items just because they're hard to read - OCR often catches faint text
5. Merge duplicates: if same item appears twice, include it once with best spelling
6. Goal: MAXIMIZE extraction - capture ALL medications, names, dates, etc.

Return JSON only:
{{"corrected_text": "all text from document with corrections", "extracted_fields": {example_json}}}"""
        else:
            # More detailed prompt for text-only
            field_descriptions = []
            for field in schema.fields:
                req = " (REQUIRED)" if field.required else ""
                field_descriptions.append(f"  - {field.name}{req}: {field.description}")
            fields_text = "\n".join(field_descriptions)
            
            return f"""DOCUMENT TYPE: {schema.display_name.upper()}

FIELDS TO EXTRACT:
{fields_text}

{schema.llm_context}

OCR TEXT (may contain errors):
---
{text}
---

Fix OCR errors using medical domain knowledge. Extract all fields.

Return JSON only:
{{"corrected_text": "full corrected text", "extracted_fields": {example_json}}}"""
    
    def _parse_json_response(self, response: str, schema: DocumentSchema) -> Dict[str, Any]:
        """Parse LLM JSON response."""
        if not response:
            return None
        
        response = response.strip()
        
        # Handle markdown code blocks
        if response.startswith("```"):
            lines = response.split("\n")
            json_lines = []
            in_json = False
            for line in lines:
                if line.startswith("```") and not in_json:
                    in_json = True
                    continue
                elif line.startswith("```") and in_json:
                    break
                elif in_json:
                    json_lines.append(line)
            response = "\n".join(json_lines)
        
        # Try to find JSON object
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            response = json_match.group()
        
        # Fix common JSON escape issues
        response = re.sub(r'\\([^"\\\/bfnrtu])', r'\\\\\1', response)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            print(f"[Gemini] JSON parse error: {e}")
            try:
                fixed = re.sub(r'\\(?!["\\/bfnrtu])', '', response)
                return json.loads(fixed)
            except json.JSONDecodeError:
                return None
    
    async def extract_structured(
        self,
        text: str,
        document_type: str,
        image_bytes: bytes = None
    ) -> Dict[str, Any]:
        """Extract structured data using Gemini with optional vision."""
        schema = get_schema(document_type)
        
        use_vision = image_bytes is not None
        
        print(f"[Gemini] Extracting structured data for document type: {document_type}")
        print(f"[Gemini] Using schema: {schema.display_name} ({len(schema.fields)} fields)")
        print(f"[Gemini] Mode: {'VISION' if use_vision else 'TEXT-ONLY'}")
        
        prompt = self._build_extraction_prompt(text, schema, use_vision)
        system_prompt = self.VISION_SYSTEM_PROMPT
        
        # Prepare images for Gemini if provided
        images = None
        if use_vision and image_bytes:
            mime_type = get_image_mime_type(image_bytes)
            img_base64 = encode_image_to_base64(image_bytes)
            images = [(img_base64, mime_type)]
        
        result = await self.client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            images=images,
            temperature=0.1,
            max_tokens=4096
        )
        
        if not result:
            return {
                "success": False,
                "original_text": text,
                "error": "Gemini generation failed",
                "model": "Gemini 2.5 Pro",
                "document_type": document_type,
                "vision_used": use_vision,
                "extracted_fields": schema.get_default_extraction()
            }
        
        parsed = self._parse_json_response(result, schema)
        
        if parsed:
            extracted = parsed.get("extracted_fields", {})
            for field in schema.fields:
                if field.name not in extracted:
                    extracted[field.name] = field.default
            
            return {
                "success": True,
                "original_text": text,
                "corrected_text": parsed.get("corrected_text", text),
                "extracted_fields": extracted,
                "confidence_notes": parsed.get("confidence_notes", ""),
                "model": "Gemini 2.5 Pro",
                "document_type": document_type,
                "document_type_name": schema.display_name,
                "schema_fields": [f.name for f in schema.fields],
                "required_fields": schema.get_required_fields(),
                "vision_used": use_vision
            }
        else:
            return {
                "success": True,
                "original_text": text,
                "corrected_text": result.strip(),
                "extracted_fields": schema.get_default_extraction(),
                "confidence_notes": "JSON parsing failed, returning raw corrected text",
                "model": "Gemini 2.5 Pro",
                "document_type": document_type,
                "document_type_name": schema.display_name,
                "parse_error": True,
                "vision_used": use_vision
            }
    
    async def process_text(
        self,
        text: str,
        document_type: str = None,
        image_bytes: bytes = None
    ) -> Dict[str, Any]:
        """Process OCR text using Gemini with optional vision."""
        return await self.extract_structured(text, document_type or "unknown", image_bytes)


# =============================================================================
# SINGLETON INSTANCES
# =============================================================================

_ollama_client: Optional[OllamaClient] = None
_llm_processor: Optional[LLMPostProcessor] = None
_gemini_client: Optional[GeminiClient] = None
_gemini_processor: Optional[GeminiPostProcessor] = None


def get_ollama_client() -> OllamaClient:
    """Get or create Ollama client singleton."""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient()
    return _ollama_client


def get_llm_processor() -> LLMPostProcessor:
    """Get or create LLM processor singleton."""
    global _llm_processor
    if _llm_processor is None:
        _llm_processor = LLMPostProcessor(get_ollama_client())
    return _llm_processor


def get_gemini_client() -> GeminiClient:
    """Get or create Gemini client singleton."""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiClient()
    return _gemini_client


def get_gemini_processor() -> GeminiPostProcessor:
    """Get or create Gemini processor singleton."""
    global _gemini_processor
    if _gemini_processor is None:
        _gemini_processor = GeminiPostProcessor(get_gemini_client())
    return _gemini_processor

"""
LLM Service for OCR Post-Processing.

Provides integration with local LLMs via Ollama for:
- Text normalization and denoising
- Medicine name recognition and correction
- Improved text grouping and structure
- OCR artifact removal

Also supports Google Gemini API when GEMINI_API_KEY is set.
"""

import asyncio
import httpx
import json
import os
import re
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from enum import Enum

from app.config.document_schemas import get_schema, DocumentSchema, FieldSchema

# Check for Gemini API key at module load
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_AVAILABLE = bool(GEMINI_API_KEY)


class LLMModel(Enum):
    """Available LLM models for post-processing."""
    DEVSTRAL = "devstral:24b"
    QWEN_2_5 = "qwen2.5:7b"
    LLAMA_MEDITRON = "meditron:7b"
    MEDGEMMA = "medgemma:latest"
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
            "meditron": cls.LLAMA_MEDITRON,
            "llama-meditron": cls.LLAMA_MEDITRON,
            "medgemma": cls.MEDGEMMA,
            "gpt-oss": cls.GPT_OSS,
            "gpt-oss-20b": cls.GPT_OSS,
        }
        return mapping.get(model_lower, cls.DEVSTRAL)
    
    @property
    def display_name(self) -> str:
        """Human-readable model name."""
        names = {
            LLMModel.DEVSTRAL: "Devstral (24B)",
            LLMModel.QWEN_2_5: "Qwen 2.5 (7B)",
            LLMModel.LLAMA_MEDITRON: "Llama-3-Meditron (7B)",
            LLMModel.MEDGEMMA: "MedGemma",
            LLMModel.GPT_OSS: "OpenAI GPT-OSS-20B",
        }
        return names.get(self, self.value)


@dataclass
class LLMConfig:
    """Configuration for LLM service."""
    ollama_base_url: str = "http://localhost:11434"
    default_model: LLMModel = LLMModel.DEVSTRAL
    timeout: float = 180.0  # LLM inference can be slow, especially for structured extraction
    max_tokens: int = 4096
    temperature: float = 0.3  # Lower for more deterministic output


class OllamaClient:
    """Client for interacting with Ollama API."""
    
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
        temperature: float = None,
        max_tokens: int = None,
        stream_to_stdout: bool = True
    ) -> Optional[str]:
        """Generate text using the specified model with optional streaming output.
        
        Args:
            model: Can be LLMModel enum or string model ID for custom models
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
        
        print(f"[LLM] Generating with {model_name}...")
        print(f"[LLM] Prompt length: {len(prompt)} chars")
        
        payload = {
            "model": model_id,
            "prompt": prompt,
            "stream": stream_to_stdout,  # Stream for real-time output
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
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
                            print(f"\n[LLM] Error: {response.status_code}")
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
                        print(f"[LLM] Error: {response.status_code} - {response.text}")
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
    LLM-based post-processor for OCR results.
    
    Capabilities:
    - Context-aware text normalization based on document type
    - Structured field extraction using document schemas
    - Medicine name recognition and correction
    - OCR artifact removal
    """
    
    BASE_SYSTEM_PROMPT = """You are an expert OCR post-processor and document data extractor.

Your task is to:
1. Clean and correct OCR-extracted text
2. Extract structured information based on the document type
3. Return results in a specific JSON format

Common OCR errors to fix:
- Character confusion: 0/O, 1/l/I, 5/S, 8/B, rn/m, cl/d
- Missing or extra spaces
- Wrong punctuation
- Garbled text or artifacts
- Split words that should be joined

IMPORTANT: Always respond with valid JSON only. No explanations or commentary outside the JSON."""

    def __init__(self, client: OllamaClient = None):
        self.client = client or OllamaClient()
    
    def _build_extraction_prompt(self, text: str, schema: DocumentSchema) -> str:
        """Build a context-aware extraction prompt based on document schema."""
        # Build detailed field descriptions with types
        field_descriptions = []
        for field in schema.fields:
            req = " (REQUIRED)" if field.required else " (optional)"
            field_type = f"[{field.field_type.value}]"
            field_descriptions.append(f"  - {field.name} {field_type}{req}: {field.description}")
        
        fields_text = "\n".join(field_descriptions)
        
        # Build example JSON structure based on actual schema fields
        example_fields = {}
        for field in schema.fields:
            if field.field_type.value == "list":
                example_fields[field.name] = [{"name": "...", "details": "..."}]
            elif field.field_type.value == "date":
                example_fields[field.name] = "YYYY-MM-DD or null"
            elif field.field_type.value == "currency":
                example_fields[field.name] = "amount as string or null"
            elif field.field_type.value == "number":
                example_fields[field.name] = "number or null"
            else:
                example_fields[field.name] = "extracted value or null"
        
        import json
        example_json = json.dumps(example_fields, indent=4)
        
        prompt = f"""=== DOCUMENT CLASSIFICATION ===
This document has been classified as: **{schema.display_name.upper()}** (type_id: {schema.type_id})

=== DOCUMENT TEMPLATE SCHEMA ===
The following schema defines exactly what fields must be extracted from this {schema.display_name}:

Document Type: {schema.display_name}
Type ID: {schema.type_id}

Fields to Extract (* = required):
{fields_text}

=== EXTRACTION CONTEXT ===
{schema.llm_context}

=== RAW OCR TEXT (may contain errors) ===
{text}
=== END OCR TEXT ===

=== YOUR TASK ===
1. This is CONFIRMED to be a {schema.display_name} document
2. Extract ALL fields defined in the schema above
3. Fix OCR errors in the text (character confusion, spacing issues)
4. Return structured data matching the schema exactly

Return ONLY this JSON (no other text):
{{
  "corrected_text": "The full OCR text with errors corrected",
  "extracted_fields": {example_json}
}}"""
        
        return prompt
    
    def _build_system_prompt(self, schema: DocumentSchema) -> str:
        """Build system prompt with document context."""
        return f"""{self.BASE_SYSTEM_PROMPT}

DOCUMENT CONTEXT:
{schema.llm_context}"""
    
    def _parse_json_response(self, response: str, schema: DocumentSchema) -> Dict[str, Any]:
        """Parse LLM JSON response, handling common issues."""
        if not response:
            return None
        
        # Try to extract JSON from response
        response = response.strip()
        
        # Handle markdown code blocks
        if response.startswith("```"):
            lines = response.split("\n")
            # Remove first and last lines (```json and ```)
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
        # Replace invalid escape sequences like \N with \\N or just N
        response = re.sub(r'\\([^"\\\/bfnrtu])', r'\\\\\1', response)
        
        try:
            parsed = json.loads(response)
            return parsed
        except json.JSONDecodeError as e:
            print(f"[LLM] JSON parse error: {e}")
            print(f"[LLM] Response was: {response[:500]}...")
            
            # Try a more aggressive fix - replace all problematic escapes
            try:
                # Replace backslash-letter combinations that aren't valid JSON escapes
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
        model: LLMModel | str = None
    ) -> Dict[str, Any]:
        """
        Extract structured data from OCR text based on document type schema.
        
        Args:
            text: Raw OCR text to process
            document_type: Type ID of the document (e.g., "prescription", "receipt")
            model: LLM model to use
            
        Returns:
            Dict with corrected text, extracted fields, and metadata
        """
        if model is None:
            model = self.client.config.default_model
        
        # Get model name for display
        model_display_name = model.display_name if isinstance(model, LLMModel) else model
        
        schema = get_schema(document_type)
        
        print(f"[LLM] Extracting structured data for document type: {document_type}")
        print(f"[LLM] Using schema: {schema.display_name} ({len(schema.fields)} fields)")
        
        # Build context-aware prompts
        prompt = self._build_extraction_prompt(text, schema)
        system_prompt = self._build_system_prompt(schema)
        
        # Generate structured extraction
        result = await self.client.generate(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
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
                "required_fields": schema.get_required_fields()
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
                "parse_error": True
            }
    
    async def process_text(
        self,
        text: str,
        model: LLMModel | str = None,
        document_type: str = None
    ) -> Dict[str, Any]:
        """
        Process OCR text using LLM with context-aware extraction.
        
        This is the main entry point that uses document schemas for structured extraction.
        """
        # Use structured extraction for all document types
        return await self.extract_structured(text, document_type or "unknown", model)
    
    async def denoise_text(
        self,
        text: str,
        model: LLMModel = None
    ) -> str:
        """
        Quick denoising of OCR text without structured extraction.
        """
        model = model or self.client.config.default_model
        
        prompt = f"""Remove noise and artifacts from this OCR text. Fix any obvious errors but preserve the original meaning:

{text}

Cleaned text:"""

        result = await self.client.generate(
            prompt=prompt,
            model=model,
            temperature=0.1
        )
        
        return result.strip() if result else text
    
    def process_text_sync(
        self,
        text: str,
        model: LLMModel = None,
        document_type: str = None
    ) -> Dict[str, Any]:
        """Synchronous wrapper for process_text."""
        return asyncio.run(self.process_text(text, model, document_type))


# =============================================================================
# GEMINI CLIENT
# =============================================================================

class GeminiClient:
    """Client for Google Gemini API."""
    
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
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Optional[str]:
        """Generate text using Gemini API."""
        if not self.available:
            print("[Gemini] API key not configured")
            return None
        
        print(f"[Gemini] Generating with {self.GEMINI_MODEL}...")
        print(f"[Gemini] Prompt length: {len(prompt)} chars")
        
        # Build the request
        url = f"{self.GEMINI_API_URL}/{self.GEMINI_MODEL}:generateContent?key={self.api_key}"
        
        # Combine system prompt and user prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        payload = {
            "contents": [{
                "parts": [{"text": full_prompt}]
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
    """LLM post-processor using Gemini API."""
    
    BASE_SYSTEM_PROMPT = LLMPostProcessor.BASE_SYSTEM_PROMPT
    
    def __init__(self, client: GeminiClient = None):
        self.client = client or GeminiClient()
    
    def _build_extraction_prompt(self, text: str, schema) -> str:
        """Build extraction prompt (same as Ollama version)."""
        # Build detailed field descriptions with types
        field_descriptions = []
        for field in schema.fields:
            req = " (REQUIRED)" if field.required else " (optional)"
            field_type = f"[{field.field_type.value}]"
            field_descriptions.append(f"  - {field.name} {field_type}{req}: {field.description}")
        
        fields_text = "\n".join(field_descriptions)
        
        # Build example JSON structure based on actual schema fields
        example_fields = {}
        for field in schema.fields:
            if field.field_type.value == "list":
                example_fields[field.name] = [{"name": "...", "details": "..."}]
            elif field.field_type.value == "date":
                example_fields[field.name] = "YYYY-MM-DD or null"
            elif field.field_type.value == "currency":
                example_fields[field.name] = "amount as string or null"
            elif field.field_type.value == "number":
                example_fields[field.name] = "number or null"
            else:
                example_fields[field.name] = "extracted value or null"
        
        example_json = json.dumps(example_fields, indent=4)
        
        prompt = f"""=== DOCUMENT CLASSIFICATION ===
This document has been classified as: **{schema.display_name.upper()}** (type_id: {schema.type_id})

=== DOCUMENT TEMPLATE SCHEMA ===
The following schema defines exactly what fields must be extracted from this {schema.display_name}:

Document Type: {schema.display_name}
Type ID: {schema.type_id}

Fields to Extract (* = required):
{fields_text}

=== EXTRACTION CONTEXT ===
{schema.llm_context}

=== RAW OCR TEXT (may contain errors) ===
{text}
=== END OCR TEXT ===

=== YOUR TASK ===
1. This is CONFIRMED to be a {schema.display_name} document
2. Extract ALL fields defined in the schema above
3. Fix OCR errors in the text (character confusion, spacing issues)
4. Return structured data matching the schema exactly

Return ONLY this JSON (no other text):
{{
  "corrected_text": "The full OCR text with errors corrected",
  "extracted_fields": {example_json}
}}"""
        return prompt
    
    def _build_system_prompt(self, schema) -> str:
        """Build system prompt with document context."""
        return f"""{self.BASE_SYSTEM_PROMPT}

DOCUMENT CONTEXT:
{schema.llm_context}"""
    
    def _parse_json_response(self, response: str, schema) -> Dict[str, Any]:
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
        document_type: str
    ) -> Dict[str, Any]:
        """Extract structured data using Gemini."""
        schema = get_schema(document_type)
        
        print(f"[Gemini] Extracting structured data for document type: {document_type}")
        print(f"[Gemini] Using schema: {schema.display_name} ({len(schema.fields)} fields)")
        
        prompt = self._build_extraction_prompt(text, schema)
        system_prompt = self._build_system_prompt(schema)
        
        result = await self.client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
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
                "required_fields": schema.get_required_fields()
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
                "parse_error": True
            }
    
    async def process_text(
        self,
        text: str,
        document_type: str = None
    ) -> Dict[str, Any]:
        """Process OCR text using Gemini."""
        return await self.extract_structured(text, document_type or "unknown")


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

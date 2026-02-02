"""
LLM Service for OCR Post-Processing.

Provides integration with local LLMs via Ollama for:
- Text normalization and denoising
- Medicine name recognition and correction
- Improved text grouping and structure
- OCR artifact removal
"""

import asyncio
import httpx
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from enum import Enum


class LLMModel(Enum):
    """Available LLM models for post-processing."""
    QWEN_2_5 = "qwen2.5:7b"
    LLAMA_MEDITRON = "meditron:7b"
    MEDGEMMA = "medgemma:latest"
    GPT_OSS = "gpt-oss-20b:latest"
    
    @classmethod
    def from_string(cls, model_name: str) -> "LLMModel":
        """Get model enum from string name."""
        mapping = {
            "qwen2.5": cls.QWEN_2_5,
            "qwen": cls.QWEN_2_5,
            "meditron": cls.LLAMA_MEDITRON,
            "llama-meditron": cls.LLAMA_MEDITRON,
            "medgemma": cls.MEDGEMMA,
            "gpt-oss": cls.GPT_OSS,
            "gpt-oss-20b": cls.GPT_OSS,
        }
        return mapping.get(model_name.lower(), cls.QWEN_2_5)
    
    @property
    def display_name(self) -> str:
        """Human-readable model name."""
        names = {
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
    default_model: LLMModel = LLMModel.QWEN_2_5
    timeout: float = 120.0  # LLM inference can be slow
    max_tokens: int = 2048
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
        model: LLMModel = None,
        system_prompt: str = None,
        temperature: float = None,
        max_tokens: int = None
    ) -> Optional[str]:
        """Generate text using the specified model."""
        model = model or self.config.default_model
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens
        
        payload = {
            "model": model.value,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get("response", "")
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
    - Text normalization and denoising
    - Medicine name recognition and correction
    - Improved text grouping
    - OCR artifact removal
    """
    
    SYSTEM_PROMPT = """You are an expert OCR post-processor. Your task is to clean and improve text that was extracted from documents using OCR (Optical Character Recognition).

Common OCR errors you should fix:
- Character confusion: 0/O, 1/l/I, 5/S, 8/B, rn/m, cl/d
- Missing or extra spaces
- Wrong punctuation
- Garbled text or artifacts
- Split words that should be joined

For medical documents:
- Correct medication names to their proper spelling
- Normalize dosage formats (e.g., "500mg" not "5OOmg")
- Preserve important medical terminology

Output ONLY the corrected text. Do not add explanations or commentary."""

    MEDICAL_SYSTEM_PROMPT = """You are a medical document specialist. Your task is to:
1. Correct OCR errors in medical text
2. Properly identify and spell medication names
3. Normalize dosage information
4. Preserve medical terminology and abbreviations
5. Group related information logically

Common medications to watch for:
- Antibiotics: Amoxicillin, Azithromycin, Ciprofloxacin, etc.
- Pain relievers: Ibuprofen, Acetaminophen, Naproxen, etc.
- Cardiovascular: Lisinopril, Metoprolol, Amlodipine, etc.
- Diabetes: Metformin, Insulin, Glipizide, etc.

Output the corrected and normalized text. Group related information together."""

    def __init__(self, client: OllamaClient = None):
        self.client = client or OllamaClient()
    
    async def process_text(
        self,
        text: str,
        model: LLMModel = None,
        document_type: str = None
    ) -> Dict[str, Any]:
        """
        Process OCR text using LLM for improvement.
        
        Args:
            text: Raw OCR text to process
            model: LLM model to use
            document_type: Type of document (e.g., "prescription", "receipt")
            
        Returns:
            Dict with processed text and metadata
        """
        model = model or self.client.config.default_model
        
        # Select appropriate system prompt
        if document_type in ["prescription", "medical", "medication"]:
            system_prompt = self.MEDICAL_SYSTEM_PROMPT
        else:
            system_prompt = self.SYSTEM_PROMPT
        
        # Build the prompt
        prompt = f"""Please clean and improve the following OCR-extracted text:

---
{text}
---

Provide the corrected text:"""

        # Generate improved text
        result = await self.client.generate(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
            temperature=0.2  # Low temperature for consistent output
        )
        
        if result:
            return {
                "success": True,
                "original_text": text,
                "processed_text": result.strip(),
                "model": model.display_name,
                "document_type": document_type
            }
        else:
            return {
                "success": False,
                "original_text": text,
                "processed_text": None,
                "error": "LLM generation failed",
                "model": model.display_name
            }
    
    async def normalize_medical_text(
        self,
        text: str,
        model: LLMModel = None
    ) -> Dict[str, Any]:
        """
        Specialized processing for medical documents.
        
        Focuses on:
        - Medication name correction
        - Dosage normalization
        - Medical terminology preservation
        """
        model = model or LLMModel.LLAMA_MEDITRON  # Prefer medical model
        
        prompt = f"""Analyze and correct this medical document text:

---
{text}
---

Please:
1. Correct any OCR errors in medication names
2. Normalize dosage formats (e.g., "500 mg" format)
3. Identify and list any medications found
4. Provide the cleaned text

Format your response as:
MEDICATIONS FOUND:
- [list medications with dosages]

CORRECTED TEXT:
[the cleaned text]"""

        result = await self.client.generate(
            prompt=prompt,
            model=model,
            system_prompt=self.MEDICAL_SYSTEM_PROMPT,
            temperature=0.1
        )
        
        if result:
            # Parse the structured response
            medications = []
            corrected_text = result
            
            if "MEDICATIONS FOUND:" in result:
                parts = result.split("CORRECTED TEXT:")
                if len(parts) == 2:
                    med_section = parts[0].replace("MEDICATIONS FOUND:", "").strip()
                    corrected_text = parts[1].strip()
                    
                    # Extract medication list
                    for line in med_section.split("\n"):
                        line = line.strip()
                        if line.startswith("-"):
                            medications.append(line[1:].strip())
            
            return {
                "success": True,
                "original_text": text,
                "processed_text": corrected_text,
                "medications": medications,
                "model": model.display_name
            }
        else:
            return {
                "success": False,
                "original_text": text,
                "error": "Medical text processing failed"
            }
    
    async def denoise_text(
        self,
        text: str,
        model: LLMModel = None
    ) -> str:
        """
        Remove OCR artifacts and noise from text.
        
        Focuses on:
        - Removing garbage characters
        - Fixing broken words
        - Correcting punctuation
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


# Singleton instances
_ollama_client: Optional[OllamaClient] = None
_llm_processor: Optional[LLMPostProcessor] = None


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

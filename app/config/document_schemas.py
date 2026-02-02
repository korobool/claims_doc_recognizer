"""
Document Type Schemas Configuration.

Loads document schemas from YAML files for easy editing and extension.
Each schema defines:
- CLIP classification prompts
- LLM extraction prompts (context-aware)
- Expected/essential fields for structured extraction
"""

import os
import yaml
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


# Directory containing schema YAML files
SCHEMAS_DIR = Path(__file__).parent / "schemas"


class FieldType(Enum):
    """Types of fields that can be extracted."""
    TEXT = "text"
    DATE = "date"
    CURRENCY = "currency"
    NUMBER = "number"
    LIST = "list"
    MEDICATION = "medication"
    PERSON_NAME = "person_name"
    ADDRESS = "address"
    PHONE = "phone"
    EMAIL = "email"
    BOOLEAN = "boolean"
    
    @classmethod
    def from_string(cls, value: str) -> "FieldType":
        """Convert string to FieldType, defaulting to TEXT."""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.TEXT


@dataclass
class FieldSchema:
    """Schema for a single extractable field."""
    name: str
    field_type: FieldType
    description: str
    required: bool = False
    aliases: List[str] = field(default_factory=list)
    default: Any = None
    list_item_schema: Optional[Dict[str, str]] = None
    
    def to_prompt_description(self) -> str:
        """Generate prompt description for this field."""
        req = "(required)" if self.required else "(optional)"
        desc = f"- {self.name} {req}: {self.description}"
        if self.list_item_schema:
            desc += f"\n  List item fields: {list(self.list_item_schema.keys())}"
        return desc
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FieldSchema":
        """Create FieldSchema from dictionary (YAML data)."""
        return cls(
            name=data.get("name", ""),
            field_type=FieldType.from_string(data.get("type", "text")),
            description=data.get("description", ""),
            required=data.get("required", False),
            aliases=data.get("aliases", []),
            default=data.get("default"),
            list_item_schema=data.get("list_item_schema")
        )


@dataclass
class DocumentSchema:
    """Complete schema for a document type."""
    type_id: str
    display_name: str
    clip_prompts: List[str]
    llm_context: str
    fields: List[FieldSchema]
    keywords: List[str] = field(default_factory=list)
    source_file: Optional[str] = None
    
    def get_fields_prompt(self) -> str:
        """Generate prompt section describing expected fields."""
        lines = ["Expected fields to extract:"]
        for f in self.fields:
            lines.append(f.to_prompt_description())
        return "\n".join(lines)
    
    def get_required_fields(self) -> List[str]:
        """Get list of required field names."""
        return [f.name for f in self.fields if f.required]
    
    def get_default_extraction(self) -> Dict[str, Any]:
        """Get default extraction result with all fields."""
        return {f.name: f.default for f in self.fields}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], source_file: str = None) -> "DocumentSchema":
        """Create DocumentSchema from dictionary (YAML data)."""
        fields = [FieldSchema.from_dict(f) for f in data.get("fields", [])]
        return cls(
            type_id=data.get("type_id", "unknown"),
            display_name=data.get("display_name", "Unknown Document"),
            clip_prompts=data.get("clip_prompts", []),
            llm_context=data.get("llm_context", ""),
            fields=fields,
            keywords=data.get("keywords", []),
            source_file=source_file
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary for YAML serialization."""
        return {
            "type_id": self.type_id,
            "display_name": self.display_name,
            "clip_prompts": self.clip_prompts,
            "keywords": self.keywords,
            "llm_context": self.llm_context,
            "fields": [
                {
                    "name": f.name,
                    "type": f.field_type.value,
                    "description": f.description,
                    "required": f.required,
                    **({"list_item_schema": f.list_item_schema} if f.list_item_schema else {})
                }
                for f in self.fields
            ]
        }


# =============================================================================
# SCHEMA LOADING FROM YAML FILES
# =============================================================================

# Cache for loaded schemas
_schemas_cache: Optional[Dict[str, DocumentSchema]] = None
_schemas_load_time: float = 0


def _load_schema_from_yaml(filepath: Path) -> Optional[DocumentSchema]:
    """Load a single schema from a YAML file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            if data:
                return DocumentSchema.from_dict(data, source_file=str(filepath))
    except Exception as e:
        print(f"[Schema] Error loading {filepath}: {e}")
    return None


def _load_all_schemas() -> Dict[str, DocumentSchema]:
    """Load all schemas from the schemas directory."""
    schemas = {}
    
    if not SCHEMAS_DIR.exists():
        print(f"[Schema] Creating schemas directory: {SCHEMAS_DIR}")
        SCHEMAS_DIR.mkdir(parents=True, exist_ok=True)
        return schemas
    
    for yaml_file in SCHEMAS_DIR.glob("*.yaml"):
        schema = _load_schema_from_yaml(yaml_file)
        if schema:
            schemas[schema.type_id] = schema
            print(f"[Schema] Loaded: {schema.type_id} ({schema.display_name})")
    
    # Ensure we always have an 'unknown' fallback
    if "unknown" not in schemas:
        schemas["unknown"] = DocumentSchema(
            type_id="unknown",
            display_name="General Document",
            clip_prompts=["a document with text"],
            llm_context="Process this document and extract any relevant information.",
            fields=[
                FieldSchema("summary", FieldType.TEXT, "Brief summary of document content")
            ]
        )
    
    return schemas


def get_schemas() -> Dict[str, DocumentSchema]:
    """Get all loaded schemas, loading from YAML if needed."""
    global _schemas_cache, _schemas_load_time
    
    # Check if we need to reload (first load or files changed)
    if _schemas_cache is None:
        _schemas_cache = _load_all_schemas()
        _schemas_load_time = os.path.getmtime(SCHEMAS_DIR) if SCHEMAS_DIR.exists() else 0
    
    return _schemas_cache


def reload_schemas() -> Dict[str, DocumentSchema]:
    """Force reload all schemas from YAML files."""
    global _schemas_cache
    _schemas_cache = None
    return get_schemas()


def get_schema(type_id: str) -> DocumentSchema:
    """Get document schema by type ID, falling back to unknown."""
    schemas = get_schemas()
    return schemas.get(type_id, schemas.get("unknown"))


def get_all_schemas() -> Dict[str, DocumentSchema]:
    """Get all registered document schemas."""
    return get_schemas()


def get_schema_for_classification(classification: dict) -> DocumentSchema:
    """Get schema based on classification result."""
    schemas = get_schemas()
    type_id = classification.get("type_id", "unknown")
    
    if type_id in schemas:
        return schemas[type_id]
    
    # Try to match by class name
    class_name = classification.get("class", "").lower()
    for schema_id, schema in schemas.items():
        if schema_id in class_name or class_name in schema.display_name.lower():
            return schema
    
    return get_schema("unknown")


def save_schema(schema: DocumentSchema, filename: str = None) -> str:
    """Save a schema to a YAML file."""
    if filename is None:
        filename = f"{schema.type_id}.yaml"
    
    filepath = SCHEMAS_DIR / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(schema.to_dict(), f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    # Invalidate cache
    global _schemas_cache
    _schemas_cache = None
    
    return str(filepath)


def list_schema_files() -> List[Dict[str, str]]:
    """List all schema files with their type IDs."""
    result = []
    if SCHEMAS_DIR.exists():
        for yaml_file in SCHEMAS_DIR.glob("*.yaml"):
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    result.append({
                        "filename": yaml_file.name,
                        "type_id": data.get("type_id", "unknown"),
                        "display_name": data.get("display_name", "Unknown"),
                        "path": str(yaml_file)
                    })
            except Exception:
                pass
    return result

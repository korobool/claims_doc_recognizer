"""
Document Type Configuration for Context-Aware Recognition.

Defines document types with CLIP prompts, expected fields, and post-processing rules.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable


@dataclass
class FieldDefinition:
    """Definition of an expected field in a document."""
    name: str
    display_name: str
    pattern: Optional[str] = None  # Regex pattern
    keywords: List[str] = field(default_factory=list)
    location: str = "any"  # top, middle, bottom, any
    required: bool = False
    data_type: str = "text"  # text, currency, date, number


@dataclass 
class DocumentType:
    """Configuration for a document type."""
    type_id: str
    display_name: str
    clip_prompt: str
    keywords_en: List[str]
    keywords_ru: List[str] = field(default_factory=list)
    expected_fields: List[FieldDefinition] = field(default_factory=list)
    # OCR corrections specific to this document type
    apply_currency_fix: bool = False
    apply_date_fix: bool = False
    apply_numeric_fix: bool = False


# Character corrections for numeric contexts
NUMERIC_CHAR_FIXES = {
    'O': '0', 'o': '0',
    'l': '1', 'I': '1', '|': '1',
    'S': '5', 's': '5',
    'B': '8',
    'Z': '2', 'z': '2',
    'G': '6',
}


# =============================================================================
# DOCUMENT TYPE DEFINITIONS
# =============================================================================

DOCUMENT_TYPES: Dict[str, DocumentType] = {
    "receipt": DocumentType(
        type_id="receipt",
        display_name="Receipt",
        clip_prompt="a photo of a receipt or invoice with prices and totals",
        keywords_en=[
            "receipt", "total", "subtotal", "tax", "payment", "cash", "change",
            "qty", "price", "amount", "item", "sale", "visa", "mastercard",
            "credit", "debit", "thank you", "store", "shop"
        ],
        keywords_ru=[
            "чек", "итого", "сумма", "оплата", "касса", "товар", "цена",
            "ндс", "скидка", "наличные", "карта", "сдача"
        ],
        expected_fields=[
            FieldDefinition("store_name", "Store Name", location="top", required=True),
            FieldDefinition("date", "Date", pattern=r"\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}", 
                          location="top", data_type="date"),
            FieldDefinition("time", "Time", pattern=r"\d{1,2}:\d{2}(?::\d{2})?",
                          location="top"),
            FieldDefinition("subtotal", "Subtotal", 
                          pattern=r"(?:sub\s*total)[:\s]*[\$€£]?\s*[\d,]+[.,]\d{2}",
                          keywords=["subtotal", "sub total"], location="bottom", data_type="currency"),
            FieldDefinition("tax", "Tax",
                          pattern=r"(?:tax|vat|ндс)[:\s]*[\$€£]?\s*[\d,]+[.,]\d{2}",
                          keywords=["tax", "vat", "ндс"], location="bottom", data_type="currency"),
            FieldDefinition("total", "Total",
                          pattern=r"(?:total|итого)[:\s]*[\$€£]?\s*[\d,]+[.,]\d{2}",
                          keywords=["total", "итого", "amount due"], location="bottom", 
                          required=True, data_type="currency"),
        ],
        apply_currency_fix=True,
        apply_date_fix=True,
        apply_numeric_fix=True,
    ),
    
    "prescription": DocumentType(
        type_id="prescription",
        display_name="Medication Prescription",
        clip_prompt="a photo of a medical prescription or medication document",
        keywords_en=[
            "prescription", "rx", "medication", "dose", "dosage", "tablet", "capsule",
            "mg", "ml", "take", "daily", "doctor", "patient", "pharmacy", "refill",
            "sig", "disp", "qty"
        ],
        keywords_ru=[
            "рецепт", "препарат", "доза", "таблетка", "капсула", "принимать",
            "врач", "пациент", "аптека", "лекарство", "мг", "мл"
        ],
        expected_fields=[
            FieldDefinition("patient_name", "Patient Name", location="top", required=True),
            FieldDefinition("doctor_name", "Doctor Name", location="top"),
            FieldDefinition("date", "Date", pattern=r"\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}",
                          location="top", data_type="date"),
            FieldDefinition("medication", "Medication", location="middle", required=True),
            FieldDefinition("dosage", "Dosage", pattern=r"\d+\s*(?:mg|ml|mcg|g|мг|мл)",
                          location="middle", required=True),
            FieldDefinition("instructions", "Instructions", 
                          keywords=["take", "принимать", "sig"], location="middle"),
            FieldDefinition("refills", "Refills", pattern=r"refill[s]?[:\s]*\d+",
                          location="bottom"),
        ],
        apply_date_fix=True,
        apply_numeric_fix=True,
    ),
    
    "form": DocumentType(
        type_id="form",
        display_name="Form",
        clip_prompt="a photo of a form or application document with fields to fill",
        keywords_en=[
            "form", "application", "name", "date", "signature", "address", "phone",
            "email", "please fill", "required", "checkbox", "field", "enter"
        ],
        keywords_ru=[
            "заявление", "форма", "анкета", "фио", "дата", "подпись", "адрес",
            "телефон", "заполните", "поле", "обязательно"
        ],
        expected_fields=[
            FieldDefinition("form_title", "Form Title", location="top"),
            FieldDefinition("name", "Name", keywords=["name", "фио", "имя"], location="top"),
            FieldDefinition("date", "Date", pattern=r"\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}",
                          data_type="date"),
            FieldDefinition("address", "Address", keywords=["address", "адрес"]),
            FieldDefinition("phone", "Phone", pattern=r"[\+]?[\d\s\-\(\)]{7,}",
                          keywords=["phone", "tel", "телефон"]),
            FieldDefinition("email", "Email", pattern=r"[\w\.-]+@[\w\.-]+\.\w+",
                          keywords=["email", "e-mail"]),
            FieldDefinition("signature", "Signature", keywords=["signature", "подпись"],
                          location="bottom"),
        ],
        apply_date_fix=True,
    ),
    
    "contract": DocumentType(
        type_id="contract",
        display_name="Contract",
        clip_prompt="a photo of a contract or legal agreement document",
        keywords_en=[
            "contract", "agreement", "party", "parties", "terms", "conditions",
            "hereby", "whereas", "witness", "signed", "effective date", "obligations",
            "clause", "article", "section"
        ],
        keywords_ru=[
            "договор", "контракт", "соглашение", "сторона", "стороны", "условия",
            "обязательства", "подписано", "дата", "пункт", "статья"
        ],
        expected_fields=[
            FieldDefinition("title", "Title", location="top"),
            FieldDefinition("contract_number", "Contract Number", 
                          pattern=r"(?:№|#|no\.?|number)[:\s]*[\w\-]+",
                          location="top"),
            FieldDefinition("date", "Date", pattern=r"\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}",
                          location="top", data_type="date"),
            FieldDefinition("party_1", "Party 1", keywords=["party", "сторона"], location="top"),
            FieldDefinition("party_2", "Party 2", location="top"),
            FieldDefinition("effective_date", "Effective Date", 
                          pattern=r"effective\s+(?:date)?[:\s]*\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}",
                          data_type="date"),
            FieldDefinition("signatures", "Signatures", location="bottom"),
        ],
        apply_date_fix=True,
    ),
    
    "unknown": DocumentType(
        type_id="unknown",
        display_name="Unknown Document",
        clip_prompt="a photo of an unknown or unclassified document",
        keywords_en=[],
        keywords_ru=[],
        expected_fields=[],
    ),
}


def get_document_type(type_id: str) -> DocumentType:
    """Get document type configuration by ID."""
    return DOCUMENT_TYPES.get(type_id, DOCUMENT_TYPES["unknown"])


def get_clip_prompts() -> List[str]:
    """Get list of CLIP prompts for all document types."""
    return [dt.clip_prompt for dt in DOCUMENT_TYPES.values()]


def get_type_names() -> List[str]:
    """Get list of document type display names."""
    return [dt.display_name for dt in DOCUMENT_TYPES.values()]


def get_type_ids() -> List[str]:
    """Get list of document type IDs."""
    return list(DOCUMENT_TYPES.keys())

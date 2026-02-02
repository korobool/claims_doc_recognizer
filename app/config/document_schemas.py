"""
Document Type Schemas Configuration.

Defines document classes with:
- CLIP classification prompts
- LLM extraction prompts (context-aware)
- Expected/essential fields for structured extraction
- Field validation rules
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


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


@dataclass
class FieldSchema:
    """Schema for a single extractable field."""
    name: str
    field_type: FieldType
    description: str
    required: bool = False
    aliases: List[str] = field(default_factory=list)
    default: Any = None
    
    def to_prompt_description(self) -> str:
        """Generate prompt description for this field."""
        req = "(required)" if self.required else "(optional)"
        return f"- {self.name} {req}: {self.description}"


@dataclass
class DocumentSchema:
    """Complete schema for a document type."""
    type_id: str
    display_name: str
    clip_prompts: List[str]
    llm_context: str
    fields: List[FieldSchema]
    keywords: List[str] = field(default_factory=list)
    
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


# =============================================================================
# DOCUMENT SCHEMAS REGISTRY
# =============================================================================

DOCUMENT_SCHEMAS: Dict[str, DocumentSchema] = {
    
    # -------------------------------------------------------------------------
    # PRESCRIPTION / MEDICAL DOCUMENTS
    # -------------------------------------------------------------------------
    "prescription": DocumentSchema(
        type_id="prescription",
        display_name="Medical Prescription",
        clip_prompts=[
            "a medical prescription document",
            "a doctor's prescription with medication names",
            "a pharmacy prescription form",
            "a handwritten or printed medical prescription"
        ],
        llm_context="""You are processing a MEDICAL PRESCRIPTION document.
This is sensitive medical data requiring accurate extraction.

Key characteristics of prescriptions:
- Contains patient name and sometimes date of birth
- Has prescriber/doctor information
- Lists medications with dosages, quantities, and instructions
- May include pharmacy information
- Often has Rx symbol or prescription number

Common OCR errors in medical documents:
- Medication names are often misspelled (e.g., "Amoxici11in" → "Amoxicillin")
- Dosages may have character confusion (e.g., "5OOmg" → "500mg", "1OO" → "100")
- "mg" vs "mcg" confusion is critical - preserve carefully
- Roman numerals in quantities (e.g., "ii" = 2, "iii" = 3)
- Latin abbreviations: "bid" (twice daily), "tid" (three times daily), "qid" (four times daily)
- "PRN" means "as needed"
""",
        fields=[
            FieldSchema("patient_name", FieldType.PERSON_NAME, "Full name of the patient", required=True),
            FieldSchema("patient_dob", FieldType.DATE, "Patient's date of birth"),
            FieldSchema("prescriber_name", FieldType.PERSON_NAME, "Doctor or prescriber name", required=True),
            FieldSchema("prescriber_license", FieldType.TEXT, "Medical license or NPI number"),
            FieldSchema("prescription_date", FieldType.DATE, "Date the prescription was written", required=True),
            FieldSchema("medications", FieldType.LIST, "List of prescribed medications with dosage and instructions", required=True),
            FieldSchema("pharmacy_name", FieldType.TEXT, "Pharmacy name if specified"),
            FieldSchema("refills", FieldType.NUMBER, "Number of refills allowed"),
            FieldSchema("diagnosis", FieldType.TEXT, "Diagnosis or condition if mentioned"),
        ],
        keywords=["rx", "prescription", "medication", "dosage", "refill", "pharmacy", "mg", "tablet", "capsule", "doctor", "md", "npi"]
    ),
    
    # -------------------------------------------------------------------------
    # RECEIPT / INVOICE
    # -------------------------------------------------------------------------
    "receipt": DocumentSchema(
        type_id="receipt",
        display_name="Receipt / Invoice",
        clip_prompts=[
            "a store receipt or sales receipt",
            "a purchase receipt with items and prices",
            "a retail transaction receipt",
            "an invoice or bill"
        ],
        llm_context="""You are processing a RECEIPT or INVOICE document.

Key characteristics:
- Contains merchant/store name and address
- Lists purchased items with quantities and prices
- Shows subtotal, tax, and total amounts
- May have payment method and transaction details
- Often includes date and time of purchase

Common OCR errors in receipts:
- Dollar amounts: "$" may be read as "S" or missing
- Decimal points may be confused with commas
- "0" and "O" confusion in prices
- Item names may be abbreviated or truncated
- Tax rates and percentages need careful extraction
""",
        fields=[
            FieldSchema("merchant_name", FieldType.TEXT, "Store or merchant name", required=True),
            FieldSchema("merchant_address", FieldType.ADDRESS, "Store address"),
            FieldSchema("transaction_date", FieldType.DATE, "Date of purchase", required=True),
            FieldSchema("transaction_time", FieldType.TEXT, "Time of purchase"),
            FieldSchema("items", FieldType.LIST, "List of purchased items with quantities and prices", required=True),
            FieldSchema("subtotal", FieldType.CURRENCY, "Subtotal before tax"),
            FieldSchema("tax_amount", FieldType.CURRENCY, "Tax amount"),
            FieldSchema("total", FieldType.CURRENCY, "Total amount paid", required=True),
            FieldSchema("payment_method", FieldType.TEXT, "Payment method (cash, card, etc.)"),
            FieldSchema("transaction_id", FieldType.TEXT, "Transaction or receipt number"),
        ],
        keywords=["total", "subtotal", "tax", "receipt", "invoice", "qty", "price", "amount", "paid", "change", "cash", "card"]
    ),
    
    # -------------------------------------------------------------------------
    # MEDICAL FORM / HEALTH RECORD
    # -------------------------------------------------------------------------
    "medical_form": DocumentSchema(
        type_id="medical_form",
        display_name="Medical Form",
        clip_prompts=[
            "a medical form or health questionnaire",
            "a patient intake form",
            "a medical history form",
            "a health insurance claim form"
        ],
        llm_context="""You are processing a MEDICAL FORM document.
This contains sensitive patient health information.

Key characteristics:
- Patient demographic information
- Medical history sections
- Checkboxes and form fields
- May include insurance information
- Often has signature lines and dates

Common OCR errors:
- Checkbox states may not be clear (checked vs unchecked)
- Handwritten entries mixed with printed text
- Medical terminology and abbreviations
- ICD codes and procedure codes
""",
        fields=[
            FieldSchema("patient_name", FieldType.PERSON_NAME, "Patient's full name", required=True),
            FieldSchema("patient_dob", FieldType.DATE, "Patient's date of birth", required=True),
            FieldSchema("patient_address", FieldType.ADDRESS, "Patient's address"),
            FieldSchema("patient_phone", FieldType.PHONE, "Patient's phone number"),
            FieldSchema("insurance_provider", FieldType.TEXT, "Insurance company name"),
            FieldSchema("insurance_id", FieldType.TEXT, "Insurance policy/member ID"),
            FieldSchema("primary_complaint", FieldType.TEXT, "Chief complaint or reason for visit"),
            FieldSchema("allergies", FieldType.LIST, "Known allergies"),
            FieldSchema("current_medications", FieldType.LIST, "Current medications"),
            FieldSchema("medical_history", FieldType.TEXT, "Relevant medical history"),
            FieldSchema("form_date", FieldType.DATE, "Date form was completed"),
        ],
        keywords=["patient", "medical", "history", "allergies", "insurance", "dob", "date of birth", "emergency contact", "physician"]
    ),
    
    # -------------------------------------------------------------------------
    # CONTRACT / LEGAL DOCUMENT
    # -------------------------------------------------------------------------
    "contract": DocumentSchema(
        type_id="contract",
        display_name="Contract / Legal Document",
        clip_prompts=[
            "a legal contract or agreement",
            "a signed contract document",
            "a business agreement or terms",
            "a legal document with signatures"
        ],
        llm_context="""You are processing a CONTRACT or LEGAL DOCUMENT.

Key characteristics:
- Formal legal language and structure
- Party names and identifying information
- Terms and conditions sections
- Signature blocks with dates
- May have notarization or witness sections

Common OCR errors:
- Legal terminology may be unfamiliar
- Section numbers and references
- Party names must be exact
- Dates are critical for validity
""",
        fields=[
            FieldSchema("document_title", FieldType.TEXT, "Title or type of contract", required=True),
            FieldSchema("parties", FieldType.LIST, "Names of all parties to the contract", required=True),
            FieldSchema("effective_date", FieldType.DATE, "Date contract becomes effective", required=True),
            FieldSchema("expiration_date", FieldType.DATE, "Date contract expires"),
            FieldSchema("contract_value", FieldType.CURRENCY, "Monetary value if applicable"),
            FieldSchema("key_terms", FieldType.TEXT, "Summary of key terms"),
            FieldSchema("signatures", FieldType.LIST, "Names of signatories"),
            FieldSchema("signature_dates", FieldType.LIST, "Dates of signatures"),
        ],
        keywords=["agreement", "contract", "party", "parties", "hereby", "whereas", "terms", "conditions", "signature", "witness", "notary"]
    ),
    
    # -------------------------------------------------------------------------
    # INSURANCE CLAIM
    # -------------------------------------------------------------------------
    "insurance_claim": DocumentSchema(
        type_id="insurance_claim",
        display_name="Insurance Claim",
        clip_prompts=[
            "an insurance claim form",
            "a health insurance claim",
            "a medical billing claim form",
            "a CMS-1500 or UB-04 form"
        ],
        llm_context="""You are processing an INSURANCE CLAIM document.
This is a medical billing document requiring precise extraction.

Key characteristics:
- Patient and insured party information
- Provider/facility information
- Diagnosis codes (ICD-10)
- Procedure codes (CPT/HCPCS)
- Dates of service
- Charges and amounts

Common OCR errors:
- Diagnosis codes are alphanumeric (e.g., "J06.9", "M54.5")
- CPT codes are 5 digits
- NPI numbers are 10 digits
- Dollar amounts need decimal precision
""",
        fields=[
            FieldSchema("patient_name", FieldType.PERSON_NAME, "Patient's name", required=True),
            FieldSchema("patient_id", FieldType.TEXT, "Patient account number"),
            FieldSchema("insured_name", FieldType.PERSON_NAME, "Name of insured (if different)"),
            FieldSchema("insurance_id", FieldType.TEXT, "Insurance member ID", required=True),
            FieldSchema("group_number", FieldType.TEXT, "Insurance group number"),
            FieldSchema("provider_name", FieldType.TEXT, "Healthcare provider name", required=True),
            FieldSchema("provider_npi", FieldType.TEXT, "Provider NPI number"),
            FieldSchema("service_date", FieldType.DATE, "Date of service", required=True),
            FieldSchema("diagnosis_codes", FieldType.LIST, "ICD-10 diagnosis codes", required=True),
            FieldSchema("procedure_codes", FieldType.LIST, "CPT/HCPCS procedure codes"),
            FieldSchema("total_charges", FieldType.CURRENCY, "Total charges", required=True),
            FieldSchema("amount_paid", FieldType.CURRENCY, "Amount already paid"),
            FieldSchema("amount_due", FieldType.CURRENCY, "Amount due from patient"),
        ],
        keywords=["claim", "insurance", "diagnosis", "procedure", "cpt", "icd", "npi", "provider", "patient", "charges", "billing"]
    ),
    
    # -------------------------------------------------------------------------
    # LAB RESULTS
    # -------------------------------------------------------------------------
    "lab_results": DocumentSchema(
        type_id="lab_results",
        display_name="Laboratory Results",
        clip_prompts=[
            "a laboratory test results document",
            "a blood test or lab report",
            "medical laboratory results",
            "a diagnostic test report"
        ],
        llm_context="""You are processing LABORATORY RESULTS.
This contains critical medical test data requiring precise extraction.

Key characteristics:
- Patient identification
- Test names and codes
- Result values with units
- Reference ranges (normal values)
- Flags for abnormal results (H=High, L=Low)
- Collection and report dates

Common OCR errors:
- Numeric values are critical - verify decimal places
- Units must be preserved exactly (mg/dL, mmol/L, etc.)
- Reference ranges often in parentheses
- "H" and "L" flags for abnormal values
""",
        fields=[
            FieldSchema("patient_name", FieldType.PERSON_NAME, "Patient's name", required=True),
            FieldSchema("patient_dob", FieldType.DATE, "Patient's date of birth"),
            FieldSchema("collection_date", FieldType.DATE, "Date specimen collected", required=True),
            FieldSchema("report_date", FieldType.DATE, "Date results reported"),
            FieldSchema("ordering_physician", FieldType.PERSON_NAME, "Ordering doctor"),
            FieldSchema("lab_name", FieldType.TEXT, "Laboratory name"),
            FieldSchema("test_results", FieldType.LIST, "List of tests with values, units, and reference ranges", required=True),
            FieldSchema("abnormal_flags", FieldType.LIST, "Tests flagged as abnormal"),
            FieldSchema("specimen_type", FieldType.TEXT, "Type of specimen (blood, urine, etc.)"),
        ],
        keywords=["lab", "laboratory", "test", "result", "reference", "range", "specimen", "blood", "urine", "normal", "abnormal"]
    ),
    
    # -------------------------------------------------------------------------
    # UNKNOWN / GENERAL DOCUMENT
    # -------------------------------------------------------------------------
    "unknown": DocumentSchema(
        type_id="unknown",
        display_name="General Document",
        clip_prompts=[
            "a document with text",
            "a scanned document",
            "a page with printed or handwritten text"
        ],
        llm_context="""You are processing a GENERAL DOCUMENT of unknown type.

Since the document type is not specifically recognized, focus on:
- Extracting any clearly identifiable information
- Preserving the document structure
- Correcting obvious OCR errors
- Identifying key entities (names, dates, amounts)

Be conservative in your corrections - only fix clear OCR errors.
""",
        fields=[
            FieldSchema("document_title", FieldType.TEXT, "Title or heading if present"),
            FieldSchema("document_date", FieldType.DATE, "Any date found in the document"),
            FieldSchema("key_entities", FieldType.LIST, "Important names, organizations, or identifiers"),
            FieldSchema("summary", FieldType.TEXT, "Brief summary of document content"),
        ],
        keywords=[]
    ),
}


def get_schema(type_id: str) -> DocumentSchema:
    """Get document schema by type ID, falling back to unknown."""
    return DOCUMENT_SCHEMAS.get(type_id, DOCUMENT_SCHEMAS["unknown"])


def get_all_schemas() -> Dict[str, DocumentSchema]:
    """Get all registered document schemas."""
    return DOCUMENT_SCHEMAS


def get_schema_for_classification(classification: dict) -> DocumentSchema:
    """Get schema based on classification result."""
    type_id = classification.get("type_id", "unknown")
    if type_id not in DOCUMENT_SCHEMAS:
        # Try to match by class name
        class_name = classification.get("class", "").lower()
        for schema_id, schema in DOCUMENT_SCHEMAS.items():
            if schema_id in class_name or class_name in schema.display_name.lower():
                return schema
    return get_schema(type_id)

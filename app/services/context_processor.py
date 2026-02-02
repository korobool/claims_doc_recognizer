"""
Context-Aware OCR Post-Processor.

Applies document-type-specific corrections and field extraction
based on the detected document type.

Includes:
- Character corrections (O->0, l->1) in appropriate contexts
- Paragraph merging (semantic line grouping)
- Whitespace and punctuation normalization
- Confidence calibration
- Pattern-based field extraction
"""

import re
from typing import List, Dict, Optional, Tuple
from app.config.document_types import (
    DocumentType, 
    FieldDefinition,
    DOCUMENT_TYPES,
    NUMERIC_CHAR_FIXES,
    get_document_type,
)


# =============================================================================
# CONFIDENCE CALIBRATION
# =============================================================================

class ConfidenceCalibrator:
    """Calibrate OCR confidence scores for better reliability."""
    
    # Empirical calibration curve (raw -> calibrated)
    CALIBRATION_CURVE = {
        0.0: 0.0,
        0.3: 0.1,
        0.5: 0.3,
        0.7: 0.6,
        0.8: 0.75,
        0.9: 0.88,
        0.95: 0.94,
        1.0: 0.98
    }
    
    @classmethod
    def calibrate(cls, raw_confidence: float) -> float:
        """Convert raw confidence to calibrated confidence."""
        points = sorted(cls.CALIBRATION_CURVE.items())
        
        for i, (raw, cal) in enumerate(points):
            if raw_confidence <= raw:
                if i == 0:
                    return cal
                prev_raw, prev_cal = points[i-1]
                ratio = (raw_confidence - prev_raw) / (raw - prev_raw) if raw != prev_raw else 0
                return prev_cal + ratio * (cal - prev_cal)
        
        return points[-1][1]
    
    @classmethod
    def get_quality_label(cls, confidence: float) -> str:
        """Get human-readable quality label."""
        if confidence >= 0.9:
            return "high"
        elif confidence >= 0.7:
            return "medium"
        elif confidence >= 0.5:
            return "low"
        else:
            return "very_low"


# =============================================================================
# PARAGRAPH MERGING
# =============================================================================

def merge_paragraph_lines(text_lines: List[dict], merge_threshold: float = 1.5) -> List[dict]:
    """
    Merge consecutive text lines that form paragraphs.
    
    Args:
        text_lines: List of text line dicts with 'text', 'bbox', 'confidence'
        merge_threshold: Max vertical gap as multiple of line height
        
    Returns:
        List of merged text blocks
    """
    if not text_lines or len(text_lines) < 2:
        return text_lines
    
    # Sort by vertical position (top to bottom)
    sorted_lines = sorted(text_lines, key=lambda l: l.get('bbox', [0,0,0,0])[1])
    
    merged = []
    current_block = None
    
    for line in sorted_lines:
        if current_block is None:
            current_block = _create_block_from_line(line)
            continue
        
        if _should_merge(current_block, line, merge_threshold):
            current_block = _merge_line_into_block(current_block, line)
        else:
            merged.append(current_block)
            current_block = _create_block_from_line(line)
    
    if current_block:
        merged.append(current_block)
    
    return merged


def _create_block_from_line(line: dict) -> dict:
    """Create a block from a single line."""
    return {
        'text': line.get('text', ''),
        'bbox': list(line.get('bbox', [0, 0, 0, 0])),
        'confidence': line.get('confidence', 0),
        'line_count': 1,
        'original_lines': [line],
        'polygon': line.get('polygon'),
        'original_text': line.get('original_text', line.get('text', '')),
    }


def _should_merge(block: dict, line: dict, threshold: float) -> bool:
    """Determine if line should be merged into block."""
    block_bbox = block.get('bbox', [0, 0, 0, 0])
    line_bbox = line.get('bbox', [0, 0, 0, 0])
    
    # Calculate line height
    block_height = block_bbox[3] - block_bbox[1]
    line_height = line_bbox[3] - line_bbox[1]
    avg_line_height = (block_height / block.get('line_count', 1) + line_height) / 2
    
    if avg_line_height <= 0:
        return False
    
    # Vertical gap
    vertical_gap = line_bbox[1] - block_bbox[3]
    if vertical_gap > avg_line_height * threshold:
        return False
    if vertical_gap < -avg_line_height * 0.5:  # Overlapping too much
        return False
    
    # Horizontal alignment (left edge)
    left_diff = abs(block_bbox[0] - line_bbox[0])
    block_width = block_bbox[2] - block_bbox[0]
    if block_width > 0 and left_diff > block_width * 0.15:  # 15% tolerance
        return False
    
    # Font size similarity (bbox height)
    if avg_line_height > 0:
        height_ratio = min(line_height, avg_line_height) / max(line_height, avg_line_height)
        if height_ratio < 0.7:  # 30% tolerance
            return False
    
    return True


def _merge_line_into_block(block: dict, line: dict) -> dict:
    """Merge a line into an existing block."""
    return {
        'text': block['text'] + ' ' + line.get('text', ''),
        'bbox': [
            min(block['bbox'][0], line.get('bbox', [0,0,0,0])[0]),
            block['bbox'][1],  # Keep original top
            max(block['bbox'][2], line.get('bbox', [0,0,0,0])[2]),
            line.get('bbox', [0,0,0,0])[3]   # Extend to new bottom
        ],
        'confidence': (block['confidence'] * block.get('line_count', 1) + line.get('confidence', 0)) / (block.get('line_count', 1) + 1),
        'line_count': block.get('line_count', 1) + 1,
        'original_lines': block.get('original_lines', []) + [line],
        'polygon': None,  # Polygon no longer valid after merge
        'original_text': block.get('original_text', '') + ' ' + line.get('original_text', line.get('text', '')),
    }


# =============================================================================
# TEXT NORMALIZATION
# =============================================================================

def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    # Multiple spaces -> single space
    text = ' '.join(text.split())
    # Remove leading/trailing whitespace
    return text.strip()


def normalize_punctuation(text: str) -> str:
    """Normalize punctuation marks."""
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace("'", "'").replace("'", "'")
    # Normalize dashes
    text = text.replace('–', '-').replace('—', '-')
    # Normalize ellipsis
    text = text.replace('…', '...')
    return text


def fix_common_ocr_errors(text: str) -> str:
    """Fix common OCR character confusion errors."""
    # These are context-independent fixes
    replacements = [
        ('|', 'I'),      # Pipe often is I
        ('¦', 'I'),      # Broken bar often is I  
        ('``', '"'),     # Double backtick to quote
        ("''", '"'),     # Double single quote to double quote
    ]
    
    for wrong, right in replacements:
        text = text.replace(wrong, right)
    
    return text


class ContextAwareProcessor:
    """
    Post-processes OCR results based on document type context.
    
    Applies:
    1. Whitespace and punctuation normalization
    2. Common OCR error fixes
    3. Character corrections (O→0, l→1) in appropriate contexts
    4. Confidence calibration
    5. Paragraph merging (optional)
    6. Pattern-based field extraction
    """
    
    def __init__(self, document_type: str, merge_paragraphs: bool = False):
        """
        Initialize processor with document type.
        
        Args:
            document_type: Document type ID (e.g., "receipt", "prescription")
            merge_paragraphs: If True, merge lines into paragraphs
        """
        self.doc_type = get_document_type(document_type)
        self.merge_paragraphs = merge_paragraphs
        self.extracted_fields: Dict[str, str] = {}
    
    def process(self, text_lines: List[dict]) -> Tuple[List[dict], Dict[str, any]]:
        """
        Process OCR results with context awareness.
        
        Args:
            text_lines: List of OCR text line dicts
            
        Returns:
            Tuple of (corrected_text_lines, extracted_fields)
        """
        # Step 1: Apply text normalization (whitespace, punctuation)
        normalized_lines = self._normalize_text(text_lines)
        
        # Step 2: Apply common OCR error fixes
        fixed_lines = self._fix_common_errors(normalized_lines)
        
        # Step 3: Apply document-type-specific corrections
        corrected_lines = self._apply_corrections(fixed_lines)
        
        # Step 4: Calibrate confidence scores
        calibrated_lines = self._calibrate_confidence(corrected_lines)
        
        # Step 5: Optionally merge paragraphs
        if self.merge_paragraphs:
            calibrated_lines = merge_paragraph_lines(calibrated_lines)
        
        # Step 6: Extract fields based on expected patterns
        extracted_fields = self._extract_fields(calibrated_lines)
        
        return calibrated_lines, extracted_fields
    
    def _normalize_text(self, text_lines: List[dict]) -> List[dict]:
        """Apply whitespace and punctuation normalization."""
        normalized = []
        for line in text_lines:
            new_line = line.copy()
            text = line.get("text", "")
            text = normalize_whitespace(text)
            text = normalize_punctuation(text)
            new_line["text"] = text
            normalized.append(new_line)
        return normalized
    
    def _fix_common_errors(self, text_lines: List[dict]) -> List[dict]:
        """Apply common OCR error fixes."""
        fixed = []
        for line in text_lines:
            new_line = line.copy()
            text = line.get("text", "")
            text = fix_common_ocr_errors(text)
            new_line["text"] = text
            fixed.append(new_line)
        return fixed
    
    def _calibrate_confidence(self, text_lines: List[dict]) -> List[dict]:
        """Calibrate confidence scores for all lines."""
        calibrated = []
        for line in text_lines:
            new_line = line.copy()
            raw_conf = line.get("confidence", 0)
            calibrated_conf = ConfidenceCalibrator.calibrate(raw_conf)
            new_line["confidence"] = round(calibrated_conf, 3)
            new_line["raw_confidence"] = raw_conf
            new_line["quality"] = ConfidenceCalibrator.get_quality_label(calibrated_conf)
            calibrated.append(new_line)
        return calibrated
    
    def _apply_corrections(self, text_lines: List[dict]) -> List[dict]:
        """Apply document-type-specific OCR corrections."""
        corrected = []
        
        for line in text_lines:
            corrected_line = line.copy()
            text = line.get("text", "")
            
            # Apply corrections based on document type settings
            if self.doc_type.apply_currency_fix:
                text = self._fix_currency(text)
            
            if self.doc_type.apply_date_fix:
                text = self._fix_date(text)
            
            if self.doc_type.apply_numeric_fix:
                text = self._fix_numeric_context(text)
            
            corrected_line["text"] = text
            corrected_line["original_text"] = line.get("text", "")
            corrected.append(corrected_line)
        
        return corrected
    
    def _fix_currency(self, text: str) -> str:
        """Fix common OCR errors in currency values."""
        # Pattern: currency symbol followed by numbers
        def fix_currency_match(match):
            value = match.group(0)
            # Apply numeric fixes within the currency value
            for wrong, right in NUMERIC_CHAR_FIXES.items():
                value = value.replace(wrong, right)
            return value
        
        # Match currency patterns: $XX.XX, €XX,XX, etc.
        currency_pattern = r'[\$€£₽]\s*[\dOolISsBZG,.\s]+'
        text = re.sub(currency_pattern, fix_currency_match, text)
        
        # Also fix standalone price patterns (XX.XX at end of line)
        price_pattern = r'(\d[\dOolISsBZG]*)[.,](\d{2})\s*$'
        def fix_price(match):
            whole = match.group(1)
            decimal = match.group(2)
            for wrong, right in NUMERIC_CHAR_FIXES.items():
                whole = whole.replace(wrong, right)
                decimal = decimal.replace(wrong, right)
            return f"{whole}.{decimal}"
        
        text = re.sub(price_pattern, fix_price, text)
        
        return text
    
    def _fix_date(self, text: str) -> str:
        """Fix common OCR errors in dates."""
        # Pattern: DD/MM/YYYY or similar
        def fix_date_match(match):
            value = match.group(0)
            for wrong, right in NUMERIC_CHAR_FIXES.items():
                value = value.replace(wrong, right)
            return value
        
        date_pattern = r'\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}'
        text = re.sub(date_pattern, fix_date_match, text)
        
        return text
    
    def _fix_numeric_context(self, text: str) -> str:
        """Fix OCR errors in numeric contexts (quantities, IDs, etc.)."""
        # Fix numbers that appear after keywords like qty, #, no., etc.
        def fix_after_keyword(match):
            keyword = match.group(1)
            number = match.group(2)
            for wrong, right in NUMERIC_CHAR_FIXES.items():
                number = number.replace(wrong, right)
            return f"{keyword}{number}"
        
        # Patterns: qty: X, #X, no. X, количество: X
        numeric_keywords = r'((?:qty|quantity|#|no\.|№|количество)[:\s]*)([OolISsBZG\d]+)'
        text = re.sub(numeric_keywords, fix_after_keyword, text, flags=re.IGNORECASE)
        
        return text
    
    def _extract_fields(self, text_lines: List[dict]) -> Dict[str, any]:
        """Extract expected fields from text lines."""
        extracted = {}
        full_text = "\n".join([line.get("text", "") for line in text_lines])
        full_text_lower = full_text.lower()
        
        # Determine line positions (top, middle, bottom)
        total_lines = len(text_lines)
        top_threshold = max(1, total_lines // 4)
        bottom_threshold = total_lines - max(1, total_lines // 4)
        
        for field_def in self.doc_type.expected_fields:
            value = None
            
            # Try pattern matching first
            if field_def.pattern:
                pattern = re.compile(field_def.pattern, re.IGNORECASE)
                
                # Search in appropriate location
                if field_def.location == "top":
                    search_lines = text_lines[:top_threshold]
                elif field_def.location == "bottom":
                    search_lines = text_lines[bottom_threshold:]
                else:
                    search_lines = text_lines
                
                search_text = "\n".join([l.get("text", "") for l in search_lines])
                match = pattern.search(search_text)
                if match:
                    value = match.group(0)
            
            # Try keyword matching if no pattern match
            if not value and field_def.keywords:
                for keyword in field_def.keywords:
                    if keyword.lower() in full_text_lower:
                        # Find the line containing the keyword
                        for i, line in enumerate(text_lines):
                            if keyword.lower() in line.get("text", "").lower():
                                # Check location constraint
                                if field_def.location == "top" and i >= top_threshold:
                                    continue
                                if field_def.location == "bottom" and i < bottom_threshold:
                                    continue
                                value = line.get("text", "")
                                break
                    if value:
                        break
            
            if value:
                # Clean up the extracted value
                value = self._clean_field_value(value, field_def)
                extracted[field_def.name] = {
                    "value": value,
                    "display_name": field_def.display_name,
                    "data_type": field_def.data_type,
                    "required": field_def.required,
                }
        
        # Mark missing required fields
        for field_def in self.doc_type.expected_fields:
            if field_def.required and field_def.name not in extracted:
                extracted[field_def.name] = {
                    "value": None,
                    "display_name": field_def.display_name,
                    "data_type": field_def.data_type,
                    "required": True,
                    "missing": True,
                }
        
        return extracted
    
    def _clean_field_value(self, value: str, field_def: FieldDefinition) -> str:
        """Clean and normalize extracted field value."""
        value = value.strip()
        
        # Remove keyword prefix if present
        for keyword in field_def.keywords:
            pattern = re.compile(rf'^{re.escape(keyword)}[:\s]*', re.IGNORECASE)
            value = pattern.sub('', value)
        
        # Type-specific cleaning
        if field_def.data_type == "currency":
            # Extract just the numeric value
            match = re.search(r'[\$€£₽]?\s*([\d,]+[.,]\d{2})', value)
            if match:
                value = match.group(0)
        
        elif field_def.data_type == "date":
            # Extract just the date
            match = re.search(r'\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}', value)
            if match:
                value = match.group(0)
        
        return value.strip()


def process_with_context(text_lines: List[dict], document_type: str) -> Tuple[List[dict], Dict[str, any]]:
    """
    Convenience function to process OCR results with context.
    
    Args:
        text_lines: OCR text lines
        document_type: Document type ID
        
    Returns:
        Tuple of (corrected_lines, extracted_fields)
    """
    processor = ContextAwareProcessor(document_type)
    return processor.process(text_lines)

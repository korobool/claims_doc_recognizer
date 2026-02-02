"""
Document preprocessing module for OCR optimization.

Includes:
- Deskew using Projection Profile Analysis
- Contrast enhancement (CLAHE)
- Noise reduction
- Sharpening
- Auto-rotation based on EXIF
"""
import cv2
import numpy as np
from PIL import Image, ExifTags
import io
from scipy import ndimage
from dataclasses import dataclass
from typing import Optional


@dataclass
class PreprocessingConfig:
    """Configuration for image preprocessing."""
    deskew: bool = True
    enhance_contrast: bool = True
    denoise: bool = True
    sharpen: bool = True
    auto_rotate_exif: bool = True
    # Advanced options
    contrast_clip_limit: float = 2.0  # CLAHE clip limit
    contrast_grid_size: int = 8       # CLAHE grid size
    denoise_strength: int = 7         # Non-local means denoising strength (reduced for handwriting)
    sharpen_amount: float = 0.5       # Unsharp mask amount (reduced for handwriting)
    blend_ratio: float = 0.5          # Blend enhanced with original (0=original, 1=full enhancement)


@dataclass
class PreprocessingResult:
    """Result of image preprocessing."""
    image: np.ndarray
    deskew_angle: float
    operations_applied: list
    quality_metrics: dict


def deskew_image(image: np.ndarray, debug: bool = False) -> tuple[np.ndarray, float, dict]:
    """
    Deskew a document image to make text baseline horizontal.
    
    Uses Projection Profile Analysis:
    1. Binarize image
    2. Test rotation angles from -15° to +15°
    3. For each angle, compute horizontal projection profile variance
    4. Best angle = maximum variance (clearest text line separation)
    5. Rotate image by best angle
    
    Args:
        image: Input BGR image as numpy array
        debug: If True, include debug information in output
        
    Returns:
        tuple: (corrected_image, angle_degrees, debug_info)
    """
    debug_info = {}
    
    # 1. PREPROCESSING
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Binarize using Otsu (inverted: text = white)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 2. FIND BEST ANGLE using projection profile
    # Search range: -15 to +15 degrees with 0.5 degree steps
    angles = np.arange(-15, 15.5, 0.5)
    best_angle = 0.0
    best_variance = 0.0
    variances = []
    
    for angle in angles:
        # Rotate binary image
        rotated = ndimage.rotate(binary, angle, reshape=False, order=0)
        
        # Compute horizontal projection (sum of pixels per row)
        projection = np.sum(rotated, axis=1)
        
        # Variance of projection - higher means clearer line separation
        variance = np.var(projection)
        variances.append(variance)
        
        if variance > best_variance:
            best_variance = variance
            best_angle = angle
    
    if debug:
        debug_info["angles_tested"] = angles.tolist()
        debug_info["variances"] = variances
        debug_info["best_angle"] = best_angle
        debug_info["best_variance"] = best_variance
        debug_info["status"] = "success"
    
    # 3. REFINE ANGLE with finer search around best angle
    fine_angles = np.arange(best_angle - 0.5, best_angle + 0.55, 0.1)
    for angle in fine_angles:
        rotated = ndimage.rotate(binary, angle, reshape=False, order=0)
        projection = np.sum(rotated, axis=1)
        variance = np.var(projection)
        
        if variance > best_variance:
            best_variance = variance
            best_angle = angle
    
    if debug:
        debug_info["refined_angle"] = best_angle
    
    # 4. ROTATE IF NEEDED
    if abs(best_angle) < 0.2:
        return image.copy(), 0.0, debug_info
    
    # Rotate the original color image
    rotated = rotate_image_no_crop(image, best_angle)
    
    return rotated, best_angle, debug_info


def rotate_image_no_crop(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate image without cropping any content.
    
    Args:
        image: Input image
        angle: Rotation angle in degrees (positive = counter-clockwise)
        
    Returns:
        Rotated image with expanded canvas
    """
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Compute new image dimensions
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust rotation matrix for new center
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # Rotate with border replication
    rotated = cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return rotated


def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8, 
                     blend_ratio: float = 0.5) -> np.ndarray:
    """
    Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    CLAHE works on small regions (tiles) and limits contrast amplification to reduce noise.
    This is particularly effective for documents with uneven lighting.
    
    The result is blended with the original to preserve thin strokes (important for handwriting).
    
    Args:
        image: Input BGR image
        clip_limit: Threshold for contrast limiting (higher = more contrast)
        grid_size: Size of grid for histogram equalization
        blend_ratio: How much of the enhanced image to use (0.0 = original, 1.0 = full enhancement)
                     Default 0.5 preserves thin handwritten strokes while improving contrast.
        
    Returns:
        Contrast-enhanced image blended with original
    """
    # Convert to LAB color space (L = lightness)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Apply CLAHE to L channel only
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    l_enhanced = clahe.apply(l_channel)
    
    # Merge channels back
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    # Blend enhanced with original to preserve thin strokes (handwriting)
    # This prevents grey/thin strokes from disappearing
    if blend_ratio < 1.0:
        blended = cv2.addWeighted(image, 1.0 - blend_ratio, enhanced, blend_ratio, 0)
        return blended
    
    return enhanced


def denoise_image(image: np.ndarray, strength: int = 10) -> np.ndarray:
    """
    Remove noise from image using Non-local Means Denoising.
    
    This algorithm replaces each pixel with a weighted average of similar pixels,
    preserving edges while reducing noise. Excellent for scanned documents.
    
    Args:
        image: Input BGR image
        strength: Filter strength (higher = more denoising, but may blur details)
        
    Returns:
        Denoised image
    """
    # For color images, use fastNlMeansDenoisingColored
    # Parameters: src, dst, h, hColor, templateWindowSize, searchWindowSize
    denoised = cv2.fastNlMeansDenoisingColored(
        image, 
        None, 
        strength,      # h - filter strength for luminance
        strength,      # hForColorComponents - filter strength for color
        7,             # templateWindowSize
        21             # searchWindowSize
    )
    return denoised


def sharpen_image(image: np.ndarray, amount: float = 0.5) -> np.ndarray:
    """
    Sharpen image using Unsharp Masking technique.
    
    Creates a blurred version, then enhances edges by subtracting the blur.
    This makes text edges crisper for better OCR recognition.
    
    Note: For handwritten text, use lower amount (0.3-0.5) to avoid artifacts.
    
    Args:
        image: Input BGR image
        amount: Sharpening strength (0.3 = subtle, 0.5 = moderate, 1.0 = strong)
        
    Returns:
        Sharpened image
    """
    if amount <= 0:
        return image
    
    # Create Gaussian blur with smaller kernel for gentler sharpening
    blurred = cv2.GaussianBlur(image, (0, 0), 2)
    
    # Unsharp mask: original + amount * (original - blurred)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    
    return sharpened


def auto_rotate_exif(image: np.ndarray, pil_image: Optional[Image.Image] = None) -> tuple[np.ndarray, int]:
    """
    Auto-rotate image based on EXIF orientation tag.
    
    Many cameras/scanners store rotation in EXIF rather than rotating pixels.
    This function applies the correct rotation.
    
    Args:
        image: Input BGR image (numpy array)
        pil_image: Optional PIL image to read EXIF from
        
    Returns:
        tuple: (rotated_image, exif_orientation_value)
    """
    if pil_image is None:
        return image, 0
    
    try:
        exif = pil_image._getexif()
        if exif is None:
            return image, 0
            
        orientation = None
        for tag, value in exif.items():
            if ExifTags.TAGS.get(tag) == 'Orientation':
                orientation = value
                break
        
        if orientation is None:
            return image, 0
        
        # Apply rotation based on EXIF orientation
        if orientation == 3:  # 180 degrees
            return cv2.rotate(image, cv2.ROTATE_180), orientation
        elif orientation == 6:  # 90 degrees CW
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), orientation
        elif orientation == 8:  # 90 degrees CCW
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), orientation
        
        return image, orientation
        
    except (AttributeError, KeyError, TypeError):
        return image, 0


def compute_quality_metrics(image: np.ndarray) -> dict:
    """
    Compute image quality metrics useful for OCR.
    
    Args:
        image: Input BGR image
        
    Returns:
        dict with quality metrics
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Contrast: standard deviation of pixel values
    contrast = float(np.std(gray))
    
    # Sharpness: variance of Laplacian (higher = sharper)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(laplacian.var())
    
    # Brightness: mean pixel value
    brightness = float(np.mean(gray))
    
    # Noise estimate: using median absolute deviation
    noise_estimate = float(np.median(np.abs(gray - np.median(gray))))
    
    return {
        "contrast": round(contrast, 2),
        "sharpness": round(sharpness, 2),
        "brightness": round(brightness, 2),
        "noise_estimate": round(noise_estimate, 2),
        "resolution": f"{image.shape[1]}x{image.shape[0]}"
    }


def preprocess_image(
    image: np.ndarray,
    config: PreprocessingConfig = None,
    pil_image: Optional[Image.Image] = None
) -> PreprocessingResult:
    """
    Apply full preprocessing pipeline to optimize image for OCR.
    
    Pipeline order:
    1. Auto-rotate based on EXIF
    2. Deskew (straighten text lines)
    3. Denoise (reduce scanner/camera noise)
    4. Enhance contrast (improve text visibility)
    5. Sharpen (crisp text edges)
    
    Args:
        image: Input BGR image
        config: Preprocessing configuration
        pil_image: Optional PIL image for EXIF reading
        
    Returns:
        PreprocessingResult with processed image and metadata
    """
    if config is None:
        config = PreprocessingConfig()
    
    operations = []
    result_image = image.copy()
    deskew_angle = 0.0
    
    # 1. Auto-rotate based on EXIF
    if config.auto_rotate_exif and pil_image is not None:
        result_image, exif_orientation = auto_rotate_exif(result_image, pil_image)
        if exif_orientation > 1:
            operations.append(f"exif_rotate_{exif_orientation}")
    
    # 2. Deskew
    if config.deskew:
        result_image, deskew_angle, _ = deskew_image(result_image, debug=False)
        if abs(deskew_angle) >= 0.2:
            operations.append(f"deskew_{deskew_angle:.1f}deg")
    
    # 3. Denoise (before contrast to avoid amplifying noise)
    if config.denoise:
        result_image = denoise_image(result_image, strength=config.denoise_strength)
        operations.append("denoise")
    
    # 4. Enhance contrast (blended with original to preserve thin strokes)
    if config.enhance_contrast:
        result_image = enhance_contrast(
            result_image, 
            clip_limit=config.contrast_clip_limit,
            grid_size=config.contrast_grid_size,
            blend_ratio=config.blend_ratio
        )
        operations.append(f"contrast_enhance_blend{int(config.blend_ratio*100)}")
    
    # 5. Sharpen
    if config.sharpen:
        result_image = sharpen_image(result_image, amount=config.sharpen_amount)
        operations.append("sharpen")
    
    # Compute quality metrics on final image
    quality_metrics = compute_quality_metrics(result_image)
    
    return PreprocessingResult(
        image=result_image,
        deskew_angle=deskew_angle,
        operations_applied=operations,
        quality_metrics=quality_metrics
    )


def normalize_image(image_bytes: bytes, enhance: bool = True) -> tuple[bytes, float, dict]:
    """
    Normalize and enhance image for optimal OCR recognition.
    
    Args:
        image_bytes: Input image as bytes
        enhance: If True, apply full enhancement pipeline. If False, only deskew.
        
    Returns:
        tuple: (normalized_image_bytes, detected_angle, preprocessing_info)
    """
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Also load as PIL for EXIF
    pil_img = Image.open(io.BytesIO(image_bytes))
    
    # Configure preprocessing
    config = PreprocessingConfig(
        deskew=True,
        enhance_contrast=enhance,
        denoise=enhance,
        sharpen=enhance,
        auto_rotate_exif=True
    )
    
    # Apply preprocessing pipeline
    result = preprocess_image(img, config, pil_img)
    
    # Encode result
    _, buffer = cv2.imencode('.png', result.image)
    
    preprocessing_info = {
        "operations": result.operations_applied,
        "quality_metrics": result.quality_metrics
    }
    
    return buffer.tobytes(), result.deskew_angle, preprocessing_info


def normalize_image_legacy(image_bytes: bytes) -> tuple[bytes, float]:
    """
    Legacy normalize function for backward compatibility.
    Only performs deskew without enhancement.
    
    Args:
        image_bytes: Input image as bytes
        
    Returns:
        tuple: (normalized_image_bytes, detected_angle)
    """
    normalized_bytes, angle, _ = normalize_image(image_bytes, enhance=False)
    return normalized_bytes, angle


def deskew_only(image_bytes: bytes) -> tuple[bytes, float]:
    """
    Deskew image only - straighten text lines without quality enhancement.
    
    Applies:
    - Auto-rotate based on EXIF orientation
    - Deskew using projection profile analysis
    
    Does NOT apply: denoise, contrast enhancement, sharpening.
    
    Args:
        image_bytes: Input image as bytes
        
    Returns:
        tuple: (deskewed_image_bytes, detected_angle)
    """
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Also load as PIL for EXIF
    pil_img = Image.open(io.BytesIO(image_bytes))
    
    # Configure preprocessing - deskew only
    config = PreprocessingConfig(
        deskew=True,
        enhance_contrast=False,
        denoise=False,
        sharpen=False,
        auto_rotate_exif=True
    )
    
    # Apply preprocessing pipeline
    result = preprocess_image(img, config, pil_img)
    
    # Encode result
    _, buffer = cv2.imencode('.png', result.image)
    
    return buffer.tobytes(), result.deskew_angle


def enhance_only(image_bytes: bytes) -> tuple[bytes, dict]:
    """
    Enhance image quality only - no rotation or deskew.
    
    Applies:
    - Denoise (reduce scanner/camera noise)
    - Contrast enhancement (CLAHE, blended 50% with original to preserve thin strokes)
    - Sharpen (crisp text edges)
    
    Does NOT apply: deskew, EXIF rotation.
    
    Args:
        image_bytes: Input image as bytes
        
    Returns:
        tuple: (enhanced_image_bytes, preprocessing_info)
    """
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Configure preprocessing - enhance only
    config = PreprocessingConfig(
        deskew=False,
        enhance_contrast=True,
        denoise=True,
        sharpen=True,
        auto_rotate_exif=False
    )
    
    # Apply preprocessing pipeline
    result = preprocess_image(img, config, None)
    
    # Encode result
    _, buffer = cv2.imencode('.png', result.image)
    
    preprocessing_info = {
        "operations": result.operations_applied,
        "quality_metrics": result.quality_metrics
    }
    
    return buffer.tobytes(), preprocessing_info


def image_to_pil(image_bytes: bytes) -> Image.Image:
    """Convert bytes to PIL Image."""
    return Image.open(io.BytesIO(image_bytes))


def pil_to_bytes(pil_image: Image.Image, format: str = "PNG") -> bytes:
    """Convert PIL Image to bytes."""
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    return buffer.getvalue()

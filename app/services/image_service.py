"""
Document deskew / orientation correction module.

Uses Projection Profile Analysis - a robust classical method for document deskew.
The algorithm rotates the image at various angles and finds the angle where
the horizontal projection profile has maximum variance (clearest separation
between text lines).
"""
import cv2
import numpy as np
from PIL import Image
import io
from scipy import ndimage


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


def normalize_image(image_bytes: bytes) -> tuple[bytes, float]:
    """
    Normalize image orientation: deskew document to make text horizontal.
    
    Args:
        image_bytes: Input image as bytes
        
    Returns:
        tuple: (normalized_image_bytes, detected_angle)
    """
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Deskew
    corrected, angle, _ = deskew_image(img, debug=False)
    
    # Encode result
    _, buffer = cv2.imencode('.png', corrected)
    return buffer.tobytes(), angle


def image_to_pil(image_bytes: bytes) -> Image.Image:
    """Convert bytes to PIL Image."""
    return Image.open(io.BytesIO(image_bytes))


def pil_to_bytes(pil_image: Image.Image, format: str = "PNG") -> bytes:
    """Convert PIL Image to bytes."""
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    return buffer.getvalue()

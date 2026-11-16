"""Independent OpenCV operations on RGB uint8 images."""

import cv2
import numpy as np


def validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.size == 0:
        raise ValueError("Expected a nonempty uint8 image.")
    if image.ndim != 2 and not (image.ndim == 3 and image.shape[2] == 3):
        raise ValueError("Expected grayscale or RGB image.")


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert RGB to grayscale while preserving an existing grayscale image."""
    validate_image(image)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image.copy()


def resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize to explicit width and height using linear interpolation."""
    validate_image(image)
    if width <= 0 or height <= 0:
        raise ValueError("Dimensions must be positive.")
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def validate_kernel(kernel_size: int) -> None:
    if type(kernel_size) is not int or kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd integer of at least 3.")


def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    validate_image(image)
    validate_kernel(kernel_size)
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_median_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    validate_image(image)
    validate_kernel(kernel_size)
    return cv2.medianBlur(image, kernel_size)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    if not np.isfinite(clip_limit) or clip_limit <= 0:
        raise ValueError("Clip limit must be positive and finite.")
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(
        convert_to_grayscale(image)
    )


def detect_canny_edges(image: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    if not 0 <= low < high <= 255:
        raise ValueError("Canny thresholds must satisfy 0 <= low < high <= 255.")
    return cv2.Canny(convert_to_grayscale(image), low, high)


def apply_threshold(image: np.ndarray, threshold: int = 127) -> np.ndarray:
    if not 0 <= threshold <= 255:
        raise ValueError("Threshold must be between 0 and 255.")
    return cv2.threshold(
        convert_to_grayscale(image), threshold, 255, cv2.THRESH_BINARY
    )[1]


def apply_morphology(
    image: np.ndarray, operation: str = "open", kernel_size: int = 3
) -> np.ndarray:
    """Apply opening or closing to a grayscale or binary image."""
    validate_kernel(kernel_size)
    operations = {"open": cv2.MORPH_OPEN, "close": cv2.MORPH_CLOSE}
    if operation not in operations:
        raise ValueError("Operation must be open or close.")
    return cv2.morphologyEx(
        convert_to_grayscale(image),
        operations[operation],
        np.ones((kernel_size, kernel_size), np.uint8),
    )

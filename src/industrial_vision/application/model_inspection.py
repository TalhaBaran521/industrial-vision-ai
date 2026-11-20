"""Uploaded image inference and portable result exports."""

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import cv2
import numpy as np
from PIL import Image

from industrial_vision.vision.image_loader import load_image


def png_bytes(image: np.ndarray) -> bytes:
    """Encode an RGB or grayscale image as PNG."""
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def inspect_upload(
    data: bytes, detector: object, metadata: dict, model_id: str
) -> dict:
    """Validate upload, predict and create a relative-scale heatmap and JSON metadata."""
    image, _ = load_image(data)
    started = perf_counter()
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "input.png"
        Image.fromarray(image).save(path)
        score, anomaly_map = detector.predict(path)
    if (
        anomaly_map.shape != image.shape[:2]
        or not np.isfinite(anomaly_map).all()
        or not np.isfinite(score)
    ):
        raise ValueError("Model returned an invalid prediction.")
    maximum = float(anomaly_map.max())
    scaled = np.clip(anomaly_map / max(maximum, 1e-12), 0, 1)
    heatmap = cv2.cvtColor(
        cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_INFERNO),
        cv2.COLOR_BGR2RGB,
    )
    alpha = scaled[..., None] * 0.6
    overlay = (image * (1 - alpha) + heatmap * alpha).astype(np.uint8)
    report = {
        "model_id": model_id,
        "image_score": score,
        "image_threshold": metadata["threshold"],
        "is_anomalous": score > metadata["threshold"],
        "device": str(detector.device),
        "processing_time_ms": (perf_counter() - started) * 1000,
        "heatmap_range": [0, maximum],
        "heatmap_scale": "per-image raw distance; not probability",
        "pixel_threshold": None,
    }
    array_buffer = BytesIO()
    np.save(array_buffer, anomaly_map, allow_pickle=False)
    return {
        "original": image,
        "heatmap": heatmap,
        "overlay": overlay,
        "report": report,
        "map_bytes": array_buffer.getvalue(),
    }

"""Render difference regions in original image coordinates."""

import cv2
import numpy as np

from industrial_vision.domain.entities import DefectRegion


def draw_regions(
    image: np.ndarray, mask: np.ndarray, regions: list[DefectRegion]
) -> tuple[np.ndarray, np.ndarray]:
    """Return RGB bounding boxes and a red overlay limited to mask pixels."""
    boxes = image.copy()
    for region in regions:
        cv2.rectangle(
            boxes,
            (region.x, region.y),
            (region.x + region.width - 1, region.y + region.height - 1),
            (255, 0, 0),
            2,
        )
    overlay = image.copy()
    overlay[mask > 0] = (image[mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(
        np.uint8
    )
    return boxes, overlay

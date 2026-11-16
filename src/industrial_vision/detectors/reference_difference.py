"""Visual differences for aligned RGB images."""

import cv2
import numpy as np

from industrial_vision.domain.entities import DefectRegion, InspectionConfiguration
from industrial_vision.vision.preprocessing import validate_image


class ReferenceDifferenceDetector:
    def detect(
        self, reference: np.ndarray, image: np.ndarray, config: InspectionConfiguration
    ) -> tuple[np.ndarray, np.ndarray, list[DefectRegion]]:
        """Threshold maximum absolute RGB-channel difference and filter components."""
        validate_image(reference)
        validate_image(image)
        if reference.shape != image.shape or image.ndim != 3:
            raise ValueError(
                "Reference and inspected image must have identical RGB dimensions."
            )
        difference = cv2.absdiff(reference, image).max(axis=2)
        mask = (difference > config.threshold).astype(np.uint8) * 255
        if config.kernel_size > 1:
            kernel = np.ones((config.kernel_size, config.kernel_size), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        filtered = np.zeros_like(mask)
        regions = []
        for label in range(1, count):
            x, y, width, height, area = map(int, stats[label])
            if area >= config.min_area:
                filtered[labels == label] = 255
                regions.append(DefectRegion(x, y, width, height, area))
        return difference, filtered, regions

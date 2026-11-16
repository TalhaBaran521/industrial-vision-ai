"""Deterministic synthetic fixtures for software demonstrations."""

import cv2
import numpy as np


class SyntheticDatasetGenerator:
    def generate(self, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return aligned reference, scratched image and exact changed-pixel mask."""
        rng = np.random.default_rng(seed)
        reference = np.full((256, 256, 3), 35, np.uint8)
        cv2.rectangle(reference, (40, 40), (215, 215), (180, 180, 180), -1)
        mask = np.zeros((256, 256), np.uint8)
        y = int(rng.integers(70, 180))
        cv2.line(mask, (65, y), (190, y + 15), 255, 7)
        defective = reference.copy()
        defective[mask > 0] = 20
        return reference, defective, mask

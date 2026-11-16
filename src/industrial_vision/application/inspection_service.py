"""Coordinate visual difference inspection without UI dependencies."""

from dataclasses import asdict
from datetime import UTC, datetime
from time import perf_counter
from uuid import uuid4

import numpy as np

from industrial_vision.detectors.reference_difference import ReferenceDifferenceDetector
from industrial_vision.domain.entities import (
    InspectionArtifacts,
    InspectionConfiguration,
    InspectionResult,
)
from industrial_vision.vision.visualization import draw_regions


class InspectionService:
    def inspect(
        self,
        reference: np.ndarray,
        image: np.ndarray,
        image_name: str = "image",
        config: InspectionConfiguration | None = None,
    ) -> InspectionArtifacts:
        """Analyze aligned images at original resolution and produce exportable artifacts."""
        started = perf_counter()
        config = config or InspectionConfiguration()
        difference, mask, regions = ReferenceDifferenceDetector().detect(
            reference, image, config
        )
        boxes, overlay = draw_regions(image, mask, regions)
        result = InspectionResult(
            inspection_id=uuid4().hex,
            timestamp=datetime.now(UTC).isoformat(),
            image_name=image_name,
            image_width=image.shape[1],
            image_height=image.shape[0],
            detector_name="ReferenceDifferenceDetector",
            detector_version="1.0",
            processing_time_ms=(perf_counter() - started) * 1000,
            anomaly_score=None,
            anomaly_threshold=None,
            is_anomalous=bool(regions),
            defect_regions=regions,
            configuration=asdict(config),
            warnings=[
                "Visual difference analysis requires aligned images with matching lighting and camera geometry.",
                "Detected regions indicate visual differences; industrial defect classification requires validation.",
            ],
        )
        return InspectionArtifacts(
            result,
            {
                "reference": reference.copy(),
                "original": image.copy(),
                "difference_map": difference,
                "binary_mask": mask,
                "result": boxes,
                "overlay": overlay,
            },
        )

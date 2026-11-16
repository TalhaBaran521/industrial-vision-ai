"""Serializable inspection data and in-memory artifacts."""

import json
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class InspectionConfiguration:
    threshold: int = 25
    min_area: int = 20
    kernel_size: int = 3

    def __post_init__(self) -> None:
        if type(self.threshold) is not int or not 0 <= self.threshold <= 255:
            raise ValueError("Threshold must be an integer between 0 and 255.")
        if type(self.min_area) is not int or self.min_area < 1:
            raise ValueError("Minimum area must be a positive integer.")
        if (
            type(self.kernel_size) is not int
            or self.kernel_size < 1
            or self.kernel_size % 2 == 0
        ):
            raise ValueError(
                "Kernel size must be a positive odd integer; 1 disables opening."
            )


@dataclass(frozen=True)
class DefectRegion:
    x: int
    y: int
    width: int
    height: int
    area: int


@dataclass
class InspectionResult:
    inspection_id: str
    timestamp: str
    image_name: str
    image_width: int
    image_height: int
    detector_name: str
    detector_version: str
    processing_time_ms: float
    anomaly_score: float | None
    anomaly_threshold: float | None
    is_anomalous: bool | None
    defect_regions: list[DefectRegion]
    configuration: dict[str, int]
    warnings: list[str]

    def to_json(self) -> str:
        """Serialize metadata only, with strict finite JSON numbers."""
        return json.dumps(asdict(self), indent=2, ensure_ascii=False, allow_nan=False)


@dataclass
class InspectionArtifacts:
    result: InspectionResult
    images: dict[str, np.ndarray]

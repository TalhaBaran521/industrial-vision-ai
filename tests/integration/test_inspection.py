import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from industrial_vision.application.inspection_service import InspectionService
from industrial_vision.datasets.synthetic import SyntheticDatasetGenerator
from industrial_vision.domain.entities import InspectionConfiguration
from industrial_vision.infrastructure.result_repository import save_inspection


def test_deterministic_pipeline_and_export(tmp_path: Path) -> None:
    reference, defective, truth = SyntheticDatasetGenerator().generate(42)
    repeated = SyntheticDatasetGenerator().generate(42)
    for actual, expected in zip((reference, defective, truth), repeated, strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(truth > 0, np.any(reference != defective, axis=2))
    artifacts = InspectionService().inspect(
        reference, defective, "../../input.png", InspectionConfiguration(kernel_size=1)
    )
    np.testing.assert_array_equal(artifacts.images["binary_mask"], truth)
    region = artifacts.result.defect_regions[0]
    ys, xs = np.where(truth > 0)
    assert (region.x, region.y, region.width, region.height, region.area) == (
        xs.min(),
        ys.min(),
        xs.max() - xs.min() + 1,
        ys.max() - ys.min() + 1,
        len(xs),
    )
    destination = save_inspection(artifacts, tmp_path)
    assert destination is not None
    report = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
    assert report["is_anomalous"] is True
    assert report["anomaly_score"] is None
    np.testing.assert_array_equal(
        np.array(Image.open(destination / "binary_mask.png")), truth
    )
    assert save_inspection(artifacts, tmp_path) is None
    assert (destination / "metadata.json").exists()


def test_identical_images_have_no_regions() -> None:
    reference, _, _ = SyntheticDatasetGenerator().generate()
    artifacts = InspectionService().inspect(reference, reference)
    assert not artifacts.result.defect_regions
    assert artifacts.result.is_anomalous is False
    assert not artifacts.images["binary_mask"].any()
    np.testing.assert_array_equal(artifacts.images["overlay"], reference)


def test_area_filter_and_shape_validation() -> None:
    reference, defective, _ = SyntheticDatasetGenerator().generate()
    artifacts = InspectionService().inspect(
        reference, defective, config=InspectionConfiguration(min_area=10000)
    )
    assert not artifacts.images["binary_mask"].any()
    with pytest.raises(ValueError):
        InspectionService().inspect(reference, defective[:20])


@pytest.mark.parametrize(
    "values",
    [{"threshold": -1}, {"threshold": 256}, {"min_area": 0}, {"kernel_size": 2}],
)
def test_invalid_configuration(values: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        InspectionConfiguration(**values)


def test_save_failure_retains_analysis(tmp_path: Path) -> None:
    reference, defective, _ = SyntheticDatasetGenerator().generate()
    artifacts = InspectionService().inspect(reference, defective)
    blocked = tmp_path / "file"
    blocked.write_text("occupied")
    assert save_inspection(artifacts, blocked) is None
    assert artifacts.result.defect_regions
    assert "Saving failed" in artifacts.result.warnings[-1]


def test_cli_demo(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "industrial_vision.cli",
            "demo",
            "--output",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Visual difference analysis" in result.stdout
    report = next(tmp_path.glob("*/metadata.json"))
    assert (
        json.loads(report.read_text())["detector_name"] == "ReferenceDifferenceDetector"
    )
    assert (report.parent / "ground_truth_mask.png").exists()

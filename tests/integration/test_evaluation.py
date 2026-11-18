import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("sklearn")
pytest.importorskip("anomalib")
from industrial_vision.application.evaluation_service import evaluate  # noqa: E402
from industrial_vision.detectors.patchcore_detector import PatchCoreDetector  # noqa: E402


def test_evaluation_exports_and_preserves_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "data"
    for name in [
        "train/good/000.png",
        "train/good/001.png",
        "test/good/000.png",
        "test/crack/000.png",
        "ground_truth/crack/000_mask.png",
    ]:
        path = root / "bottle" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (8, 8), 255 if "crack" in name else 0).save(path)
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.pt").write_bytes(b"fixture")
    metadata = {
        "training": ["bottle/train/good/000.png"],
        "validation": ["bottle/train/good/001.png"],
        "threshold": 0.5,
        "category": "bottle",
        "test_used": False,
    }
    calls = []

    def predict(path: Path) -> tuple[float, np.ndarray]:
        calls.append(path.relative_to(root).as_posix())
        value = float("crack" in path.parts)
        return value, np.full((8, 8), value, np.float32)

    detector = SimpleNamespace(device="cpu", predict=predict)
    monkeypatch.setattr(PatchCoreDetector, "load", lambda *args: (detector, metadata))
    destination = evaluate(root, model, tmp_path / "reports")
    report = json.loads((destination / "report.json").read_text())
    assert report["image_metrics"]["confusion_matrix"] == [[1, 0], [0, 1]]
    assert report["pixel_metrics"]["iou"] == 1
    assert calls[0] == metadata["validation"][0]
    assert metadata["training"][0] not in calls
    assert (destination / "metrics.csv").is_file()
    assert (destination / "predictions.csv").is_file()
    assert report["examples"]["FP"] == []
    metadata["validation"] = metadata["training"]
    with pytest.raises(ValueError, match="manifest"):
        evaluate(root, model, tmp_path / "reports")

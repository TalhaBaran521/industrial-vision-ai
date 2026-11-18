import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("anomalib")
from industrial_vision.detectors.patchcore_detector import (  # noqa: E402
    PatchCoreDetector,
    select_device,
)


def test_cpu_memory_bank_save_reload(tmp_path: Path) -> None:
    torch.manual_seed(42)
    detector = PatchCoreDetector("cpu", 32, pretrained=False)
    image = tmp_path / "image.png"
    Image.fromarray(
        np.random.default_rng(42).integers(0, 256, (40, 48, 3), dtype=np.uint8)
    ).save(image)
    tensor, _ = detector.tensor(image)
    detector.model.train()
    with torch.no_grad():
        detector.model(tensor)
    detector.model.subsample_embedding(sampling_ratio=0.5)
    score, anomaly_map = detector.predict(image)
    assert np.isfinite(score) and np.isfinite(anomaly_map).all()
    assert anomaly_map.shape == (40, 48)
    torch.save(detector.model.state_dict(), tmp_path / "model.pt")
    (tmp_path / "metadata.json").write_text(json.dumps({"image_size": 32}))
    loaded, _ = PatchCoreDetector.load(tmp_path, "cpu")
    restored_score, restored_map = loaded.predict(image)
    assert restored_score == pytest.approx(score)
    np.testing.assert_allclose(restored_map, anomaly_map)


def test_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert str(select_device("auto")) == "cpu"
    assert str(select_device("cuda")) == "cpu"

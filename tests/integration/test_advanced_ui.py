from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from industrial_vision.application.model_inspection import inspect_upload, png_bytes


def test_upload_report_and_map() -> None:
    buffer = BytesIO()
    Image.new("RGB", (12, 8), "white").save(buffer, format="PNG")
    detector = SimpleNamespace(
        device="cpu", predict=lambda path: (2.0, np.ones((8, 12), np.float32))
    )
    result = inspect_upload(buffer.getvalue(), detector, {"threshold": 1.0}, "fixture")
    assert result["report"]["is_anomalous"] is True
    assert result["overlay"].shape == (8, 12, 3)
    assert png_bytes(result["overlay"]).startswith(b"\x89PNG")
    np.testing.assert_array_equal(
        np.load(BytesIO(result["map_bytes"])), np.ones((8, 12))
    )
    with pytest.raises(ValueError):
        inspect_upload(b"broken", detector, {"threshold": 1.0}, "fixture")


@pytest.mark.parametrize("page", ["2_Image_Inspection.py", "3_Experiment_Results.py"])
def test_page_startup(page: str) -> None:
    app = AppTest.from_file(
        Path(__file__).resolve().parents[2] / "pages" / page, default_timeout=60
    ).run()
    assert not app.exception
    assert app.title


def test_actual_model_upload() -> None:
    pytest.importorskip("anomalib")
    from industrial_vision.detectors.patchcore_detector import PatchCoreDetector

    model = Path("models/patchcore/51ad27ecf5134c64abddbdb9cd2ba44e")
    image = Path("data/mvtec_ad/bottle/test/good/000.png")
    if not model.is_dir() or not image.is_file():
        pytest.skip("Local trained model and bottle sample required")
    detector, metadata = PatchCoreDetector.load(model)
    result = inspect_upload(image.read_bytes(), detector, metadata, model.name)
    assert np.isfinite(result["report"]["image_score"])
    assert result["overlay"].shape == (900, 900, 3)

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from industrial_vision.datasets.mvtec import DatasetLoader


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    for name in [
        "train/good/000.png",
        "test/good/000.png",
        "test/crack/000.png",
        "ground_truth/crack/000_mask.png",
    ]:
        path = tmp_path / "bottle" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (16, 16), 255).save(path)
    return tmp_path


def test_splits_and_mask_pairing(dataset: Path) -> None:
    loader = DatasetLoader(dataset)
    index = loader.scan()
    assert index.errors == []
    assert index.statistics() == {"test/crack": 1, "test/good": 1, "train/good": 1}
    for sample in index.samples:
        image, mask = loader.load_sample(sample)
        assert image.shape == (16, 16, 3)
        assert bool(mask.any()) == sample.is_anomalous
        if sample.split == "train":
            assert sample.mask_path is None
    assert loader.scan().samples == index.samples


@pytest.mark.parametrize(
    "failure", ["missing", "corrupt", "dimensions", "nonbinary", "empty"]
)
def test_invalid_mask(dataset: Path, failure: str) -> None:
    path = dataset / "bottle/ground_truth/crack/000_mask.png"
    if failure == "missing":
        path.unlink()
    elif failure == "corrupt":
        path.write_bytes(b"broken")
    elif failure == "dimensions":
        Image.new("L", (8, 8), 255).save(path)
    else:
        Image.new("L", (16, 16), 127 if failure == "nonbinary" else 0).save(path)
    index = DatasetLoader(dataset).scan()
    assert index.errors
    assert not any(sample.is_anomalous for sample in index.samples)


def test_corruption_orphan_and_training_defect(dataset: Path) -> None:
    (dataset / "bottle/train/good/000.png").write_bytes(b"broken")
    (dataset / "bottle/train/crack").mkdir()
    Image.new("L", (16, 16)).save(dataset / "bottle/ground_truth/crack/orphan_mask.png")
    index = DatasetLoader(dataset).scan()
    assert any("Training group" in error for error in index.errors)
    assert any("Unmatched" in error for error in index.errors)
    assert not any(sample.split == "train" for sample in index.samples)


def test_missing_dataset_and_category_validation(tmp_path: Path) -> None:
    assert DatasetLoader(tmp_path).scan().errors
    with pytest.raises(ValueError):
        DatasetLoader(tmp_path, "../bottle")


def test_explorer(dataset: Path) -> None:
    app = AppTest.from_file(
        Path(__file__).resolve().parents[2] / "pages/1_Dataset_Explorer.py",
        default_timeout=60,
    ).run()
    app.text_input[0].set_value(str(dataset))
    app.button[0].click().run()
    assert not app.exception
    assert app.success
    app.selectbox[1].select("test").run()
    assert not app.exception


def test_external_sample_rejected(dataset: Path, tmp_path: Path) -> None:
    from industrial_vision.datasets.mvtec import DatasetSample

    outside = tmp_path / "outside.png"
    Image.fromarray(np.zeros((2, 2, 3), np.uint8)).save(outside)
    with pytest.raises(ValueError, match="leaves category"):
        DatasetLoader(dataset).load_sample(DatasetSample(outside, "test", "good", None))

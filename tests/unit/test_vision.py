from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from industrial_vision.vision import preprocessing as ops
from industrial_vision.vision.image_loader import load_image


def encode(mode="RGB", color="red", format="PNG"):
    buffer = BytesIO()
    Image.new(mode, (20, 10), color).save(buffer, format=format)
    return buffer.getvalue()


def test_rgb_and_metadata():
    image, metadata = load_image(encode())
    assert image.shape == (10, 20, 3)
    assert image[0, 0].tolist() == [255, 0, 0]
    assert metadata.width == 20
    assert ops.convert_to_grayscale(image)[0, 0] == 76


def test_transparency():
    image, _ = load_image(encode("RGBA", (0, 0, 0, 0)))
    assert np.all(image == 255)


@pytest.mark.parametrize("data", [b"", b"broken", encode(format="GIF"), encode()[:30]])
def test_bad_image(data):
    with pytest.raises(ValueError):
        load_image(data)


def test_limits():
    with pytest.raises(ValueError):
        load_image(encode(), max_pixels=100)
    with pytest.raises(ValueError):
        load_image(encode(), max_bytes=1)


def test_operations():
    image = np.zeros((20, 20, 3), np.uint8)
    image[5:15, 5:15] = 255
    assert ops.resize_image(image, 8, 6).shape == (6, 8, 3)
    assert ops.detect_canny_edges(image).any()
    assert set(np.unique(ops.apply_threshold(image))) == {0, 255}
    assert ops.apply_gaussian_blur(image)[5, 5, 0] < 255
    assert ops.apply_median_blur(image).shape == image.shape
    assert ops.apply_clahe(image).shape == (20, 20)
    noise = np.zeros((20, 20), np.uint8)
    noise[10, 10] = 255
    assert not ops.apply_morphology(noise).any()


@pytest.mark.parametrize(
    "operation",
    [
        lambda x: ops.apply_gaussian_blur(x, 2),
        lambda x: ops.detect_canny_edges(x, 150, 50),
        lambda x: ops.apply_threshold(x, 256),
        lambda x: ops.resize_image(x, 0, 1),
        lambda x: ops.apply_morphology(x, "invalid"),
    ],
)
def test_invalid_parameters(operation):
    with pytest.raises(ValueError):
        operation(np.zeros((10, 10, 3), np.uint8))


def test_streamlit_startup():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(Path(__file__).resolve().parents[2] / "app.py").run()
    assert not app.exception
    assert app.title[0].value == "INDUSTRIAL VISION AI"

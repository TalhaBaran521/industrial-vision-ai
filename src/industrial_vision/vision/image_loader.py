"""Validated image decoding with explicit RGB representation."""

import warnings
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


@dataclass(frozen=True)
class ImageMetadata:
    width: int
    height: int
    format: str


def load_image(
    data: bytes, max_bytes: int = 10_000_000, max_pixels: int = 20_000_000
) -> tuple[np.ndarray, ImageMetadata]:
    """Decode PNG/JPEG, orient using EXIF and composite transparency on white."""
    if max_bytes <= 0 or max_pixels <= 0:
        raise ValueError("Image limits must be positive.")
    if not data or len(data) > max_bytes:
        raise ValueError("Image is empty or exceeds the upload size limit.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as source:
                if source.format not in {"PNG", "JPEG"}:
                    raise ValueError("Only PNG and JPEG images are supported.")
                if source.width * source.height > max_pixels:
                    raise ValueError("Image exceeds the pixel limit.")
                metadata = ImageMetadata(source.width, source.height, source.format)
                source.verify()
            with Image.open(BytesIO(data)) as source:
                oriented = ImageOps.exif_transpose(source).convert("RGBA")
                background = Image.new("RGBA", oriented.size, "white")
                return np.array(
                    Image.alpha_composite(background, oriented).convert("RGB")
                ), metadata
    except (
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        raise ValueError("The image is damaged or too large to decode.") from error

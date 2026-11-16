"""Publish complete inspection directories with metadata and PNG artifacts."""

import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID

from PIL import Image

from industrial_vision.domain.entities import InspectionArtifacts

LOGGER = logging.getLogger(__name__)


def save_inspection(artifacts: InspectionArtifacts, output_root: Path) -> Path | None:
    """Stage a complete result before publication; report IO failures on the result."""
    identifier = artifacts.result.inspection_id
    if UUID(identifier).hex != identifier:
        raise ValueError("Invalid inspection identifier.")
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        destination = output_root / identifier
        with TemporaryDirectory(prefix=".inspection-", dir=output_root) as staging:
            staged = Path(staging) / "result"
            staged.mkdir()
            for name in (
                "reference",
                "original",
                "difference_map",
                "binary_mask",
                "result",
                "overlay",
            ):
                Image.fromarray(artifacts.images[name]).save(staged / f"{name}.png")
            (staged / "metadata.json").write_text(
                artifacts.result.to_json(), encoding="utf-8"
            )
            staged.rename(destination)
        return destination
    except OSError:
        LOGGER.exception("Inspection save failed")
        artifacts.result.warnings.append(
            "Saving failed. Inspection remains available in memory."
        )
        return None

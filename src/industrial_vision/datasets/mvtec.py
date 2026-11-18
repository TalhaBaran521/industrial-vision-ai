"""Read and validate one local MVTec AD category without downloading data."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from industrial_vision.vision.image_loader import load_image

CATEGORIES = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)


@dataclass(frozen=True)
class DatasetSample:
    image_path: Path
    split: str
    defect_type: str
    mask_path: Path | None

    @property
    def is_anomalous(self) -> bool:
        return self.defect_type != "good"


@dataclass
class DatasetIndex:
    samples: list[DatasetSample]
    errors: list[str]

    def statistics(self) -> dict[str, int]:
        """Count successfully validated samples by split and defect type."""
        return dict(
            sorted(
                Counter(
                    f"{sample.split}/{sample.defect_type}" for sample in self.samples
                ).items()
            )
        )


class DatasetLoader:
    def __init__(self, root: Path, category: str = "bottle") -> None:
        if category not in CATEGORIES:
            raise ValueError("Unknown MVTec AD category.")
        self.root = Path(root).resolve()
        self.category_path = self.root / category

    def _read(self, path: Path) -> np.ndarray:
        """Validate resolved containment and decode a bounded local image."""
        if not path.resolve().is_relative_to(self.category_path.resolve()):
            raise ValueError(f"Path leaves category directory: {path}")
        if not path.resolve().is_relative_to(self.root):
            raise ValueError(f"Path leaves dataset directory: {path}")
        with path.open("rb") as source:
            data = source.read(10_000_001)
        return load_image(data)[0]

    def load_sample(self, sample: DatasetSample) -> tuple[np.ndarray, np.ndarray]:
        """Return RGB image and binary ground truth, including zero masks for good images."""
        image = self._read(sample.image_path)
        if sample.mask_path is None:
            if sample.is_anomalous:
                raise ValueError("An anomalous sample requires ground truth.")
            return image, np.zeros(image.shape[:2], np.uint8)
        mask = self._read(sample.mask_path)
        if mask.shape[:2] != image.shape[:2]:
            raise ValueError("Image and ground truth dimensions differ.")
        if not np.array_equal(mask[..., 0], mask[..., 1]) or not np.array_equal(
            mask[..., 0], mask[..., 2]
        ):
            raise ValueError("Ground truth must be grayscale.")
        if not np.isin(mask, [0, 255]).all():
            raise ValueError("Ground truth must contain only 0 and 255.")
        if sample.is_anomalous and not mask.any():
            raise ValueError("Anomalous ground truth is empty.")
        return image, mask[..., 0].copy()

    def scan(self) -> DatasetIndex:
        """Collect valid samples and all discovered integrity errors, preserving official splits."""
        result = DatasetIndex([], [])
        expected_masks: set[Path] = set()
        for split in ("train", "test"):
            folder = self.category_path / split
            if not folder.is_dir():
                result.errors.append(f"Missing directory: {folder}")
                continue
            if not (folder / "good").is_dir():
                result.errors.append(f"Missing normal directory: {folder / 'good'}")
            for group in sorted(folder.iterdir()):
                if not group.is_dir():
                    result.errors.append(f"Unexpected entry: {group}")
                    continue
                if split == "train" and group.name != "good":
                    result.errors.append(f"Training group must be good: {group}")
                    continue
                files = sorted(group.iterdir())
                if not files:
                    result.errors.append(f"Empty image directory: {group}")
                for path in files:
                    if not path.is_file() or path.suffix.lower() != ".png":
                        result.errors.append(f"Expected PNG file: {path}")
                        continue
                    mask = None
                    if group.name != "good":
                        mask = (
                            self.category_path
                            / "ground_truth"
                            / group.name
                            / f"{path.stem}_mask.png"
                        )
                        expected_masks.add(mask)
                    sample = DatasetSample(path, split, group.name, mask)
                    try:
                        self.load_sample(sample)
                    except (OSError, ValueError) as error:
                        result.errors.append(f"{path}: {error}")
                        continue
                    result.samples.append(sample)
        truth_folder = self.category_path / "ground_truth"
        if truth_folder.is_dir():
            for path in sorted(truth_folder.rglob("*")):
                if path.is_file() and path not in expected_masks:
                    result.errors.append(f"Unmatched ground truth: {path}")
        if not any(
            sample.split == "test" and sample.is_anomalous for sample in result.samples
        ):
            result.errors.append("No valid anomalous test samples found.")
        return result

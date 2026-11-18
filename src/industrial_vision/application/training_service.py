"""Normal-only PatchCore memory bank fitting and validation calibration."""

import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from uuid import uuid4

import numpy as np
import torch

from industrial_vision.datasets.training_split import split_normal
from industrial_vision.detectors.patchcore_detector import PatchCoreDetector


def train(
    root: Path,
    output: Path,
    device: str = "auto",
    seed: int = 42,
    image_size: int = 128,
) -> Path:
    """Fit only train/good, calibrate on held-out good images, then publish artifacts."""
    started = perf_counter()
    training, validation = split_normal(root, seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    detector = PatchCoreDetector(device, image_size)
    detector.model.train()
    for index, path in enumerate(training):
        tensor, _ = detector.tensor(path)
        with torch.no_grad():
            detector.model(tensor)
        if index % 20 == 0:
            print(f"Extracted {index + 1}/{len(training)} normal images", flush=True)
    detector.model.subsample_embedding(sampling_ratio=0.01)
    scores = [detector.predict(path)[0] for path in validation]
    threshold = float(np.quantile(scores, 0.99))
    metadata = {
        "category": "bottle",
        "model": "PatchCore",
        "backbone": "resnet18",
        "image_size": image_size,
        "seed": seed,
        "device": str(detector.device),
        "coreset_sampling_ratio": 0.01,
        "num_neighbors": 1,
        "threshold": threshold,
        "threshold_method": "normal validation 99th percentile",
        "training": [str(path.relative_to(root.resolve())) for path in training],
        "validation": [str(path.relative_to(root.resolve())) for path in validation],
        "validation_scores": scores,
        "test_used": False,
        "versions": {
            name: version(name) for name in ["torch", "torchvision", "anomalib"]
        },
        "timestamp": datetime.now(UTC).isoformat(),
        "duration_seconds": perf_counter() - started,
        "memory_bank_shape": list(detector.model.memory_bank.shape),
    }
    output.mkdir(parents=True, exist_ok=True)
    destination = output / uuid4().hex
    with TemporaryDirectory(dir=output) as staging:
        staged = Path(staging) / "model"
        staged.mkdir()
        torch.save(detector.model.cpu().state_dict(), staged / "model.pt")
        (staged / "metadata.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
        )
        staged.rename(destination)
    return destination

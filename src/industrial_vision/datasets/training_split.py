"""Deterministic normal-only split without opening test images."""

import random
from pathlib import Path


def split_normal(root: Path, seed: int = 42) -> tuple[list[Path], list[Path]]:
    """Reserve 20 percent of bottle train/good for normal validation."""
    folder = (root / "bottle/train/good").resolve()
    paths = sorted(folder.glob("*.png"))
    if len(paths) < 5:
        raise ValueError("At least five bottle/train/good PNG images are required.")
    if any(not path.resolve().is_relative_to(folder) for path in paths):
        raise ValueError("Training image resolves outside train/good.")
    random.Random(seed).shuffle(paths)
    count = max(1, len(paths) // 5)
    return paths[count:], paths[:count]

from pathlib import Path

import pytest

from industrial_vision.datasets.training_split import split_normal


def test_normal_split_is_disjoint_and_deterministic(tmp_path: Path) -> None:
    folder = tmp_path / "bottle/train/good"
    folder.mkdir(parents=True)
    for index in range(10):
        (folder / f"{index:03}.png").touch()
    training, validation = split_normal(tmp_path)
    assert len(training) == 8 and len(validation) == 2
    assert not set(training) & set(validation)
    assert (training, validation) == split_normal(tmp_path)
    assert all("train/good" in path.as_posix() for path in training + validation)


def test_missing_training_data(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="At least five"):
        split_normal(tmp_path)

import numpy as np
import pytest

pytest.importorskip("sklearn")
from industrial_vision.application.evaluation_service import binary_metrics  # noqa: E402


def test_perfect_metrics() -> None:
    result = binary_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), 0.5)
    assert (
        result["auroc"]
        == result["average_precision"]
        == result["f1"]
        == result["iou"]
        == 1
    )
    assert result["confusion_matrix"] == [[2, 0], [0, 2]]


def test_undefined_metrics() -> None:
    result = binary_metrics(np.zeros(2), np.zeros(2), 0.5)
    assert result["auroc"] is None
    assert result["average_precision"] is None
    assert result["precision"] is None
    assert result["recall"] is None


def test_confusion_and_threshold_equality() -> None:
    result = binary_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 0.5, 1]), 0.5)
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["precision"] == result["recall"] == 0.5
    assert result["iou"] == pytest.approx(1 / 3)


def test_invalid_scores() -> None:
    with pytest.raises(ValueError):
        binary_metrics(np.array([0]), np.array([np.nan]), 1)

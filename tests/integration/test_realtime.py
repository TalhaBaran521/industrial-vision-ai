from pathlib import Path

import cv2
import numpy as np
import pytest

from industrial_vision.application.realtime_service import run_video


def make_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened()
    try:
        for _ in range(8):
            frame = np.zeros((48, 64, 3), np.uint8)
            frame[..., 2] = 255
            writer.write(frame)
    finally:
        writer.release()


def test_video_decoding_color_and_limits(tmp_path: Path) -> None:
    path = tmp_path / "fixture.avi"
    make_video(path)
    calls = []

    def predict(frame: np.ndarray) -> tuple[float, np.ndarray]:
        assert frame[0, 0, 0] > 240 and frame[0, 0, 2] < 10
        calls.append(1)
        return 1.0, np.zeros(frame.shape[:2], np.float32)

    report = run_video(str(path), predict, 2, display=False)
    assert report["frames"] == 8
    assert len(calls) == 11
    assert report["processing_fps"] > 0
    assert report["stop_reason"] == "end_of_stream"
    assert report["camera_dropped_frames"] is None
    assert run_video(str(path), predict, 2, False, 2)["frames"] == 2


def test_cleanup_on_prediction_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class Capture:
        released = False

        def isOpened(self):
            return True

        def get(self, key):
            return 10

        def read(self):
            return True, np.zeros((8, 8, 3), np.uint8)

        def release(self):
            self.released = True

    capture = Capture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda source: capture)

    def fail(frame):
        raise RuntimeError("prediction failed")

    with pytest.raises(RuntimeError, match="prediction failed"):
        run_video(0, fail, 1, False)
    assert capture.released


def test_missing_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Cannot open"):
        run_video(str(tmp_path / "missing.avi"), lambda frame: None, 1, False)

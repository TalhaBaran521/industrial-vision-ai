"""Synchronous local video inspection with deterministic resource cleanup."""

from collections.abc import Callable
from time import perf_counter

import cv2
import numpy as np


def run_video(
    source: str | int,
    predict: Callable,
    threshold: float,
    display: bool = True,
    max_frames: int | None = None,
    warmup: int = 3,
) -> dict:
    """Inspect every decoded frame; report throughput and read-to-result latency."""
    if max_frames is not None and max_frames < 1:
        raise ValueError("Maximum frames must be positive.")
    if warmup < 0 or not np.isfinite(threshold):
        raise ValueError("Invalid warmup or threshold.")
    capture = cv2.VideoCapture(source)
    latencies = []
    scores = []
    stop_reason = "end_of_stream"
    started = None
    source_fps = 0.0
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video source: {source}")
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        expected_frames = (
            int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if isinstance(source, str) else 0
        )
        while max_frames is None or len(scores) < max_frames:
            frame_started = perf_counter()
            ok, frame = capture.read()
            if not ok:
                if isinstance(source, int):
                    raise RuntimeError(
                        "Camera frame read failed or camera disconnected."
                    )
                if not scores or (
                    expected_frames > 0 and len(scores) < expected_frames
                ):
                    raise RuntimeError("Video could not be decoded completely.")
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if started is None:
                for _ in range(warmup):
                    predict(rgb)
                started = perf_counter()
                frame_started = started
            score, anomaly_map = predict(rgb)
            if (
                not np.isfinite(score)
                or anomaly_map.shape != rgb.shape[:2]
                or not np.isfinite(anomaly_map).all()
            ):
                raise ValueError("Invalid model output.")
            latencies.append((perf_counter() - frame_started) * 1000)
            scores.append(float(score))
            if display:
                scale = max(float(anomaly_map.max()), 1e-12)
                intensity = np.clip(anomaly_map / scale, 0, 1)
                heatmap = cv2.applyColorMap(
                    (intensity * 255).astype(np.uint8), cv2.COLORMAP_INFERNO
                )
                alpha = intensity[..., None] * 0.5
                overlay = (frame * (1 - alpha) + heatmap * alpha).astype(np.uint8)
                fps = len(scores) / max(perf_counter() - started, 1e-9)
                cv2.putText(
                    overlay,
                    f"{'ANOMALOUS' if score > threshold else 'NORMAL'} score={score:.3f} FPS={fps:.1f}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    overlay,
                    f"Distance scale 0-{scale:.2f} | Q / Esc: exit",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
                cv2.imshow("Industrial Vision AI - Video", overlay)
                if (
                    cv2.waitKey(1) & 0xFF in (ord("q"), 27)
                    or cv2.getWindowProperty(
                        "Industrial Vision AI - Video", cv2.WND_PROP_VISIBLE
                    )
                    < 1
                ):
                    stop_reason = "user_exit"
                    break
        else:
            stop_reason = "frame_limit"
    except KeyboardInterrupt:
        stop_reason = "interrupted"
    finally:
        capture.release()
        if display:
            cv2.destroyAllWindows()
    elapsed = perf_counter() - started if started is not None else 0
    return {
        "frames": len(scores),
        "source_fps": source_fps if np.isfinite(source_fps) else None,
        "processing_fps": len(scores) / elapsed if elapsed else None,
        "latency_mean_ms": float(np.mean(latencies)) if latencies else None,
        "latency_p50_ms": float(np.percentile(latencies, 50)) if latencies else None,
        "latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
        "warmup_passes": warmup,
        "duration_seconds": elapsed,
        "application_skipped_frames": 0,
        "camera_dropped_frames": None,
        "stop_reason": stop_reason,
        "scores": scores,
        "threshold": threshold,
    }

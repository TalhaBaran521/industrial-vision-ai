"""Evaluate frozen PatchCore predictions on the official test split."""

import csv
import hashlib
import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from uuid import uuid4

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from industrial_vision.datasets.mvtec import DatasetLoader


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Compute binary ranking and decision metrics; undefined quantities are null."""
    labels, scores = np.asarray(labels).ravel(), np.asarray(scores).ravel()
    if (
        not len(labels)
        or labels.shape != scores.shape
        or not np.isin(labels, [0, 1]).all()
        or not np.isfinite(scores).all()
        or not np.isfinite(threshold)
    ):
        raise ValueError(
            "Expected matching nonempty binary labels and finite scores/threshold."
        )
    tn, fp, fn, tp = map(
        int, confusion_matrix(labels, scores > threshold, labels=[0, 1]).ravel()
    )
    return {
        "auroc": float(roc_auc_score(labels, scores))
        if len(np.unique(labels)) == 2
        else None,
        "average_precision": float(average_precision_score(labels, scores))
        if labels.any()
        else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
        "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def evaluate(
    root: Path, model_directory: Path, output: Path, device: str = "auto"
) -> Path:
    """Calibrate pixel threshold on held-out normal images, then evaluate every test pixel."""
    from industrial_vision.detectors.patchcore_detector import PatchCoreDetector

    started = perf_counter()
    root = root.resolve()
    detector, metadata = PatchCoreDetector.load(model_directory, device)
    loader = DatasetLoader(root, "bottle")
    index = loader.scan()
    if index.errors:
        raise ValueError("Dataset integrity failed: " + "; ".join(index.errors))
    training = {(root / path).resolve() for path in metadata["training"]}
    validation = [(root / path).resolve() for path in metadata["validation"]]
    normal_root = (root / "bottle/train/good").resolve()
    if (
        not training
        or not validation
        or len(validation) != len(set(validation))
        or metadata.get("test_used") is not False
        or metadata.get("category") != "bottle"
        or training.intersection(validation)
        or any(
            not path.is_relative_to(normal_root) for path in training.union(validation)
        )
    ):
        raise ValueError("Invalid or overlapping normal training/validation manifest.")
    # Each image contributes its 99.5th pixel percentile; aggregate at the 99th percentile.
    pixel_threshold = float(
        np.quantile(
            [np.quantile(detector.predict(path)[1], 0.995) for path in validation], 0.99
        )
    )
    samples = [sample for sample in index.samples if sample.split == "test"]
    rows, pixel_labels, pixel_scores = [], [], []
    output.mkdir(parents=True, exist_ok=True)
    run_id = uuid4().hex
    destination = output / run_id
    with TemporaryDirectory(dir=output) as temporary:
        staged = Path(temporary) / "report"
        staged.mkdir()
        examples: dict[str, list[str]] = {name: [] for name in ("TP", "FP", "TN", "FN")}
        for sample in samples:
            image, mask = loader.load_sample(sample)
            score, anomaly_map = detector.predict(sample.image_path)
            if anomaly_map.shape != mask.shape:
                raise ValueError("Prediction/ground truth shape mismatch.")
            if not np.isfinite(score) or not np.isfinite(anomaly_map).all():
                raise ValueError("Prediction contains nonfinite values.")
            predicted = score > metadata["threshold"]
            outcome = ("T" if predicted == sample.is_anomalous else "F") + (
                "P" if predicted else "N"
            )
            relative = sample.image_path.relative_to(root).as_posix()
            rows.append(
                {
                    "image": relative,
                    "defect_type": sample.defect_type,
                    "label": int(sample.is_anomalous),
                    "score": score,
                    "prediction": int(predicted),
                    "outcome": outcome,
                }
            )
            pixel_labels.append((mask > 0).ravel())
            pixel_scores.append(anomaly_map.astype(np.float32).ravel())
            if len(examples[outcome]) < 3:
                name = f"{outcome}_{len(examples[outcome])}"
                Image.fromarray(image).save(staged / f"{name}_original.png")
                Image.fromarray(mask).save(staged / f"{name}_ground_truth.png")
                Image.fromarray(
                    (anomaly_map > pixel_threshold).astype(np.uint8) * 255
                ).save(staged / f"{name}_prediction.png")
                examples[outcome].append(relative)
            print(f"Evaluated {len(rows)}/{len(samples)}", flush=True)
        image_metrics = binary_metrics(
            np.array([row["label"] for row in rows]),
            np.array([row["score"] for row in rows]),
            metadata["threshold"],
        )
        pixel_metrics = binary_metrics(
            np.concatenate(pixel_labels), np.concatenate(pixel_scores), pixel_threshold
        )
        report = {
            "run_id": run_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "category": "bottle",
            "model_metadata": metadata,
            "model_sha256": hashlib.sha256(
                (model_directory / "model.pt").read_bytes()
            ).hexdigest(),
            "device": str(detector.device),
            "versions": {
                name: version(name)
                for name in ("torch", "anomalib", "numpy", "scikit-learn")
            },
            "image_threshold": metadata["threshold"],
            "pixel_threshold": pixel_threshold,
            "pixel_threshold_method": "99th percentile of held-out normal images' 99.5th pixel percentiles",
            "pixel_aggregation": "all original-resolution test pixels pooled, no sampling",
            "test_count": len(rows),
            "image_metrics": image_metrics,
            "pixel_metrics": pixel_metrics,
            "examples": examples,
            "duration_seconds": perf_counter() - started,
            "undefined_metrics": "null means N/A",
        }
        (staged / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
        with (staged / "predictions.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with (staged / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["level", "metric", "value"])
            for level, metrics in (("image", image_metrics), ("pixel", pixel_metrics)):
                for metric, value in metrics.items():
                    writer.writerow(
                        [level, metric, "N/A" if value is None else json.dumps(value)]
                    )
        staged.rename(destination)
    return destination

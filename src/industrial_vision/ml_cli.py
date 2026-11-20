"""Optional model commands, isolated from the base application imports."""

import argparse
import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="PatchCore training and inference")
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--data", type=Path, default=Path("data/mvtec_ad"))
    fit.add_argument("--output", type=Path, default=Path("models/patchcore"))
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--image-size", type=int, default=128)
    predict = commands.add_parser("predict")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--image", type=Path, required=True)
    predict.add_argument("--output", type=Path, default=Path("outputs/predictions"))
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--data", type=Path, default=Path("data/mvtec_ad"))
    evaluation.add_argument("--model", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, default=Path("outputs/evaluations"))
    evaluation.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    video = commands.add_parser("video")
    video.add_argument("--model", type=Path, required=True)
    source = video.add_mutually_exclusive_group(required=True)
    source.add_argument("--camera", type=int)
    source.add_argument("--video", type=Path)
    video.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    video.add_argument("--headless", action="store_true")
    video.add_argument("--max-frames", type=int)
    video.add_argument("--output", type=Path, default=Path("outputs/video"))
    commands.add_parser("device")
    for command in (fit, predict):
        command.add_argument(
            "--device", choices=["auto", "cuda", "cpu"], default="auto"
        )
    args = parser.parse_args()
    try:
        from industrial_vision.detectors.patchcore_detector import (
            PatchCoreDetector,
            select_device,
        )

        if args.command == "device":
            import torch

            print(
                json.dumps(
                    {
                        "cuda_available": torch.cuda.is_available(),
                        "device": str(select_device()),
                        "gpu": torch.cuda.get_device_name(0)
                        if torch.cuda.is_available()
                        else None,
                    }
                )
            )
        elif args.command == "train":
            from industrial_vision.application.training_service import train

            print(
                train(
                    args.data.resolve(),
                    args.output,
                    args.device,
                    args.seed,
                    args.image_size,
                )
            )
        elif args.command == "video":
            from industrial_vision.application.realtime_service import run_video

            if args.video is not None and not args.video.is_file():
                raise ValueError("Video file does not exist.")
            if args.camera is not None and args.camera < 0:
                raise ValueError("Camera index must be nonnegative.")
            detector, metadata = PatchCoreDetector.load(args.model, args.device)
            report = run_video(
                str(args.video) if args.video is not None else args.camera,
                detector.predict_rgb,
                metadata["threshold"],
                not args.headless,
                args.max_frames,
            )
            report.update(
                {
                    "model": str(args.model),
                    "device": str(detector.device),
                    "source": str(args.video)
                    if args.video is not None
                    else args.camera,
                }
            )
            args.output.mkdir(parents=True, exist_ok=True)
            destination = args.output / f"{uuid4().hex}.json"
            destination.write_text(
                json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
            )
            print(destination)
        elif args.command == "evaluate":
            from industrial_vision.application.evaluation_service import evaluate

            print(evaluate(args.data, args.model, args.output, args.device))
        else:
            detector, metadata = PatchCoreDetector.load(args.model, args.device)
            score, anomaly_map = detector.predict(args.image)
            destination = args.output / uuid4().hex
            destination.mkdir(parents=True)
            np.save(destination / "anomaly_map.npy", anomaly_map, allow_pickle=False)
            report = {
                "image": str(args.image),
                "model": str(args.model),
                "anomaly_score": score,
                "threshold": metadata["threshold"],
                "is_anomalous": score > metadata["threshold"],
                "map_shape": list(anomaly_map.shape),
                "device": str(detector.device),
            }
            (destination / "prediction.json").write_text(
                json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
            )
            print(destination)
    except (ImportError, OSError, ValueError, RuntimeError, cv2.error) as error:
        parser.exit(1, f"Model operation failed: {error}\n")


if __name__ == "__main__":
    main()

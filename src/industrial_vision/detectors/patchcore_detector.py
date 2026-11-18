"""Anomalib PatchCore adapter; imported only by ML commands."""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from anomalib.models.image.patchcore.torch_model import PatchcoreModel

from industrial_vision.vision.image_loader import load_image


def select_device(requested: str = "auto") -> torch.device:
    """Use CUDA when available and otherwise fall back to CPU."""
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError("Device must be auto, cuda or cpu.")
    return torch.device(
        "cuda" if requested != "cpu" and torch.cuda.is_available() else "cpu"
    )


class PatchCoreDetector:
    def __init__(
        self, device: str = "auto", image_size: int = 128, pretrained: bool = True
    ) -> None:
        if not 32 <= image_size <= 256:
            raise ValueError("Image size must be between 32 and 256.")
        self.device = select_device(device)
        self.image_size = image_size
        self.model = PatchcoreModel(
            layers=["layer2", "layer3"],
            backbone="resnet18",
            pre_trained=pretrained,
            num_neighbors=1,
        ).to(self.device)

    def tensor(self, path: Path) -> tuple[torch.Tensor, tuple[int, int]]:
        """Load RGB, resize without cropping, and normalize with ImageNet statistics."""
        with path.open("rb") as source:
            image, _ = load_image(source.read(10_000_001))
        height, width = image.shape[:2]
        resized = cv2.resize(
            image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
        )
        tensor = torch.from_numpy(resized.copy()).permute(2, 0, 1).float().div(255)
        tensor = (
            tensor - torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        ) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        return tensor.unsqueeze(0).to(self.device), (height, width)

    def predict(self, path: Path) -> tuple[float, np.ndarray]:
        """Return raw distance score and continuous map in original image coordinates."""
        tensor, shape = self.tensor(path)
        self.model.eval()
        with torch.inference_mode():
            prediction = self.model(tensor)
        score = float(prediction.pred_score.item())
        anomaly_map = prediction.anomaly_map.squeeze().cpu().numpy()
        return score, cv2.resize(
            anomaly_map, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR
        )

    @classmethod
    def load(
        cls, directory: Path, device: str = "auto"
    ) -> tuple["PatchCoreDetector", dict]:
        """Load tensor-only state from a locally produced model directory."""
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        detector = cls(device, metadata["image_size"], pretrained=False)
        state = torch.load(
            directory / "model.pt", map_location=detector.device, weights_only=True
        )
        detector.model.load_state_dict(state)
        return detector, metadata

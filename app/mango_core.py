"""Inference architectures and preprocessing reconstructed from the five supplied notebooks.

Not a training script: all neural networks require the matching trained checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torchvision import models, transforms

NUM_CLASSES = 6
INPUT_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class ModelSpec:
    display: str
    filename: str
    architecture: str
    source: str
    kind: str


MODEL_SPECS: Dict[str, ModelSpec] = {
    "Custom CNN": ModelSpec("Custom CNN", "custom_cnn_model.pth", "Custom 4-block CNN", "Notebook 1", "From scratch"),
    "ResNet50": ModelSpec("ResNet50", "transfer_learning_resnet50.pth", "ResNet-50", "Notebook 2", "ImageNet-initialized fine-tuning"),
    "DenseNet121": ModelSpec("DenseNet121", "transfer_learning_densenet121.pth", "DenseNet-121", "Notebook 3", "ImageNet-initialized fine-tuning"),
    "EfficientNet-B0": ModelSpec("EfficientNet-B0", "transfer_learning_efficientnet_b0.pth", "EfficientNet-B0", "Notebook 4", "ImageNet-initialized fine-tuning"),
    "MobileNetV2": ModelSpec("MobileNetV2", "transfer_learning_mobilenetv2.pth", "MobileNet-V2", "Notebook 5", "ImageNet-initialized fine-tuning"),
}


class CustomCNN(nn.Module):
    """Exact notebook-1 CustomCNN layer order: do not add extra conv layers."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.4), nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(model_name: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    """No pretrained downloads: state_dict supplies the entire learned network."""
    if model_name == "Custom CNN":
        return CustomCNN(num_classes)
    if model_name == "ResNet50":
        net = models.resnet50(weights=None)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        return net
    if model_name == "DenseNet121":
        net = models.densenet121(weights=None)
        net.classifier = nn.Linear(net.classifier.in_features, num_classes)
        return net
    if model_name == "EfficientNet-B0":
        net = models.efficientnet_b0(weights=None)
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, num_classes)
        return net
    if model_name == "MobileNetV2":
        net = models.mobilenet_v2(weights=None)
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, num_classes)
        return net
    raise ValueError(f"Unknown model: {model_name}")


def load_checkpoint(model_name: str, checkpoint: Path, device: torch.device = DEVICE) -> nn.Module:
    """Strict state_dict loading prevents inaccurate predictions from random layers.

    Only load checkpoints produced by this project's trusted notebooks.
    """
    net = build_model(model_name)
    try:
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not state:
            raise ValueError("Checkpoint must be a non-empty PyTorch state_dict.")
        # Simple notebooks save raw state_dict, but handle a standard wrapped state_dict too.
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        if state and all(isinstance(k, str) and k.startswith("module.") for k in state):
            state = {k[len("module."):]: v for k, v in state.items()}
        net.load_state_dict(state, strict=True)
    except Exception as exc:
        raise ValueError(
            f"Cannot load '{checkpoint.name}' for {model_name}. Confirm that this is "
            f"the exact checkpoint saved from the matching notebook and that "
            f"its trained output has six classes. Details: {exc}"
        ) from exc
    net.to(device)
    net.eval()
    return net


# The notebook evaluation pipeline uses (0.5,) for all RGB channels via broadcasting.
# Explicit 3-channel means/std is equivalent and less error-prone.
PREPROCESS = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])


def prepare_image(image: Image.Image, device: torch.device = DEVICE):
    """Return RGB corrected image, exact 224px view, input tensor, and RGB [0,1] array."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    resized = image.resize((INPUT_SIZE, INPUT_SIZE), resample=Image.Resampling.BILINEAR)
    tensor = PREPROCESS(image).unsqueeze(0).to(device)
    rgb_float = np.asarray(resized, dtype=np.float32) / 255.0
    return image, resized, tensor, rgb_float


def predict(model: nn.Module, tensor: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] != NUM_CLASSES:
            raise ValueError(f"Expected output [1, {NUM_CLASSES}], received {list(logits.shape)}")
        return torch.softmax(logits.float(), dim=1).cpu().numpy()[0]


def get_target_layer(model_name: str, model: nn.Module) -> nn.Module:
    """Notebook-specific XAI target layers (not a generic 'last conv' heuristic)."""
    if model_name == "Custom CNN":
        return model.features[-4]
    if model_name == "ResNet50":
        return model.layer4[-1]
    if model_name == "DenseNet121":
        return model.features.norm5
    if model_name in ("EfficientNet-B0", "MobileNetV2"):
        return model.features[-1]
    raise ValueError(f"Unknown model for target-layer selection: {model_name}")

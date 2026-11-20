"""Run `python check_app.py` to inspect setup without requiring trained weights."""
from __future__ import annotations

import importlib.util
import sys
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ["app.py", "mango_core.py", "explainability.py", "paper_figure.py",
         "requirements.txt", "assets/theme.css", "class_names.json", "results/model_metrics.csv"]
WEIGHTS = [
    "custom_cnn_model.pth", "transfer_learning_resnet50.pth",
    "transfer_learning_densenet121.pth", "transfer_learning_efficientnet_b0.pth",
    "transfer_learning_mobilenetv2.pth",
]
PACKAGES = ["streamlit", "torch", "torchvision", "pytorch_grad_cam", "lime", "skimage", "pandas", "matplotlib", "PIL"]


def main():
    print("=== MangoSight Leaf: setup check ===")
    print("\nProject files")
    for file in FILES:
        print(f"  {'OK' if (HERE / file).is_file() else 'MISSING'}  {file}")
    print("\nPython dependencies")
    for package in PACKAGES:
        print(f"  {'OK' if importlib.util.find_spec(package) else 'MISSING'}  {package}")
    print("\nTrained model checkpoints")
    for name in WEIGHTS:
        path = HERE / "models" / name
        print(f"  {'FOUND' if path.is_file() else 'NOT INCLUDED'}  models/{name}")
    print("\nClass label config")
    names = json.loads((HERE / "class_names.json").read_text(encoding="utf-8"))
    print("  ", names)
    print("  Verify these names and order against dataset.classes printed by EACH training notebook.")
    if "--verify-weights" in sys.argv:
        print("\nStrict checkpoint validation (real weights are required)")
        from mango_core import load_checkpoint
        import torch
        for model, name in zip(("Custom CNN", "ResNet50", "DenseNet121", "EfficientNet-B0", "MobileNetV2"), WEIGHTS):
            path = HERE / "models" / name
            if not path.is_file():
                continue
            try:
                instance = load_checkpoint(model, path, torch.device("cpu"))
                print(f"  OK {model}: architecture matches checkpoint")
                del instance
            except Exception as error:
                print(f"  ERROR {model}: {error}")
    print("\nThe app works as an information dashboard without weights, but prediction requires them.")
    print("Start with: python -m streamlit run app.py")


if __name__ == "__main__":
    main()

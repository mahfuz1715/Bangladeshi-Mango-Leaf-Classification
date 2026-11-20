"""Small local tests that do not require the genuine trained checkpoints."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mango_core import MODEL_SPECS, CustomCNN, build_model, get_target_layer, prepare_image, predict, load_checkpoint


def test_custom_architecture_and_key_positions():
    net = CustomCNN()
    assert len([layer for layer in net.modules() if isinstance(layer, torch.nn.Conv2d)]) == 4
    assert get_target_layer("Custom CNN", net) is net.features[-4]
    assert net.classifier[-1].out_features == 6
    assert "features.12.weight" in net.state_dict()
    assert "features.15.weight" not in net.state_dict()


def test_preprocess_shapes_and_range():
    img = Image.new("RGB", (320, 180), (128, 80, 25))
    rgb, resized, tensor, float_rgb = prepare_image(img, torch.device("cpu"))
    assert resized.size == (224, 224)
    assert tuple(tensor.shape) == (1, 3, 224, 224)
    assert float_rgb.dtype == np.float32 and float_rgb.min() >= 0 and float_rgb.max() <= 1
    assert -1.01 <= float(tensor.min()) < float(tensor.max()) <= 1.01


def test_synthetic_checkpoint_strict_roundtrip(tmp_path):
    net = CustomCNN()
    path = tmp_path / "custom_cnn_model.pth"
    torch.save(net.state_dict(), path)
    loaded = load_checkpoint("Custom CNN", path, torch.device("cpu"))
    for key, weight in net.state_dict().items():
        assert torch.equal(weight, loaded.state_dict()[key]), key
    weights = torch.load(path, weights_only=True)
    weights.pop("features.0.weight")
    torch.save(weights, path)
    try:
        load_checkpoint("Custom CNN", path, torch.device("cpu"))
    except ValueError:
        pass
    else:
        raise AssertionError("strict loading should reject missing weights")


def test_five_architectures_and_target_layers():
    assert len(MODEL_SPECS) == 5
    for name in MODEL_SPECS:
        model = build_model(name)
        assert get_target_layer(name, model) is not None
        if name == "Custom CNN":
            scores = predict(model.eval(), torch.zeros((1,3,224,224)))
            assert scores.shape == (6,) and np.isclose(scores.sum(), 1.0)
        del model

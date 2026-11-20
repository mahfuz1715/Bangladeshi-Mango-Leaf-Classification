"""CAM and LIME methods from the mango-leaf notebooks, with safer inference handling."""
from __future__ import annotations

from typing import Iterable
import numpy as np
from PIL import Image
import torch

from mango_core import PREPROCESS, get_target_layer

CAM_METHODS = ("Grad-CAM", "Grad-CAM++", "Eigen-CAM", "Ablation-CAM")
ALL_METHODS = CAM_METHODS + ("LIME",)


def cam_explanation(model, model_name: str, input_tensor: torch.Tensor,
                    rgb_float: np.ndarray, predicted_index: int, method: str) -> np.ndarray:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM, AblationCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    names = {
        "Grad-CAM": GradCAM,
        "Grad-CAM++": GradCAMPlusPlus,
        "Eigen-CAM": EigenCAM,
        "Ablation-CAM": AblationCAM,
    }
    if method not in names:
        raise ValueError(f"Unsupported CAM method: {method}")
    target_layer = get_target_layer(model_name, model)
    # Explicit class target: all explanations are for the selected predicted class.
    target = [ClassifierOutputTarget(int(predicted_index))]
    model.eval()
    with torch.enable_grad():
        with names[method](model=model, target_layers=[target_layer]) as cam:
            result = cam(input_tensor=input_tensor.clone().detach(), targets=target)[0]
    result = np.asarray(result, dtype=np.float32)
    result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)
    return show_cam_on_image(rgb_float.astype(np.float32), np.clip(result, 0, 1), use_rgb=True)


def lime_explanation(model, model_input_image: Image.Image, input_tensor: torch.Tensor,
                     predicted_index: int, samples: int = 100) -> np.ndarray:
    from lime import lime_image
    from skimage.segmentation import mark_boundaries

    rgb_input = np.asarray(model_input_image.convert("RGB"), dtype=np.uint8)

    def batch_predict(batch_images: Iterable[np.ndarray]) -> np.ndarray:
        images = []
        for item in batch_images:
            array = np.clip(np.asarray(item), 0, 255).astype(np.uint8)
            images.append(PREPROCESS(Image.fromarray(array, mode="RGB")))
        batch = torch.stack(images, dim=0).to(input_tensor.device)
        with torch.no_grad():
            outputs = model(batch)
            return torch.softmax(outputs.float(), dim=1).cpu().numpy()

    model.eval()
    explainer = lime_image.LimeImageExplainer(random_state=42)
    explanation = explainer.explain_instance(
        rgb_input, batch_predict, labels=(int(predicted_index),),
        top_labels=1, hide_color=0, num_samples=int(samples),
    )
    highlighted, mask = explanation.get_image_and_mask(
        label=int(predicted_index), positive_only=True, hide_rest=False,
        num_features=10, min_weight=0.01,
    )
    # LIME can return arrays scaled to 0..255 or 0..1 depending on upstream input.
    highlighted = np.asarray(highlighted, dtype=np.float32)
    if highlighted.max() > 1.0:
        highlighted /= 255.0
    outlined = mark_boundaries(np.clip(highlighted, 0, 1), mask)
    return np.uint8(np.clip(outlined * 255.0, 0, 255))

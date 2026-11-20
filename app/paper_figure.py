"""Export a 300-DPI study figure without generating or changing test metrics."""
from __future__ import annotations

from io import BytesIO
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def make_paper_figure(original: Image.Image, processed: Image.Image,
                      class_names: list[str], probabilities: np.ndarray,
                      chosen_model: str, explanation_name: str | None = None,
                      explanation_image: np.ndarray | None = None) -> bytes:
    fig = plt.figure(figsize=(12.2, 4.6), facecolor="white")
    grid = fig.add_gridspec(1, 4 if explanation_image is not None else 3,
                           width_ratios=[1, 1, 1, 1.20] if explanation_image is not None else [1, 1, 1.3])
    axes = [fig.add_subplot(grid[0, index]) for index in range(grid.ncols)]
    axes[0].imshow(original)
    axes[0].set_title("Uploaded image", fontsize=11, fontweight="bold")
    axes[1].imshow(processed)
    axes[1].set_title("224 × 224 model input", fontsize=11, fontweight="bold")
    if explanation_image is not None:
        axes[2].imshow(explanation_image)
        axes[2].set_title(explanation_name or "Explanation", fontsize=11, fontweight="bold")
    for ax in axes[:-1]:
        ax.axis("off")
    ax = axes[-1]
    order = np.argsort(probabilities)
    ax.barh(np.arange(len(class_names)), probabilities[order] * 100,
            color="#19866A", height=0.62)
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_yticklabels([textwrap.fill(class_names[index], 16) for index in order], fontsize=9)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Softmax score (%)", fontsize=9)
    ax.set_title("Class scores", fontsize=11, fontweight="bold")
    for i, v in enumerate(probabilities[order]):
        ax.text(min(v * 100 + 1, 100), i, f"{v*100:.1f}%", va="center", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Mango Leaf Classification  |  {chosen_model}", fontsize=14, fontweight="bold", y=1.02)
    fig.text(.5, -.015, "Research demo · CAM/LIME visualizations are explanatory estimates, not proof of causation.",
             ha="center", fontsize=8, color="#6b7280")
    fig.tight_layout(pad=1.6)
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buffer.getvalue()

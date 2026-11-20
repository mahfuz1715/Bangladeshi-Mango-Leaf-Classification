# Bangladeshi Mango Leaf Classification

### Comparative Deep Learning and Explainable AI for Mango Leaf Variety Recognition

This project investigates image-based recognition of **six Bangladeshi mango varieties** using five convolutional neural network (CNN) architectures. It combines model-specific training experiments, classification evaluation, five visual explanation methods, and **MangoSight Leaf**, a Streamlit research prototype for interactive inference and model inspection.

---

## Overview

Mango varieties can have visually similar leaves, making variety recognition from photographs an interesting fine-grained image classification problem. This project compares a custom CNN trained from scratch with four transfer-learning architectures, then uses Explainable AI (XAI) techniques to inspect which image regions contribute to predictions.

The workflow includes:
- loading and preprocessing mango leaf photographs;
- training five separate CNN-based classifiers;
- evaluating accuracy, weighted precision, weighted recall, and weighted F1-score;
- generating Grad-CAM, Grad-CAM++, Eigen-CAM, Ablation-CAM, and LIME explanations;
- and demonstrating model inference through a Streamlit application.

---

## Research Highlights

- **6-class** mango leaf *variety* classification (not disease detection)
- **5 models:** Custom CNN, ResNet50, DenseNet121, EfficientNet-B0, and MobileNetV2
- **224 × 224** RGB model input
- **70:20:10** intended training/validation/testing split in each notebook
- **5 XAI techniques** for qualitative interpretation
- **MangoSight Leaf** application with model selection, top predictions, explanation views, and paper-oriented PNG figure export

---

## Model Performance

The notebooks implement metric computation, but the supplied `.ipynb` files do not contain saved numerical execution results. To avoid misrepresenting the experiments, **no accuracy or F1-score is asserted here**.

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1-score |
|---|---:|---:|---:|---:|
| Custom CNN | Not verified | Not verified | Not verified | Not verified |
| ResNet50 | Not verified | Not verified | Not verified | Not verified |
| DenseNet121 | Not verified | Not verified | Not verified | Not verified |
| EfficientNet-B0 | Not verified | Not verified | Not verified | Not verified |
| MobileNetV2 | Not verified | Not verified | Not verified | Not verified |

The values can be updated after retrieving the actual evaluation outputs from the completed experiments. The app's `app/results/model_metrics.csv` is a placeholder for verified results, not a source of measured performance.

---

## Methodology

1. Load the folder-organized mango leaf dataset using PyTorch `ImageFolder`.
2. Resize RGB images to **224 × 224** and normalize the channels using mean and standard deviation of **0.5**.
3. Create training, validation, and test subsets using a **70:20:10** allocation.
4. Train **Custom CNN**, **ResNet50**, **DenseNet121**, **EfficientNet-B0**, and **MobileNetV2** separately with six-class output heads.
5. Evaluate predictions using accuracy and support-weighted precision, recall, and F1-score.
6. Generate five complementary XAI visualizations for qualitative inspection.
7. Load matching trained checkpoints in the Streamlit prototype to classify user-uploaded images.

The supplied training scripts specify Adam, a `0.001` learning rate, cross-entropy loss, batch size `32`, up to `50` epochs, and early stopping patience `5`. These describe the implemented training configuration; they do not guarantee that all intended augmentation or checkpoint-selection behavior worked as designed.

---

## Dataset

The notebooks refer to the **Image Dataset of Bangladeshi Mango Leaf**, loaded from a Kaggle input directory using `ImageFolder`.

The six mango varieties shown in the application are:

`Amrapali` · `Banana` · `Chaunsa` · `Fazli` · `Haribhanga` · `Himsagar`

**Important:** Check that `app/class_names.json` follows the exact index order returned by `dataset.classes` during training. A different label order can produce incorrect displayed predictions even when model inference succeeds.

The original dataset download URL, license, total image count, and class distribution should be verified from the dataset provider before they are added here. Dataset images are not included in this source-code package.

---

## Explainability

Five XAI techniques are incorporated to inspect model behavior:

| Method | Explanation produced |
|---|---|
| Grad-CAM | Gradient-based, class-oriented activation heatmap |
| Grad-CAM++ | Alternative gradient-weighted class activation map |
| Eigen-CAM | Salient structure in intermediate feature maps |
| Ablation-CAM | Feature-map importance based on ablation |
| LIME | Local superpixel-based explanation through image perturbations |

These visualizations can help explore model attention but **do not establish botanical correctness** or prove that a highlighted region is causally responsible for the classification.

---

## Streamlit Prototype — MangoSight Leaf

The `app/` directory contains a Streamlit research interface organized into three sections:

- **Leaf Analysis:** Upload a JPG/PNG image, choose one of the five models, view class scores and top predictions, and generate five XAI visualizations.
- **Study Results:** Display model comparison tables and charts once verified metrics have been entered into `app/results/model_metrics.csv`.
- **Model Card:** Review architecture, preprocessing, checkpoints, and methodological notes.

The application also supports a **300-DPI PNG figure export** for research presentations. Displayed softmax scores should not be interpreted as calibrated confidence or guarantees of correctness.

The app requires the corresponding `.pth` weights saved by the training notebooks. The source files alone cannot perform trained-model inference.

---

## Repository Structure

```text
Bangladeshi-Mango-Leaf-Classification/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── notebook1-custom-cnn-model.ipynb
│   ├── notebook2-resnet50-model.ipynb
│   ├── notebook3-densenet121-model.ipynb
│   ├── notebook4-efficientnet-b0-model.ipynb
│   └── notebook5-mobilenet-v2-model.ipynb
└── app/
    ├── app.py
    ├── mango_core.py
    ├── explainability.py
    ├── paper_figure.py
    ├── class_names.json
    ├── requirements.txt
    ├── README.md
    ├── check_app.py
    ├── assets/theme.css
    ├── .streamlit/config.toml
    ├── models/PUT_CHECKPOINTS_HERE.txt
    ├── results/model_metrics.csv
    └── tests/test_mango_core.py
```

This layout assumes the application files are directly inside `app/`, **not** nested under `app/MangoLeaf_Research_App/`. The launch scripts `app/run_windows.bat` and `app/run_linux_mac.sh` may also be retained.

---

## Running the Application

**Requirements:** Python 3.11 recommended, an internet connection for installing packages, and at least one matching trained checkpoint.

```bash
git clone https://github.com/mahfuz1715/Bangladeshi-Mango-Leaf-Classification.git
cd Bangladeshi-Mango-Leaf-Classification/app
python -m venv .venv
```

On **Windows (PowerShell)**:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe check_app.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

On **Linux/macOS**:

```bash
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python check_app.py
./.venv/bin/python -m streamlit run app.py
```

Place checkpoint files under `app/models/` using these filenames:

| Model | Checkpoint filename |
|---|---|
| Custom CNN | `custom_cnn_model.pth` |
| ResNet50 | `transfer_learning_resnet50.pth` |
| DenseNet121 | `transfer_learning_densenet121.pth` |
| EfficientNet-B0 | `transfer_learning_efficientnet_b0.pth` |
| MobileNetV2 | `transfer_learning_mobilenetv2.pth` |

Then open `http://localhost:8501`. If checkpoints are omitted from GitHub, a cloud deployment must obtain them through a separate, permitted storage mechanism before prediction can work.

---

## Project Status and Reproducibility

This repository presents an **unpublished academic implementation**, not an accepted/published paper. Accordingly, it does not include a conference claim, DOI, BibTeX citation, or unverifiable model ranking.

A few implementation details merit attention when reproducing or extending the experiments:

- The notebook splits are not guaranteed to be identical between models because a shared fixed random seed was not established.
- Reassigning the shared dataset transform can affect whether the intended training augmentation is applied.
- Early stopping does not necessarily restore the best-validation checkpoint before saving.
- Model-specific checkpoint architecture, class order, and XAI target layers must match the training setup.

Before making comparative performance claims, document the dataset version, test split, class mapping, per-class metrics, saved checkpoints, and evaluation outputs.

---

## Research Scope

MangoSight Leaf is a **research and demonstration prototype**, not a field-validated agricultural identification system. Future work could examine fixed and shared data splits, corrected augmentation, class-wise error analysis, confidence calibration, robustness across imaging conditions, and external validation on photographs collected outside the training dataset.

Dataset source and reuse terms should be acknowledged once verified. This README will be updated when complete experiment records and publication information, if any, become available.

# 🥭 MangoSight Leaf — Bangladeshi Mango Leaf Research Web App

A **Streamlit** six-class research classifier based on the **five supplied mango-leaf training notebooks**, redesigned with the visual hierarchy of the supplied **SpineSight MRI** application. The analysis tab, study results tab, model card, and 300-DPI PNG export are modeled on that reference experience; **classification models and transformations are strictly mango-specific**.

## What's included

- **Leaf Analysis:** image upload, exact evaluation-time preprocessing, five selectable models, six-class ranked softmax scores, top three predictions, four CAM explanations plus LIME, and a 300-DPI downloadable paper panel.
- **Study Results:** placeholder table/optional comparison chart populated *only* when you enter your verified notebook metrics (no guessed results).
- **Model Card:** trained architectures, checkpoint status, preprocessing, evaluation protocol, class-order checklist, and caveats.
- **No checkpoint:** all informational tabs still open, and a clear message tells you the required file location. No fake predictions are made.

## 1. Folder layout

```text
MangoLeaf_Research_App/
├── app.py                   # Main Streamlit user interface
├── mango_core.py            # Model definitions, strict checkpoint loading, transforms
├── explainability.py        # Grad-CAM / ++ / Eigen / Ablation / LIME
├── paper_figure.py          # 300-DPI PNG figure export
├── class_names.json         # SIX class labels in training-index order (VERIFY!)
├── requirements.txt         # Python packages
├── check_app.py             # Checks files, checkpoints and dependencies
├── run_windows.bat          # Start on Windows after dependencies are installed
├── run_linux_mac.sh         # Start on Linux/macOS
├── assets/theme.css         # Green-and-gold research app styling
├── .streamlit/config.toml
├── models/
│   ├── PUT_CHECKPOINTS_HERE.txt
│   ├── custom_cnn_model.pth                    # YOU ADD THIS
│   ├── transfer_learning_resnet50.pth         # YOU ADD THIS
│   ├── transfer_learning_densenet121.pth      # YOU ADD THIS
│   ├── transfer_learning_efficientnet_b0.pth  # YOU ADD THIS
│   └── transfer_learning_mobilenetv2.pth      # YOU ADD THIS
└── results/
    └── model_metrics.csv    # Fill in real numerical study results if available
```

**IMPORTANT:** The original five `.pth` model weight files were **not included** in the uploaded material. They must be exported/downloaded from the matching completed Kaggle notebooks and copied into `models/`. The ZIP is a complete **application source package**, not a pretrained inference bundle.

### Exact checkpoint names (from each training notebook)

| Training notebook | Expected checkpoint in `models/` |
|---|---|
| Notebook 1 — Custom CNN | `custom_cnn_model.pth` |
| Notebook 2 — ResNet50 | `transfer_learning_resnet50.pth` |
| Notebook 3 — DenseNet121 | `transfer_learning_densenet121.pth` |
| Notebook 4 — EfficientNet-B0 | `transfer_learning_efficientnet_b0.pth` |
| Notebook 5 — MobileNetV2 | `transfer_learning_mobilenetv2.pth` |

In Kaggle's notebook output / `/kaggle/working/`, the training code saves the `.pth` files using the names above. Download these actual weights. **Do not** substitute the old `model_fix.py` script, use `strict=False`, or rename one architecture's weights to match another.

## 2. Very easy Windows setup (recommended)

1. Install **Python 3.11 (64-bit)** on Windows; enable **Add Python to PATH** in the installer. Python 3.10 or 3.12 should also work.
2. Extract the ZIP into a regular folder, e.g. `D:\MangoLeaf_Research_App`. Avoid running it directly inside the ZIP.
3. Put your real `.pth` files into `D:\MangoLeaf_Research_App\models\` (you may start with just one).
4. Open this extracted folder in File Explorer. Click the address bar, type `cmd`, then press Enter.
5. In the Command Prompt, run the following commands, one by one:

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe check_app.py
.venv\Scripts\python.exe -m streamlit run app.py
```

6. Your default browser should open the app (usually at `http://localhost:8501`). If not, paste this address into Chrome/Edge. The terminal must remain open while the app is running.

**Next time:** double-click `run_windows.bat`; it uses the `.venv` you created. Press `Ctrl+C` in the terminal to stop the app.

If `python` isn't recognized, install Python and restart Command Prompt. On some Windows installations, use `py -3.11` instead of `python` when creating the environment.

## 3. macOS / Linux

```bash
cd MangoLeaf_Research_App
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python check_app.py
.venv/bin/python -m streamlit run app.py
```

On machines with unsupported hardware, you may need to use a platform-specific PyTorch installation command from the official PyTorch selector before installing the other requirements. CPU operation is supported; XAI will be slower.

## 4. Critical: verify class label order

The source notebooks use **`class_names = dataset.classes` from `torchvision.datasets.ImageFolder`**. ImageFolder determines class indices from **alphabetical folder order**. The notebooks' execution outputs were empty, so the exact printed folder names cannot be recovered from the uploaded files. We used the six variety names from the *old Streamlit app*, in the order shown there:

```json
[
  "Amrapali Mango", "Banana Mango", "Chaunsa Mango",
  "Fazli Mango", "Haribhanga Mango", "Himsagar Mango"
]
```

To ensure the predictions are labeled correctly, **open every trained notebook**, find the output of `print("Classes found:", class_names)`, and compare the printed list against `class_names.json`. If the actual folder names are different, edit **the labels and their order** in `class_names.json` to match; labels are not embedded in plain `state_dict` files. If the five notebooks used different class orders, they cannot all share a single verified class mapping without model-specific label metadata, and you should first resolve that before publishing the app.

## 5. Add test results to the Study Results tab (optional)

Open `results/model_metrics.csv` in Excel, Notepad, or Google Sheets. The rows are already present, but **values are intentionally blank** because the notebooks did not contain executed output. Fill with verified results obtained from notebook evaluation.

Example **format only** (these are NOT real scores):

```csv
Model,Accuracy,Precision,Recall,F1 Score
Custom CNN,0.80,0.81,0.82,0.81
```

Metric values should be decimals **between 0 and 1**: 91.23% corresponds to `0.9123`. Do not use the example numbers as research results. The app uses weighted precision, recall and F1 to match the notebooks.

## 6. How to use the app

1. Open **Leaf Analysis**.
2. Choose one of the five architectures. If its checkpoint is missing, add the file to `models/` and rerun.
3. Upload an example mango leaf photograph (`.jpg`, `.jpeg`, `.png` up to 15 MB).
4. View the uploaded image, the resized model input, the predicted variety, softmax score, and six-class distribution.
5. Choose any of the five XAI methods and select **Generate selected explanations**. Start with Grad-CAM; LIME and Ablation-CAM can be slower.
6. Save your **300-DPI PNG** analysis figure using the download button (includes the first successfully generated XAI visualization, if present).
7. Visit **Study Results** and **Model Card** to review the study protocol and model status.

## 7. Exact links to the five notebooks' training choices

| Setting | Source behavior retained |
|---|---|
| Task | Six-class mango leaf variety classification |
| Input | RGB, resize `224×224`, ToTensor, Normalize `0.5`/`0.5` |
| Model types | Notebook 1 CustomCNN; notebook 2 ResNet50; notebook 3 DenseNet121; notebook 4 EfficientNet-B0; notebook 5 MobileNetV2 |
| Saved weights | `torch.save(model.state_dict(), name)` |
| XAI | Grad-CAM, Grad-CAM++, Eigen-CAM, Ablation-CAM, LIME |
| Target layers | Custom `features[-4]`; ResNet `layer4[-1]`; DenseNet `features.norm5`; EfficientNet/MobileNet `features[-1]` |
| Training (context only) | Adam lr=0.001, CrossEntropyLoss, up to 50 epochs, batch 32, early stopping patience 5 |
| Evaluation (context only) | Accuracy, weighted precision, weighted recall, weighted F1 |

**Corrections relative to the previous mango app:** We reconstructed the **exact 4-convolution CustomCNN** from notebook 1 (not the incompatible 6-convolution version), chose each notebook's XAI target layers, fixed the LIME input scale, and implemented **strict** checkpoint matching instead of silently accepting partial/random model weights.

**Research limitations worth recording:** The notebook snippets indicate unseeded split selection and shared dataset transform reassignment; training augmentation therefore was not applied as intended. They save the last model instead of restoring the best validation checkpoint. The inference app does **not** change/retrain those models and does not claim comparable identical test splits. Validate on the intended test data before paper publication.

## 8. Troubleshooting

- `ModuleNotFoundError: No module named 'streamlit'`: activate/install the project's venv; use `.venv\Scripts\python.exe -m pip install -r requirements.txt` on Windows.
- `No module named pytorch_grad_cam` or `lime`: install the project requirements; both are explicitly included.
- `Model checkpoint not found`: confirm exact filename and location in the `models` folder; filename extensions must not silently become `.pth.pth`.
- `Model architecture mismatch`: verify the file came from the **matching** notebook; no partial loading is attempted.
- App is slow: choose Grad-CAM first, use fewer CAM methods and avoid large images. CPU LIME and Ablation-CAM can be slow.
- Predictions look incorrect: check class ordering (`class_names.json` versus `dataset.classes`), input image quality, and training/checkpoint compatibility.
- `torchvision` version error: use a fresh Python virtual environment and the matching versions from `requirements.txt`.

## Research disclaimer

This is a **research demonstration** based on a specific dataset and set of trained models. Model scores are uncalibrated, and visual explanations are exploratory. The software does not guarantee a correct mango variety identification, and it is not a substitute for independent testing.

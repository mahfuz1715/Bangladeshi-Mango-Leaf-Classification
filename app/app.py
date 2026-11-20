"""MangoSight Leaf: research showcase and reproducible six-class image inference.

UI layout inspired by the user's LumbarMRI_Streamlit_App.zip; models and transforms
are taken from the five mango-leaf notebooks, never the MRI pipeline.
"""
from __future__ import annotations

import hashlib
import html
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import streamlit as st

from mango_core import DEVICE, MODEL_SPECS, NUM_CLASSES, build_model, load_checkpoint, prepare_image, predict
from explainability import ALL_METHODS, CAM_METHODS, cam_explanation, lime_explanation
from paper_figure import make_paper_figure

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
RESULTS_FILE = ROOT / "results" / "model_metrics.csv"
CLASSES_FILE = ROOT / "class_names.json"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 25_000_000

st.set_page_config(page_title="MangoSight Leaf | Research Classifier",
                   page_icon="🥭", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>" + (ROOT / "assets" / "theme.css").read_text(encoding="utf-8") + "</style>",
            unsafe_allow_html=True)


def text_safe(value: object) -> str:
    return html.escape(str(value), quote=True)


def class_names() -> list[str]:
    try:
        values = json.loads(CLASSES_FILE.read_text(encoding="utf-8"))
        if not isinstance(values, list) or len(values) != NUM_CLASSES:
            raise ValueError("class_names.json must contain exactly six names.")
        values = [str(x).strip() for x in values]
        if not all(values) or len(set(values)) != NUM_CLASSES:
            raise ValueError("Class names must be unique and non-empty.")
        return values
    except (ValueError, OSError, TypeError) as exc:
        st.error(f"Class label configuration error: {exc}")
        st.stop()


@st.cache_resource(show_spinner=False)
def cached_model(name: str, filename: str, modified_ns: int, device_name: str):
    # modified_ns busts the cache after a checkpoint is replaced.
    return load_checkpoint(name, Path(filename), DEVICE)


def card_stat(label: str, value: str):
    st.markdown(f"<div class='stat-card'><div class='stat-value'>{text_safe(value)}</div>"
                f"<div class='stat-label'>{text_safe(label)}</div></div>", unsafe_allow_html=True)


def probability_html(labels: list[str], probabilities: np.ndarray) -> str:
    rows = []
    for idx in np.argsort(-probabilities):
        pct = float(probabilities[idx]) * 100
        rows.append(
            "<div class='pbar-row'>"
            f"<div class='pbar-caption'><span>{text_safe(labels[idx])}</span><span>{pct:.2f}%</span></div>"
            f"<div class='pbar-track'><div class='pbar-fill' style='width:{pct:.4f}%'></div></div>"
            "</div>"
        )
    return "".join(rows)


def checkpoint_for(name: str) -> Path:
    return MODEL_DIR / MODEL_SPECS[name].filename


def render_checkpoint_status() -> None:
    available = [name for name in MODEL_SPECS if checkpoint_for(name).is_file()]
    st.markdown("<div class='mini-label'>Available checkpoints</div>", unsafe_allow_html=True)
    pills = []
    for name in MODEL_SPECS:
        ok = name in available
        pills.append(f"<span class='{'status-good' if ok else 'status-bad'}'>"
                     f"{'●' if ok else '○'} {text_safe(name)}</span>")
    st.markdown("<div class='status-pills'>" + "".join(pills) + "</div>", unsafe_allow_html=True)
    if not available:
        st.caption("No trained .pth checkpoints are bundled. The interface and reference tabs are still available.")


def read_metrics():
    """Never fill missing metrics with fabricated data."""
    if not RESULTS_FILE.is_file():
        return pd.DataFrame(), None
    try:
        table = pd.read_csv(RESULTS_FILE)
    except Exception as exc:
        return pd.DataFrame(), f"Could not read results/model_metrics.csv: {exc}"
    expected = ["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
    if not set(expected).issubset(table.columns):
        return pd.DataFrame(), "Metrics file must contain: " + ", ".join(expected)
    table = table[expected].copy()
    for col in expected[1:]:
        table[col] = pd.to_numeric(table[col], errors="coerce")
    metrics = table[expected[1:]]
    bad = metrics.notna() & ~(metrics.ge(0) & metrics.le(1))
    if bad.any().any():
        return pd.DataFrame(), "Metrics values must be decimal fractions from 0 to 1, not percentages."
    table = table.loc[table[expected[1:]].notna().any(axis=1)].copy()
    return table, None


def study_results_tab():
    st.markdown("<div class='section-kicker'>Experiment summary</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Five-architecture study results</div>", unsafe_allow_html=True)
    st.markdown("<div class='lead'>The original mango-leaf notebooks compute test accuracy and weighted precision, recall and F1. "
                "Their uploaded copies do not include saved numeric execution output, so values are shown only if you supply your verified results.</div>",
                unsafe_allow_html=True)
    table, error = read_metrics()
    if error:
        st.error(error)
    if table.empty:
        st.info("No verified numerical results have been entered yet. Add the numbers you obtained from your completed notebook runs to `results/model_metrics.csv` (values from 0 to 1).")
    else:
        printable = table.copy()
        for col in ("Accuracy", "Precision", "Recall", "F1 Score"):
            printable[col] = printable[col].map(lambda v: f"{v*100:.2f}%" if pd.notna(v) else "—")
        st.dataframe(printable, hide_index=True, use_container_width=True)
        complete = table.dropna(subset=["F1 Score"]).sort_values("F1 Score", ascending=False)
        if not complete.empty:
            st.markdown("**Weighted F1 comparison (reported values)**")
            st.bar_chart(complete.set_index("Model")[["F1 Score"]], height=340)
        st.caption("These are user-entered, notebook-reported metrics; the app does not recompute test-set results.")
    st.download_button("Download blank metrics CSV template", data=(ROOT / "results" / "model_metrics.csv").read_bytes(),
                       file_name="mango_model_metrics_template.csv", mime="text/csv")

    st.divider()
    st.markdown("<div class='section-title'>Experiment protocol in the supplied notebooks</div>", unsafe_allow_html=True)
    columns = st.columns(4, gap="medium")
    protocol = [("Train / val / test", "70% / 20% / 10%"), ("Input resolution", "224 × 224"),
                ("Batch size", "32"), ("Max. epochs", "50")]
    for col, (name, value) in zip(columns, protocol):
        with col:
            card_stat(name, value)
    st.markdown("<div class='info-note'><b>Evaluation in notebooks:</b> Accuracy, weighted precision, weighted recall, weighted F1. "
                "The five notebooks use separate, unseeded random splits; their results should not be described as a strictly controlled same-test-set comparison without additional validation.</div>",
                unsafe_allow_html=True)


def model_card_tab():
    st.markdown("<div class='section-kicker'>Reproducibility and deployment</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Model card · Bangladeshi mango leaf classification</div>", unsafe_allow_html=True)
    left, right = st.columns([1.05, .95], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("#### Architectures")
            for name, item in MODEL_SPECS.items():
                st.markdown(f"**{name}** · {item.kind}  ")
                st.caption(f"Source: {item.source} · Saved file: `models/{item.filename}`")
        with st.container(border=True):
            st.markdown("#### Input pipeline")
            st.write("RGB conversion → resize to 224 × 224 → ToTensor → normalize each channel with mean 0.5 and standard deviation 0.5.")
            st.caption("Matches the notebooks' deterministic `transform_test` (the original one-value normalization broadcasts over RGB channels).")
        with st.container(border=True):
            st.markdown("#### Explainability")
            st.write("Grad-CAM, Grad-CAM++, Eigen-CAM, Ablation-CAM, and LIME target the model's predicted category.")
            st.caption("Target layers are selected separately for each model, matching the training notebooks.")
    with right:
        with st.container(border=True):
            st.markdown("#### Training setup (as written in notebooks)")
            st.markdown("**Loss:** CrossEntropyLoss  \n**Optimizer:** Adam, lr 0.001  \n**Batch size:** 32  \n**Maximum epochs:** 50  \n**Early stopping patience:** 5  \n**Checkpoint:** final model state saved after loop")
        with st.container(border=True):
            st.markdown("#### Deployment status")
            st.write(f"Runtime device: **{DEVICE.type.upper()}**")
            for name, item in MODEL_SPECS.items():
                p = checkpoint_for(name)
                st.markdown(f"{'✅' if p.is_file() else '⬜'} **{name}** — `{item.filename}`")
        with st.container(border=True):
            st.markdown("#### Class label order")
            st.write("The trained logits follow alphabetical `ImageFolder(...).classes` order from each training dataset.")
            st.code("\n".join(f"{i}: {name}" for i, name in enumerate(class_names())), language=None)
            st.warning("Confirm `class_names.json` matches `dataset.classes` in EVERY notebook before trusting predicted labels. The raw training data/class printouts were not supplied.")
    st.markdown("<div class='warning-note'><b>Reproducibility caveats:</b> The notebooks use unseeded random splits. "
                "The transform on the shared ImageFolder dataset is overwritten by the validation/test transform, "
                "so intended augmentation is not applied in the training subset. Early stopping does not restore the best checkpoint. "
                "The app preserves the saved-model inference behavior instead of silently changing the study.</div>",
                unsafe_allow_html=True)


def analysis_tab():
    left, right = st.columns([1.25, .75], gap="large")
    with left:
        st.markdown("<div class='section-kicker'>01 · Live model inference</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Analyze a mango leaf image</div>", unsafe_allow_html=True)
        chosen_model = st.selectbox("Choose a trained model", list(MODEL_SPECS), key="analysis_model")
        uploaded = st.file_uploader("Upload a mango leaf photograph (JPG, JPEG or PNG)",
                                    type=["jpg", "jpeg", "png"], key="leaf_upload")
    with right:
        st.markdown("<div class='info-note'><b>Research workflow</b><br>1. Select the model<br>2. Upload a mango leaf image<br>3. Inspect class probabilities<br>4. Generate optional XAI explanations and a 300-DPI paper panel</div>",
                    unsafe_allow_html=True)
        render_checkpoint_status()
    ckpt = checkpoint_for(chosen_model)
    if not ckpt.is_file():
        st.markdown(f"<div class='warning-note'><b>{text_safe(chosen_model)} checkpoint not found.</b> "
                    f"Place <code>{text_safe(ckpt.name)}</code> inside this project's <code>models/</code> folder. "
                    "Do not rename a different model's weights or use random replacement weights. Other tabs remain usable.</div>",
                    unsafe_allow_html=True)
    if uploaded is None:
        with st.container(border=True):
            st.markdown("### Ready for analysis")
            st.write("Upload a mango leaf image to preview the exact model input, generate classification scores and inspect visual explanations.")
            st.caption("The app is a research demonstration, not a plant health diagnosis or a guarantee of correct variety identification.")
        return

    image_bytes = uploaded.getvalue()
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        st.error("Image is too large. Please upload an image smaller than 15 MB.")
        return
    try:
        original, view224, tensor, rgb_float = prepare_image(Image.open(io.BytesIO(image_bytes)))
    except (UnidentifiedImageError, ValueError, OSError, Image.DecompressionBombError) as exc:
        st.error(f"Cannot open the uploaded image: {exc}")
        return

    st.markdown("<div class='section-kicker'>02 · Image preview</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("<div class='mini-label'>Uploaded image</div>", unsafe_allow_html=True)
        st.image(original, use_container_width=True)
        st.caption(f"Original image size: {original.width} × {original.height} pixels")
    with col2:
        st.markdown("<div class='mini-label'>Model input preview</div>", unsafe_allow_html=True)
        st.image(view224, use_container_width=True)
        st.caption("224 × 224 RGB · values normalized to [-1, 1] for inference")

    if not ckpt.is_file():
        st.info("Prediction and XAI will be available once the selected model checkpoint is added.")
        return
    try:
        with st.spinner(f"Loading {chosen_model} model checkpoint…"):
            model = cached_model(chosen_model, str(ckpt), ckpt.stat().st_mtime_ns, str(DEVICE))
        probs = predict(model, tensor)
    except Exception as exc:
        st.error(f"Model inference failed: {exc}")
        return

    names = class_names()
    pred_id = int(np.argmax(probs))
    confidence = float(probs[pred_id]) * 100
    pred_name = names[pred_id]

    st.divider()
    st.markdown("<div class='section-kicker'>03 · Classification</div>", unsafe_allow_html=True)
    pred_col, score_col = st.columns([.7, 1.3], gap="large")
    with pred_col:
        st.markdown(f"<div class='pred-box'><div class='pred-note'>Predicted mango variety</div>"
                    f"<div class='pred-name'>{text_safe(pred_name)}</div>"
                    f"<div class='pred-pct'>{confidence:.2f}%</div>"
                    f"<div class='pred-note'>Softmax score · {text_safe(chosen_model)}</div></div>",
                    unsafe_allow_html=True)
        st.caption("Scores are uncalibrated model outputs, not guarantees of classification correctness.")
    with score_col:
        st.markdown("<div class='mini-label'>Six-class probability distribution</div>", unsafe_allow_html=True)
        st.markdown(probability_html(names, probs), unsafe_allow_html=True)
    top3 = np.argsort(-probs)[:3]
    st.caption("Top three: " + " · ".join(f"{names[int(i)]} ({100*float(probs[i]):.1f}%)" for i in top3))

    st.divider()
    st.markdown("<div class='section-kicker'>04 · Explainable AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Where does the model focus?</div>", unsafe_allow_html=True)
    st.write("Choose which explanations to calculate. Grad-CAM is selected by default; Ablation-CAM and LIME can be slower on CPU.")
    methods = st.multiselect("Explanation methods", options=list(ALL_METHODS), default=["Grad-CAM"], key="xai_methods")
    pressed = st.button("Generate selected explanations", type="primary", disabled=not methods)
    img_hash = hashlib.sha256(image_bytes).hexdigest()
    sig = (chosen_model, str(ckpt), ckpt.stat().st_mtime_ns, img_hash, pred_id, tuple(methods))
    if pressed:
        images, errors = {}, {}
        for method in methods:
            try:
                with st.spinner(f"Generating {method}…"):
                    if method in CAM_METHODS:
                        images[method] = cam_explanation(model, chosen_model, tensor, rgb_float, pred_id, method)
                    else:
                        images[method] = lime_explanation(model, view224, tensor, pred_id, samples=100)
            except Exception as exc:
                errors[method] = str(exc)
        st.session_state["xai_session"] = {"key": sig, "images": images, "errors": errors}
    result = st.session_state.get("xai_session")
    xai_images = {}
    if result is not None and result.get("key") == sig:
        xai_images = result["images"]
        for name, message in result.get("errors", {}).items():
            st.warning(f"{name} could not be generated: {message}")
        ordered = [name for name in methods if name in xai_images]
        if ordered:
            st.markdown("<div class='mini-label'>Visual explanation gallery</div>", unsafe_allow_html=True)
            cols = st.columns(min(3, len(ordered)), gap="medium")
            for i, method in enumerate(ordered):
                with cols[i % len(cols)]:
                    st.image(xai_images[method], caption=f"{method} · {pred_name}", use_container_width=True)
    elif methods:
        st.caption("Press the button to generate the selected explanations. No artificial heatmaps are displayed.")

    st.divider()
    st.markdown("<div class='section-kicker'>05 · Figure export</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Paper-ready prediction panel</div>", unsafe_allow_html=True)
    first = next((method for method in methods if method in xai_images), None)
    try:
        fig_bytes = make_paper_figure(original, view224, names, probs, chosen_model,
                                      explanation_name=first,
                                      explanation_image=xai_images[first] if first else None)
        st.download_button("Download 300-DPI PNG study panel", data=fig_bytes,
                           file_name=f"mango_leaf_{chosen_model.lower().replace(' ', '_')}_study_panel.png",
                           mime="image/png", type="primary", use_container_width=True)
        st.caption("Figure includes the uploaded image, model input, six class scores, and the first successfully generated XAI visualization (when available).")
    except Exception as exc:
        st.warning(f"Could not export the paper figure: {exc}")
    st.markdown("<div class='warning-note'><b>Research-use note:</b> Predictions are estimates derived from the dataset. "
                "CAM/LIME heatmaps are model explanations, not externally validated biological ground truth. "
                "Confirm image quality, dataset compatibility and class order before interpreting any output.</div>",
                unsafe_allow_html=True)


# Page frame
st.markdown("""
<div class='hero'>
  <div class='eyebrow'>Deep learning · Explainable AI · Research prototype</div>
  <div class='headline'>🥭 MangoSight Leaf</div>
  <div class='subtitle'>Bangladeshi mango leaf variety classification with five CNN-based architectures and five visual explanation methods — presented in a clean, paper-oriented interface.</div>
  <div class='badge-row'>
    <span class='hero-badge'>6 mango varieties</span>
    <span class='hero-badge'>5 deep learning models</span>
    <span class='hero-badge'>5 XAI methods</span>
    <span class='hero-badge'>224 × 224 RGB input</span>
  </div>
</div>
""", unsafe_allow_html=True)
stat_cols = st.columns(4, gap="medium")
for col, (name, value) in zip(stat_cols, [("Architectures", "05"), ("Output classes", "06"),
                                         ("XAI methods", "05"), ("Image input", "224 × 224")]):
    with col:
        card_stat(name, value)
st.write("")
analysis, results, modelcard = st.tabs(["🥭 Leaf Analysis", "📊 Study Results", "📘 Model Card"])
with analysis:
    analysis_tab()
with results:
    study_results_tab()
with modelcard:
    model_card_tab()

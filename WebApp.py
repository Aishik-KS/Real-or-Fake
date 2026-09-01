"""Local Streamlit frontend for the trained AIGC image detector."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

from RunModel import (
    IMAGE_EXTENSIONS,
    find_checkpoint,
    find_images,
    load_detector,
    output_image_path,
    predict_images,
)


APP_DIR = Path(__file__).resolve().parent
TEST_IMAGES_DIR = APP_DIR / "Test Images"
SUPPORTED_TYPES = sorted(extension.removeprefix(".") for extension in IMAGE_EXTENSIONS)


st.set_page_config(
    page_title="AIGC Image Detector",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        :root {
            --ink: #17211d;
            --muted: #617069;
            --paper: #f4f7f5;
            --panel: #ffffff;
            --line: #dbe4df;
            --accent: #087f6b;
            --accent-soft: #dff3ed;
        }
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 85% 0%, rgba(8,127,107,.10), transparent 28rem),
                var(--paper);
            color: var(--ink);
        }
        [data-testid="stHeader"] { background: transparent; }
        .block-container { max-width: 1120px; padding-top: 2.2rem; padding-bottom: 4rem; }
        h1, h2, h3 { letter-spacing: -0.035em; color: var(--ink); }
        h1 { font-size: clamp(2.2rem, 5vw, 4rem) !important; line-height: 1 !important; }
        .eyebrow {
            color: var(--accent); font-weight: 750; font-size: .75rem;
            letter-spacing: .14em; text-transform: uppercase; margin-bottom: .75rem;
        }
        .lead { color: var(--muted); max-width: 700px; font-size: 1.05rem; margin-bottom: 1.6rem; }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255,255,255,.88); border-color: var(--line) !important;
            border-radius: 18px !important; box-shadow: 0 12px 35px rgba(23,33,29,.05);
        }
        .status-ready, .status-missing {
            display: inline-flex; align-items: center; gap: .5rem; border-radius: 999px;
            padding: .42rem .75rem; font-size: .84rem; font-weight: 650;
        }
        .status-ready { background: var(--accent-soft); color: #076454; }
        .status-missing { background: #fde8e5; color: #a33428; }
        .status-ready::before, .status-missing::before {
            content: ""; width: .48rem; height: .48rem; border-radius: 50%; background: currentColor;
        }
        div.stButton > button[kind="primary"] {
            background: var(--accent); border-color: var(--accent); border-radius: 10px;
            min-height: 2.8rem; font-weight: 700;
        }
        div.stDownloadButton > button { border-radius: 10px; min-height: 2.8rem; font-weight: 700; }
        [data-testid="stMetric"] {
            background: var(--panel); border: 1px solid var(--line); border-radius: 14px;
            padding: .9rem 1rem;
        }
        [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *,
        [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
            color: var(--ink) !important;
            opacity: 1 !important;
        }
        [data-testid="stMetricValue"] { font-weight: 750 !important; }
        [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *,
        [data-testid="stFileUploader"] label, [data-testid="stFileUploader"] label * {
            color: var(--ink) !important;
            opacity: 1 !important;
        }
        [data-testid="stFileUploaderDropzone"] {
            background: #eef4f1 !important;
            border: 1px dashed #9bb7ad !important;
        }
        [data-testid="stFileUploaderDropzone"] * {
            color: var(--ink) !important;
            opacity: 1 !important;
        }
        [data-testid="stFileUploaderDropzone"] button {
            background: #ffffff !important;
            color: var(--ink) !important;
            border: 1px solid #9bb7ad !important;
        }
        [data-testid="stButtonGroup"] button {
            background: #ffffff !important;
            color: var(--ink) !important;
            border-color: #b9cbc4 !important;
            min-height: 2.65rem;
            font-weight: 650;
        }
        [data-testid="stButtonGroup"] button * {
            color: inherit !important;
            opacity: 1 !important;
        }
        [data-testid="stButtonGroup"] button[role="radio"][aria-checked="true"] {
            background: var(--accent) !important;
            color: #ffffff !important;
            border-color: var(--accent) !important;
        }
        [data-testid="stAlert"] * {
            color: #173d34 !important;
            opacity: 1 !important;
        }
        [data-testid="stDownloadButton"] button {
            background: var(--ink) !important;
            color: #ffffff !important;
            border: 1px solid var(--ink) !important;
        }
        [data-testid="stDownloadButton"] button * {
            color: #ffffff !important;
            opacity: 1 !important;
        }
        [data-testid="stDownloadButton"] button:hover {
            background: #26352f !important;
            border-color: #26352f !important;
        }
        .footer-note { color: var(--muted); font-size: .82rem; margin-top: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_runtime(checkpoint_path: str) -> dict[str, Any]:
    """Load the large model once and reuse it across Streamlit reruns."""
    return load_detector(Path(checkpoint_path), device_choice="auto", script_dir=APP_DIR)


def verdict(score: float) -> str:
    """Convert a probability into the same three practical score bands."""
    if score >= 0.65:
        return "AI-generated"
    if score <= 0.35:
        return "Real"
    return "Uncertain"


def save_uploads(uploaded_files: list[Any], directory: Path) -> tuple[list[Path], list[str]]:
    """Copy browser uploads into a temporary inference directory."""
    paths: list[Path] = []
    names: list[str] = []
    for index, uploaded in enumerate(uploaded_files):
        original_name = str(uploaded.name).replace("\\", "/")
        safe_name = Path(original_name).name
        destination = directory / f"{index:05d}_{safe_name}"
        destination.write_bytes(uploaded.getvalue())
        paths.append(destination)
        names.append(original_name)
    return paths, names


def run_detection(
    checkpoint: Path,
    image_paths: list[Path],
    display_names: list[str],
) -> list[dict[str, Any]]:
    """Load/cache the model, show progress, and run the shared predictor."""
    with st.spinner("Loading the model. The first load can take a few minutes..."):
        runtime = get_runtime(str(checkpoint))

    progress = st.progress(0.0, text="Preparing images...")

    def update_progress(index: int, total: int, name: str, probability: float) -> None:
        del probability
        progress.progress(index / total, text=f"Processing {index} of {total}: {Path(name).name}")

    results = predict_images(
        runtime,
        image_paths,
        display_names=display_names,
        progress_callback=update_progress,
    )
    progress.empty()
    st.session_state["runtime_device"] = runtime["device"].type.upper()
    return results


st.markdown('<div class="eyebrow">Local image analysis</div>', unsafe_allow_html=True)
st.title("AIGC image detector")
st.markdown(
    '<p class="lead">Select images, run the trained detector, and download a JSON file containing the AIGC confidence score for every image.</p>',
    unsafe_allow_html=True,
)

try:
    checkpoint_path = find_checkpoint(None, APP_DIR)
    checkpoint_error = None
except (OSError, RuntimeError) as exc:
    checkpoint_path = None
    checkpoint_error = str(exc)

status_col, detail_col = st.columns([1, 2.6], vertical_alignment="center")
with status_col:
    status_class = "status-ready" if checkpoint_path else "status-missing"
    status_text = "Model ready" if checkpoint_path else "Model missing"
    st.markdown(f'<span class="{status_class}">{status_text}</span>', unsafe_allow_html=True)
with detail_col:
    if checkpoint_path:
        st.caption(f"Checkpoint: {checkpoint_path.name} · Device selected automatically")
    else:
        st.caption(checkpoint_error or "Place a .pt checkpoint inside the Model folder.")

st.write("")

with st.container(border=True):
    st.subheader("1. Choose images")
    source = st.segmented_control(
        "Image source",
        options=("Upload files", "Upload folder", "Use Test Images", "Use folder path"),
        default="Upload files",
        selection_mode="single",
        label_visibility="collapsed",
    )

    uploaded_files: list[Any] = []
    folder_argument: Path | None = None

    if source == "Upload files":
        uploaded_files = st.file_uploader(
            "Upload one or more images",
            type=SUPPORTED_TYPES,
            accept_multiple_files=True,
        )
        if uploaded_files:
            st.caption(f"{len(uploaded_files)} image(s) selected")
    elif source == "Upload folder":
        uploaded_files = st.file_uploader(
            "Choose a folder containing images",
            type=SUPPORTED_TYPES,
            accept_multiple_files="directory",
        )
        if uploaded_files:
            st.caption(f"{len(uploaded_files)} image(s) found in the selected folder")
    elif source == "Use Test Images":
        folder_argument = TEST_IMAGES_DIR
        try:
            _, available_images = find_images(TEST_IMAGES_DIR)
            st.info(f"Ready to scan {len(available_images)} image(s) from Test Images.")
        except (OSError, RuntimeError) as exc:
            st.warning(str(exc))
    else:
        entered_path = st.text_input(
            "Folder path",
            placeholder=r"C:\path\to\images",
            help="This path must be accessible on the computer running the app.",
        )
        if entered_path.strip():
            folder_argument = Path(entered_path.strip().strip('"'))

    run_clicked = st.button(
        "Run detection",
        type="primary",
        use_container_width=True,
        disabled=checkpoint_path is None,
    )

if run_clicked and checkpoint_path is not None:
    try:
        if uploaded_files:
            with tempfile.TemporaryDirectory(prefix="aigc-uploads-") as temporary:
                image_paths, display_names = save_uploads(uploaded_files, Path(temporary))
                results = run_detection(checkpoint_path, image_paths, display_names)
        elif folder_argument is not None:
            image_root, image_paths = find_images(folder_argument)
            display_names = [
                output_image_path(path, folder_argument, image_root) for path in image_paths
            ]
            results = run_detection(checkpoint_path, image_paths, display_names)
        else:
            raise RuntimeError("Select at least one image or provide an image folder.")

        st.session_state["results"] = results
        st.session_state["result_source"] = source
        st.success(f"Detection complete — {len(results)} image(s) processed.")
    except (OSError, RuntimeError, ValueError) as exc:
        st.error(str(exc))

results = st.session_state.get("results")
if results:
    st.write("")
    with st.container(border=True):
        st.subheader("2. Results")

        ai_count = sum(result["pred"] >= 0.65 for result in results)
        real_count = sum(result["pred"] <= 0.35 for result in results)
        uncertain_count = len(results) - ai_count - real_count

        metric_columns = st.columns(4)
        metric_columns[0].metric("Images", len(results))
        metric_columns[1].metric("AI-generated", ai_count)
        metric_columns[2].metric("Real", real_count)
        metric_columns[3].metric("Uncertain", uncertain_count)

        table_rows = [
            {
                "Image": result["image_path"],
                "Result": verdict(result["pred"]),
                "AI confidence": result["pred"],
            }
            for result in results
        ]
        st.dataframe(
            table_rows,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Image": st.column_config.TextColumn("Image", width="large"),
                "Result": st.column_config.TextColumn("Result", width="medium"),
                "AI confidence": st.column_config.ProgressColumn(
                    "AI confidence",
                    help="0 means more likely real; 1 means more likely AI-generated.",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.3f",
                ),
            },
        )

        json_output = json.dumps(results, indent=2, ensure_ascii=False) + "\n"
        download_col, device_col = st.columns([1, 2], vertical_alignment="center")
        with download_col:
            st.download_button(
                "Download predictions.json",
                data=json_output,
                file_name="predictions.json",
                mime="application/json",
                use_container_width=True,
            )
        with device_col:
            device_used = st.session_state.get("runtime_device", "AUTO")
            st.caption(f"Processed locally on {device_used}. Files are not sent to an external service.")
else:
    st.info("Results will appear here after detection is complete.")

st.markdown(
    '<p class="footer-note">Scores are model estimates, not proof of an image\'s origin. Treat scores near 0.5 as uncertain.</p>',
    unsafe_allow_html=True,
)

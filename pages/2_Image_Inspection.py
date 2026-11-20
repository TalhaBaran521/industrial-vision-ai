"""Session-isolated PatchCore inspection UI."""

import hashlib
import json
from pathlib import Path

import streamlit as st

from industrial_vision.application.model_inspection import inspect_upload, png_bytes

st.set_page_config(page_title="Image Inspection", layout="wide")
st.title("Image Inspection")
st.caption("PatchCore · Research prototype · Raw anomaly distance")
models = sorted(
    path.parent
    for path in Path("models/patchcore").glob("*/metadata.json")
    if (path.parent / "model.pt").is_file()
)
if not models:
    st.info(
        "Train a bottle model using the CLI to begin inspection. See README for setup."
    )
    st.stop()
selected = st.selectbox("Model", models, format_func=lambda path: path.name)
device = st.selectbox("Device", ["auto", "cpu", "cuda"])
try:
    metadata = json.loads((selected / "metadata.json").read_text(encoding="utf-8"))
    with st.expander("Model information"):
        st.json(metadata)
    st.info(
        "This model was fitted on MVTec AD bottle. Use comparable bottle images. Image threshold comes from normal validation data."
    )
    uploaded = st.file_uploader("Inspection image", type=["png", "jpg", "jpeg"])
    signature = (str(selected), device, (selected / "model.pt").stat().st_mtime_ns)
    data = uploaded.getvalue() if uploaded else None
    result_key = (signature, hashlib.sha256(data).hexdigest() if data else None)
    if st.button("Analyze image", disabled=data is None):
        st.session_state.pop("inspection_result", None)
        with st.spinner("Loading model and analyzing image"):
            cached = st.session_state.get("inspection_model")
            if cached is None or cached[0] != signature:
                from industrial_vision.detectors.patchcore_detector import (
                    PatchCoreDetector,
                )

                st.session_state.pop("inspection_model", None)
                detector, model_metadata = PatchCoreDetector.load(selected, device)
                st.session_state["inspection_model"] = (
                    signature,
                    detector,
                    model_metadata,
                )
            _, detector, model_metadata = st.session_state["inspection_model"]
            result = inspect_upload(data, detector, model_metadata, selected.name)
            st.session_state["inspection_result"] = (result_key, result)
    stored = st.session_state.get("inspection_result")
    if stored is not None and stored[0] == result_key:
        result = stored[1]
        report = result["report"]
        cols = st.columns(4)
        cols[0].metric("Anomaly score", f"{report['image_score']:.4f}")
        cols[1].metric("Image threshold", f"{report['image_threshold']:.4f}")
        cols[2].metric("Decision", "Anomalous" if report["is_anomalous"] else "Normal")
        cols[3].metric("Processing time", f"{report['processing_time_ms']:.0f} ms")
        st.caption(
            f"Device: {report['device']} · Heatmap range: 0–{report['heatmap_range'][1]:.4f} raw distance. Scale is specific to this image."
        )
        for column, name in zip(
            st.columns(3), ["original", "heatmap", "overlay"], strict=True
        ):
            column.image(result[name], caption=name.title())
        st.caption(
            "Pixel threshold is unset. Overlay shows continuous distance intensity."
        )
        st.download_button(
            "Download overlay PNG",
            png_bytes(result["overlay"]),
            "overlay.png",
            "image/png",
        )
        st.download_button(
            "Download report JSON",
            json.dumps(report, indent=2, allow_nan=False),
            "inspection.json",
            "application/json",
        )
        st.download_button(
            "Download raw anomaly map",
            result["map_bytes"],
            "anomaly_map.npy",
            "application/octet-stream",
        )
except (ImportError, OSError, ValueError, RuntimeError, KeyError) as error:
    st.error(f"Inspection unavailable: {error}")

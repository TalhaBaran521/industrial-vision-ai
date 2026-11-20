"""Browse saved evaluation metrics and error examples."""

import json
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Experiment Results", layout="wide")
st.title("Experiment Results")
reports = sorted(Path("outputs/evaluations").glob("*/report.json"), reverse=True)
if not reports:
    st.info("Run the evaluate CLI command to create an experiment report.")
    st.stop()
selected = st.selectbox(
    "Evaluation run", reports, format_func=lambda path: path.parent.name
)
try:
    report = json.loads(selected.read_text(encoding="utf-8"))
    st.caption(
        f"{report['timestamp']} · {report['category']} · {report['device']} · {report['test_count']} test images"
    )
    rows = []
    for level in ["image", "pixel"]:
        for metric, value in report[f"{level}_metrics"].items():
            if metric != "confusion_matrix":
                rows.append(
                    {
                        "Level": level,
                        "Metric": metric,
                        "Value": "N/A" if value is None else f"{value:.6f}",
                    }
                )
    st.dataframe(rows, hide_index=True)
    st.subheader("Image confusion matrix")
    st.caption("Rows: actual normal/anomalous. Columns: predicted normal/anomalous.")
    st.table(report["image_metrics"]["confusion_matrix"])
    st.write(
        "Image threshold",
        report["image_threshold"],
        "Pixel threshold",
        report["pixel_threshold"],
    )
    st.caption(report["pixel_threshold_method"])
    outcome = st.selectbox("Example outcome", ["TP", "FP", "TN", "FN"])
    examples = report["examples"][outcome]
    if not examples:
        st.info("No saved examples for this outcome.")
    for index, source in enumerate(examples):
        st.write(source)
        for column, suffix, label in zip(
            st.columns(3),
            ["original", "ground_truth", "prediction"],
            ["Original", "Ground truth", "Predicted mask"],
            strict=True,
        ):
            path = selected.parent / f"{outcome}_{index}_{suffix}.png"
            if path.is_file():
                column.image(str(path), caption=label)
    with st.expander("Model and experiment metadata"):
        st.json(report)
    for name in ["report.json", "metrics.csv", "predictions.csv"]:
        path = selected.parent / name
        if path.is_file():
            st.download_button(f"Download {name}", path.read_bytes(), name)
except (OSError, ValueError, KeyError, TypeError) as error:
    st.error(f"Cannot read this experiment: {error}")

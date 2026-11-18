"""Inspect a local MVTec AD category and its ground truth."""

from pathlib import Path

import streamlit as st

from industrial_vision.datasets.mvtec import CATEGORIES, DatasetLoader

st.set_page_config(page_title="Dataset Explorer", layout="wide")
st.title("Dataset Explorer")
st.caption("MVTec AD · Ground truth annotations · Research dataset")
root = st.text_input("MVTec AD root", "data/mvtec_ad")
category = st.selectbox("Category", CATEGORIES)
st.markdown(
    "Download the category from the [official MVTec AD page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad), then extract it under the selected root. License: CC BY-NC-SA 4.0."
)
st.code(
    f"{root}/{category}/train/good/\n{root}/{category}/test/good/\n{root}/{category}/test/<defect_type>/\n{root}/{category}/ground_truth/<defect_type>/<image_stem>_mask.png"
)
key = (root, category)
if st.button("Scan and validate"):
    try:
        st.session_state["dataset_scan"] = (
            key,
            DatasetLoader(Path(root), category).scan(),
        )
    except (OSError, ValueError) as error:
        st.error(str(error))
        st.session_state.pop("dataset_scan", None)
stored = st.session_state.get("dataset_scan")
if stored is not None and stored[0] == key:
    index = stored[1]
    st.write("Validated sample counts", index.statistics())
    if index.errors:
        st.warning(
            f"Integrity errors: {len(index.errors)}. Counts include only validated samples."
        )
        st.dataframe({"Error": index.errors})
    else:
        st.success("All discovered samples passed validation.")
    split = st.selectbox("Split", ["train", "test"])
    samples = [sample for sample in index.samples if sample.split == split]
    if samples:
        sample = st.selectbox(
            "Sample",
            samples,
            format_func=lambda item: f"{item.defect_type}/{item.image_path.name}",
        )
        try:
            image, mask = DatasetLoader(Path(root), category).load_sample(sample)
            left, right = st.columns(2)
            left.image(image, caption=f"{sample.split} / {sample.defect_type}")
            right.image(
                mask,
                caption="Ground truth"
                if sample.is_anomalous
                else "Normal sample — empty ground truth",
            )
        except (OSError, ValueError) as error:
            st.error(str(error))
    else:
        st.info("No validated samples in this split.")

#!/usr/bin/env python
"""Streamlit app: interactive contract clause search."""
from pathlib import Path
import streamlit as st

from baseline import BM25Search
from lib_common import load_config, read_contract

st.set_page_config(page_title="Contract Intelligence", layout="wide")
st.title("Contract Intelligence – Clause Search")

st.sidebar.header("Configuration")
config_options = {
    "Final (configs/final.yaml)": "configs/final.yaml",
    "Baseline (configs/baseline.yaml)": "configs/baseline.yaml",
}
selected_label = st.sidebar.selectbox(
    "Profile",
    options=list(config_options.keys()),
    index=0,
)
config_path = config_options[selected_label]


@st.cache_resource
def get_searcher(path: str):
    cfg = load_config(path)
    return BM25Search(cfg), cfg


searcher, cfg = get_searcher(config_path)

question = st.text_input("Question")
default_k = int(cfg.get("k", cfg.get("top_k", 4)))
top_k = st.slider("Top-K", min_value=1, max_value=16, value=default_k)

if st.button("Search") and question:
    results = searcher.search(question, k=top_k)

    if not results:
        st.warning("No relevant clause found")
    else:
        for fp, start, end, score in results:
            text = read_contract(fp, cfg)
            preview = text[start:end][:200].replace("\n", " ")
            st.markdown(f"**doc:** `{fp}`")
            st.text(f"offsets: {start}-{end}  score: {score:.4f}")
            st.text(f"preview: {preview}")
            st.divider()


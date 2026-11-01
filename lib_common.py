"""Shared utilities: data loading, subsetting, splitting, chunking."""
import hashlib
import json
import random
from pathlib import Path

import yaml


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_benchmark(cfg=None):
    cfg = cfg or load_config()
    with open(cfg["benchmark_path"], "r", encoding="utf-8") as f:
        return json.load(f)["tests"]


def subset_tests(tests, max_n=194):
    """Sort by random.random() seeded on first snippet file_path, keep first max_n."""
    if len(tests) <= max_n:
        return tests
    seed_val = tests[0]["snippets"][0]["file_path"]
    random.seed(seed_val)
    indexed = list(enumerate(range(len(tests))))
    keys = [random.random() for _ in tests]
    order = sorted(range(len(tests)), key=lambda i: keys[i])
    return [tests[i] for i in order[:max_n]]


def split_label(query: str) -> str:
    """sha256 of stripped query -> 'test' if %10<3 else 'train'."""
    h = hashlib.sha256(query.strip().encode()).digest()
    val = int.from_bytes(h, "big")
    return "test" if val % 10 < 3 else "train"


def contracts_used(tests, cfg=None):
    """Return set of file_paths referenced by tests."""
    return {s["file_path"] for t in tests for s in t["snippets"]}


def read_contract(file_path, cfg=None):
    """Read a contract text file. file_path is relative to data_dir (e.g. 'cuad/...')."""
    cfg = cfg or load_config()
    full = Path(cfg["data_dir"]) / file_path
    with open(full, "r", encoding="utf-8") as f:
        return f.read()


def make_chunks(text, chunk_size, chunk_overlap):
    """Fixed-size character chunks with char offsets. Returns list of (start, end, chunk_text)."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append((start, end, text[start:end]))
        if end == len(text):
            break
        start += chunk_size - chunk_overlap
    return chunks


def build_index(tests, cfg=None):
    """Build chunk index for all contracts referenced by tests.

    Returns:
        chunks: list of (file_path, char_start, char_end, text)
        documents: dict file_path -> full text
    """
    cfg = cfg or load_config()
    fps = contracts_used(tests, cfg)
    documents = {}
    chunks = []
    for fp in sorted(fps):
        text = read_contract(fp, cfg)
        documents[fp] = text
        for start, end, chunk_text in make_chunks(
            text, cfg["chunk_size"], cfg["chunk_overlap"]
        ):
            chunks.append((fp, start, end, chunk_text))
    return chunks, documents


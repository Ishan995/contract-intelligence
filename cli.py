#!/usr/bin/env python
"""CLI: ask a question against the contract corpus."""
import argparse
from pathlib import Path

from baseline import BM25Search
from lib_common import load_config, read_contract


def main():
    parser = argparse.ArgumentParser(description="Query contract corpus via BM25")
    parser.add_argument("question", help="Natural-language question")
    parser.add_argument("--top-k", type=int, default=None, help="Number of results (default from config or 4)")
    parser.add_argument(
        "--config",
        default="configs/final.yaml" if Path("configs/final.yaml").exists() else "config.yaml",
        help="Path to YAML configuration (default: configs/final.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    searcher = BM25Search(cfg)
    top_k = args.top_k if args.top_k is not None else int(cfg.get("k", cfg.get("top_k", 4)))
    results = searcher.search(args.question, k=top_k)

    if not results:
        print("No relevant clause found")
        return

    for fp, start, end, score in results:
        text = read_contract(fp, cfg)
        preview = text[start:end][:200].replace("\n", " ")
        print(f"doc: {fp}")
        print(f"  offsets: {start}-{end}  score: {score:.4f}")
        print(f"  preview: {preview}")
        print()


if __name__ == "__main__":
    main()



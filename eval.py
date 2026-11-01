#!/usr/bin/env python
"""Evaluation: char-level precision/recall at k=1,2,4,8,16 + abstention rates."""
import argparse
import datetime
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from tabulate import tabulate

from baseline import BM25Search
from lib_common import load_config, split_label


def char_overlap(pred_spans, gold_spans):
    """Compute char-level precision and recall between predicted and gold spans.

    Each span is (file_path, start, end).
    Uses interval intersection for efficiency.
    Returns (precision, recall).
    """
    pred_total = sum(e - s for _, s, e in pred_spans)
    gold_total = sum(e - s for _, s, e in gold_spans)

    if pred_total == 0 and gold_total == 0:
        return 1.0, 1.0
    if pred_total == 0:
        return 1.0, 0.0  # abstained; precision is vacuously 1
    if gold_total == 0:
        return 0.0, 1.0  # no gold; recall is vacuously 1

    pred_by_fp = defaultdict(list)
    gold_by_fp = defaultdict(list)
    for fp, s, e in pred_spans:
        pred_by_fp[fp].append((s, e))
    for fp, s, e in gold_spans:
        gold_by_fp[fp].append((s, e))

    overlap_total = 0
    for fp in pred_by_fp:
        if fp not in gold_by_fp:
            continue
        for ps, pe in pred_by_fp[fp]:
            for gs, ge in gold_by_fp[fp]:
                ov_start = max(ps, gs)
                ov_end = min(pe, ge)
                if ov_start < ov_end:
                    overlap_total += ov_end - ov_start

    precision = overlap_total / pred_total
    recall = overlap_total / gold_total
    return precision, recall


def get_next_run_id(log_path="test_runs.log"):
    if not os.path.exists(log_path):
        return 1
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()
    matches = [int(m) for m in re.findall(r"Run(?:\s*counter)?:\s*(\d+)", content, re.IGNORECASE)]
    if matches:
        return max(matches) + 1
    test_blocks = len(re.findall(r"Split:\s*test", content, re.IGNORECASE))
    return test_blocks + 1


def evaluate(split_name: str, config_path: str = None, log_to_file: bool = True):
    if config_path:
        cfg = load_config(config_path)
    elif Path("configs/final.yaml").exists():
        cfg = load_config("configs/final.yaml")
    else:
        cfg = load_config("config.yaml")

    searcher = BM25Search(cfg)

    # Load unanswerable
    with open(cfg["unanswerable_path"], "r", encoding="utf-8") as f:
        unanswerable = json.load(f)

    # Filter tests and unanswerable by split
    tests = [t for t in searcher.tests if split_label(t["query"]) == split_name]
    unans = [u for u in unanswerable if u["split"] == split_name]

    ks = [1, 2, 4, 8, 16]

    # Pre-compute scores once per query
    test_scores = [(t, searcher.score_query(t["query"])) for t in tests]
    unans_scores = [(u, searcher.score_query(u["query"])) for u in unans]

    rows = []
    for k in ks:
        precisions = []
        recalls = []
        real_abstain = 0

        for t, scores in test_scores:
            results = searcher.search(t["query"], k=k, scores=scores)
            pred_spans = [(fp, s, e) for fp, s, e, _sc in results]
            gold_spans = [
                (s["file_path"], s["span"][0], s["span"][1]) for s in t["snippets"]
            ]
            p, r = char_overlap(pred_spans, gold_spans)
            precisions.append(p)
            recalls.append(r)
            if not results:
                real_abstain += 1

        avg_p = sum(precisions) / len(precisions) if precisions else 0.0
        avg_r = sum(recalls) / len(recalls) if recalls else 0.0
        real_abstain_rate = real_abstain / len(tests) if tests else 0.0

        # Abstention on unanswerable (correct abstain)
        unans_abstain = sum(
            1 for u, scores in unans_scores
            if not searcher.search(u["query"], k=k, scores=scores)
        )
        unans_abstain_rate = unans_abstain / len(unans) if unans else 0.0

        rows.append([
            k,
            f"{avg_p:.4f}",
            f"{avg_r:.4f}",
            f"{unans_abstain_rate:.4f}",
            f"{real_abstain_rate:.4f}",
        ])

    headers = ["k", "Precision", "Recall", "Abstain(unans)", "Abstain(real)"]
    table = tabulate(rows, headers=headers, tablefmt="simple")
    print(f"\n=== {split_name.upper()} split ({len(tests)} queries, {len(unans)} unanswerable) ===")
    print(table)

    if log_to_file and split_name == "test":
        run_counter = get_next_run_id("test_runs.log")
        with open("test_runs.log", "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Run: {run_counter}\n")
            f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
            f.write(f"Split: {split_name}\n")
            f.write(f"Queries: {len(tests)}, Unanswerable: {len(unans)}\n")
            f.write(f"Config: chunk_size={cfg['chunk_size']}, overlap={cfg['chunk_overlap']}, threshold={cfg['bm25_abstain_threshold']}\n")
            f.write(table + "\n")

    return rows, len(tests), len(unans)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["train", "test"])
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--no-log", action="store_true", help="Do not write to test_runs.log")
    args = parser.parse_args()
    evaluate(args.split, config_path=args.config, log_to_file=not args.no_log)


if __name__ == "__main__":
    main()

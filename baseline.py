#!/usr/bin/env python
"""BM25 baseline search over CUAD contract chunks."""
import numpy as np
from rank_bm25 import BM25Okapi
from lib_common import (
    load_config,
    load_benchmark,
    subset_tests,
    build_index,
)


class BM25Search:
    """Fixed-size chunk BM25 search with abstention."""

    def __init__(self, cfg=None):
        self.cfg = cfg or load_config()
        self.tests = subset_tests(load_benchmark(self.cfg), self.cfg["subset_size"])
        self.chunks, self.documents = build_index(self.tests, self.cfg)
        tokenized = [c[3].lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)
        self.threshold = self.cfg["bm25_abstain_threshold"]

    def score_query(self, query: str):
        """Return raw BM25 scores array for a query (cached-friendly entry point)."""
        tokens = query.lower().split()
        return self.bm25.get_scores(tokens)

    def search(self, query: str, k: int = 16, scores=None):
        """Return top-k results as list of (file_path, start, end, score).

        Pass pre-computed scores to avoid re-scoring.
        Returns empty list if top score < threshold (abstain).
        """
        if scores is None:
            scores = self.score_query(query)
        top_indices = np.argsort(scores)[::-1][:k]

        if scores[top_indices[0]] < self.threshold:
            return []  # abstain

        results = []
        for idx in top_indices:
            fp, start, end, _text = self.chunks[idx]
            results.append((fp, start, end, float(scores[idx])))
        return results


if __name__ == "__main__":
    searcher = BM25Search()
    print(f"Indexed {len(searcher.chunks)} chunks from {len(searcher.documents)} contracts")
    q = searcher.tests[0]["query"]
    results = searcher.search(q, k=4)
    print(f"\nQuery: {q[:80]}...")
    for fp, s, e, score in results:
        print(f"  [{score:.2f}] {fp}  chars {s}-{e}")

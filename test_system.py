"""Tests for chunk offsets, split function, and metrics."""
import hashlib

import pytest

from lib_common import make_chunks, split_label
from eval import char_overlap


# ── Chunk offset tests ──


class TestChunkOffsets:
    def test_chunks_cover_full_text(self):
        text = "A" * 1000
        chunks = make_chunks(text, chunk_size=200, chunk_overlap=50)
        covered = set()
        for start, end, _ in chunks:
            covered.update(range(start, end))
        assert covered == set(range(1000))

    def test_chunk_text_matches_offsets(self):
        text = "abcdefghijklmnopqrstuvwxyz" * 40  # 1040 chars
        chunks = make_chunks(text, chunk_size=100, chunk_overlap=20)
        for start, end, chunk_text in chunks:
            assert text[start:end] == chunk_text

    def test_no_negative_starts(self):
        text = "hello world"
        chunks = make_chunks(text, chunk_size=5, chunk_overlap=2)
        for start, end, _ in chunks:
            assert start >= 0
            assert end <= len(text)
            assert start < end

    def test_single_chunk_small_text(self):
        text = "short"
        chunks = make_chunks(text, chunk_size=512, chunk_overlap=128)
        assert len(chunks) == 1
        assert chunks[0] == (0, 5, "short")

    def test_overlap_correct(self):
        text = "A" * 100
        chunks = make_chunks(text, chunk_size=30, chunk_overlap=10)
        for i in range(len(chunks) - 1):
            # The start of the next chunk should be chunk_size - overlap after current start
            expected_next_start = chunks[i][0] + 30 - 10
            assert chunks[i + 1][0] == expected_next_start


# ── Split function tests ──


class TestSplitLabel:
    def test_deterministic(self):
        assert split_label("some query") == split_label("some query")

    def test_strips_whitespace(self):
        assert split_label("  hello  ") == split_label("hello")

    def test_returns_valid_labels(self):
        for q in ["a", "b", "contract law", "indemnification clause"]:
            assert split_label(q) in ("train", "test")

    def test_distribution_roughly_correct(self):
        """With many queries, ~30% should be test."""
        labels = [split_label(f"query_{i}") for i in range(1000)]
        test_frac = labels.count("test") / len(labels)
        assert 0.2 < test_frac < 0.4  # roughly 30%

    def test_sha256_matches_manual(self):
        q = "test query"
        h = hashlib.sha256(q.encode()).digest()
        val = int.from_bytes(h, "big")
        expected = "test" if val % 10 < 3 else "train"
        assert split_label(q) == expected


# ── Metric tests ──


class TestCharOverlap:
    def test_perfect_match(self):
        gold = [("f.txt", 0, 10)]
        pred = [("f.txt", 0, 10)]
        p, r = char_overlap(pred, gold)
        assert p == 1.0 and r == 1.0

    def test_no_overlap(self):
        gold = [("f.txt", 0, 10)]
        pred = [("f.txt", 20, 30)]
        p, r = char_overlap(pred, gold)
        assert p == 0.0 and r == 0.0

    def test_partial_overlap(self):
        gold = [("f.txt", 0, 10)]
        pred = [("f.txt", 5, 15)]
        p, r = char_overlap(pred, gold)
        assert p == 0.5  # 5 overlap / 10 predicted
        assert r == 0.5  # 5 overlap / 10 gold

    def test_empty_both(self):
        p, r = char_overlap([], [])
        assert p == 1.0 and r == 1.0

    def test_abstain_on_real(self):
        """Empty pred, non-empty gold -> precision=1 (vacuous), recall=0."""
        p, r = char_overlap([], [("f.txt", 0, 10)])
        assert p == 1.0 and r == 0.0

    def test_different_files_no_overlap(self):
        gold = [("a.txt", 0, 10)]
        pred = [("b.txt", 0, 10)]
        p, r = char_overlap(pred, gold)
        assert p == 0.0 and r == 0.0

    def test_multiple_spans(self):
        gold = [("f.txt", 0, 10), ("f.txt", 20, 30)]
        pred = [("f.txt", 5, 25)]
        p, r = char_overlap(pred, gold)
        # overlap: 5-10 (5 chars) + 20-25 (5 chars) = 10
        # pred chars: 20, gold chars: 20
        assert p == 0.5 and r == 0.5


# ── Abstention logic tests ──


class TestAbstentionLogic:
    def test_bm25_abstains_below_threshold(self):
        """When top score is below threshold, search returns empty list (abstains)."""
        from baseline import BM25Search

        cfg = {
            "chunk_size": 200,
            "chunk_overlap": 50,
            "bm25_k1": 1.5,
            "bm25_b": 0.75,
            "bm25_abstain_threshold": 1000.0,
            "subset_size": 194,
            "benchmark_path": "data/benchmarks/cuad.json",
            "data_dir": "data/corpus",
            "unanswerable_path": "queries/unanswerable.json",
        }
        searcher = BM25Search(cfg)
        res = searcher.search("query unlikely to score above 1000")
        assert res == []

    def test_bm25_returns_results_above_threshold(self):
        """When top score exceeds threshold, search returns ranked passages with offsets."""
        from baseline import BM25Search

        cfg = {
            "chunk_size": 512,
            "chunk_overlap": 128,
            "bm25_k1": 1.5,
            "bm25_b": 0.75,
            "bm25_abstain_threshold": 0.0,
            "subset_size": 194,
            "benchmark_path": "data/benchmarks/cuad.json",
            "data_dir": "data/corpus",
            "unanswerable_path": "queries/unanswerable.json",
        }
        searcher = BM25Search(cfg)
        res = searcher.search("agreement", k=4)
        assert len(res) == 4
        for fp, start, end, score in res:
            assert score >= 0.0
            assert start < end



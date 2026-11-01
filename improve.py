#!/usr/bin/env python
"""
Automated retrieval-quality improvement loop.

One command:  python improve.py
  - Evaluates on TRAIN split only (never touches test).
  - Classifies failures into categories.
  - Proposes next config change from top failure category.
  - Accepts only if recall@8 improves AND precision@1 / abstention accuracy
    drop < 2 percentage points.
  - Logs every iteration to experiments/log.md.
  - Saves configs/baseline.yaml and configs/final.yaml.
"""
import copy, datetime, hashlib, json, math, os, pickle, re, sys, shutil
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import yaml
from rank_bm25 import BM25Okapi

# ── lazy imports for dense retrieval ──
_st_model = None

def _get_st_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer(model_name)
    return _st_model

# ─────────────────────── data helpers (inline, no import of baseline) ──────
def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_config(cfg, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

def load_benchmark(cfg):
    with open(cfg["benchmark_path"], "r", encoding="utf-8") as f:
        return json.load(f)["tests"]

def split_label(query):
    h = hashlib.sha256(query.strip().encode()).digest()
    return "test" if int.from_bytes(h, "big") % 10 < 3 else "train"

def subset_tests(tests, max_n=194):
    import random as _rand
    if len(tests) <= max_n:
        return tests
    seed_val = tests[0]["snippets"][0]["file_path"]
    _rand.seed(seed_val)
    keys = [_rand.random() for _ in tests]
    order = sorted(range(len(tests)), key=lambda i: keys[i])
    return [tests[i] for i in order[:max_n]]

def read_contract(file_path, cfg):
    full = Path(cfg["data_dir"]) / file_path
    with open(full, "r", encoding="utf-8") as f:
        return f.read()

def contracts_used(tests):
    return {s["file_path"] for t in tests for s in t["snippets"]}

# ─────────────────────── chunking ──────────────────────────────────────────
CLAUSE_RE = re.compile(
    r"(?:^|\n)"
    r"(?="
    r"(?:\d+\.[\d.]*\s)"           # numbered headings: "1. ", "2.3 "
    r"|(?:Section\s+\d)"           # "Section 1"
    r"|(?:ARTICLE\s+[IVXLC\d])"   # "ARTICLE IV"
    r"|(?:[A-Z][A-Z ]{3,}:)"      # "ALL-CAPS HEADING:"
    r"|(?:[A-Z][A-Z ]{3,}\n)"     # "ALL-CAPS HEADING\n"
    r")"
)

def make_chunks_fixed(text, chunk_size, overlap):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append((start, end, text[start:end]))
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks

def make_chunks_clause(text, chunk_size, overlap):
    """Split on clause/section boundaries, then sub-chunk large pieces."""
    boundaries = [m.start() for m in CLAUSE_RE.finditer(text)]
    if not boundaries or boundaries[0] != 0:
        boundaries.insert(0, 0)
    boundaries.append(len(text))

    raw_sections = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        if e > s:
            raw_sections.append((s, e, text[s:e]))

    # sub-chunk sections that are too big
    chunks = []
    for s, e, sec_text in raw_sections:
        if len(sec_text) <= chunk_size:
            chunks.append((s, e, sec_text))
        else:
            for sub in make_chunks_fixed(sec_text, chunk_size, overlap):
                chunks.append((s + sub[0], s + sub[1], sub[2]))
    return chunks

def make_chunks_dispatch(text, cfg):
    mode = cfg.get("chunking_mode", "fixed")
    cs = cfg["chunk_size"]
    co = cfg["chunk_overlap"]
    if mode == "clause":
        return make_chunks_clause(text, cs, co)
    return make_chunks_fixed(text, cs, co)

# ─────────────────────── indexing + caching ────────────────────────────────
def _cache_key(cfg):
    """Deterministic key from chunking-relevant config fields."""
    parts = f"{cfg['chunk_size']}_{cfg['chunk_overlap']}_{cfg.get('chunking_mode','fixed')}"
    return hashlib.md5(parts.encode()).hexdigest()[:12]

def build_index(tests, cfg):
    cache_dir = Path(cfg.get("cache_dir", "cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    ck = _cache_key(cfg)
    cache_path = cache_dir / f"chunks_{ck}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    fps = contracts_used(tests)
    documents, chunks = {}, []
    for fp in sorted(fps):
        text = read_contract(fp, cfg)
        documents[fp] = text
        for start, end, ct in make_chunks_dispatch(text, cfg):
            chunks.append((fp, start, end, ct))

    with open(cache_path, "wb") as f:
        pickle.dump((chunks, documents), f)
    return chunks, documents

# ─────────────────────── embeddings (cached) ───────────────────────────────
def _emb_cache_path(cfg):
    cache_dir = Path(cfg.get("cache_dir", "cache"))
    ck = _cache_key(cfg)
    return cache_dir / f"embeddings_{ck}.npy"

def compute_embeddings(chunks, cfg):
    ep = _emb_cache_path(cfg)
    if ep.exists():
        return np.load(str(ep))
    model = _get_st_model(cfg.get("dense_model", "sentence-transformers/all-MiniLM-L6-v2"))
    texts = [c[3] for c in chunks]
    embs = model.encode(texts, show_progress_bar=True, batch_size=256,
                        normalize_embeddings=True)
    embs = np.array(embs, dtype=np.float32)
    np.save(str(ep), embs)
    return embs

# ─────────────────────── searcher ──────────────────────────────────────────
class ConfigurableSearcher:
    def __init__(self, cfg, tests):
        self.cfg = cfg
        self.chunks, self.documents = build_index(tests, cfg)
        tokenized = [c[3].lower().split() for c in self.chunks]
        k1 = cfg.get("bm25_k1", 1.5)
        b = cfg.get("bm25_b", 0.75)
        self.bm25 = BM25Okapi(tokenized, k1=k1, b=b)
        self.mode = cfg.get("retrieval_mode", "bm25")
        self.embeddings = None
        if self.mode in ("dense", "hybrid"):
            self.embeddings = compute_embeddings(self.chunks, cfg)

    def _bm25_scores(self, query):
        return self.bm25.get_scores(query.lower().split())

    def _dense_scores(self, query):
        model = _get_st_model(self.cfg.get("dense_model",
                              "sentence-transformers/all-MiniLM-L6-v2"))
        qe = model.encode([query], normalize_embeddings=True)
        return (self.embeddings @ qe.T).flatten()

    def score_query(self, query):
        mode = self.mode
        if mode == "bm25":
            return self._bm25_scores(query)
        elif mode == "dense":
            return self._dense_scores(query)
        else:  # hybrid
            alpha = self.cfg.get("hybrid_alpha", 0.5)
            bm25 = self._bm25_scores(query)
            dense = self._dense_scores(query)
            # min-max normalise each
            def _norm(a):
                mn, mx = a.min(), a.max()
                return (a - mn) / (mx - mn + 1e-9)
            return alpha * _norm(bm25) + (1 - alpha) * _norm(dense)

    def _threshold(self):
        mode = self.mode
        if mode == "bm25":
            return self.cfg.get("bm25_abstain_threshold", 5.0)
        elif mode == "dense":
            return self.cfg.get("dense_abstain_threshold", 0.25)
        else:
            return self.cfg.get("hybrid_abstain_threshold", 3.0)

    def search(self, query, k=16, scores=None):
        if scores is None:
            scores = self.score_query(query)
        top_idx = np.argsort(scores)[::-1][:k]
        if scores[top_idx[0]] < self._threshold():
            return []
        return [(self.chunks[i][0], self.chunks[i][1], self.chunks[i][2],
                 float(scores[i])) for i in top_idx]

# ─────────────────────── metrics ───────────────────────────────────────────
def char_overlap(pred_spans, gold_spans):
    pred_total = sum(e - s for _, s, e in pred_spans)
    gold_total = sum(e - s for _, s, e in gold_spans)
    if pred_total == 0 and gold_total == 0:
        return 1.0, 1.0
    if pred_total == 0:
        return 1.0, 0.0
    if gold_total == 0:
        return 0.0, 1.0
    pred_by_fp = defaultdict(list)
    gold_by_fp = defaultdict(list)
    for fp, s, e in pred_spans:
        pred_by_fp[fp].append((s, e))
    for fp, s, e in gold_spans:
        gold_by_fp[fp].append((s, e))
    overlap = 0
    for fp in pred_by_fp:
        if fp not in gold_by_fp:
            continue
        for ps, pe in pred_by_fp[fp]:
            for gs, ge in gold_by_fp[fp]:
                os_ = max(ps, gs)
                oe = min(pe, ge)
                if os_ < oe:
                    overlap += oe - os_
    return overlap / pred_total, overlap / gold_total

# ─────────────────────── failure classification ───────────────────────────
def classify_failure(results, gold_spans, is_unanswerable=False):
    """Return failure category string or None if correct enough.

    Categories:
      wrong_doc        – top-1 result is from wrong document
      right_doc_wrong_section – right doc but zero char overlap at k=8
      partial_span     – some overlap but recall < 0.3
      wrong_abstain    – system abstained but should have returned results
      missed_abstain   – system returned results for unanswerable query
    """
    if is_unanswerable:
        if results:  # should have abstained
            return "missed_abstain"
        return None  # correctly abstained

    if not results:
        return "wrong_abstain"  # abstained on answerable

    gold_fps = {fp for fp, _, _ in gold_spans}
    top1_fp = results[0][0]

    if top1_fp not in gold_fps:
        return "wrong_doc"

    # check overlap at k=8
    pred_spans_k8 = [(fp, s, e) for fp, s, e, _ in results[:8]]
    _, recall = char_overlap(pred_spans_k8, gold_spans)
    if recall == 0:
        return "right_doc_wrong_section"
    if recall < 0.3:
        return "partial_span"
    return None  # good enough

# ─────────────────────── evaluate config ──────────────────────────────────
def evaluate_config(cfg):
    """Return dict of metrics + failure counts.  TRAIN split only."""
    tests_all = subset_tests(load_benchmark(cfg), cfg.get("subset_size", 194))
    tests = [t for t in tests_all if split_label(t["query"]) == "train"]

    with open(cfg["unanswerable_path"], "r", encoding="utf-8") as f:
        unanswerable = [u for u in json.load(f) if u["split"] == "train"]

    searcher = ConfigurableSearcher(cfg, tests_all)

    # pre-compute scores
    test_scores = [(t, searcher.score_query(t["query"])) for t in tests]
    unans_scores = [(u, searcher.score_query(u["query"])) for u in unanswerable]

    ks = [1, 2, 4, 8, 16]
    metrics = {}
    for k in ks:
        precs, recs = [], []
        for t, scores in test_scores:
            res = searcher.search(t["query"], k=k, scores=scores)
            pred = [(fp, s, e) for fp, s, e, _ in res]
            gold = [(s["file_path"], s["span"][0], s["span"][1]) for s in t["snippets"]]
            p, r = char_overlap(pred, gold)
            precs.append(p)
            recs.append(r)
        metrics[f"precision@{k}"] = sum(precs) / len(precs) if precs else 0
        metrics[f"recall@{k}"] = sum(recs) / len(recs) if recs else 0

    # abstention accuracy on unanswerable
    correct_abstain = sum(1 for u, sc in unans_scores
                         if not searcher.search(u["query"], k=8, scores=sc))
    metrics["abstain_accuracy"] = correct_abstain / len(unanswerable) if unanswerable else 1.0

    # real abstain rate (answerable queries)
    real_abstain = sum(1 for t, sc in test_scores
                       if not searcher.search(t["query"], k=8, scores=sc))
    metrics["real_abstain_rate"] = real_abstain / len(tests) if tests else 0

    # failure classification at k=8
    fail_counts = defaultdict(int)
    for t, sc in test_scores:
        res = searcher.search(t["query"], k=8, scores=sc)
        gold = [(s["file_path"], s["span"][0], s["span"][1]) for s in t["snippets"]]
        cat = classify_failure(res, gold, is_unanswerable=False)
        if cat:
            fail_counts[cat] += 1
    for u, sc in unans_scores:
        cat = classify_failure(
            searcher.search(u["query"], k=8, scores=sc), [], is_unanswerable=True)
        if cat:
            fail_counts[cat] += 1

    metrics["failures"] = dict(fail_counts)
    metrics["n_train"] = len(tests)
    metrics["n_unans_train"] = len(unanswerable)
    return metrics

# ─────────────────────── proposal rules ───────────────────────────────────
# Each rule: (applicable_failure_category, function(cfg) -> list[cfg_variants])

def _variants_chunk_size(cfg):
    out = []
    for cs, co in [(256, 64), (384, 96), (768, 192), (1024, 256)]:
        if cs == cfg["chunk_size"]:
            continue
        c = copy.deepcopy(cfg)
        c["chunk_size"] = cs
        c["chunk_overlap"] = co
        out.append((c, f"chunk_size={cs}, overlap={co}"))
    return out

def _variants_clause_chunking(cfg):
    if cfg.get("chunking_mode") == "clause":
        return []
    c = copy.deepcopy(cfg)
    c["chunking_mode"] = "clause"
    return [(c, "clause-aware chunking")]

def _variants_bm25_params(cfg):
    out = []
    for k1 in [0.5, 1.0, 1.2, 1.5, 2.0]:
        for b in [0.3, 0.5, 0.75, 1.0]:
            if k1 == cfg.get("bm25_k1", 1.5) and b == cfg.get("bm25_b", 0.75):
                continue
            c = copy.deepcopy(cfg)
            c["bm25_k1"] = k1
            c["bm25_b"] = b
            out.append((c, f"bm25 k1={k1}, b={b}"))
    return out

def _variants_dense(cfg):
    if cfg.get("retrieval_mode") == "dense":
        return []
    c = copy.deepcopy(cfg)
    c["retrieval_mode"] = "dense"
    return [(c, "dense retrieval (all-MiniLM-L6-v2)")]

def _variants_hybrid(cfg):
    out = []
    for alpha in [0.3, 0.4, 0.5, 0.6, 0.7]:
        c = copy.deepcopy(cfg)
        c["retrieval_mode"] = "hybrid"
        c["hybrid_alpha"] = alpha
        out.append((c, f"hybrid BM25+dense alpha={alpha}"))
    return out

def _sweep_threshold(cfg):
    """Find threshold on TRAIN only that maximizes abstain_accuracy - real_abstain_rate."""
    tests_all = subset_tests(load_benchmark(cfg), cfg.get("subset_size", 194))
    train_tests = [t for t in tests_all if split_label(t["query"]) == "train"]
    with open(cfg["unanswerable_path"], "r", encoding="utf-8") as f:
        train_unans = [u for u in json.load(f) if u["split"] == "train"]

    searcher = ConfigurableSearcher(cfg, tests_all)
    train_scores = np.array([float(np.max(searcher.score_query(t["query"]))) for t in train_tests])
    unans_scores = np.array([float(np.max(searcher.score_query(u["query"]))) for u in train_unans])

    mode = cfg.get("retrieval_mode", "bm25")
    key = {"bm25": "bm25_abstain_threshold",
           "dense": "dense_abstain_threshold",
           "hybrid": "hybrid_abstain_threshold"}[mode]

    all_scores = np.sort(np.unique(np.concatenate([train_scores, unans_scores])))
    candidates = (all_scores[:-1] + all_scores[1:]) / 2.0
    candidates = np.concatenate([[all_scores[0] - 0.1], candidates, [all_scores[-1] + 0.1]])

    best_obj = -999.0
    best_th = None
    best_abs_acc = 0.0
    best_real_abs = 0.0

    for th in candidates:
        abs_acc = float(np.mean(unans_scores < th))
        real_abs = float(np.mean(train_scores < th))
        obj = abs_acc - real_abs
        if obj > best_obj or (np.isclose(obj, best_obj) and real_abs < best_real_abs):
            best_obj = obj
            best_th = th
            best_abs_acc = abs_acc
            best_real_abs = real_abs

    out = []
    if best_th is not None:
        best_th_val = round(float(best_th), 2)
        if best_th_val != cfg.get(key):
            c = copy.deepcopy(cfg)
            c[key] = best_th_val
            out.append((c, f"threshold sweep ({key}={best_th_val}, obj={best_obj:.4f}, unans_abs={best_abs_acc:.4f}, real_abs={best_real_abs:.4f})"))
    return out

def _variants_threshold(cfg):
    out = _sweep_threshold(cfg)
    if out:
        return out
    mode = cfg.get("retrieval_mode", "bm25")
    key = {"bm25": "bm25_abstain_threshold",
           "dense": "dense_abstain_threshold",
           "hybrid": "hybrid_abstain_threshold"}[mode]
    cur = cfg.get(key, 5.0)
    for delta in [-5.0, -2.0, -1.0, 1.0, 2.0, 5.0]:
        nv = round(cur + delta, 2)
        if nv <= 0:
            continue
        c = copy.deepcopy(cfg)
        c[key] = nv
        out.append((c, f"{key}={nv}"))
    return out

# Map failure categories → ordered list of proposer functions
PROPOSER_MAP = {
    "wrong_doc":               [_variants_hybrid, _variants_dense, _variants_chunk_size, _variants_bm25_params],
    "right_doc_wrong_section": [_variants_clause_chunking, _variants_chunk_size, _variants_bm25_params],
    "partial_span":            [_variants_chunk_size, _variants_clause_chunking, _variants_bm25_params],
    "wrong_abstain":           [_sweep_threshold, _variants_threshold, _variants_bm25_params, _variants_hybrid],
    "missed_abstain":          [_sweep_threshold, _variants_threshold, _variants_hybrid, _variants_dense],
}

def propose_changes(cfg, failures):
    """Yield (candidate_cfg, description, proposer_name) from failure categories."""
    if failures.get("missed_abstain", 0) > 0 or failures.get("wrong_abstain", 0) > 0:
        for c, desc in _sweep_threshold(cfg):
            yield c, desc, "abstention_failure -> _sweep_threshold"

    if not failures:
        yield from _expand("(no failures)", _sweep_threshold, cfg)
        yield from _expand("(no failures)", _variants_hybrid, cfg)
        return

    sorted_cats = sorted(failures.keys(), key=lambda k: failures[k], reverse=True)
    for cat in sorted_cats:
        proposers = PROPOSER_MAP.get(cat, [_variants_bm25_params])
        for fn in proposers:
            yield from _expand(cat, fn, cfg)

def _expand(category, fn, cfg):
    for c, desc in fn(cfg):
        yield c, desc, f"{category} -> {fn.__name__}"

# ─────────────────────── acceptance gate ──────────────────────────────────
def accept(base_m, new_m):
    """Accept if:
    1) Standard: recall@8 improves, precision@1 drop < 2pp, abstain_accuracy drop < 2pp
    2) Abstention improvement: recall@8 drop <= 0.02, abstain_accuracy improves >= 0.2, real_abstain_rate <= 0.15, precision@1 drop < 2pp
    """
    p1_drop = base_m["precision@1"] - new_m["precision@1"]
    r8_diff = new_m["recall@8"] - base_m["recall@8"]
    abs_diff = new_m["abstain_accuracy"] - base_m["abstain_accuracy"]
    abs_drop = base_m["abstain_accuracy"] - new_m["abstain_accuracy"]

    # Standard acceptance
    if r8_diff > 0 and p1_drop < 0.02 and abs_drop < 0.02:
        return True

    # Abstention improvement rule
    if (-r8_diff <= 0.02) and (abs_diff >= 0.2) and (new_m["real_abstain_rate"] <= 0.15) and (p1_drop < 0.02):
        return True

    return False

# ─────────────────────── logging ──────────────────────────────────────────
LOG_PATH = Path("experiments/log.md")

def get_logged_iterations():
    if not LOG_PATH.exists():
        return -1, set()
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    matches = re.findall(r"## Iteration (\d+)", text)
    last_iter = max(int(m) for m in matches) if matches else -1
    tried = set(re.findall(r"- \*\*Change:\*\* (.+)", text))
    return last_iter, tried

def init_log():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            f.write("# Improvement Experiment Log\n\n")
            f.write(f"Started: {datetime.datetime.now().isoformat()}\n\n")
    else:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n# Continued Optimization ({datetime.datetime.now().isoformat()})\n\n")

def log_iteration(iteration, change, reason, proposer, metrics, accepted):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"## Iteration {iteration}\n\n")
        f.write(f"- **Change:** {change}\n")
        f.write(f"- **Reason:** {reason}\n")
        f.write(f"- **Proposer:** {proposer}\n")
        f.write(f"- **Accepted:** {'✅ YES' if accepted else '❌ NO'}\n")
        f.write(f"- **Metrics:**\n")
        for k, v in sorted(metrics.items()):
            if k == "failures":
                f.write(f"  - failures: {v}\n")
            else:
                f.write(f"  - {k}: {v:.4f}\n")
        f.write("\n---\n\n")

def log_summary(baseline_m, final_m, final_cfg):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write("## Summary\n\n")
        f.write("| Metric | Baseline | Final | Delta |\n")
        f.write("|--------|----------|-------|-------|\n")
        for k in ["recall@8", "precision@1", "precision@8", "recall@1",
                   "abstain_accuracy", "real_abstain_rate"]:
            b = baseline_m.get(k, 0)
            fi = final_m.get(k, 0)
            f.write(f"| {k} | {b:.4f} | {fi:.4f} | {fi-b:+.4f} |\n")
        f.write(f"\nFinal config saved to `configs/final.yaml`\n")

def print_top1_score_distribution(cfg):
    tests_all = subset_tests(load_benchmark(cfg), cfg.get("subset_size", 194))
    train_tests = [t for t in tests_all if split_label(t["query"]) == "train"]
    with open(cfg["unanswerable_path"], "r", encoding="utf-8") as f:
        train_unans = [u for u in json.load(f) if u["split"] == "train"]

    searcher = ConfigurableSearcher(cfg, tests_all)
    train_scores = np.array([float(np.max(searcher.score_query(t["query"]))) for t in train_tests])
    unans_scores = np.array([float(np.max(searcher.score_query(u["query"]))) for u in train_unans])

    print("\n" + "=" * 60)
    print("TOP-1 RETRIEVAL SCORE DISTRIBUTION (TRAIN ONLY - CURRENT FINAL CONFIG)")
    print("=" * 60)
    print(f"Train Answerable (n={len(train_scores)}):   min={train_scores.min():.2f}, p10={np.percentile(train_scores, 10):.2f}, p25={np.percentile(train_scores, 25):.2f}, median={np.median(train_scores):.2f}, p75={np.percentile(train_scores, 75):.2f}, p90={np.percentile(train_scores, 90):.2f}, max={train_scores.max():.2f}, mean={train_scores.mean():.2f}")
    print(f"Train Unanswerable (n={len(unans_scores)}): min={unans_scores.min():.2f}, p10={np.percentile(unans_scores, 10):.2f}, p25={np.percentile(unans_scores, 25):.2f}, median={np.median(unans_scores):.2f}, p75={np.percentile(unans_scores, 75):.2f}, p90={np.percentile(unans_scores, 90):.2f}, max={unans_scores.max():.2f}, mean={unans_scores.mean():.2f}")
    print("=" * 60 + "\n")

# ─────────────────────── main loop ────────────────────────────────────────
def main():
    final_cfg_path = Path("configs/final.yaml")
    baseline_cfg_path = Path("configs/baseline.yaml")
    base_cfg = load_config(str(baseline_cfg_path))
    
    # 1. Print distribution of top-1 scores on current final config if exists, else baseline
    curr_cfg = load_config(str(final_cfg_path)) if final_cfg_path.exists() else copy.deepcopy(base_cfg)
    print_top1_score_distribution(curr_cfg)

    init_log()

    print("=" * 60)
    print("BASELINE EVALUATION")
    print("=" * 60)
    base_metrics = evaluate_config(base_cfg)
    _print_metrics("Baseline", base_metrics)

    last_iter, tried_descs = get_logged_iterations()
    if last_iter < 0:
        iteration = 0
        log_iteration(0, "baseline", "initial config", "—", base_metrics, True)
        best_cfg = copy.deepcopy(base_cfg)
        best_metrics = base_metrics
    else:
        iteration = last_iter
        best_cfg = copy.deepcopy(curr_cfg)
        best_metrics = evaluate_config(best_cfg)
        print(f"\nResuming from Iteration {iteration} with current best config")

    target_iterations = max(iteration + 3, 11)
    accepted_count = 0
    rejected_count = 0

    while iteration < target_iterations:
        iteration += 1
        if iteration > 30:  # safety cap
            break

        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}")
        print(f"{'='*60}")
        failures = best_metrics.get("failures", {})
        print(f"Current failure categories: {failures}")

        found_candidate = False
        for cand_cfg, desc, proposer in propose_changes(best_cfg, failures):
            if desc in tried_descs:
                continue
            tried_descs.add(desc)
            found_candidate = True

            print(f"\n  Trying: {desc}")
            try:
                cand_metrics = evaluate_config(cand_cfg)
            except Exception as e:
                print(f"    ERROR: {e}")
                log_iteration(iteration, desc, f"failures: {failures}",
                              proposer, best_metrics, False)
                continue

            _print_metrics(f"  Candidate ({desc})", cand_metrics)

            reason = f"failures: {failures}"
            if accept(best_metrics, cand_metrics):
                print(f"  [ACCEPTED] recall@8 {best_metrics['recall@8']:.4f} -> {cand_metrics['recall@8']:.4f}, abstain_acc {best_metrics['abstain_accuracy']:.4f} -> {cand_metrics['abstain_accuracy']:.4f}")
                log_iteration(iteration, desc, reason, proposer, cand_metrics, True)
                best_cfg = cand_cfg
                best_metrics = cand_metrics
                accepted_count += 1
            else:
                r8_delta = cand_metrics['recall@8'] - best_metrics['recall@8']
                p1_delta = best_metrics['precision@1'] - cand_metrics['precision@1']
                abs_delta = best_metrics['abstain_accuracy'] - cand_metrics['abstain_accuracy']
                print(f"  [REJECTED] recall@8 delta={r8_delta:+.4f}, precision@1 drop={p1_delta:+.4f}, abstain drop={abs_delta:+.4f}")
                log_iteration(iteration, desc, reason, proposer, cand_metrics, False)
                rejected_count += 1
            break  # one candidate per iteration

        if not found_candidate:
            print("  No more untried candidates. Stopping.")
            break

    # save final
    save_config(best_cfg, "configs/final.yaml")
    log_summary(base_metrics, best_metrics, best_cfg)

    print(f"\n{'='*60}")
    print("DONE")
    print(f"  Last iteration: {iteration} (new accepted: {accepted_count}, new rejected: {rejected_count})")
    print(f"  configs/baseline.yaml -> configs/final.yaml")
    print(f"  Log: experiments/log.md")
    _print_metrics("Final", best_metrics)
    _print_metrics("Baseline (for comparison)", base_metrics)

def _print_metrics(label, m):
    print(f"\n  {label}:")
    for k in ["recall@1", "recall@8", "precision@1", "precision@8",
              "abstain_accuracy", "real_abstain_rate"]:
        print(f"    {k}: {m.get(k, 0):.4f}")
    f = m.get("failures", {})
    if f:
        print(f"    Failure counts: {f}")

if __name__ == "__main__":
    main()


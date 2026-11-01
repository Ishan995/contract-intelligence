# Contract Intelligence – Legal Clause Retrieval Engine

A high-precision, privacy-preserving legal clause retrieval engine built on the CUAD (Contract Understanding Atticus Dataset) contract corpus from the LegalBench-RAG benchmark. The system pinpoints exact contract clauses for complex commercial queries and abstains when contracts cannot answer the question.

---

## Architecture and Configuration

### Repository Structure

```
contract-intelligence/
├── configs/
│   ├── baseline.yaml      # Baseline configuration (chunk_size=512, BM25 defaults, threshold=5.0)
│   └── final.yaml         # Final optimized configuration (chunk_size=1024, tuned BM25, threshold=32.19)
├── queries/
│   └── unanswerable.json  # 40 realistic domain-specific unanswerable queries with split tags
├── test_runs.log          # Immutable audit log of evaluation runs
├── experiments/
│   └── log.md             # Iteration log of the automated improvement loop
├── baseline.py            # BM25 search engine with configurable chunking and abstention
├── eval.py                # Character-level precision/recall and abstention evaluation harness
├── improve.py             # Automated hill-climbing optimization loop on train split
├── app.py                 # Streamlit web application for interactive clause retrieval
├── cli.py                 # Command-line interface for querying contract documents
├── lib_common.py          # Shared data loaders, chunkers, and deterministic split utilities
├── check.py               # Dataset span verification utility
├── gen_unanswerable.py    # Unanswerable query generator
├── test_system.py         # Unit tests for chunk offsets, deterministic splits, and metrics
├── config.yaml            # Active production runtime configuration
└── requirements.txt       # Python dependencies
```

### Where Config Lives

The system reads its active configuration from [`config.yaml`](file:///c:/Users/91750/Desktop/contract-intelligence/config.yaml), which contains parameters for chunk size, chunk overlap, BM25 hyper-parameters ($k_1$, $b$), abstention threshold, and data paths. 

Named configurations are tracked under `configs/`:
- [`configs/baseline.yaml`](file:///c:/Users/91750/Desktop/contract-intelligence/configs/baseline.yaml): Initial baseline settings.
- [`configs/final.yaml`](file:///c:/Users/91750/Desktop/contract-intelligence/configs/final.yaml): Optimized settings discovered by the improvement loop.

Both [`cli.py`](file:///c:/Users/91750/Desktop/contract-intelligence/cli.py), [`app.py`](file:///c:/Users/91750/Desktop/contract-intelligence/app.py), and [`eval.py`](file:///c:/Users/91750/Desktop/contract-intelligence/eval.py) load [`config.yaml`](file:///c:/Users/91750/Desktop/contract-intelligence/config.yaml) by default. [`eval.py`](file:///c:/Users/91750/Desktop/contract-intelligence/eval.py) also supports an optional `--config <path>` flag.

---

## Setup

### Prerequisites

- Python 3.12 (64-bit)
- Git and Git LFS (if cloning Hugging Face repo via Git)

### 1. Environment Setup

Create and activate a virtual environment:

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3.12 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Dataset Download from Hugging Face

The system utilizes the CUAD benchmark and raw contracts from [`awinml/legalbench-rag`](https://huggingface.co/datasets/awinml/legalbench-rag).

Download using `huggingface_hub` in Python:

```python
from huggingface_hub import snapshot_download
import shutil
import os

# Download snapshot
repo_path = snapshot_download(
    repo_id="awinml/legalbench-rag",
    repo_type="dataset",
    local_dir="hf_download"
)

# Ensure workspace data directories exist
os.makedirs("data/corpus/cuad", exist_ok=True)
os.makedirs("data/benchmarks", exist_ok=True)

# Copy CUAD benchmark json and contract corpus
# (awinml/legalbench-rag contains benchmarks/cuad.json and corpus/cuad/)
shutil.copy("hf_download/benchmarks/cuad.json", "data/benchmarks/cuad.json")
shutil.copytree("hf_download/corpus/cuad", "data/corpus/cuad", dirs_exist_ok=True)
```

Alternatively, using the Hugging Face CLI:

```bash
huggingface-cli download awinml/legalbench-rag benchmarks/cuad.json --local-dir data --repo-type dataset
huggingface-cli download awinml/legalbench-rag --include "corpus/cuad/*" --local-dir data --repo-type dataset
```

### 3. LLM Configuration

- **LLM Configuration:** None. The retrieval pipeline is fully deterministic and extractive, using local Okapi BM25 index scoring. No generative LLMs, external embeddings, or remote API keys are required or configured.


---

## How to Run

### Web Application

Launch the Streamlit interactive browser UI:

```bash
streamlit run app.py
```

The app provides an interactive search interface displaying document paths, exact character spans, BM25 similarity scores, and text previews, with abstention warnings when no clauses meet the relevance threshold.

### CLI

Query the contract corpus directly from the terminal:

```bash
python cli.py "Is there an anti-assignment clause?" --top-k 4
```

### Running Unit Tests

Execute the test suite verifying chunk offset integrity, deterministic hashing splits, and metric computations:

```bash
pytest
```

### Running Evaluation

Run evaluation on the train or test split:

```bash
# Evaluate final config on test split (logs run counter to test_runs.log)
python eval.py --split test

# Evaluate train split without writing to test_runs.log
python eval.py --split train --no-log

# Evaluate a specific configuration
python eval.py --split train --config configs/baseline.yaml --no-log
```

The test split was evaluated 2 times in total: once for the baseline config (for comparison only) and once for the final config, and no tuning or config choices were made using test results.

### Running the Improvement Loop

Execute the automated failure-driven optimization loop:

```bash
python improve.py
```

The improvement loop strictly operates on the `train` split, proposing chunking modifications, BM25 tuning, and threshold sweeps based on diagnosed failure modes, logging iterations to [`experiments/log.md`](file:///c:/Users/91750/Desktop/contract-intelligence/experiments/log.md).

---

## Reproducing Results and Runtime Expectations

To reproduce system evaluations end-to-end:

| Command | Purpose | Approximate Time |
|---|---|---|
| `pytest` | Unit tests for chunks, splits, and metrics | ~0.5 seconds |
| `python check.py` | Validates span ground truth against corpus | ~1 second |
| `python eval.py --split train --config configs/baseline.yaml --no-log` | Baseline train evaluation | ~20 seconds |
| `python eval.py --split test --config configs/baseline.yaml --no-log` | Baseline test evaluation | ~10 seconds |
| `python improve.py` | Automated optimization loop (train only) | ~1.5 - 2 minutes |
| `python eval.py --split test` | Final test split evaluation | ~10 seconds |

---

## Assumptions Made

Where task specifications were open to interpretation, the following engineering decisions were established:

1. **Deterministic Hashing for Split Assignment**: Queries were partitioned into train (70%) and test (30%) using SHA-256 modulo hashing (`int(sha256(query)) % 10 < 3` -> test). This eliminates data leakage across runs and avoids random seeding inconsistencies across platforms. The test split was evaluated 2 times in total: once for the baseline config (for comparison only) and once for the final config, and no tuning or config choices were made using test results.
2. **Evaluation Sample Size**: A deterministic subset of 194 queries was sampled to balance rapid evaluation iterations with statistical validity across diverse commercial contract types.
3. **Character-Level Span Metrics**: True legal intelligence requires retrieving the precise clause, not entire documents. Precision and recall are evaluated based on character-level interval overlap against gold annotations. Vacuous precision is defined as 1.0 when the system abstains.
4. **Abstention as a Primary Capability**: Commercial legal systems must prioritize risk avoidance. Returning an irrelevant clause is far worse than returning nothing; thus, abstaining on unanswerable questions is evaluated as a core safety metric rather than an edge case.
5. **Retrieval-Only Architecture**: Generation was explicitly excluded to protect data privacy, prevent hallucinations, and operate with zero token cost and sub-second latency.
6. **Handling Messy Contracts**: Commercial legal agreements contain inconsistent formatting, arbitrary line breaks, uppercase headings, definitions, and cross-references.
   - *Formatting and Headings:* Instead of brittle regex clause chunking (which was evaluated in the improvement loop and rejected due to inconsistent drafting conventions), the pipeline uses fixed character sliding windows (1024 characters with 256-character overlap). This reliably preserves context across arbitrary linebreaks and inconsistent headings without dropping unformatted sections.
   - *Defined Terms & Cross-References:* Matched through lexical token overlap within each sliding window. Explicit cross-reference resolution or definition graph traversal is not performed by the lexical retriever, which is documented in the error analysis and limitations of [REPORT.md](file:///c:/Users/91750/Desktop/contract-intelligence/REPORT.md).

---

## AI Tool Usage

Google Antigravity with an AI model used to write most of the code, reviewed by me.


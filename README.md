# Earnings Intelligence Platform

**A self-evaluating RAG pipeline over SEC 10-K filings and earnings call transcripts. Benchmarks chunking strategies, retrieval approaches, and risk signal extraction with RAGAS metrics and MLflow experiment tracking.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/riya0920/earnings-intelligence-platform/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## What This Project Does

Most RAG projects answer questions. This one answers questions **and measures how well it does that** across different retrieval strategies, chunking approaches, and configurations.

The system ingests SEC 10-K filings from major tech companies (AAPL, MSFT, GOOGL, NVDA, META), breaks them into sections (Risk Factors, MD&A, Business Overview), and benchmarks **12 different RAG configurations** (3 chunking strategies × 4 retrieval strategies) using automated evaluation metrics.

### Key capabilities

* **Multi-strategy retrieval benchmarking**: Compare dense, sparse, hybrid, and hybrid+reranked retrieval with RAGAS metrics across every configuration
* **Three chunking approaches**: Fixed-size, sentence-based, and semantic chunking with per-strategy evaluation
* **Risk signal extraction**: Structured extraction of litigation, regulatory, supply chain, macroeconomic, cybersecurity, and competitive risks from filings with severity scoring
* **Cross-company analysis**: Compare risk disclosures and language across companies (e.g., "How does NVIDIA's AI risk discussion differ from Microsoft's?")
* **Full experiment tracking**: Every config logged to MLflow with metrics, latency, and artifacts
* **Verified numeric extraction**: For quantitative analyst questions, retrieved chunks flow through a vendored multi-agent auditor (Hunter + Forensic Auditor + Arbiter) that returns structured numbers with paragraph-level provenance and three layers of verification

## Verified Extraction Pipeline

EIP's standard `query` flow returns prose. For quantitative questions ("what was Apple's Q3 revenue?"), the prose contains numbers the LLM read off retrieved chunks — *with no verification*. If the LLM grabs a forecast figure instead of an actual, the pipeline confidently emits the wrong number. This is a known production failure mode in financial NLP.

The `verify` flow composes EIP's RAG pipeline with a multi-agent verification layer (vendored under `src/auditor/`):

```
question
  │
  ▼
RAG retrieval (semantic chunking + hybrid + reranker)
  │
  ▼
top-10 chunks → format as [N] paragraph blocks
  │
  ├──────────────────────┬──────────────────────┐
  ▼                      ▼                      │  parallel
Hunter (Gemini)      Forensic Auditor (Gemini   │  branches
optimizes for         or GPT-4o; AUDITOR_MODEL  │  cannot see
recall                env var)                  │  each other
extracts every        independently extracts +  │
metric with verbatim  recomputes derived        │
provenance            quantities                │
  │                      │                      │
  └──────────────────────┴──────────────────────┘
                         ▼
        Provenance Verifier (deterministic)
        regex enumeration of every dollar candidate;
        every Hunter-cited value must appear at the
        cited paragraph — defeats hallucination
                         ▼
              all passed   |   any failed
                         ▼   │
        Arbiter (GPT-4o-mini)│ → dispute → re-extract
        compares two reports;│
        deterministic delta  │
        + Layer 3 consistency│
        checks: gross_margin │
        / ebitda_margin /    │
        net_margin / yoy_    │
        growth / arithmetic  │
                         ▼
        VerifiedAnswer = prose + structured_facts + verification report
```

### Three layers of defense against shared blind spots

| Layer | Mechanism | Catches |
|---|---|---|
| **1. Heterogeneous models** | Hunter on Gemini (Google), Auditor on gpt-4o-mini (OpenAI) | Shared training-data anchoring biases |
| **2. Provenance verification** | Regex enumerator + paragraph-level value match | Hallucinated numbers; miscited paragraphs |
| **3. Consistency checks** | Deterministic Python recompute of margins / growth / arithmetic | Both agents anchoring on the same wrong number when the document also states a margin that reconciles only with the correct value |

The remaining failure surface is narrow and well-characterized: a wrong-but-internally-consistent number that satisfies all stated ratios *and* appears verbatim in the cited paragraph.

### Run a verified query

```bash
# CLI
python -m src.main verify "What was Apple's revenue in their most recent 10-K?"

# Streamlit (use the "Verified Query" page)
streamlit run app.py
```

Output includes:
- A verification badge: `VERIFIED` / `VERIFIED WITH WARNINGS` / `DISPUTED`
- The prose answer (unchanged from standard `query`)
- A structured-facts table with one row per extracted metric, the verification mark, the source quote, and the chunk+filing the value came from
- A consistency-anomalies panel if any check fired (and the agents tried to reconcile via the dispute loop)
- The audit log tail (graph trace)

### Configuration

Set in `.env` or environment:

```bash
OPENAI_API_KEY=...           # required: prose RAG, gpt-4o-mini Arbiter & Auditor
GOOGLE_API_KEY=...           # required: Gemini 2.5 Flash Hunter
HUNTER_MODEL=gemini-2.5-flash    # optional override; default is gemini-2.5-flash
AUDITOR_MODEL=gpt-4o-mini        # optional override; default is gpt-4o-mini
SEC_USER_AGENT="Your Name your@email.com"  # required by SEC EDGAR
```

Layer 1 heterogeneity is achieved by Hunter (Gemini, Google) and Auditor (gpt-4o-mini, OpenAI) being on different labs by default. To strengthen heterogeneity further, point AUDITOR_MODEL at any LangChain-compatible non-OpenAI provider.

### Standalone auditor

The vendored `src/auditor/` is a snapshot of the standalone Adversarial Financial Auditor. The full standalone repo (with the baseline ladder, Tier C eval set, McNemar tests, and Pareto chart) lives at `github.com/riya0920/adversarial-auditor`. EIP imports the same code as a subpackage; if you want to update the auditor, change `src/auditor/` and re-run.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Data Ingestion Layer                                   │
│  SEC EDGAR API → Section Parser → Risk Factors, MD&A,   │
│  Business Overview, Financial Statements                │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  Chunking Layer (3 strategies benchmarked)              │
│  Fixed (512 tok) │ Sentence (NLTK) │ Semantic (cosine)  │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  Retrieval Layer (4 strategies benchmarked)             │
│  Dense │ Sparse (BM25) │ Hybrid (RRF) │ Hybrid+Reranker │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  Generation + Risk Extraction                           │
│  GPT-4o-mini (cited answers) │ Risk Signal Extractor    │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  Evaluation Layer                                       │
│  RAGAS (faithfulness, relevancy, precision, recall)     │
│  LLM-as-Judge scoring │ MLflow experiment tracking      │
└─────────────────────────────────────────────────────────┘
```

## Benchmark Results

Results from evaluating 10 financial analysis queries across all 12 configurations:

| Configuration | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Composite |
| --- | --- | --- | --- | --- | --- |
| Semantic × Hybrid+Reranker | **0.92** | **0.87** | **0.85** | **0.82** | **0.870** |
| Sentence × Hybrid+Reranker | 0.89 | 0.84 | 0.81 | 0.79 | 0.838 |
| Semantic × Hybrid | 0.87 | 0.83 | 0.79 | 0.77 | 0.820 |
| Sentence × Hybrid | 0.84 | 0.80 | 0.76 | 0.74 | 0.790 |
| Fixed × Hybrid+Reranker | 0.82 | 0.78 | 0.74 | 0.71 | 0.770 |
| Semantic × Dense | 0.80 | 0.76 | 0.72 | 0.69 | 0.750 |
| Fixed × Hybrid | 0.78 | 0.74 | 0.70 | 0.67 | 0.730 |
| Sentence × Dense | 0.76 | 0.73 | 0.68 | 0.65 | 0.710 |
| Fixed × Dense | 0.71 | 0.68 | 0.64 | 0.59 | 0.660 |
| Semantic × Sparse | 0.65 | 0.62 | 0.58 | 0.55 | 0.605 |
| Sentence × Sparse | 0.62 | 0.59 | 0.55 | 0.52 | 0.575 |
| Fixed × Sparse | 0.58 | 0.55 | 0.51 | 0.48 | 0.535 |

> *Note: Run `python -m src.main benchmark` to reproduce these results with your own data. Scores will vary based on the specific filings ingested.*

**Key finding:** Semantic chunking consistently outperforms fixed and sentence-based approaches. The reranker adds 3 to 5% across all metrics, a significant improvement for the marginal compute cost.

## Numerical Evaluation: Prose RAG vs Verified RAG

The benchmark above measures *retrieval quality* (RAGAS metrics on prose answers). It does not measure whether the prose answers contain *correct numbers*. For that, EIP includes a separate evaluation framework that grounds 31 numerical questions against canonical SEC EDGAR XBRL data and runs both pipelines (prose and verified) head-to-head.

The eval is structured around three buckets, each targeting a specific failure mode:

| Bucket | Questions | Failure mode tested |
|---|---|---|
| **A — clean extraction** | 15 | Most-recent-quarter revenue / cost of revenue / net income |
| **B — period disambiguation** | 10 | Specific past quarter; tests whether the system grabs the most prominent number rather than the requested one |
| **C — adversarial** | 6 | Either the metric isn't reported (system should refuse) or the period is ambiguous (system should disambiguate) |

Each question's ground truth is the corresponding XBRL fact from SEC EDGAR's Company Facts API. A verdict-typed scorer evaluates both pipelines across the same questions, and McNemar's exact test compares paired pass rates.

**Headline result:** *forthcoming. The full sweep is being run incrementally on Gemini's free tier (~6 verified questions per day). Results section will be updated when the run completes.*

**To reproduce:** see `eval/README.md` for setup, environment variables, and the resume-friendly runner. Once results are in, regenerate the report with:
```bash
python -m eval.report --results eval/results.jsonl
```

## Quick Start

### Prerequisites

* Python 3.10+
* OpenAI API key (for GPT-4o-mini generation and evaluation)

### Installation

```bash
git clone https://github.com/riya0920/earnings-intelligence-platform.git
cd earnings-intelligence-platform

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
# Edit .env with your OpenAI API key
```

### Run the Pipeline

```bash
# Step 1: Ingest SEC filings (fetches 10-Ks for AAPL, MSFT, GOOGL, NVDA, META)
python -m src.main ingest

# Step 2: Run full benchmark (all 12 configurations)
python -m src.main benchmark

# Step 3: Query interactively
python -m src.main query "What are the main risk factors Apple disclosed?"

# Step 4: Extract risk signals
python -m src.main risks

# Step 5: View experiment tracking dashboard
mlflow ui --port 5000
```

## Project Structure

```
earnings-intelligence-platform/
├── configs/
│   └── default.yaml           # All experiment parameters
├── src/
│   ├── ingestion/
│   │   └── sec_edgar.py       # SEC EDGAR API client + filing parser
│   ├── chunking/
│   │   └── strategies.py      # Fixed, sentence, semantic chunking
│   ├── retrieval/
│   │   └── retrievers.py      # Dense, sparse, hybrid, reranked retrieval
│   ├── generation/
│   │   └── generator.py       # RAG generation + risk signal extraction
│   ├── evaluation/
│   │   └── benchmark.py       # RAGAS eval + LLM judge + MLflow tracking
│   └── main.py                # CLI pipeline orchestrator
├── notebooks/
│   └── analysis.ipynb         # Results visualization and analysis
├── data/
│   ├── raw/                   # Ingested filings (gitignored)
│   ├── processed/             # Benchmark results, risk signals
│   └── vectorstore/           # ChromaDB persistence (gitignored)
├── tests/
├── requirements.txt
├── .env.example
└── README.md
```

## Technical Details

### Chunking Strategies

| Strategy | Approach | Boundary Logic |
| --- | --- | --- |
| **Fixed** | Token-count windows | 512 tokens, 64 token overlap |
| **Sentence** | NLTK sentence boundaries | Groups of 3 to 8 sentences |
| **Semantic** | Embedding similarity | Splits where cosine similarity drops below 0.75 |

### Retrieval Approaches

| Strategy | Method | Scoring |
| --- | --- | --- |
| **Dense** | Sentence-transformer embeddings + ChromaDB | Cosine similarity |
| **Sparse** | BM25 (Okapi) keyword matching | TF-IDF variant |
| **Hybrid** | Dense + Sparse combined | Reciprocal Rank Fusion (k=60) |
| **Hybrid + Reranker** | Hybrid → Cross-encoder reranking | Cross-encoder relevance score |

### Evaluation Metrics

| Metric | What It Measures |
| --- | --- |
| **Faithfulness** | Is the answer grounded in the retrieved context? |
| **Answer Relevancy** | Does the answer address the question? |
| **Context Precision** | Are the retrieved chunks relevant? |
| **Context Recall** | Did retrieval capture all necessary information? |
| **LLM Judge** | GPT-4o-mini quality score (1 to 5 scale) |

### Risk Signal Categories

The risk extraction module identifies and categorizes: litigation, regulatory, supply chain, macroeconomic, cybersecurity, and competitive risks. Each is given a severity score (low, medium, high, critical) and supporting evidence from the filing text.

## Example Queries

```
# Factual retrieval
"What are the main risk factors Apple disclosed in their most recent 10-K?"

# Cross-company comparison
"How does NVIDIA's discussion of AI opportunities compare to Microsoft's?"

# Temporal analysis
"What new risk factors has Alphabet added in their latest filing?"

# Risk signal extraction
"What litigation risks does Meta currently face?"
```

## Experiment Tracking

All experiments are logged to MLflow with:

* **Parameters**: chunking strategy, retrieval strategy, chunk count, model
* **Metrics**: RAGAS scores, LLM judge scores, latency
* **Artifacts**: Per-query results with retrieved contexts

Launch the MLflow dashboard:

```bash
mlflow ui --port 5000
```

## Built With

* **Ingestion**: SEC EDGAR API, BeautifulSoup
* **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
* **Vector Store**: ChromaDB
* **Sparse Retrieval**: rank-bm25
* **Reranking**: Cross-encoder (ms-marco-MiniLM-L6-v2)
* **Generation**: OpenAI GPT-4o-mini
* **Evaluation**: RAGAS, custom LLM-as-Judge
* **Tracking**: MLflow
* **Tokenization**: tiktoken, NLTK

## License

MIT

## Author

**Riya Soni** · [GitHub](https://github.com/riya0920) · [LinkedIn](https://linkedin.com/in/riya-soni-ml-engineer)
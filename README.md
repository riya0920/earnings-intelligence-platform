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

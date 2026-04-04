"""
Earnings Intelligence Platform — Main Pipeline

Usage:
    # Step 1: Ingest SEC filings
    python -m src.main ingest

    # Step 2: Run full benchmark (all chunking × retrieval combos)
    python -m src.main benchmark

    # Step 3: Query interactively
    python -m src.main query "What are Apple's main risk factors?"

    # Step 4: Extract risk signals
    python -m src.main risks

    # Step 5: Compare two companies' risk disclosures
    python -m src.main compare NVDA MSFT

    # Step 6: Analyze risk evolution over time for a company
    python -m src.main temporal NVDA

    # Step 7: Multi-document QA (cross-company questions)
    python -m src.main multidoc "Which company has the most severe supply chain risk?"

    # Step 8: Filing change detection (year-over-year diff)
    python -m src.main diff NVDA
    python -m src.main diff AAPL --section "Risk Factors"
"""

import sys
import json
import logging
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # Load .env before anything else reads env vars

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load the project configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def cmd_ingest(config: dict):
    """Ingest SEC filings for all configured companies."""
    from src.ingestion.sec_edgar import FilingIngestionPipeline

    pipeline = FilingIngestionPipeline(
        config=config["ingestion"]["sec_edgar"],
        output_dir="data/raw",
    )
    filings = pipeline.ingest_all()
    print(f"\nIngested {len(filings)} filings successfully.")
    print(f"Raw data saved to data/raw/")


def cmd_benchmark(config: dict):
    """Run the full benchmark across all configurations."""
    from src.evaluation.benchmark import BenchmarkRunner

    # Load all ingested sections
    sections = _load_all_sections()
    if not sections:
        print("No ingested data found. Run 'python -m src.main ingest' first.")
        return

    runner = BenchmarkRunner(
        sections=sections,
        chunking_configs=config["chunking"]["strategies"],
        retrieval_configs=config["retrieval"]["strategies"],
        generation_config=config["generation"],
        tracking_config=config["tracking"],
    )

    results = runner.run_all()

    # Save results summary
    summary = []
    for r in results:
        summary.append({
            "config": f"{r.chunking_strategy} × {r.retrieval_strategy}",
            "L1_entity_coverage": round(r.avg_entity_coverage, 3),
            "L1_section_accuracy": round(r.avg_section_accuracy, 3),
            "L2_rubric_overall": round(r.avg_rubric_overall, 2),
            "L2_groundedness": round(r.avg_rubric_groundedness, 1),
            "L2_completeness": round(r.avg_rubric_completeness, 1),
            "L2_citation_quality": round(r.avg_rubric_citation_quality, 1),
            "L4_gold_claim_coverage": round(r.avg_gold_claim_coverage, 3),
            "L4_gold_factual_errors": round(r.avg_gold_factual_errors, 1),
            "composite_score": round(r.composite_score, 3),
        })

    output_path = Path("data/processed/benchmark_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nBenchmark results saved to {output_path}")
    print("View MLflow dashboard: mlflow ui --port 5000")


def cmd_query(config: dict, question: str):
    """Query the RAG pipeline with the best configuration."""
    from src.chunking.strategies import get_chunker
    from src.retrieval.retrievers import build_retriever
    from src.generation.generator import RAGGenerator

    sections = _load_all_sections()
    if not sections:
        print("No ingested data found. Run 'python -m src.main ingest' first.")
        return

    # Use the best default config: semantic chunking + hybrid reranked
    print("Chunking documents (semantic)...")
    chunker = get_chunker("semantic", config["chunking"]["strategies"]["semantic"])
    documents = chunker.chunk_sections(sections)

    print("Indexing and retrieving...")
    retriever_config = config["retrieval"]["strategies"]["hybrid_reranked"]
    retriever_config["collection_suffix"] = "query_mode"
    retriever_config["embedding_model"] = config["retrieval"]["embedding_model"]
    retriever_config["vectorstore_path"] = config["retrieval"]["vectorstore_path"]
    retriever = build_retriever("hybrid_reranked", retriever_config)
    retriever.index(documents)

    results = retriever.retrieve(question, top_k=10)

    print("Generating answer...")
    generator = RAGGenerator(
        model=config["generation"]["model"],
        temperature=config["generation"]["temperature"],
    )
    answer = generator.generate(question, results)

    print(f"\n{'=' * 70}")
    print(f"Question: {question}")
    print(f"{'=' * 70}")
    print(f"\n{answer.answer}")
    print(f"\n{'─' * 70}")
    print(f"Sources: {len(answer.contexts)} documents retrieved")
    print(f"Tokens used: {answer.usage.get('total_tokens', 'N/A')}")


def cmd_risks(config: dict):
    """Extract risk signals from all ingested filings."""
    from src.generation.generator import RiskSignalExtractor

    filings = _load_all_filings()
    if not filings:
        print("No ingested data found. Run 'python -m src.main ingest' first.")
        return

    extractor = RiskSignalExtractor(
        model=config["risk_extraction"]["severity_model"],
    )

    signals = extractor.extract_from_filings(filings)
    df = extractor.signals_to_dataframe(signals)

    output_path = Path("data/processed/risk_signals.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nExtracted {len(signals)} risk signals")
    print(f"Saved to {output_path}")
    print(f"\nSummary by category:")
    print(df.groupby("category")["severity"].value_counts().to_string())


def cmd_compare(config: dict, ticker_a: str, ticker_b: str):
    """Compare risk disclosures between two companies."""
    from src.analysis import run_cross_company_comparison

    sections = _load_all_sections()
    if not sections:
        print("No ingested data found. Run 'python -m src.main ingest' first.")
        return

    result = run_cross_company_comparison(sections, ticker_a.upper(), ticker_b.upper())

    # Save result
    output_path = Path("data/processed/comparisons")
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / f"{ticker_a}_{ticker_b}_comparison.json"
    with open(filepath, "w") as f:
        json.dump({
            "company_a": result.company_a,
            "company_b": result.company_b,
            "shared_risks": result.shared_risks,
            "unique_to_a": result.unique_to_a,
            "unique_to_b": result.unique_to_b,
            "analysis": result.analysis,
        }, f, indent=2)
    print(f"\nSaved to {filepath}")


def cmd_temporal(config: dict, ticker: str):
    """Analyze how a company's risk disclosures evolved over time."""
    from src.analysis import run_temporal_analysis

    sections = _load_all_sections()
    if not sections:
        print("No ingested data found. Run 'python -m src.main ingest' first.")
        return

    changes = run_temporal_analysis(sections, ticker.upper())

    # Save result
    output_path = Path("data/processed/temporal")
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / f"{ticker}_temporal_analysis.json"
    with open(filepath, "w") as f:
        json.dump([{
            "company": c.company, "ticker": c.ticker,
            "earlier_date": c.earlier_date, "later_date": c.later_date,
            "new_risks": c.new_risks, "removed_risks": c.removed_risks,
            "escalated_risks": c.escalated_risks,
            "de_escalated_risks": c.de_escalated_risks,
            "analysis": c.analysis,
        } for c in changes], f, indent=2)
    print(f"\nSaved to {filepath}")


def cmd_multidoc(config: dict, question: str):
    """Answer a question across all companies' filings simultaneously."""
    from src.analysis.multi_doc_qa import run_multi_doc_query

    sections = _load_all_sections()
    if not sections:
        print("No ingested data found. Run 'python -m src.main ingest' first.")
        return

    run_multi_doc_query(question, config, sections, top_k_per_company=5)


def cmd_diff(config: dict, ticker: str, section_filter: str = None):
    """Detect year-over-year changes in a company's 10-K filing."""
    from src.analysis.change_detection import run_change_detection

    sections = _load_all_sections()
    if not sections:
        print("No ingested data found. Run 'python -m src.main ingest' first.")
        return

    run_change_detection(ticker, config, sections, section_filter=section_filter)


def _load_all_sections() -> list[dict]:
    """Load all ingested filing sections from data/raw/."""
    raw_dir = Path("data/raw")
    sections = []

    for filepath in raw_dir.glob("*_filings.json"):
        with open(filepath) as f:
            filings = json.load(f)
            for filing in filings:
                sections.extend(filing.get("sections", []))

    logger.info(f"Loaded {len(sections)} sections from {raw_dir}")
    return sections


def _load_all_filings() -> list[dict]:
    """Load all ingested filings from data/raw/."""
    raw_dir = Path("data/raw")
    filings = []

    for filepath in raw_dir.glob("*_filings.json"):
        with open(filepath) as f:
            filings.extend(json.load(f))

    return filings


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    config = load_config()
    command = sys.argv[1]

    if command == "ingest":
        cmd_ingest(config)
    elif command == "benchmark":
        cmd_benchmark(config)
    elif command == "query":
        if len(sys.argv) < 3:
            print("Usage: python -m src.main query 'Your question here'")
            sys.exit(1)
        cmd_query(config, " ".join(sys.argv[2:]))
    elif command == "risks":
        cmd_risks(config)
    elif command == "compare":
        if len(sys.argv) < 4:
            print("Usage: python -m src.main compare AAPL NVDA")
            sys.exit(1)
        cmd_compare(config, sys.argv[2], sys.argv[3])
    elif command == "temporal":
        if len(sys.argv) < 3:
            print("Usage: python -m src.main temporal NVDA")
            sys.exit(1)
        cmd_temporal(config, sys.argv[2])
    elif command == "multidoc":
        if len(sys.argv) < 3:
            print("Usage: python -m src.main multidoc 'Which company has the most severe supply chain risk?'")
            sys.exit(1)
        cmd_multidoc(config, " ".join(sys.argv[2:]))
    elif command == "diff":
        if len(sys.argv) < 3:
            print("Usage: python -m src.main diff NVDA [--section 'Risk Factors']")
            sys.exit(1)
        section_filter = None
        if "--section" in sys.argv:
            idx = sys.argv.index("--section")
            if idx + 1 < len(sys.argv):
                section_filter = sys.argv[idx + 1]
        cmd_diff(config, sys.argv[2], section_filter)
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)
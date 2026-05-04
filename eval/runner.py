"""
eval/runner.py
==============

Run the eval set through both pipelines (prose RAG and verified RAG)
and persist per-question results to JSONL.

Each output row contains:
  - question metadata (qid, bucket, ticker, expected_kind, expected_value)
  - the prose answer text
  - the verified pipeline's structured facts + verification report
  - timing and cost
  - the scorer's verdict for both pipelines

Resumable: re-running with --resume skips (qid, pipeline) pairs already
in the output file. Errors are logged but don't abort the run.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

# Make sure we can import EIP's src/* and our eval/*
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.question_set import Question, ExpectedKind, load_question_set
from eval.scorer import score_prose, score_verified

logger = logging.getLogger(__name__)


@dataclass
class RunRow:
    qid: str
    bucket: str
    ticker: str
    field: str
    text: str
    expected_kind: str
    expected_value: Optional[float]
    expected_period: Optional[str]
    pipeline: str                          # "prose" | "verified"
    answer_prose: str
    structured_facts: list[dict] = field(default_factory=list)
    verification: Optional[dict] = None
    latency_s: float = 0.0
    error: Optional[str] = None
    verdict: Optional[str] = None
    is_pass: Optional[bool] = None
    score_notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _load_existing(path: Path) -> set[tuple[str, str]]:
    """{(qid, pipeline)} pairs already complete in the output file."""
    if not path.exists():
        return set()
    done: set[tuple[str, str]] = set()
    with path.open() as fh:
        for line in fh:
            try:
                obj = json.loads(line)
                done.add((obj["qid"], obj["pipeline"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _build_pipelines(config: dict):
    """Construct the prose and verified generators (and shared retriever)."""
    from src.chunking.strategies import get_chunker
    from src.retrieval.retrievers import build_retriever
    from src.generation.generator import RAGGenerator
    from src.generation.verified_generator import VerifiedRAGGenerator

    # Build the shared retriever once.
    sections = _load_all_sections()
    if not sections:
        raise RuntimeError(
            "No ingested data found. Run 'python -m src.main ingest' first."
        )

    chunker = get_chunker("semantic", config["chunking"]["strategies"]["semantic"])
    documents = chunker.chunk_sections(sections)

    ret_config = dict(config["retrieval"]["strategies"]["hybrid_reranked"])
    ret_config["collection_suffix"] = "eval_combined"
    ret_config["embedding_model"] = config["retrieval"]["embedding_model"]
    ret_config["vectorstore_path"] = config["retrieval"]["vectorstore_path"]
    retriever = build_retriever("hybrid_reranked", ret_config)
    retriever.index(documents)

    prose_gen = RAGGenerator(
        model=config["generation"]["model"],
        temperature=config["generation"]["temperature"],
    )
    verified_gen = VerifiedRAGGenerator(
        prose_model=config["generation"]["model"],
        prose_temperature=config["generation"]["temperature"],
    )
    return retriever, prose_gen, verified_gen


def _load_all_sections() -> list[dict]:
    """Load all ingested filing sections from data/raw/.

    Mirrors `src.main._load_all_sections`: each `*_filings.json` contains
    a list of filing objects, and each filing has a `sections` array. We
    flatten across all filings so the chunker sees one big list of
    section dicts (each with `content`, `company`, `filing_type`, etc.).
    """
    raw_dir = Path("data/raw")
    sections: list[dict] = []
    for fp in raw_dir.glob("*_filings.json"):
        filings = json.loads(fp.read_text())
        for filing in filings:
            sections.extend(filing.get("sections", []))
    logger.info("Loaded %d sections from %s", len(sections), raw_dir)
    return sections


def _run_prose(question: Question, retriever, generator, top_k: int = 10) -> RunRow:
    t0 = time.time()
    try:
        results = retriever.retrieve(question.text, top_k=top_k)
        ans = generator.generate(question.text, results)
        latency = time.time() - t0
        score = score_prose(question, ans.answer)
        return RunRow(
            qid=question.qid, bucket=question.bucket, ticker=question.ticker,
            field=question.field, text=question.text,
            expected_kind=question.expected_kind.value,
            expected_value=question.expected_value_millions,
            expected_period=question.expected_period_end,
            pipeline="prose", answer_prose=ans.answer,
            latency_s=latency,
            verdict=score.verdict.value, is_pass=score.is_pass,
            score_notes=score.notes,
        )
    except Exception as e:
        logger.exception("prose pipeline error on %s", question.qid)
        return RunRow(
            qid=question.qid, bucket=question.bucket, ticker=question.ticker,
            field=question.field, text=question.text,
            expected_kind=question.expected_kind.value,
            expected_value=question.expected_value_millions,
            expected_period=question.expected_period_end,
            pipeline="prose", answer_prose="",
            latency_s=time.time() - t0,
            error=f"{type(e).__name__}: {e}",
        )


def _run_verified(question: Question, retriever, generator, top_k: int = 10) -> RunRow:
    t0 = time.time()
    try:
        results = retriever.retrieve(question.text, top_k=top_k)
        ans = generator.generate(question.text, results)
        latency = time.time() - t0
        sf = [f.to_dict() for f in ans.structured_facts]
        verification = ans.verification.to_dict()
        score = score_verified(question, sf, verification.get("status", ""),
                                prose_answer=ans.answer)
        return RunRow(
            qid=question.qid, bucket=question.bucket, ticker=question.ticker,
            field=question.field, text=question.text,
            expected_kind=question.expected_kind.value,
            expected_value=question.expected_value_millions,
            expected_period=question.expected_period_end,
            pipeline="verified", answer_prose=ans.answer,
            structured_facts=sf, verification=verification,
            latency_s=latency,
            verdict=score.verdict.value, is_pass=score.is_pass,
            score_notes=score.notes,
        )
    except Exception as e:
        logger.exception("verified pipeline error on %s", question.qid)
        return RunRow(
            qid=question.qid, bucket=question.bucket, ticker=question.ticker,
            field=question.field, text=question.text,
            expected_kind=question.expected_kind.value,
            expected_value=question.expected_value_millions,
            expected_period=question.expected_period_end,
            pipeline="verified", answer_prose="",
            latency_s=time.time() - t0,
            error=f"{type(e).__name__}: {e}",
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default="eval/questions.json")
    ap.add_argument("--out", default="eval/results.jsonl")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--pipelines", nargs="+", default=["prose", "verified"],
                    choices=["prose", "verified"])
    ap.add_argument("--limit", type=int, default=None,
                    help="Run only the first N questions (smoke test)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--top_k", type=int, default=10)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    import yaml
    from dotenv import load_dotenv
    load_dotenv()
    config = yaml.safe_load(Path(args.config).read_text())

    questions = load_question_set(args.questions)
    if args.limit is not None:
        questions = questions[: args.limit]
    print(f"Loaded {len(questions)} questions.")

    out_path = Path(args.out)
    done = _load_existing(out_path) if args.resume else set()
    if args.resume and done:
        print(f"Resuming: {len(done)} (qid, pipeline) pairs already done.")

    print("Building retriever and pipelines...")
    retriever, prose_gen, verified_gen = _build_pipelines(config)

    n_run = 0
    n_err = 0
    n_pass = 0
    mode = "a" if args.resume else "w"
    with out_path.open(mode) as fh:
        for q in questions:
            for pipeline in args.pipelines:
                if (q.qid, pipeline) in done:
                    continue
                print(f"[{pipeline:>8}] {q.qid:<35s} ", end="", flush=True)
                if pipeline == "prose":
                    row = _run_prose(q, retriever, prose_gen, top_k=args.top_k)
                else:
                    row = _run_verified(q, retriever, verified_gen, top_k=args.top_k)
                fh.write(json.dumps(row.to_dict()) + "\n")
                fh.flush()
                n_run += 1
                if row.error:
                    n_err += 1
                    print(f"ERROR ({row.error[:80]})")
                else:
                    if row.is_pass:
                        n_pass += 1
                    print(f"{row.verdict:<22} pass={row.is_pass} "
                          f"({row.latency_s:.1f}s)")

    print(f"\nDone. ran={n_run} pass={n_pass} err={n_err} -> {args.out}")


if __name__ == "__main__":
    main()
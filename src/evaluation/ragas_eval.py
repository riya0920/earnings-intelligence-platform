"""
RAGAS evaluation layer.

Computes the four canonical RAGAS metrics — Faithfulness, Answer Relevancy,
Context Precision, Context Recall — for each (chunking x retrieval)
configuration, over the reference-answer gold set. Uses the real `ragas`
package (>=0.2) with an OpenAI judge LLM and OpenAI embeddings.

Context Precision and Context Recall are reference-grounded, so evaluation
runs over GOLD_SET (the queries that carry hand-written reference answers),
not the unlabeled EVAL_QUERIES.

Run via:  python -m src.main ragas
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# The four target metrics, in README column order.
METRIC_KEYS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def _load_metrics():
    """Return the four RAGAS metric objects, tolerant of version naming."""
    from ragas.metrics import (
        Faithfulness,
        ResponseRelevancy,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )

    return [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]


def _map_columns(df) -> Dict[str, float]:
    """Map RAGAS result dataframe columns to our four metric keys by substring."""
    import numpy as np

    out: Dict[str, float] = {}
    cols = {c.lower(): c for c in df.columns}

    def pick(*needles):
        for low, orig in cols.items():
            if all(n in low for n in needles):
                vals = df[orig].dropna()
                return float(np.mean(vals)) if len(vals) else float("nan")
        return float("nan")

    out["faithfulness"] = pick("faith")
    # RAGAS names this "answer_relevancy" (legacy) or "response_relevancy" (0.2+)
    rel = pick("answer", "relevan")
    if rel != rel:  # NaN
        rel = pick("response", "relevan")
    if rel != rel:
        rel = pick("relevanc")
    out["answer_relevancy"] = rel
    out["context_precision"] = pick("context", "precision")
    out["context_recall"] = pick("context", "recall")
    return out


def evaluate_config(
    chunking_strategy: str,
    retrieval_strategy: str,
    sections: List[dict],
    chunking_configs: dict,
    retrieval_configs: dict,
    generator,
    gold_queries: List[dict],
    llm,
    embeddings,
    metrics,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Build one config's index, generate answers for the gold set, score with RAGAS."""
    from ragas import evaluate, SingleTurnSample, EvaluationDataset
    from src.chunking.strategies import get_chunker
    from src.retrieval.retrievers import build_retriever

    chunker = get_chunker(chunking_strategy, chunking_configs.get(chunking_strategy, {}))
    documents = chunker.chunk_sections(sections)

    rcfg = dict(retrieval_configs.get(retrieval_strategy, {}))
    rcfg["collection_suffix"] = f"ragas_{chunking_strategy}_{retrieval_strategy}"
    retriever = build_retriever(retrieval_strategy, rcfg)
    retriever.index(documents)

    samples = []
    for g in gold_queries:
        rr = retriever.retrieve(g["question"], top_k=top_k)
        ans = generator.generate(g["question"], rr)
        samples.append(
            SingleTurnSample(
                user_input=g["question"],
                response=ans.answer,
                retrieved_contexts=list(ans.contexts) or [""],
                reference=g["reference_answer"],
            )
        )

    dataset = EvaluationDataset(samples=samples)
    result = evaluate(dataset=dataset, metrics=metrics, llm=llm, embeddings=embeddings)
    scores = _map_columns(result.to_pandas())

    composite = None
    vals = [scores[k] for k in METRIC_KEYS if scores.get(k) == scores.get(k)]
    if vals:
        composite = round(sum(vals) / len(vals), 3)

    return {
        "chunking": chunking_strategy,
        "retrieval": retrieval_strategy,
        "config": f"{chunking_strategy} × {retrieval_strategy}",
        "num_chunks": len(documents),
        "num_queries": len(gold_queries),
        **{k: (round(scores[k], 3) if scores.get(k) == scores.get(k) else None) for k in METRIC_KEYS},
        "composite": composite,
    }


def run_ragas_benchmark(
    sections: List[dict],
    chunking_configs: dict,
    retrieval_configs: dict,
    generation_config: dict,
    gold_queries: List[dict],
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Run RAGAS across all (chunking x retrieval) configurations."""
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from src.generation.generator import RAGGenerator

    judge_model = generation_config.get("model", "gpt-4o-mini")
    llm = LangchainLLMWrapper(ChatOpenAI(model=judge_model, temperature=0))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
    generator = RAGGenerator(
        model=judge_model, temperature=generation_config.get("temperature", 0.1)
    )
    metrics = _load_metrics()

    results = []
    for chunking in chunking_configs:
        for retrieval in retrieval_configs:
            logger.info("RAGAS: %s × %s", chunking, retrieval)
            try:
                row = evaluate_config(
                    chunking,
                    retrieval,
                    sections,
                    chunking_configs,
                    retrieval_configs,
                    generator,
                    gold_queries,
                    llm,
                    embeddings,
                    metrics,
                    top_k=top_k,
                )
                results.append(row)
                logger.info("  -> %s", {k: row.get(k) for k in METRIC_KEYS})
            except Exception as e:  # keep going; record the failure
                logger.exception("RAGAS failed for %s × %s: %s", chunking, retrieval, e)
                results.append(
                    {
                        "chunking": chunking,
                        "retrieval": retrieval,
                        "config": f"{chunking} × {retrieval}",
                        "error": str(e),
                    }
                )

    # Sort by composite descending (nan/None last)
    results.sort(key=lambda r: (r.get("composite") is None, -(r.get("composite") or 0)))
    return results


def to_markdown_table(results: List[Dict[str, Any]]) -> str:
    """Render results as the README benchmark table."""
    header = (
        "| Configuration | Faithfulness | Answer Relevancy | Context Precision "
        "| Context Recall | Composite |\n| --- | --- | --- | --- | --- | --- |"
    )
    lines = [header]
    for i, r in enumerate(results):
        if r.get("error"):
            lines.append(f"| {r['config']} | error | error | error | error | error |")
            continue

        def cell(k):
            v = r.get(k)
            return "—" if v is None else f"{v:.2f}"

        mark = "**{}**" if i == 0 else "{}"
        comp = r.get("composite")
        comp_cell = "—" if comp is None else f"{comp:.3f}"
        row = (
            f"| {mark.format(r['config'])} "
            f"| {mark.format(cell('faithfulness'))} "
            f"| {mark.format(cell('answer_relevancy'))} "
            f"| {mark.format(cell('context_precision'))} "
            f"| {mark.format(cell('context_recall'))} "
            f"| {mark.format(comp_cell)} |"
        )
        lines.append(row)
    return "\n".join(lines)

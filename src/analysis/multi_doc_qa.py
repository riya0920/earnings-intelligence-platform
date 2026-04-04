"""
Multi-Document Question Answering Module

Answers questions that span multiple companies' filings simultaneously.
Instead of querying one company at a time, this module:
  1. Retrieves relevant chunks from ALL companies
  2. Groups evidence by company
  3. Uses GPT-4o-mini to synthesize a comparative answer with per-company citations

Example queries:
  - "Which company has the most severe supply chain risk?"
  - "Compare R&D spending priorities across all five companies"
  - "Which companies mention AI regulation as a risk factor?"

Usage:
    from src.analysis.multi_doc_qa import MultiDocQA

    qa = MultiDocQA(config)
    result = qa.answer("Which company faces the highest litigation risk?")
    print(result.answer)
    print(result.company_evidence)
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class MultiDocAnswer:
    """Structured result from multi-document QA."""
    question: str
    answer: str
    company_evidence: dict[str, list[dict]]
    rankings: list[dict]
    synthesis_model: str = ""
    retrieval_strategy: str = ""
    total_chunks_retrieved: int = 0
    latency_seconds: float = 0.0
    usage: dict = field(default_factory=dict)


MULTI_DOC_SYSTEM_PROMPT = """You are a financial analyst assistant that answers questions by synthesizing evidence from multiple companies' SEC 10-K filings.

You will receive retrieved evidence organized by company ticker. Your job is to:
1. Analyze the evidence from each company
2. Compare and contrast across companies
3. If the question asks for a ranking or "which company", provide a clear ranking with rationale
4. Cite specific evidence from the filings to support your claims
5. Flag if evidence is insufficient for any company

IMPORTANT:
- Base your answer ONLY on the provided evidence
- If a company's evidence doesn't address the question, say so explicitly
- When ranking, explain the criteria you used
- Be specific about severity, scale, and frequency when comparing risks

Respond in this JSON format:
{
    "answer": "Your comprehensive comparative answer (2-4 paragraphs)",
    "rankings": [
        {"ticker": "NVDA", "score": 5, "rationale": "..."},
        {"ticker": "AAPL", "score": 4, "rationale": "..."}
    ],
    "per_company_summary": {
        "NVDA": "One-sentence summary of findings for this company",
        "AAPL": "One-sentence summary of findings for this company"
    },
    "confidence": "high|medium|low",
    "evidence_gaps": ["List any companies or topics with insufficient evidence"]
}

If the question is NOT a ranking question, return an empty list for "rankings".
Return ONLY valid JSON, no markdown fences."""


class MultiDocQA:
    """
    Multi-document question answering across all company filings.

    Retrieves from all companies simultaneously, groups by ticker,
    and synthesizes a comparative answer.
    """

    def __init__(self, config: dict):
        self.config = config
        self.client = OpenAI()
        self.model = config.get("generation", {}).get("model", "gpt-4o-mini")
        self.temperature = config.get("generation", {}).get("temperature", 0.1)

    def answer(
        self,
        question: str,
        sections: list[dict],
        top_k_per_company: int = 5,
        chunking_strategy: str = "semantic",
        retrieval_strategy: str = "hybrid",
    ) -> MultiDocAnswer:
        start_time = time.time()

        # Step 1: Group sections by company
        companies = self._group_by_company(sections)
        logger.info(f"Found {len(companies)} companies: {list(companies.keys())}")

        # Step 2: Retrieve relevant chunks per company
        all_evidence = {}
        total_chunks = 0

        for ticker, company_sections in companies.items():
            chunks = self._retrieve_for_company(
                question, company_sections, top_k_per_company,
                chunking_strategy, retrieval_strategy,
            )
            all_evidence[ticker] = chunks
            total_chunks += len(chunks)
            logger.info(f"  {ticker}: retrieved {len(chunks)} chunks")

        # Step 3: Build the prompt with organized evidence
        evidence_prompt = self._format_evidence(question, all_evidence)

        # Step 4: Generate comparative answer
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": MULTI_DOC_SYSTEM_PROMPT},
                {"role": "user", "content": evidence_prompt},
            ],
            response_format={"type": "json_object"},
        )

        raw_output = response.choices[0].message.content
        parsed = json.loads(raw_output)

        latency = time.time() - start_time

        return MultiDocAnswer(
            question=question,
            answer=parsed.get("answer", ""),
            company_evidence=all_evidence,
            rankings=parsed.get("rankings", []),
            synthesis_model=self.model,
            retrieval_strategy=retrieval_strategy,
            total_chunks_retrieved=total_chunks,
            latency_seconds=round(latency, 2),
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "per_company_summary": parsed.get("per_company_summary", {}),
                "confidence": parsed.get("confidence", "unknown"),
                "evidence_gaps": parsed.get("evidence_gaps", []),
            },
        )

    def _group_by_company(self, sections: list[dict]) -> dict[str, list[dict]]:
        """Group filing sections by company ticker."""
        companies: dict[str, list[dict]] = {}
        for section in sections:
            ticker = section.get("ticker", "UNKNOWN")
            if ticker not in companies:
                companies[ticker] = []
            companies[ticker].append(section)
        return dict(sorted(companies.items()))

    def _retrieve_for_company(
        self,
        question: str,
        sections: list[dict],
        top_k: int,
        chunking_strategy: str,
        retrieval_strategy: str,
    ) -> list[dict]:
        """
        Chunk and retrieve relevant passages for a single company.

        Returns list of {text, section, filing_date, score} dicts.
        """
        from src.chunking.strategies import get_chunker
        from src.retrieval.retrievers import build_retriever

        chunker_config = self.config.get("chunking", {}).get("strategies", {}).get(
            chunking_strategy, {"chunk_size": 512, "chunk_overlap": 50}
        )
        chunker = get_chunker(chunking_strategy, chunker_config)
        documents = chunker.chunk_sections(sections)

        if not documents:
            return []

        ticker = sections[0].get("ticker", "UNK")
        ret_config = self.config.get("retrieval", {}).get("strategies", {}).get(
            retrieval_strategy, {}
        ).copy()
        ret_config["collection_suffix"] = f"multidoc_{ticker}"
        ret_config["embedding_model"] = self.config.get("retrieval", {}).get(
            "embedding_model", "text-embedding-3-small"
        )
        ret_config["vectorstore_path"] = self.config.get("retrieval", {}).get(
            "vectorstore_path", "data/vectorstore"
        )

        retriever = build_retriever(retrieval_strategy, ret_config)
        retriever.index(documents)
        results = retriever.retrieve(question, top_k=top_k)

        return [
            {
                "text": r.content,
                "section": r.metadata.get("section", "unknown"),
                "filing_date": r.metadata.get("filing_date", "unknown"),
                "score": r.score,
            }
            for r in results
        ]

    def _format_evidence(
        self, question: str, evidence: dict[str, list[dict]]
    ) -> str:
        """Format retrieved evidence into a structured prompt."""
        parts = [f"QUESTION: {question}\n"]
        parts.append("EVIDENCE FROM SEC 10-K FILINGS:\n")

        for ticker, chunks in evidence.items():
            parts.append(f"{'=' * 50}")
            parts.append(f"COMPANY: {ticker}")
            parts.append(f"{'=' * 50}")

            if not chunks:
                parts.append("  [No relevant evidence found for this company]\n")
                continue

            for i, chunk in enumerate(chunks, 1):
                parts.append(f"\n--- {ticker} Evidence #{i} ---")
                parts.append(f"Section: {chunk['section']}")
                parts.append(f"Filing Date: {chunk['filing_date']}")
                parts.append(f"Relevance Score: {chunk['score']:.3f}")
                text = chunk["text"]
                if len(text) > 1500:
                    text = text[:1500] + "... [truncated]"
                parts.append(f"Content:\n{text}\n")

        parts.append(
            "\nBased on the evidence above, provide a comprehensive "
            "comparative analysis answering the question."
        )
        return "\n".join(parts)


def run_multi_doc_query(
    question: str,
    config: dict,
    sections: list[dict],
    top_k_per_company: int = 5,
) -> MultiDocAnswer:
    """
    Convenience function to run a multi-document query.
    """
    qa = MultiDocQA(config)
    result = qa.answer(question, sections, top_k_per_company=top_k_per_company)

    # Pretty-print results
    print(f"\n{'=' * 70}")
    print(f"MULTI-DOCUMENT QA")
    print(f"{'=' * 70}")
    print(f"Question: {question}")
    print(f"Companies analyzed: {list(result.company_evidence.keys())}")
    print(f"Total chunks retrieved: {result.total_chunks_retrieved}")
    print(f"Latency: {result.latency_seconds}s")
    print(f"Confidence: {result.usage.get('confidence', 'N/A')}")
    print(f"\n{'─' * 70}")
    print(f"\n{result.answer}")

    if result.rankings:
        print(f"\n{'─' * 70}")
        print("RANKINGS:")
        for i, r in enumerate(result.rankings, 1):
            print(f"  {i}. {r['ticker']} (score: {r['score']}/5) — {r['rationale']}")

    per_company = result.usage.get("per_company_summary", {})
    if per_company:
        print(f"\n{'─' * 70}")
        print("PER-COMPANY SUMMARY:")
        for ticker, summary in per_company.items():
            print(f"  {ticker}: {summary}")

    gaps = result.usage.get("evidence_gaps", [])
    if gaps:
        print(f"\n  Evidence gaps: {', '.join(gaps)}")

    print(f"\n{'=' * 70}")
    return result
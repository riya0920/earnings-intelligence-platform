"""
Generation & Risk Extraction Module

Two generation paths:
1. RAG Answer Generation: Takes retrieved context + user query,
   produces a cited answer using GPT-4o-mini.
2. Risk Signal Extraction: Structured extraction of risk categories
   from filing sections with severity scoring.

Both paths are designed for evaluation — every generation call
returns metadata needed by the RAGAS eval pipeline.
"""

import json
import logging
from dataclasses import dataclass, field

from openai import OpenAI

from src.retrieval.retrievers import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class GeneratedAnswer:
    """A generated answer with full provenance for evaluation."""
    query: str
    answer: str
    contexts: list[str]
    context_metadata: list[dict]
    model: str
    usage: dict = field(default_factory=dict)


@dataclass
class RiskSignal:
    """A structured risk signal extracted from a filing."""
    category: str
    severity: str  # "low", "medium", "high", "critical"
    summary: str
    evidence: str
    company: str
    ticker: str
    filing_date: str
    section: str


class RAGGenerator:
    """
    RAG answer generation with structured context injection.

    Formats retrieved documents into a context window, sends to
    GPT-4o-mini with a financial analyst system prompt, and returns
    a structured response with citations.
    """

    SYSTEM_PROMPT = """You are a senior financial analyst assistant. Your job is to answer 
questions about SEC filings and earnings call transcripts using ONLY the provided context.

Rules:
1. ONLY use information from the provided context documents. Never use outside knowledge.
2. ALWAYS cite which company, filing type, date, and section your information comes from.
3. If the context doesn't contain enough information to fully answer, say so explicitly.
4. Use precise financial language. Quote specific figures when available.
5. Structure your answer clearly with the most important findings first.

Format citations as: [Company, Filing Type, Date, Section]"""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ):
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _format_context(self, results: list[RetrievalResult]) -> str:
        """Format retrieval results into a numbered context block."""
        context_parts = []
        for i, result in enumerate(results, 1):
            meta = result.metadata
            header = (
                f"[Document {i}] "
                f"{meta.get('company', 'Unknown')} | "
                f"{meta.get('filing_type', '')} | "
                f"{meta.get('filing_date', '')} | "
                f"Section: {meta.get('section', 'unknown')}"
            )
            context_parts.append(f"{header}\n{result.content}")

        return "\n\n---\n\n".join(context_parts)

    def generate(
        self,
        query: str,
        retrieval_results: list[RetrievalResult],
    ) -> GeneratedAnswer:
        """
        Generate a cited answer from retrieved context.

        Args:
            query: User question
            retrieval_results: Retrieved documents from any retriever

        Returns:
            GeneratedAnswer with full provenance chain
        """
        context = self._format_context(retrieval_results)

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Context documents:\n\n{context}\n\n"
                    f"---\n\nQuestion: {query}\n\n"
                    f"Provide a thorough, well-cited answer based solely on the context above."
                ),
            },
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        answer = response.choices[0].message.content
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        return GeneratedAnswer(
            query=query,
            answer=answer,
            contexts=[r.content for r in retrieval_results],
            context_metadata=[r.metadata for r in retrieval_results],
            model=self.model,
            usage=usage,
        )


class RiskSignalExtractor:
    """
    Structured risk signal extraction from filing sections.

    Uses GPT-4o-mini with a structured output prompt to identify
    and categorize risk factors from 10-K filings, scoring them
    by severity and extracting supporting evidence.
    """

    EXTRACTION_PROMPT = """Analyze the following SEC filing section and extract risk signals.

For each risk signal found, provide:
- category: One of [litigation, regulatory, supply_chain, macroeconomic, cybersecurity, competitive]
- severity: One of [low, medium, high, critical]
- summary: One-sentence description of the risk (max 50 words)
- evidence: The key phrase or sentence from the text that supports this risk signal

Return your analysis as a JSON array. If no clear risk signals are found, return an empty array [].

Example output format:
[
  {{
    "category": "litigation",
    "severity": "high",
    "summary": "Ongoing antitrust lawsuit from DOJ could result in significant penalties",
    "evidence": "The Company is currently subject to..."
  }}
]

IMPORTANT: Return ONLY the JSON array, no other text."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ):
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature

    def extract_from_section(
        self,
        content: str,
        metadata: dict,
    ) -> list[RiskSignal]:
        """
        Extract risk signals from a single filing section.

        Args:
            content: Raw text of the filing section
            metadata: Section metadata (company, date, etc.)

        Returns:
            List of structured RiskSignal objects
        """
        # Truncate very long sections to fit context window
        max_chars = 12000
        if len(content) > max_chars:
            content = content[:max_chars] + "\n[...truncated...]"

        messages = [
            {
                "role": "user",
                "content": (
                    f"{self.EXTRACTION_PROMPT}\n\n"
                    f"Filing: {metadata.get('company', '')} "
                    f"{metadata.get('filing_type', '')} "
                    f"({metadata.get('filing_date', '')})\n"
                    f"Section: {metadata.get('section_name', '')}\n\n"
                    f"Text:\n{content}"
                ),
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=2048,
            )

            raw = response.choices[0].message.content.strip()
            # Clean potential markdown fencing
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

            signals_data = json.loads(raw)

            signals = []
            for s in signals_data:
                signals.append(RiskSignal(
                    category=s.get("category", "unknown"),
                    severity=s.get("severity", "medium"),
                    summary=s.get("summary", ""),
                    evidence=s.get("evidence", ""),
                    company=metadata.get("company", ""),
                    ticker=metadata.get("ticker", ""),
                    filing_date=metadata.get("filing_date", ""),
                    section=metadata.get("section_name", ""),
                ))

            logger.info(
                f"Extracted {len(signals)} risk signals from "
                f"{metadata.get('ticker', '?')} {metadata.get('filing_date', '?')}"
            )
            return signals

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse risk extraction response: {e}")
            return []

    def extract_from_filings(
        self,
        filings: list[dict],
        target_sections: list[str] | None = None,
    ) -> list[RiskSignal]:
        """
        Extract risk signals from all sections across multiple filings.

        Args:
            filings: List of filing dicts (from ingestion pipeline)
            target_sections: Which sections to analyze (default: risk_factors only)

        Returns:
            Aggregated list of RiskSignal objects
        """
        if target_sections is None:
            target_sections = ["risk_factors"]

        all_signals = []
        for filing in filings:
            for section in filing.get("sections", []):
                if section.get("section_name") in target_sections:
                    signals = self.extract_from_section(
                        content=section["content"],
                        metadata=section,
                    )
                    all_signals.extend(signals)

        return all_signals

    @staticmethod
    def signals_to_dataframe(signals: list[RiskSignal]):
        """Convert risk signals to a pandas DataFrame for analysis."""
        import pandas as pd
        return pd.DataFrame([
            {
                "company": s.company,
                "ticker": s.ticker,
                "filing_date": s.filing_date,
                "category": s.category,
                "severity": s.severity,
                "summary": s.summary,
                "evidence": s.evidence[:200],
            }
            for s in signals
        ])

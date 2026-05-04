"""
Verified RAG Generator
======================

Composes EIP's existing RAG generation (prose answer over retrieved
chunks) with the vendored Adversarial Financial Auditor (structured
numeric extraction with three-layer verification).

Why this layer exists
---------------------
EIP's `RAGGenerator` returns a prose answer. For quantitative analyst
questions ("what was Apple's Q3 revenue?"), the prose contains numbers
that the LLM read off retrieved chunks — with no verification. If the
LLM grabs a forecast figure instead of an actual, EIP confidently
emits the wrong number.

`VerifiedRAGGenerator` runs the same retrieval, then dual-tracks:
  1. Prose track: existing RAGGenerator behavior, unchanged.
  2. Structured track: format the retrieved chunks as a mini-document
     with `[N]` paragraph markers (where N = chunk rank), feed to the
     auditor's three-agent graph, get back verified numbers with
     paragraph citations that map back to the source chunks.

The "paragraph number" in the auditor's output is the *chunk index*
in the retrieval results, so the analyst sees provenance like:
  "revenue $85.8B verified — chunk 3 (AAPL 10-Q, 2024-08-01)"

Composition, not replacement: prose still flows from RAGGenerator;
structured facts come from the auditor; the verification panel shows
both alongside.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from src.generation.generator import GeneratedAnswer, RAGGenerator
from src.retrieval.retrievers import RetrievalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type — extends GeneratedAnswer with structured + verification.
# ---------------------------------------------------------------------------


@dataclass
class StructuredFact:
    """One verified numeric fact extracted from the retrieved context."""
    field_name: str
    value: float
    unit: str  # e.g. "USD_millions", "percent"
    source_quote: str
    chunk_index: int  # index into VerifiedAnswer.contexts
    chunk_metadata: dict  # company, filing_type, filing_date, section
    is_actual: bool
    confidence: str  # "high" | "medium" | "low"
    provenance_verified: bool  # passed Layer 2

    def to_dict(self) -> dict:
        return {
            "field_name": self.field_name,
            "value": self.value,
            "unit": self.unit,
            "source_quote": self.source_quote,
            "chunk_index": self.chunk_index,
            "chunk_metadata": self.chunk_metadata,
            "is_actual": self.is_actual,
            "confidence": self.confidence,
            "provenance_verified": self.provenance_verified,
        }


@dataclass
class VerificationReport:
    """Summary of the auditor's verification verdict."""
    consensus_met: bool
    iterations: int
    provenance_all_passed: bool
    provenance_failed_count: int
    consistency_anomaly_count: int
    consistency_anomaly_checks: list[str]  # which checks fired
    audit_log_tail: list[str]
    rationale: Optional[str] = None

    @property
    def status(self) -> str:
        """One-word verdict for UI badges."""
        if self.consensus_met and self.provenance_all_passed and self.consistency_anomaly_count == 0:
            return "verified"
        if self.consensus_met and self.consistency_anomaly_count == 0:
            return "verified_with_warnings"
        return "disputed"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "consensus_met": self.consensus_met,
            "iterations": self.iterations,
            "provenance_all_passed": self.provenance_all_passed,
            "provenance_failed_count": self.provenance_failed_count,
            "consistency_anomaly_count": self.consistency_anomaly_count,
            "consistency_anomaly_checks": self.consistency_anomaly_checks,
            "audit_log_tail": self.audit_log_tail,
            "rationale": self.rationale,
        }


@dataclass
class VerifiedAnswer:
    """Prose answer + verified structured facts + verification report."""
    query: str
    answer: str  # prose, from RAGGenerator
    contexts: list[str]
    context_metadata: list[dict]
    structured_facts: list[StructuredFact]
    verification: VerificationReport
    consistency_anomalies: list[dict] = field(default_factory=list)
    model: str = ""
    usage: dict = field(default_factory=dict)

    @property
    def has_quantitative_facts(self) -> bool:
        return len(self.structured_facts) > 0


# ---------------------------------------------------------------------------
# Chunk-to-document conversion.
# ---------------------------------------------------------------------------


def format_chunks_as_document(retrieval_results: list[RetrievalResult]) -> str:
    """Format retrieved chunks into the auditor's expected document format.

    The auditor splits documents on `[N]` paragraph markers and uses N
    as the paragraph_number. By emitting one chunk per `[N]` block, the
    auditor's "paragraph number" in the final report is exactly the
    rank-1-indexed position of the chunk in the retrieval results,
    which is how downstream UIs map verified facts back to source
    metadata.
    """
    blocks: list[str] = []
    for i, r in enumerate(retrieval_results, start=1):
        meta = r.metadata or {}
        header = (
            f"({meta.get('company', 'Unknown')} | "
            f"{meta.get('filing_type', '?')} | "
            f"{meta.get('filing_date', '?')} | "
            f"{meta.get('section', 'unknown')})"
        )
        blocks.append(f"[{i}] {header}\n{r.content}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Generator.
# ---------------------------------------------------------------------------


# Fields the auditor's HunterReport may populate. Order matches the
# canonical ordering used in the standalone auditor's eval.
_DOLLAR_FIELDS = ["revenue", "cogs", "gross_profit", "ebitda", "net_income",
                  "prior_period_revenue"]
_PCT_FIELDS = ["gross_margin_pct", "ebitda_margin_pct", "net_margin_pct",
               "yoy_revenue_growth_pct"]


class VerifiedRAGGenerator:
    """RAG generator that composes prose generation with verified extraction.

    Drop-in replacement for `RAGGenerator` for callers that want
    structured + verified numbers alongside the prose answer.

    On qualitative queries (no numbers in the retrieved chunks), the
    auditor returns nulls everywhere, the verification report shows zero
    structured facts, and the response degrades gracefully to prose-only.
    Cost on qualitative queries is the auditor's fixed overhead
    (~$0.008-0.015/query) — accepted as the price of always-on
    verification, because routing-around adds a misclassification
    failure mode that defeats the entire point.
    """

    def __init__(
        self,
        prose_model: str = "gpt-4o-mini",
        prose_temperature: float = 0.1,
        prose_max_tokens: int = 1024,
    ):
        self.prose_generator = RAGGenerator(
            model=prose_model,
            temperature=prose_temperature,
            max_tokens=prose_max_tokens,
        )

    def generate(
        self,
        query: str,
        retrieval_results: list[RetrievalResult],
    ) -> VerifiedAnswer:
        """Generate a verified answer from retrieved chunks.

        Runs prose generation and structured extraction in sequence.
        (Could be parallelized with asyncio, but the auditor's internal
        Hunter+Auditor branches already parallelize the slow LLM calls,
        and serial execution makes debugging tractable.)
        """
        # 1. Prose track — existing RAGGenerator, unchanged behavior.
        prose = self.prose_generator.generate(query, retrieval_results)

        # 2. Structured track — auditor over the retrieved chunks.
        # Imported lazily so EIP can run prose-only flows even if the
        # auditor's heavier deps (langgraph, langchain-google-genai) are
        # missing.
        try:
            from src.auditor import run_audit
        except ImportError as e:
            logger.warning(
                "Auditor package unavailable (%s); returning prose-only answer.", e
            )
            return self._prose_only(query, prose, retrieval_results,
                                    reason=f"auditor unavailable: {e}")

        document = format_chunks_as_document(retrieval_results)
        try:
            final = run_audit(document)
        except Exception as e:
            logger.warning("Auditor run failed (%s); returning prose-only.", e)
            return self._prose_only(query, prose, retrieval_results,
                                    reason=f"auditor failed: {e}")

        structured = self._extract_structured_facts(final, retrieval_results)
        verification = self._build_verification_report(final)

        return VerifiedAnswer(
            query=query,
            answer=prose.answer,
            contexts=prose.contexts,
            context_metadata=prose.context_metadata,
            structured_facts=structured,
            verification=verification,
            consistency_anomalies=final.get("consistency_anomalies") or [],
            model=prose.model,
            usage=prose.usage,
        )

    # ------------------------------------------------------------------
    # Internals.
    # ------------------------------------------------------------------

    def _prose_only(
        self,
        query: str,
        prose: GeneratedAnswer,
        retrieval_results: list[RetrievalResult],
        *,
        reason: str,
    ) -> VerifiedAnswer:
        """Graceful degradation when the auditor can't run."""
        return VerifiedAnswer(
            query=query,
            answer=prose.answer,
            contexts=prose.contexts,
            context_metadata=prose.context_metadata,
            structured_facts=[],
            verification=VerificationReport(
                consensus_met=False,
                iterations=0,
                provenance_all_passed=False,
                provenance_failed_count=0,
                consistency_anomaly_count=0,
                consistency_anomaly_checks=[],
                audit_log_tail=[f"[skip] {reason}"],
                rationale=reason,
            ),
            model=prose.model,
            usage=prose.usage,
        )

    def _extract_structured_facts(
        self,
        final_state: dict,
        retrieval_results: list[RetrievalResult],
    ) -> list[StructuredFact]:
        """Walk the Hunter's report and emit one StructuredFact per non-null field."""
        hunter = final_state.get("hunter_report") or {}
        if not isinstance(hunter, dict):
            return []

        # Map provenance verdicts by field for quick lookup.
        prov = (final_state.get("provenance_report") or {}).get("verdicts") or []
        verdict_by_field = {v["field_name"]: v["passed"] for v in prov}

        facts: list[StructuredFact] = []
        for fname in _DOLLAR_FIELDS + _PCT_FIELDS:
            m = hunter.get(fname)
            if not isinstance(m, dict) or m.get("value") is None:
                continue
            chunk_idx = m.get("paragraph_number")
            # Auditor's paragraph_number is 1-indexed and equals the
            # chunk's rank-1-indexed position. Convert to 0-indexed for
            # list lookup; clamp defensively.
            chunk_meta: dict = {}
            zero_idx = -1
            if isinstance(chunk_idx, int) and 1 <= chunk_idx <= len(retrieval_results):
                zero_idx = chunk_idx - 1
                chunk_meta = retrieval_results[zero_idx].metadata or {}
            facts.append(StructuredFact(
                field_name=fname,
                value=float(m["value"]),
                unit="USD_millions" if fname in _DOLLAR_FIELDS else "percent",
                source_quote=str(m.get("source_quote") or ""),
                chunk_index=zero_idx,
                chunk_metadata=chunk_meta,
                is_actual=bool(m.get("is_actual", True)),
                confidence=str(m.get("confidence", "medium")),
                provenance_verified=verdict_by_field.get(fname, False),
            ))
        return facts

    def _build_verification_report(self, final_state: dict) -> VerificationReport:
        prov = final_state.get("provenance_report") or {}
        anomalies = final_state.get("consistency_anomalies") or []
        decision = final_state.get("arbiter_decision") or {}
        log_tail = (final_state.get("audit_log") or [])[-3:]
        return VerificationReport(
            consensus_met=bool(final_state.get("consensus_met")),
            iterations=int(final_state.get("iterations", 0)),
            provenance_all_passed=bool(prov.get("all_passed", False)) if prov else False,
            provenance_failed_count=int(prov.get("failed_count", 0)) if prov else 0,
            consistency_anomaly_count=len(anomalies),
            consistency_anomaly_checks=[a.get("check", "") for a in anomalies],
            audit_log_tail=log_tail,
            rationale=str(decision.get("rationale", "")) if decision else None,
        )

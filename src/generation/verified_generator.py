"""
Verified RAG Generator
======================

Composes EIP's existing RAG generation (prose answer over retrieved
chunks) with the vendored Adversarial Financial Auditor (structured
numeric extraction with three-layer verification) and the runtime
XBRL Reconciliation Agent (canonical-data anchor + delta classifier).

Why this layer exists
---------------------
EIP's `RAGGenerator` returns a prose answer. For quantitative analyst
questions ("what was Apple's Q3 revenue?"), the prose contains numbers
that the LLM read off retrieved chunks â€” with no verification. If the
LLM grabs a forecast figure instead of an actual, EIP confidently
emits the wrong number.

`VerifiedRAGGenerator` runs the same retrieval, then four-tracks:
  1. Prose track: existing RAGGenerator behavior, unchanged.
  2. Structured track: format the retrieved chunks as a mini-document
     with `[N]` paragraph markers, feed to the auditor's three-agent
     graph, get back verified numbers with paragraph citations.
  3. Reconciliation track (new): for each verified number, look up
     the canonical us-gaap XBRL value and classify any delta into
     one of five known reporting patterns.
  4. Verification report: assembles status / consensus / anomaly counts.

The reconciliation step turns "verified extraction" into "verified
extraction reconciled against structured truth." See
src/reconciliation/ for the agent and the five-pattern classifier.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from src.generation.generator import GeneratedAnswer, RAGGenerator
from src.retrieval.retrievers import RetrievalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types.
# ---------------------------------------------------------------------------


@dataclass
class StructuredFact:
    """One verified numeric fact extracted from the retrieved context."""
    field_name: str
    value: float
    unit: str
    source_quote: str
    chunk_index: int
    chunk_metadata: dict
    is_actual: bool
    confidence: str
    provenance_verified: bool

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
    consistency_anomaly_checks: list[str]
    audit_log_tail: list[str]
    rationale: Optional[str] = None

    @property
    def status(self) -> str:
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
    """Prose + verified structured facts + verification + reconciliation."""
    query: str
    answer: str
    contexts: list[str]
    context_metadata: list[dict]
    structured_facts: list[StructuredFact]
    verification: VerificationReport
    consistency_anomalies: list[dict] = field(default_factory=list)
    # ReconciliationReport from src.reconciliation. Typed as Optional to
    # support graceful degradation if the agent fails or is disabled.
    reconciliation: Optional[object] = None
    model: str = ""
    usage: dict = field(default_factory=dict)

    @property
    def has_quantitative_facts(self) -> bool:
        return len(self.structured_facts) > 0


# ---------------------------------------------------------------------------
# Chunk-to-document conversion.
# ---------------------------------------------------------------------------


def format_chunks_as_document(retrieval_results: list[RetrievalResult]) -> str:
    """Format retrieved chunks as a [N]-marked document for the auditor."""
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


_DOLLAR_FIELDS = ["revenue", "cogs", "gross_profit", "ebitda", "net_income",
                  "prior_period_revenue"]
_PCT_FIELDS = ["gross_margin_pct", "ebitda_margin_pct", "net_margin_pct",
               "yoy_revenue_growth_pct"]


class VerifiedRAGGenerator:
    """RAG generator: prose + verified extraction + XBRL reconciliation.

    Composition order (each step is fail-soft):
      1. Prose track via RAGGenerator (unchanged from base EIP).
      2. Structured track via the auditor (Hunter+Auditor+Verifier+Arbiter).
      3. Reconciliation track via the ReconciliationAgent: for each
         verified fact, looks up the corresponding us-gaap XBRL value
         and classifies any delta.

    Failure modes:
      â€¢ Auditor import/run fails  -> prose-only response with skip reason
      â€¢ ReconciliationAgent fails -> verified response without reconciliation
      â€¢ XBRL lookup fails for a field -> that finding is LOOKUP_FAILED
    """

    def __init__(
        self,
        prose_model: str = "gpt-4o-mini",
        prose_temperature: float = 0.1,
        prose_max_tokens: int = 1024,
        reconciliation_enabled: bool = True,
    ):
        self.prose_generator = RAGGenerator(
            model=prose_model,
            temperature=prose_temperature,
            max_tokens=prose_max_tokens,
        )
        self.reconciliation_enabled = reconciliation_enabled

    def generate(
        self,
        query: str,
        retrieval_results: list[RetrievalResult],
    ) -> VerifiedAnswer:
        # 1. Prose track.
        prose = self.prose_generator.generate(query, retrieval_results)

        # 2. Structured track (auditor).
        try:
            from src.auditor import run_audit
        except ImportError as e:
            logger.warning("Auditor unavailable (%s); prose-only.", e)
            return self._prose_only(query, prose, retrieval_results,
                                    reason=f"auditor unavailable: {e}")

        document = format_chunks_as_document(retrieval_results)
        try:
            final = run_audit(document)
        except Exception as e:
            logger.warning("Auditor run failed (%s); prose-only.", e)
            return self._prose_only(query, prose, retrieval_results,
                                    reason=f"auditor failed: {e}")

        structured = self._extract_structured_facts(final, retrieval_results)
        verification = self._build_verification_report(final)

        # 3. Reconciliation track (new).
        reconciliation = None
        if self.reconciliation_enabled and structured:
            reconciliation = self._reconcile(structured)

        return VerifiedAnswer(
            query=query,
            answer=prose.answer,
            contexts=prose.contexts,
            context_metadata=prose.context_metadata,
            structured_facts=structured,
            verification=verification,
            consistency_anomalies=final.get("consistency_anomalies") or [],
            reconciliation=reconciliation,
            model=prose.model,
            usage=prose.usage,
        )

    # ------------------------------------------------------------------
    # Internals.
    # ------------------------------------------------------------------

    def _reconcile(self, facts: list[StructuredFact]):
        """Run XBRL reconciliation. Fail-soft: returns None on error."""
        try:
            from src.reconciliation import (
                ProseFactInput,
                ReconciliationAgent,
            )
        except ImportError as e:
            logger.warning("Reconciliation unavailable (%s); skipping.", e)
            return None

        try:
            agent = ReconciliationAgent()
            inputs = [
                ProseFactInput(
                    field_name=f.field_name,
                    value=f.value,
                    unit=f.unit,
                    source_quote=f.source_quote,
                    chunk_metadata=f.chunk_metadata,
                )
                for f in facts
            ]
            return agent.reconcile(inputs)
        except Exception as e:  # noqa: BLE001
            logger.warning("Reconciliation failed (%s); skipping.", e)
            return None

    def _prose_only(
        self,
        query: str,
        prose: GeneratedAnswer,
        retrieval_results: list[RetrievalResult],
        *,
        reason: str,
    ) -> VerifiedAnswer:
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
            reconciliation=None,
            model=prose.model,
            usage=prose.usage,
        )

    def _extract_structured_facts(
        self,
        final_state: dict,
        retrieval_results: list[RetrievalResult],
    ) -> list[StructuredFact]:
        hunter = final_state.get("hunter_report") or {}
        if not isinstance(hunter, dict):
            return []
        prov = (final_state.get("provenance_report") or {}).get("verdicts") or []
        verdict_by_field = {v["field_name"]: v["passed"] for v in prov}

        facts: list[StructuredFact] = []
        for fname in _DOLLAR_FIELDS + _PCT_FIELDS:
            m = hunter.get(fname)
            if not isinstance(m, dict) or m.get("value") is None:
                continue
            chunk_idx = m.get("paragraph_number")
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
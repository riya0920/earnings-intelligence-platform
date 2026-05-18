"""
src/reconciliation
==================

Runtime XBRL reconciliation: the second-layer defense for verified RAG.

The verified pipeline (src/auditor + src/generation/verified_generator)
proves a number was extracted faithfully from the retrieved chunks.
This package compares each verified number to its canonical us-gaap
XBRL value at runtime and classifies any delta into one of five
known reporting patterns, surfacing UNEXPLAINED deltas for human
review.

Public API
----------
    XBRLLookup            â€” cache-friendly, fail-soft XBRL lookups
    ReconciliationAgent   â€” five-pattern delta classifier
    ProseFactInput        â€” minimal input shape from the verified pipeline
    DeltaPattern          â€” enum of the six output classifications
    ReconciliationFinding â€” per-fact result
    ReconciliationReport  â€” aggregated result with histogram

Typical usage from the verified generator:

    from src.reconciliation import ReconciliationAgent, ProseFactInput

    agent = ReconciliationAgent()
    inputs = [ProseFactInput(field_name=f.field_name,
                              value=f.value,
                              unit=f.unit,
                              source_quote=f.source_quote,
                              chunk_metadata=f.chunk_metadata)
              for f in verified_answer.structured_facts]
    report = agent.reconcile(inputs)
    print(report.pattern_counts)
"""
from src.reconciliation.agent import ProseFactInput, ReconciliationAgent
from src.reconciliation.schema import (
    DeltaPattern,
    ReconciliationFinding,
    ReconciliationReport,
)
from src.reconciliation.xbrl import (
    Fact,
    FactKey,
    TICKER_TO_CIK,
    XBRLLookup,
    XBRL_TAG_MAP,
    fetch_company_facts,
    index_facts,
    latest_period_facts,
)

__all__ = [
    # Agent
    "ReconciliationAgent",
    "ProseFactInput",
    # Schema
    "DeltaPattern",
    "ReconciliationFinding",
    "ReconciliationReport",
    # XBRL
    "XBRLLookup",
    "Fact",
    "FactKey",
    "TICKER_TO_CIK",
    "XBRL_TAG_MAP",
    "fetch_company_facts",
    "index_facts",
    "latest_period_facts",
]
"""
src/auditor — Adversarial Financial Auditor (vendored).

Multi-agent verification layer for structured numeric extraction with
three layers of defense against shared-blind-spot failures:

  Layer 1 — Heterogeneous models (Hunter Gemini 1.5 Pro, Auditor
            configurable via AUDITOR_MODEL env var).
  Layer 2 — Deterministic provenance verification (every Hunter-cited
            value must appear at the cited paragraph in the source).
  Layer 3 — Consistency checks (extracted primaries must reconcile
            against any margins / growth rates stated in the document).

Public API
----------
  run_audit(document) -> dict
      Run the full graph end-to-end on a document. Returns the final
      LangGraph state with hunter_report, auditor_report, arbiter_decision,
      provenance_report, consistency_anomalies, consensus_met, iterations.

  AgentState — graph state TypedDict (for advanced usage)
  HunterReport, AuditorReport, ArbiterDecision — Pydantic schemas

Standalone CLI version: github.com/riya0920/adversarial-auditor
"""
from .runner import build_graph, run_audit
from .state import (
    AgentState,
    ArbiterDecision,
    AuditorReport,
    FinancialMetric,
    HunterReport,
    DELTA_THRESHOLD,
    MAX_ITERATIONS,
)

__all__ = [
    "run_audit",
    "build_graph",
    "AgentState",
    "ArbiterDecision",
    "AuditorReport",
    "FinancialMetric",
    "HunterReport",
    "DELTA_THRESHOLD",
    "MAX_ITERATIONS",
]

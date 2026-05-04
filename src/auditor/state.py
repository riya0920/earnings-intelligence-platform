"""
state.py
========

Schemas and graph-state definitions for the Adversarial Financial Auditor.

Adversarial Logic
-----------------
The system separates *extraction* (Hunter), *independent verification*
(Forensic Auditor), and *adjudication* (Arbiter) into three isolated agents
that never share intermediate reasoning. The Pydantic models below enforce
strict provenance: every numeric claim must carry a verbatim source quote and
a paragraph index so disputes can be resolved by pointing to ground truth in
the document, not to model intuition.

The graph state is a `TypedDict` because LangGraph reducers (e.g.
`operator.add` for the audit log) compose more cleanly with TypedDict than
with Pydantic models. The agent reports themselves are stored as
`dict` (the `model_dump()` of a Pydantic model) so they can be serialized
through LangGraph's persistence layer without surprises.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, Optional, TypedDict

from pydantic import BaseModel, Field, field_validator


# ----------------------------------------------------------------------------
# Pydantic v2 schemas — used for structured LLM output validation.
# ----------------------------------------------------------------------------


class FinancialMetric(BaseModel):
    """A single extracted financial metric with full provenance.

    Every metric must include a verbatim quote and an `is_actual` flag. The
    `is_actual` flag is the primary defense against the "Tricky Transcript"
    failure mode where forecasts and guidance are mistaken for reported
    actuals.
    """

    field_name: str = Field(description="Canonical metric name, lowercase snake_case.")
    value: float = Field(description="Numeric value, normalized to millions USD.")
    source_quote: str = Field(
        description="Verbatim quote from the source document. No paraphrasing."
    )
    paragraph_number: Optional[int] = Field(
        default=None,
        description="1-indexed paragraph containing the source quote.",
    )
    is_actual: bool = Field(
        description=(
            "True if this is a *reported actual* result. "
            "False if it is a forecast, guidance, target, or expectation."
        )
    )
    confidence: Literal["high", "medium", "low"] = Field(default="medium")

    @field_validator("field_name")
    @classmethod
    def _normalize_name(cls, v: str) -> str:
        return v.strip().lower().replace(" ", "_")


class HunterReport(BaseModel):
    """High-recall extraction output.

    The Hunter is OPTIMISTIC: it reports every metric it can find. Missing
    fields are explicitly `None` rather than silently absent, so the
    Arbiter can distinguish between "not present" and "not reported."

    Layer 3 fields (margins, growth, prior period) are extracted when the
    document explicitly states them. They are NOT computed by the Hunter;
    they enable the Auditor's deterministic consistency checks to catch
    cases where Hunter and Auditor agree on a wrong primitive (e.g., both
    pick the forecast revenue) but the document also states a margin or
    growth rate that is consistent only with the *correct* primitive.
    """

    revenue: Optional[FinancialMetric] = None
    cogs: Optional[FinancialMetric] = None
    gross_profit: Optional[FinancialMetric] = None
    ebitda: Optional[FinancialMetric] = None
    net_income: Optional[FinancialMetric] = None

    # Layer 3 — stated ratios and comparison-period values. Optional;
    # populated only when the document explicitly states them.
    gross_margin_pct: Optional[FinancialMetric] = None
    ebitda_margin_pct: Optional[FinancialMetric] = None
    net_margin_pct: Optional[FinancialMetric] = None
    yoy_revenue_growth_pct: Optional[FinancialMetric] = None
    prior_period_revenue: Optional[FinancialMetric] = None

    notes: str = Field(default="", description="Free-text caveats for the Arbiter.")


class AuditorReport(BaseModel):
    """Independent verification output.

    The Auditor performs its own extraction and then *recomputes* derived
    quantities from primitives. Any deviation between what the document
    states and what the Auditor's arithmetic yields is recorded as a
    `flagged_anomaly`. This is the second line of defense against tricky
    transcripts: even if both agents misread a value, an internal
    inconsistency (e.g., gross_profit != revenue - cogs) will surface here.
    """

    independently_extracted: dict[str, Optional[float]] = Field(
        description="Metric name -> value, extracted without seeing the Hunter."
    )
    derived_calculations: dict[str, Optional[float]] = Field(
        description="Metric name -> value, computed from primitives.",
        default_factory=dict,
    )
    skeptical_critiques: list[str] = Field(default_factory=list)
    flagged_anomalies: list[str] = Field(default_factory=list)


class ArbiterDecision(BaseModel):
    """Adjudication output.

    The Arbiter never reads the document directly. It compares the Hunter's
    and Auditor's reports and emits a structured decision. Routing logic
    keys off `consensus_met`; if false, `dispute_instructions` are passed
    back to both extractors for the next iteration.
    """

    consensus_met: bool
    field_deltas: dict[str, float] = Field(
        description="Metric -> relative delta (e.g., 0.0001 = 0.01%)."
    )
    disputed_fields: list[str] = Field(default_factory=list)
    dispute_instructions: Optional[str] = Field(
        default=None,
        description="Specific guidance for Hunter and Auditor on the next pass.",
    )
    rationale: str = Field(description="Human-readable summary of the decision.")


# ----------------------------------------------------------------------------
# LangGraph state.
# ----------------------------------------------------------------------------


def _merge_error_messages(left: Optional[str], right: Optional[str]) -> Optional[str]:
    """Reducer for `last_error` so Hunter and Auditor can both write in parallel.

    LangGraph requires every state field touched by concurrent branches to
    declare a merge reducer. Without it, a parallel write triggers
    INVALID_CONCURRENT_GRAPH_UPDATE. This reducer keeps either error if one
    is null, or concatenates both when both branches fail simultaneously
    (e.g., shared upstream API outage).
    """
    if left is None:
        return right
    if right is None:
        return left
    if left == right:
        return left
    return f"{left} | {right}"


class AgentState(TypedDict, total=False):
    """Cyclic graph state.

    The `audit_log`, `last_error`, and `error_count` fields all use
    reducers because the parallel Hunter+Auditor branches can write to
    them concurrently.

    Layer 2/3 additions:
      - `provenance_report`: output of provenance.verify_hunter_report.
        Populated by the verifier node between extraction and arbitration.
      - `consistency_anomalies`: list of deterministic consistency-check
        anomalies (e.g., gross_margin recomputed from primitives doesn't
        match the gross_margin stated in the document).
    """

    raw_document: str
    hunter_report: Optional[dict]
    auditor_report: Optional[dict]
    arbiter_decision: Optional[dict]
    audit_log: Annotated[list[str], operator.add]
    consensus_met: bool
    iterations: int
    dispute_instructions: Optional[str]
    last_error: Annotated[Optional[str], _merge_error_messages]
    error_count: Annotated[int, operator.add]

    # Layer 2 — provenance verification.
    provenance_report: Optional[dict]
    # Layer 3 — deterministic consistency checks.
    consistency_anomalies: Optional[list]


# ----------------------------------------------------------------------------
# Constants.
# ----------------------------------------------------------------------------

#: Relative delta threshold above which the Arbiter triggers a dispute.
#: 0.0001 == 0.01%, matching the spec.
DELTA_THRESHOLD: float = 0.0001

#: Hard ceiling on dispute iterations to prevent unbounded loops.
MAX_ITERATIONS: int = 3

#: Hard ceiling on JSON self-heal retries per node, per iteration.
MAX_ERROR_RETRIES: int = 2
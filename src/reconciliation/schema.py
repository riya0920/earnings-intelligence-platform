"""
src/reconciliation/schema.py
============================

Types for the runtime XBRL reconciliation step.

The verified pipeline produces prose-extracted numbers. The reconciliation
layer compares each one against the corresponding XBRL fact (canonical
SEC-tagged value) and classifies any delta into one of five well-known
reporting patterns, plus a catch-all for things that genuinely don't fit.

Design intent
-------------
Perfect transparency, not perfect accuracy. A `ReconciliationFinding`
either explains the delta or surfaces it for human review. We do not
silently pick a winner.

The five patterns were chosen because they cover ~all of the legitimate
reasons a prose number and an XBRL number differ in public filings:
  1. NON_GAAP_VS_GAAP   â€” "Adjusted EBITDA $X" vs us-gaap:NetIncome
  2. PERIODICITY        â€” TTM or YTD vs single-quarter
  3. HIERARCHY          â€” segment / division total vs consolidated
  4. VERSIONING         â€” originally-reported vs restated
  5. METRIC_MAPPING     â€” the model confused two related metrics
                            (e.g. operating income reported as EBITDA)

Anything else is UNEXPLAINED â€” the agent's honest signal that a human
should look at this case.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DeltaPattern(str, Enum):
    """How a prose-XBRL discrepancy is classified."""
    NO_DELTA = "no_delta"                    # values match within tolerance
    NON_GAAP_VS_GAAP = "non_gaap_vs_gaap"    # prose is an adjusted figure
    PERIODICITY = "periodicity"              # TTM/YTD vs quarter mismatch
    HIERARCHY = "hierarchy"                  # segment vs consolidated
    VERSIONING = "versioning"                # original vs restated
    METRIC_MAPPING = "metric_mapping"        # wrong metric type confused for the right one
    UNEXPLAINED = "unexplained"              # surface for human review
    LOOKUP_FAILED = "lookup_failed"          # no XBRL fact available

    @property
    def is_legitimate(self) -> bool:
        """True if the delta has a known reporting-context explanation.

        Legitimate deltas are differences the analyst should understand
        but not "fix"; an unexplained delta is a candidate hallucination
        and should be flagged for human review.
        """
        return self in {
            DeltaPattern.NO_DELTA,
            DeltaPattern.NON_GAAP_VS_GAAP,
            DeltaPattern.PERIODICITY,
            DeltaPattern.HIERARCHY,
            DeltaPattern.VERSIONING,
            DeltaPattern.METRIC_MAPPING,
        }


@dataclass
class ReconciliationFinding:
    """One field's worth of prose-vs-XBRL comparison."""

    field_name: str                          # e.g. "revenue"
    prose_value: float                       # what the auditor extracted, USD_millions
    xbrl_value: Optional[float]              # canonical XBRL value, USD_millions
    xbrl_tag: Optional[str]                  # e.g. "Revenues" or "NetIncomeLoss"
    xbrl_period_end: Optional[str]           # YYYY-MM-DD
    delta_pct: Optional[float]               # signed % delta; None if no XBRL match
    pattern: DeltaPattern
    rationale: str                           # one-sentence explanation
    human_review_recommended: bool = False   # set True iff pattern is UNEXPLAINED

    @property
    def matches(self) -> bool:
        """Does the prose value agree with XBRL within tolerance?"""
        return self.pattern == DeltaPattern.NO_DELTA

    def to_dict(self) -> dict:
        return {
            "field_name": self.field_name,
            "prose_value": self.prose_value,
            "xbrl_value": self.xbrl_value,
            "xbrl_tag": self.xbrl_tag,
            "xbrl_period_end": self.xbrl_period_end,
            "delta_pct": self.delta_pct,
            "pattern": self.pattern.value,
            "rationale": self.rationale,
            "human_review_recommended": self.human_review_recommended,
        }


@dataclass
class ReconciliationReport:
    """Aggregated reconciliation across all extracted facts."""

    findings: list[ReconciliationFinding] = field(default_factory=list)
    ticker: Optional[str] = None
    period_end_used: Optional[str] = None

    @property
    def total(self) -> int:
        return len(self.findings)

    @property
    def matches(self) -> int:
        return sum(1 for f in self.findings if f.pattern == DeltaPattern.NO_DELTA)

    @property
    def legitimate_deltas(self) -> int:
        """Deltas the agent successfully classified (excludes NO_DELTA)."""
        return sum(1 for f in self.findings
                   if f.pattern.is_legitimate and f.pattern != DeltaPattern.NO_DELTA)

    @property
    def human_review_count(self) -> int:
        return sum(1 for f in self.findings if f.human_review_recommended)

    @property
    def lookup_failed_count(self) -> int:
        return sum(1 for f in self.findings
                   if f.pattern == DeltaPattern.LOOKUP_FAILED)

    @property
    def pattern_counts(self) -> dict[str, int]:
        """Histogram by pattern, useful for the README governance table."""
        counts: dict[str, int] = {}
        for f in self.findings:
            counts[f.pattern.value] = counts.get(f.pattern.value, 0) + 1
        return counts

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "period_end_used": self.period_end_used,
            "total": self.total,
            "matches": self.matches,
            "legitimate_deltas": self.legitimate_deltas,
            "human_review_count": self.human_review_count,
            "lookup_failed_count": self.lookup_failed_count,
            "pattern_counts": self.pattern_counts,
            "findings": [f.to_dict() for f in self.findings],
        }
"""
eval/reconciliation_scorer.py
=============================

Score the Reconciliation Agent's pattern classification on Bucket D.

Where eval/scorer.py answers "did the system extract the right NUMBER",
this module answers "did the system correctly classify the CAUSE of a
prose-vs-XBRL delta into one of five known patterns".

Scoring is partial-credit (locked in earlier in the planning session):
  â€¢ 1.0   â€” exact pattern match
  â€¢ 0.5   â€” any other 'legitimate' pattern (not UNEXPLAINED, not NO_DELTA)
  â€¢ 0.0   â€” UNEXPLAINED (agent flagged for human review when it should have classified)
             NO_DELTA (agent saw no delta when there should have been one)
             LOOKUP_FAILED (XBRL had no fact; pre-classification failure)
             None / missing report (agent didn't run at all)

Why partial credit
------------------
A "wrong but legitimate" classification is meaningfully different from
"unable to classify". An interviewer asking about the system's behavior
on the Apple Services example wants to distinguish:
  (a) agent said HIERARCHY â€” correct                       -> 1.0
  (b) agent said NON_GAAP â€” wrong but still in the family   -> 0.5
  (c) agent said UNEXPLAINED â€” flagged for human review     -> 0.0
  (d) agent said NO_DELTA â€” missed entirely                 -> 0.0

Tracking (a)/(b)/(c)/(d) separately captures whether the agent is
"trying and missing" vs "punting" vs "missing the delta entirely".
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from eval.bucket_d import LEGITIMATE_PATTERNS, extract_expected_pattern
from eval.question_set import Question


class ReconciliationVerdict(str, Enum):
    """Per-question scoring outcome for Bucket D classification."""
    PATTERN_MATCH = "pattern_match"
    PATTERN_LEGITIMATE_ALTERNATIVE = "pattern_legitimate_alternative"
    INCORRECTLY_FLAGGED_UNEXPLAINED = "incorrectly_flagged_unexplained"
    NO_DELTA_WHEN_EXPECTED = "no_delta_when_expected"
    LOOKUP_FAILED = "lookup_failed"
    NO_REPORT = "no_report"               # agent didn't produce a finding
    NOT_BUCKET_D = "not_bucket_d"         # question wasn't designed for D scoring


@dataclass
class ReconciliationScore:
    """One question's classification outcome."""
    qid: str
    expected_pattern: Optional[str]
    actual_pattern: Optional[str]
    verdict: ReconciliationVerdict
    credit: float                           # 0.0 / 0.5 / 1.0
    matched_field: Optional[str] = None     # which fact's pattern we scored
    rationale: str = ""                     # agent's rationale (for audit trail)

    def to_dict(self) -> dict:
        return {
            "qid": self.qid,
            "expected_pattern": self.expected_pattern,
            "actual_pattern": self.actual_pattern,
            "verdict": self.verdict.value,
            "credit": self.credit,
            "matched_field": self.matched_field,
            "rationale": self.rationale,
        }


def score_reconciliation(
    question: Question,
    reconciliation_report: Optional[dict],
) -> ReconciliationScore:
    """Score one question's reconciliation outcome.

    Args:
        question: the eval Question (must be bucket == "D_reconciliation").
        reconciliation_report: the agent's report as a dict (already
            serialized via .to_dict()), or None if the agent didn't run.

    Returns:
        A ReconciliationScore with verdict + credit + audit fields.
    """
    expected = extract_expected_pattern(question)
    if expected is None:
        return ReconciliationScore(
            qid=question.qid,
            expected_pattern=None,
            actual_pattern=None,
            verdict=ReconciliationVerdict.NOT_BUCKET_D,
            credit=0.0,
            rationale="question is not Bucket D",
        )

    if not reconciliation_report:
        return ReconciliationScore(
            qid=question.qid,
            expected_pattern=expected,
            actual_pattern=None,
            verdict=ReconciliationVerdict.NO_REPORT,
            credit=0.0,
            rationale="reconciliation agent did not produce a report",
        )

    # Find the finding whose field_name matches the question's field.
    # If the question's field isn't represented (e.g. agent couldn't
    # find an EBITDA fact), fall back to the first finding.
    findings = reconciliation_report.get("findings", []) or []
    if not findings:
        return ReconciliationScore(
            qid=question.qid,
            expected_pattern=expected,
            actual_pattern=None,
            verdict=ReconciliationVerdict.NO_REPORT,
            credit=0.0,
            rationale="reconciliation report contained zero findings",
        )

    target_field = question.field
    finding = next(
        (f for f in findings if f.get("field_name") == target_field),
        findings[0],
    )

    actual = finding.get("pattern")
    rationale = finding.get("rationale", "")

    # NO_DELTA when we expected a delta = agent missed entirely.
    if actual == "no_delta":
        return ReconciliationScore(
            qid=question.qid,
            expected_pattern=expected,
            actual_pattern=actual,
            verdict=ReconciliationVerdict.NO_DELTA_WHEN_EXPECTED,
            credit=0.0,
            matched_field=finding.get("field_name"),
            rationale=rationale,
        )

    # LOOKUP_FAILED is a pre-classification failure â€” doesn't count
    # for or against the agent, but we track it for the governance table.
    if actual == "lookup_failed":
        return ReconciliationScore(
            qid=question.qid,
            expected_pattern=expected,
            actual_pattern=actual,
            verdict=ReconciliationVerdict.LOOKUP_FAILED,
            credit=0.0,
            matched_field=finding.get("field_name"),
            rationale=rationale,
        )

    # UNEXPLAINED when we expected a known pattern = human-review punt.
    if actual == "unexplained":
        return ReconciliationScore(
            qid=question.qid,
            expected_pattern=expected,
            actual_pattern=actual,
            verdict=ReconciliationVerdict.INCORRECTLY_FLAGGED_UNEXPLAINED,
            credit=0.0,
            matched_field=finding.get("field_name"),
            rationale=rationale,
        )

    # Exact pattern match: full credit.
    if actual == expected:
        return ReconciliationScore(
            qid=question.qid,
            expected_pattern=expected,
            actual_pattern=actual,
            verdict=ReconciliationVerdict.PATTERN_MATCH,
            credit=1.0,
            matched_field=finding.get("field_name"),
            rationale=rationale,
        )

    # Different legitimate pattern: partial credit.
    if actual in LEGITIMATE_PATTERNS:
        return ReconciliationScore(
            qid=question.qid,
            expected_pattern=expected,
            actual_pattern=actual,
            verdict=ReconciliationVerdict.PATTERN_LEGITIMATE_ALTERNATIVE,
            credit=0.5,
            matched_field=finding.get("field_name"),
            rationale=rationale,
        )

    # Anything else (e.g. malformed pattern from a future code change)
    # we treat as zero credit, marked as unexplained for safety.
    return ReconciliationScore(
        qid=question.qid,
        expected_pattern=expected,
        actual_pattern=actual,
        verdict=ReconciliationVerdict.INCORRECTLY_FLAGGED_UNEXPLAINED,
        credit=0.0,
        matched_field=finding.get("field_name"),
        rationale=f"unrecognized pattern '{actual}'",
    )


def aggregate_reconciliation_scores(
    scores: list[ReconciliationScore],
) -> dict:
    """Aggregate Bucket D scores into a summary suitable for the report.

    Returns a dict ready to print or render in the README governance table.
    """
    bucket_d = [s for s in scores if s.verdict != ReconciliationVerdict.NOT_BUCKET_D]
    if not bucket_d:
        return {
            "total": 0, "credit_sum": 0.0, "accuracy_pct": 0.0,
            "exact_matches": 0, "legitimate_alternatives": 0,
            "unexplained": 0, "missed_no_delta": 0, "lookup_failed": 0,
        }

    credit_sum = sum(s.credit for s in bucket_d)
    return {
        "total": len(bucket_d),
        "credit_sum": round(credit_sum, 2),
        "accuracy_pct": round(100.0 * credit_sum / len(bucket_d), 1),
        "exact_matches": sum(
            1 for s in bucket_d
            if s.verdict == ReconciliationVerdict.PATTERN_MATCH
        ),
        "legitimate_alternatives": sum(
            1 for s in bucket_d
            if s.verdict == ReconciliationVerdict.PATTERN_LEGITIMATE_ALTERNATIVE
        ),
        "unexplained": sum(
            1 for s in bucket_d
            if s.verdict == ReconciliationVerdict.INCORRECTLY_FLAGGED_UNEXPLAINED
        ),
        "missed_no_delta": sum(
            1 for s in bucket_d
            if s.verdict == ReconciliationVerdict.NO_DELTA_WHEN_EXPECTED
        ),
        "lookup_failed": sum(
            1 for s in bucket_d
            if s.verdict == ReconciliationVerdict.LOOKUP_FAILED
        ),
        "no_report": sum(
            1 for s in bucket_d
            if s.verdict == ReconciliationVerdict.NO_REPORT
        ),
    }
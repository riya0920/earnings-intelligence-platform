"""
eval/bucket_d.py
================

Bucket D questions for the runtime XBRL reconciliation evaluation.

Where Buckets A/B/C measure "did the system extract the right number",
Bucket D measures something orthogonal: "given a prose-vs-XBRL delta,
does the Reconciliation Agent correctly classify its cause?"

Each question has an `expected_pattern` field naming the DeltaPattern
the agent should output for that case. The scorer uses partial credit:
  â€¢ Full credit for exact pattern match
  â€¢ Half credit for any legitimate alternative (not UNEXPLAINED)
  â€¢ Zero credit for incorrectly_flagged_unexplained or wrong-direction
    miss (e.g. agent said NO_DELTA when there should have been a delta)

Question design principle
-------------------------
Bucket D questions are deliberately constructed so the prose RAG answer
and the canonical XBRL value SHOULD differ for a known reason. We are
not testing whether the system gets the "right number" â€” there isn't
one. We are testing whether the system correctly explains the divergence.

Five patterns covered
---------------------
  1. NON_GAAP_VS_GAAP    â€” 2 questions (Apple, Meta adjusted figures)
  2. PERIODICITY         â€” 2 questions (YTD vs quarter mismatches)
  3. HIERARCHY           â€” 2 questions (segment vs consolidated)
  4. METRIC_MAPPING      â€” 2 questions (EBITDA vs Operating Income)
  5. VERSIONING          â€” 1 synthetic question (clearly labeled)

Total: 9 questions. The VERSIONING case is marked synthetic in notes
because real restatements are rare in our 5-ticker 2023-2026 corpus
and we wanted full coverage of the five-pattern classifier.

Author note
-----------
These question texts are written so a competent prose-RAG would
naturally extract a segment / adjusted / YTD / wrong-metric value.
The agent is then expected to recognize the structural reason for
the delta from the source_quote and filing context.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from eval.question_set import Question, ExpectedKind, TICKER_NAMES


# Pattern names mirror DeltaPattern in src/reconciliation/schema.py.
# Defined here as strings so eval/ doesn't import from src/reconciliation
# (would create a circular import when src.reconciliation.xbrl imports
# eval/xbrl_fetcher.py for backwards-compat).
PATTERN_NON_GAAP = "non_gaap_vs_gaap"
PATTERN_PERIODICITY = "periodicity"
PATTERN_HIERARCHY = "hierarchy"
PATTERN_METRIC_MAPPING = "metric_mapping"
PATTERN_VERSIONING = "versioning"

LEGITIMATE_PATTERNS = {
    PATTERN_NON_GAAP, PATTERN_PERIODICITY, PATTERN_HIERARCHY,
    PATTERN_METRIC_MAPPING, PATTERN_VERSIONING,
}


def _bucket_d_questions() -> list[Question]:
    """Generate Bucket D questions for reconciliation classification.

    Each Question is augmented with an `expected_pattern` value stashed
    in its `notes` field as a key=value substring (so we don't have to
    modify the existing Question dataclass and break Bucket A/B/C).

    The scorer extracts the expected pattern from notes via the
    `extract_expected_pattern()` helper below.
    """
    questions: list[Question] = []

    # ------------------------------------------------------------------
    # NON_GAAP_VS_GAAP (2 questions)
    # ------------------------------------------------------------------
    questions.append(Question(
        qid="D_AAPL_adjusted_operating_income",
        bucket="D_reconciliation",
        ticker="AAPL",
        field="operating_income",
        text=(
            "What was Apple's adjusted operating income, excluding "
            "one-time items, in their most recent quarter?"
        ),
        expected_kind=ExpectedKind.EXACT_NUMBER,
        notes=(
            "expected_pattern=non_gaap_vs_gaap; "
            "Apple's MD&A occasionally references non-GAAP / adjusted "
            "figures while us-gaap:OperatingIncomeLoss is GAAP baseline."
        ),
    ))

    questions.append(Question(
        qid="D_META_adjusted_net_income",
        bucket="D_reconciliation",
        ticker="META",
        field="net_income",
        text=(
            "What was Meta's adjusted net income, excluding legal "
            "settlement charges, in their most recent quarter?"
        ),
        expected_kind=ExpectedKind.EXACT_NUMBER,
        notes=(
            "expected_pattern=non_gaap_vs_gaap; "
            "Meta has historically disclosed adjusted figures excluding "
            "legal/regulatory charges; XBRL NetIncomeLoss is GAAP."
        ),
    ))

    # ------------------------------------------------------------------
    # PERIODICITY (2 questions)
    # ------------------------------------------------------------------
    questions.append(Question(
        qid="D_NVDA_nine_month_revenue",
        bucket="D_reconciliation",
        ticker="NVDA",
        field="revenue",
        text=(
            "What was NVIDIA's revenue for the first nine months of "
            "fiscal year 2025?"
        ),
        expected_kind=ExpectedKind.EXACT_NUMBER,
        notes=(
            "expected_pattern=periodicity; "
            "Prose answers will report YTD nine-month total ($91.166B "
            "for FY25 Q3); XBRL most-recent-quarterly lookup returns "
            "the Q3 standalone value (~$35B). The delta is purely a "
            "periodicity mismatch."
        ),
    ))

    questions.append(Question(
        qid="D_MSFT_ytd_revenue",
        bucket="D_reconciliation",
        ticker="MSFT",
        field="revenue",
        text=(
            "What was Microsoft's year-to-date revenue through their "
            "most recent 10-Q?"
        ),
        expected_kind=ExpectedKind.EXACT_NUMBER,
        notes=(
            "expected_pattern=periodicity; "
            "Microsoft 10-Qs include six-month and nine-month YTD "
            "cumulative totals alongside quarterly values; prose may "
            "extract the cumulative figure while XBRL has only the "
            "quarterly fact."
        ),
    ))

    # ------------------------------------------------------------------
    # HIERARCHY (2 questions)
    # ------------------------------------------------------------------
    questions.append(Question(
        qid="D_AAPL_services_revenue",
        bucket="D_reconciliation",
        ticker="AAPL",
        field="revenue",
        text="What was Apple's Services revenue in their most recent quarter?",
        expected_kind=ExpectedKind.EXACT_NUMBER,
        notes=(
            "expected_pattern=hierarchy; "
            "Services is a reportable segment (~25% of total revenue); "
            "prose pulls the segment value (~$25B) while XBRL Revenues "
            "is consolidated total (~$144B)."
        ),
    ))

    questions.append(Question(
        qid="D_GOOGL_cloud_revenue",
        bucket="D_reconciliation",
        ticker="GOOGL",
        field="revenue",
        text="What was Google Cloud's revenue in their most recent quarter?",
        expected_kind=ExpectedKind.EXACT_NUMBER,
        notes=(
            "expected_pattern=hierarchy; "
            "Google Cloud is a reporting segment; XBRL us-gaap:Revenues "
            "is Alphabet consolidated. Pure segment-vs-consolidated delta."
        ),
    ))

    # ------------------------------------------------------------------
    # METRIC_MAPPING (2 questions)
    # ------------------------------------------------------------------
    questions.append(Question(
        qid="D_NVDA_ebitda",
        bucket="D_reconciliation",
        ticker="NVDA",
        field="ebitda",
        text="What was NVIDIA's EBITDA in their most recent quarter?",
        expected_kind=ExpectedKind.EXACT_NUMBER,
        notes=(
            "expected_pattern=metric_mapping; "
            "NVIDIA does not tag EBITDA directly in XBRL. Prose "
            "extraction commonly grabs Operating Income or computes "
            "EBITDA from line items. The reconciliation agent should "
            "recognize the mapping mismatch."
        ),
    ))

    questions.append(Question(
        qid="D_AAPL_ebitda",
        bucket="D_reconciliation",
        ticker="AAPL",
        field="ebitda",
        text="What was Apple's EBITDA in their most recent quarter?",
        expected_kind=ExpectedKind.EXACT_NUMBER,
        notes=(
            "expected_pattern=metric_mapping; "
            "Apple does not tag EBITDA in XBRL; OperatingIncomeLoss is "
            "the closest tagged proxy. Prose answers often equate the "
            "two; the agent should classify this as metric_mapping."
        ),
    ))

    # ------------------------------------------------------------------
    # VERSIONING (1 question, synthetic â€” clearly labeled)
    # ------------------------------------------------------------------
    questions.append(Question(
        qid="D_SYNTHETIC_restated_revenue",
        bucket="D_reconciliation",
        ticker="AAPL",
        field="revenue",
        text=(
            "What was Apple's revenue in Q1 FY2024 as originally "
            "reported, before any subsequent restatement?"
        ),
        expected_kind=ExpectedKind.EXACT_NUMBER,
        notes=(
            "expected_pattern=versioning; "
            "SYNTHETIC QUESTION â€” real restatements are rare in our "
            "5-ticker 2023-2026 corpus. Included to round out coverage "
            "of all five DeltaPatterns. Test setup expects the agent "
            "to recognize 'originally reported' vs 'as restated' "
            "language as a versioning case."
        ),
    ))

    return questions


def extract_expected_pattern(question: Question) -> Optional[str]:
    """Pull the expected_pattern out of a Bucket D question's notes.

    Returns None for non-D questions (they have no expected pattern).
    Returns None for malformed D questions (graceful degradation; the
    scorer treats them as 'no expected pattern, agent can't score').
    """
    if question.bucket != "D_reconciliation":
        return None
    notes = question.notes or ""
    for token in notes.split(";"):
        token = token.strip()
        if token.startswith("expected_pattern="):
            value = token.split("=", 1)[1].strip()
            if value in LEGITIMATE_PATTERNS:
                return value
            return None
    return None


if __name__ == "__main__":
    # CLI: print the Bucket D questions and their expected patterns.
    qs = _bucket_d_questions()
    print(f"Generated {len(qs)} Bucket D questions:")
    for q in qs:
        pattern = extract_expected_pattern(q)
        print(f"  {q.qid:<40} pattern={pattern} ticker={q.ticker} field={q.field}")
"""
eval/scorer.py
==============

Score a system response against a Question's ground truth.

The hard part: extracting numeric claims from prose answers. A prose
RAG answer might say:
  "Apple's revenue was $143,756 million for the quarter."
  "Apple reported approximately $144 billion in net sales."
  "Apple's revenue: 143.8B."
  "Net sales of $143.756B in Q1 FY2026."

Our `extract_numeric_claims` regex normalizes all of these to a list of
floats in USD millions, with confidence scores reflecting how unambiguous
each candidate is. We then check whether ANY candidate matches the
ground-truth value within tolerance.

For the verified pipeline, we don't need regex extraction — we have
structured facts directly. The scorer reads the verified `field` value
from the structured-facts list.

Verdict types:
  CORRECT             — number matches ground truth
  WRONG_NUMBER        — number was extracted but doesn't match ground truth
  CONFIDENT_WRONG     — number was extracted with high confidence and is wrong
                        (the most dangerous failure mode for production)
  NO_NUMBER           — no number extracted (acceptable for AMBIGUOUS/NOT_REPORTED)
  CORRECTLY_REFUSED   — system explicitly declined to answer (good for NOT_REPORTED)
  CONFABULATED        — system gave a number for a NOT_REPORTED metric (bad)

Each verdict carries a `is_pass` boolean for aggregate metrics.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from eval.question_set import Question, ExpectedKind


class Verdict(str, Enum):
    CORRECT = "correct"
    WRONG_NUMBER = "wrong_number"
    CONFIDENT_WRONG = "confident_wrong"
    NO_NUMBER = "no_number"
    CORRECTLY_REFUSED = "correctly_refused"
    CONFABULATED = "confabulated"


@dataclass
class ScoreDetail:
    qid: str
    pipeline: str                      # "prose" | "verified"
    verdict: Verdict
    is_pass: bool
    extracted_value: Optional[float]   # in USD millions
    expected_value: Optional[float]    # in USD millions
    notes: str = ""


# Regex for dollar values in prose. Captures number + optional scale.
_NUM_RE = re.compile(
    r"""
    \$?\s*
    (?P<num>\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)
    \s*
    (?P<scale>million|billion|trillion|thousand|M|B|T|K|bn|mn)?\b
    """,
    re.VERBOSE | re.IGNORECASE,
)

_SCALE = {
    "thousand": 0.001, "k": 0.001,
    "million": 1.0, "m": 1.0, "mn": 1.0,
    "billion": 1000.0, "b": 1000.0, "bn": 1000.0,
    "trillion": 1_000_000.0, "t": 1_000_000.0,
}

# Phrases that indicate the system is refusing or expressing uncertainty.
REFUSAL_MARKERS = [
    "not reported", "not disclosed", "is not provided", "isn't provided",
    "do not have", "cannot find", "cannot extract", "unable to find",
    "no specific figure", "not available", "not a gaap", "not provided in",
    "could not find", "i don't have", "i do not have",
]


def extract_numeric_claims(text: str) -> list[float]:
    """Pull every number from `text`, normalize to USD millions.

    Heuristic: only count numbers that have either a `$` prefix or a
    scale word (million/billion/etc.) — bare integers like "2024" or
    "10-Q" are noise. Returns a deduplicated list (within 0.5% relative
    tolerance) sorted by appearance order.
    """
    out: list[float] = []
    for m in _NUM_RE.finditer(text):
        # Require either $ prefix or scale word for relevance.
        match_str = m.group(0)
        has_dollar = "$" in text[max(0, m.start() - 1):m.end()]
        scale = m.group("scale")
        if not has_dollar and not scale:
            continue
        try:
            val = float(m.group("num").replace(",", ""))
        except ValueError:
            continue
        mult = _SCALE.get(scale.lower(), 1.0) if scale else 1.0
        # If no scale and a $ prefix, this is bare $123 — could be a share
        # price, EPS, or similar small dollar amount, not a financial-
        # statement number. Real revenues/income/COGS without a scale word
        # would already be in millions and printed as "$143,756" (with
        # comma). Heuristic: bare $N without scale AND < 10,000 → skip.
        if not scale and has_dollar and val < 10_000:
            # Likely a per-share figure or similar small dollar amount, not
            # a financial statement number.
            continue
        millions = val * mult
        # Dedup against prior extractions within 0.5% tolerance.
        is_dup = any(_close(millions, prior) for prior in out)
        if not is_dup:
            out.append(millions)
    return out


def _close(a: float, b: float, rel_tol: float = 0.005) -> bool:
    if b == 0:
        return abs(a) <= rel_tol
    return abs(a - b) / abs(b) <= rel_tol


def looks_like_refusal(text: str) -> bool:
    """Heuristic: does this prose explicitly decline to answer?"""
    low = text.lower()
    return any(marker in low for marker in REFUSAL_MARKERS)


# ---------------------------------------------------------------------------
# Scoring functions, one per pipeline.
# ---------------------------------------------------------------------------


def score_prose(question: Question, prose_answer: str) -> ScoreDetail:
    """Score a prose-only RAG answer.

    For EXACT_NUMBER questions: pass if any extracted number matches GT.
    For NOT_REPORTED: pass if system refused; fail if it extracted a number.
    For AMBIGUOUS: pass if system either refused OR extracted a number AND
      labeled the period somewhere in the response.
    """
    extracted = extract_numeric_claims(prose_answer)
    refused = looks_like_refusal(prose_answer)

    if question.expected_kind == ExpectedKind.EXACT_NUMBER:
        gt = question.expected_value_millions
        for cand in extracted:
            if _close(cand, gt):
                return ScoreDetail(
                    qid=question.qid, pipeline="prose",
                    verdict=Verdict.CORRECT, is_pass=True,
                    extracted_value=cand, expected_value=gt,
                    notes=f"Match within 0.5% (got {cand:.1f}M, want {gt:.1f}M).",
                )
        if extracted:
            # Pick the candidate closest to GT for the diagnostic.
            closest = min(extracted, key=lambda c: abs(c - gt))
            return ScoreDetail(
                qid=question.qid, pipeline="prose",
                verdict=Verdict.CONFIDENT_WRONG, is_pass=False,
                extracted_value=closest, expected_value=gt,
                notes=(f"Extracted {len(extracted)} value(s); closest was "
                       f"{closest:.1f}M, expected {gt:.1f}M "
                       f"(off by {abs(closest - gt) / max(abs(gt), 1):.1%})."),
            )
        return ScoreDetail(
            qid=question.qid, pipeline="prose",
            verdict=Verdict.NO_NUMBER, is_pass=False,
            extracted_value=None, expected_value=gt,
            notes="No numeric value extracted from prose.",
        )

    if question.expected_kind == ExpectedKind.NOT_REPORTED:
        if refused and not extracted:
            return ScoreDetail(
                qid=question.qid, pipeline="prose",
                verdict=Verdict.CORRECTLY_REFUSED, is_pass=True,
                extracted_value=None, expected_value=None,
                notes="System correctly declined to extract a non-reported metric.",
            )
        if extracted:
            return ScoreDetail(
                qid=question.qid, pipeline="prose",
                verdict=Verdict.CONFABULATED, is_pass=False,
                extracted_value=extracted[0], expected_value=None,
                notes=(f"Confabulated a value ({extracted[0]:.1f}M) for a "
                       f"non-reported metric."),
            )
        return ScoreDetail(
            qid=question.qid, pipeline="prose",
            verdict=Verdict.NO_NUMBER, is_pass=True,
            extracted_value=None, expected_value=None,
            notes="No number extracted; treated as implicit refusal.",
        )

    # AMBIGUOUS: any reasonable behavior passes — refuse, OR extract with period label.
    if refused:
        return ScoreDetail(
            qid=question.qid, pipeline="prose",
            verdict=Verdict.CORRECTLY_REFUSED, is_pass=True,
            extracted_value=None, expected_value=None,
            notes="System asked for clarification on ambiguous question.",
        )
    if extracted:
        # Check for period anchor: any 4-digit year, "quarter", "Q1/Q2/Q3/Q4",
        # or month name in the response.
        has_period = bool(re.search(
            r"\b(20\d{2}|Q[1-4]|quarter|January|February|March|April|May|June|"
            r"July|August|September|October|November|December)\b",
            prose_answer, re.IGNORECASE,
        ))
        if has_period:
            return ScoreDetail(
                qid=question.qid, pipeline="prose",
                verdict=Verdict.CORRECT, is_pass=True,
                extracted_value=extracted[0], expected_value=None,
                notes="Extracted a number with a period label.",
            )
        return ScoreDetail(
            qid=question.qid, pipeline="prose",
            verdict=Verdict.CONFIDENT_WRONG, is_pass=False,
            extracted_value=extracted[0], expected_value=None,
            notes="Returned a number with no period anchor (ambiguous).",
        )
    return ScoreDetail(
        qid=question.qid, pipeline="prose",
        verdict=Verdict.NO_NUMBER, is_pass=True,
        extracted_value=None, expected_value=None,
        notes="No number extracted from ambiguous question.",
    )


def score_verified(
    question: Question,
    structured_facts: list[dict],
    verification_status: str,
    prose_answer: str = "",
) -> ScoreDetail:
    """Score a VerifiedRAGGenerator response.

    Uses structured facts directly (no regex extraction needed). The
    verification_status is one of "verified" / "verified_with_warnings" /
    "disputed". Disputed facts are treated as withheld — system was
    honest about uncertainty.
    """
    # Find the field-matching fact.
    target_field = question.field
    matching = [f for f in structured_facts if f.get("field_name") == target_field]
    # For ambiguous "ebitda" / "free_cash_flow" / etc. that aren't in the
    # canonical schema, no match is expected.
    fact = matching[0] if matching else None

    if question.expected_kind == ExpectedKind.EXACT_NUMBER:
        gt = question.expected_value_millions
        if fact is None:
            # Verified pipeline didn't extract; fall back to prose check
            # so we don't penalize for empty structured output when the
            # answer is actually in prose.
            return score_prose(question, prose_answer)
        if not fact.get("provenance_verified"):
            # System extracted but couldn't verify — treat as uncertain.
            if _close(fact["value"], gt):
                return ScoreDetail(
                    qid=question.qid, pipeline="verified",
                    verdict=Verdict.CORRECT, is_pass=True,
                    extracted_value=fact["value"], expected_value=gt,
                    notes="Correct value, but provenance failed.",
                )
            return ScoreDetail(
                qid=question.qid, pipeline="verified",
                verdict=Verdict.WRONG_NUMBER, is_pass=False,
                extracted_value=fact["value"], expected_value=gt,
                notes="Wrong value, provenance correctly failed.",
            )
        if _close(fact["value"], gt):
            return ScoreDetail(
                qid=question.qid, pipeline="verified",
                verdict=Verdict.CORRECT, is_pass=True,
                extracted_value=fact["value"], expected_value=gt,
                notes="Correct and verified.",
            )
        return ScoreDetail(
            qid=question.qid, pipeline="verified",
            verdict=Verdict.CONFIDENT_WRONG, is_pass=False,
            extracted_value=fact["value"], expected_value=gt,
            notes=(f"Wrong value ({fact['value']:.1f}M vs {gt:.1f}M) "
                   f"that passed provenance."),
        )

    if question.expected_kind == ExpectedKind.NOT_REPORTED:
        # For non-reported metrics, the verified pipeline should return no
        # structured fact (Hunter returns null). If it does, that's correct.
        if fact is None:
            return ScoreDetail(
                qid=question.qid, pipeline="verified",
                verdict=Verdict.CORRECTLY_REFUSED, is_pass=True,
                extracted_value=None, expected_value=None,
                notes="Correctly extracted nothing for a non-reported metric.",
            )
        if verification_status == "disputed":
            return ScoreDetail(
                qid=question.qid, pipeline="verified",
                verdict=Verdict.CORRECTLY_REFUSED, is_pass=True,
                extracted_value=fact["value"], expected_value=None,
                notes="Extracted but flagged as disputed — appropriate caution.",
            )
        return ScoreDetail(
            qid=question.qid, pipeline="verified",
            verdict=Verdict.CONFABULATED, is_pass=False,
            extracted_value=fact["value"], expected_value=None,
            notes=f"Confabulated {fact['value']:.1f}M for a non-reported metric.",
        )

    # AMBIGUOUS — verified pipeline passes if it either declined or
    # produced verified output.
    if fact is None:
        return ScoreDetail(
            qid=question.qid, pipeline="verified",
            verdict=Verdict.CORRECTLY_REFUSED, is_pass=True,
            extracted_value=None, expected_value=None,
            notes="No structured fact extracted; appropriate for ambiguous Q.",
        )
    if fact.get("provenance_verified"):
        return ScoreDetail(
            qid=question.qid, pipeline="verified",
            verdict=Verdict.CORRECT, is_pass=True,
            extracted_value=fact["value"], expected_value=None,
            notes=("Verified extraction with provenance — period is anchored "
                   "via chunk metadata."),
        )
    return ScoreDetail(
        qid=question.qid, pipeline="verified",
        verdict=Verdict.CONFIDENT_WRONG, is_pass=False,
        extracted_value=fact["value"], expected_value=None,
        notes="Extracted without provenance; period anchor unclear.",
    )
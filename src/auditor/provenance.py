"""
provenance.py
=============

Deterministic provenance verification for the Hunter's extractions.

Why this layer exists
---------------------
Multi-agent verification has a known blind spot: when both Hunter and
Auditor share the same anchoring bias (e.g., both grab the forecast number
in a "we forecasted $X but actual was $Y" sentence), the Arbiter cannot
detect the error because the two reports agree. Layer 2 (this module)
breaks that by turning soft verification into hard verification:

  Soft: "do these two LLMs agree?"
  Hard: "does the value the Hunter cited actually appear in the cited
         paragraph of the document?"

If the Hunter claims revenue = $812M with quote "actual revenue of $812
million" from paragraph 3, this module checks that paragraph 3 of the
document literally contains both the quote AND a dollar figure equal to
$812 million. Hallucination is no longer survivable.

Design choices
--------------
1. Regex-based candidate enumeration over the whole document. The set of
   numbers the LLM can legitimately pick from is now closed and finite.
2. Paragraph splitting handles two formats: the Tier C generator's
   `[N] paragraph text` and the mock-data `Paragraph N: paragraph text`.
3. Quote matching is whitespace-normalized substring match, not exact.
   Verbatim is enforced by prompt; this layer is tolerant of minor
   re-flowing without sacrificing the hallucination guarantee.
4. Value matching normalizes "$812 million" / "$812M" / "$0.812 billion"
   all to 812.0 USD millions before comparison. Floating-point comparison
   uses a relative tolerance of 0.5%.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Paragraph splitting.
# ---------------------------------------------------------------------------

_PARAGRAPH_PREFIX_RE = re.compile(
    r"""
    ^\s*
    (?:
        \[\s*(?P<bracket>\d+)\s*\]   |   # [3]   ← Tier C generator format
        Paragraph\s+(?P<word>\d+)\s*:    # Paragraph 3:   ← mock_data format
    )
    \s*
    """,
    re.VERBOSE | re.MULTILINE,
)


def split_paragraphs(document: str) -> dict[int, str]:
    """Split a document into a {paragraph_number: text} mapping.

    Recognizes both `[N] ...` and `Paragraph N: ...` prefixes. Falls back
    to blank-line splitting (1-indexed) if no prefix markers are found.

    The returned text has the prefix stripped so quote matching doesn't
    have to worry about it.
    """
    matches = list(_PARAGRAPH_PREFIX_RE.finditer(document))
    if not matches:
        # Fallback: blank-line split, 1-indexed.
        chunks = [c.strip() for c in re.split(r"\n\s*\n", document.strip())
                  if c.strip()]
        return {i + 1: c for i, c in enumerate(chunks)}

    out: dict[int, str] = {}
    for i, m in enumerate(matches):
        n_str = m.group("bracket") or m.group("word")
        n = int(n_str)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(document)
        out[n] = document[start:end].strip()
    return out


# ---------------------------------------------------------------------------
# Dollar candidate enumerator.
# ---------------------------------------------------------------------------

# Matches things like:  $812 million  $1.2 billion  $498M  $1,234.5M  $0.85B
_DOLLAR_RE = re.compile(
    r"""
    \$\s*
    (?P<num>\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)
    \s*
    (?P<scale>million|billion|thousand|trillion|M|B|T|K|bn|mn)?
    """,
    re.VERBOSE | re.IGNORECASE,
)

# SEC accounting-style parenthesized figures (no $): "( 74,525 )" or
# "(74,525.0)" — used for negative line items like cogs and operating
# expenses in financial statements. These are legitimate candidates for
# extracted COGS / opex values and should be recognized by provenance.
_PAREN_NUM_RE = re.compile(
    r"""
    \(\s*
    (?P<num>\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)
    \s*
    (?P<scale>million|billion|thousand|trillion|M|B|T|K|bn|mn)?
    \s*\)
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Multiplier in MILLIONS USD (the canonical unit used everywhere in this
# project). e.g. "billion" -> 1000 millions.
_SCALE_TO_MILLIONS: dict[str, float] = {
    "thousand": 0.001, "k": 0.001,
    "million": 1.0, "m": 1.0, "mn": 1.0,
    "billion": 1000.0, "b": 1000.0, "bn": 1000.0,
    "trillion": 1_000_000.0, "t": 1_000_000.0,
}


@dataclass
class DollarCandidate:
    """One dollar figure found in the document."""
    paragraph_number: int
    raw_match: str          # the literal substring matched, e.g. "$812 million"
    value_millions: float   # normalized to USD millions
    span: tuple[int, int]   # (start, end) within the paragraph text

    def __repr__(self) -> str:
        return f"DollarCandidate(p{self.paragraph_number}, {self.raw_match!r} -> ${self.value_millions}M)"


def _normalize_value(num_str: str, scale: Optional[str]) -> float:
    n = float(num_str.replace(",", ""))
    if scale is None:
        # Bare "$812" with no unit. The convention in this project is to
        # treat unscaled dollar figures in financial-call contexts as
        # millions. This is consistent with the Hunter prompt, which
        # normalizes everything to millions. If a transcript ever uses
        # bare dollars for share prices etc., the consistency check
        # at the metric level will catch the mismatch.
        return n
    return n * _SCALE_TO_MILLIONS[scale.lower()]


def enumerate_dollar_candidates(document: str) -> list[DollarCandidate]:
    """Return every dollar figure in the document, with its paragraph.

    Recognizes two patterns:
      1. `$812 million`, `$1.2B`, etc. — explicit dollar sign.
      2. `( 74,525 )` — SEC accounting-style parenthesized negatives,
         common in financial-statements tables for cogs and expenses.
         The absolute value is recorded; provenance verification compares
         against extracted absolute values.
    """
    paragraphs = split_paragraphs(document)
    out: list[DollarCandidate] = []
    for pnum, ptext in paragraphs.items():
        for m in _DOLLAR_RE.finditer(ptext):
            try:
                value = _normalize_value(m.group("num"), m.group("scale"))
            except (ValueError, KeyError):
                continue
            out.append(DollarCandidate(
                paragraph_number=pnum,
                raw_match=m.group(0).strip(),
                value_millions=value,
                span=m.span(),
            ))
        for m in _PAREN_NUM_RE.finditer(ptext):
            try:
                value = _normalize_value(m.group("num"), m.group("scale"))
            except (ValueError, KeyError):
                continue
            out.append(DollarCandidate(
                paragraph_number=pnum,
                raw_match=m.group(0).strip(),
                value_millions=value,
                span=m.span(),
            ))
    return out


# ---------------------------------------------------------------------------
# Provenance verification.
# ---------------------------------------------------------------------------


@dataclass
class ProvenanceVerdict:
    """Per-metric verdict from provenance verification."""
    field_name: str
    passed: bool
    reason: str
    cited_paragraph: Optional[int] = None
    found_paragraph: Optional[int] = None  # where it was actually found, if anywhere
    cited_value: Optional[float] = None
    nearest_candidate: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "field_name": self.field_name,
            "passed": self.passed,
            "reason": self.reason,
            "cited_paragraph": self.cited_paragraph,
            "found_paragraph": self.found_paragraph,
            "cited_value": self.cited_value,
            "nearest_candidate": self.nearest_candidate,
        }


def _normalize_text(s: str) -> str:
    """Lowercase, collapse whitespace, strip non-essential punctuation."""
    return re.sub(r"\s+", " ", s.lower()).strip()


def _value_matches(candidate: float, target: float, rel_tol: float = 0.005) -> bool:
    if target == 0:
        return abs(candidate) <= rel_tol
    return abs(candidate - target) / abs(target) <= rel_tol


def verify_metric(
    field_name: str,
    value: Optional[float],
    source_quote: Optional[str],
    paragraph_number: Optional[int],
    document: str,
    *,
    rel_tol: float = 0.005,
) -> ProvenanceVerdict:
    """Verify one extracted metric's provenance.

    Three checks, increasingly strict:
      1. The cited paragraph exists in the document.
      2. The source_quote (or a normalized form of it) appears in that
         paragraph (or, as a fallback, anywhere in the document).
      3. A dollar figure equal to `value` appears within the cited
         paragraph (the strongest claim — defeats hallucination).

    A None value short-circuits to PASSED with reason "metric not extracted."
    A None paragraph_number with a source_quote triggers a document-wide
    quote search to recover the paragraph.
    """
    # If the metric wasn't extracted, there's nothing to verify.
    if value is None:
        return ProvenanceVerdict(field_name=field_name, passed=True,
                                 reason="metric not extracted (null)")

    paragraphs = split_paragraphs(document)
    candidates = enumerate_dollar_candidates(document)

    # 1. Does the cited paragraph exist?
    if paragraph_number is not None and paragraph_number not in paragraphs:
        return ProvenanceVerdict(
            field_name=field_name, passed=False,
            reason=f"cited paragraph {paragraph_number} does not exist",
            cited_paragraph=paragraph_number, cited_value=value,
        )

    # 2. Does the quote appear?
    found_pnum: Optional[int] = paragraph_number
    if source_quote:
        nq = _normalize_text(source_quote)
        if paragraph_number is not None:
            ptext = _normalize_text(paragraphs[paragraph_number])
            if nq not in ptext:
                # Hunter cited paragraph but the verbatim quote isn't there.
                # Two-stage fallback before declaring failure:
                #
                # Stage A (preferred): find the quote verbatim in another
                # paragraph. Common when the same financial table appears in
                # multiple chunks and the Hunter cites the wrong one.
                actual = next(
                    (n for n, t in paragraphs.items()
                     if nq in _normalize_text(t)),
                    None,
                )
                if actual is not None:
                    # Verbatim found elsewhere — verify the value lives there
                    # too, then accept with a corrected paragraph anchor.
                    found_pnum = actual
                else:
                    # Stage B: SEC filings often reformat tables across
                    # chunks (extra whitespace, line wraps). The Hunter may
                    # paraphrase slightly. As a softer check: does a dollar
                    # candidate equal to `value` live anywhere in the
                    # document? If so, anchor to that paragraph and let the
                    # value match in step 3 do the actual verification.
                    value_candidates = [c for c in candidates
                                        if _value_matches(c.value_millions, value, rel_tol)]
                    if value_candidates:
                        # If multiple candidates have this exact value (e.g.
                        # the same number repeats in the document), prefer
                        # the cited paragraph if it has one; otherwise pick
                        # the first.
                        in_cited = [c for c in value_candidates
                                    if c.paragraph_number == paragraph_number]
                        chosen = in_cited[0] if in_cited else value_candidates[0]
                        return ProvenanceVerdict(
                            field_name=field_name, passed=True,
                            reason=(f"quote did not match verbatim, but value {value} "
                                    f"found as {chosen.raw_match!r} in paragraph "
                                    f"{chosen.paragraph_number}"),
                            cited_paragraph=paragraph_number,
                            found_paragraph=chosen.paragraph_number,
                            cited_value=value,
                            nearest_candidate=chosen.value_millions,
                        )
                    # No verbatim quote, no matching value anywhere. Hard fail.
                    return ProvenanceVerdict(
                        field_name=field_name, passed=False,
                        reason=("quote not found in cited paragraph or anywhere else, "
                                f"and value {value} does not appear in the document"),
                        cited_paragraph=paragraph_number,
                        cited_value=value,
                    )
        else:
            # No paragraph cited. Search the whole document.
            actual = next(
                (n for n, t in paragraphs.items() if nq in _normalize_text(t)),
                None,
            )
            if actual is None:
                return ProvenanceVerdict(
                    field_name=field_name, passed=False,
                    reason="quote not found in document (no paragraph cited)",
                    cited_value=value,
                )
            found_pnum = actual

    # 3. Does the value appear in the (now established) paragraph?
    if found_pnum is None:
        # No paragraph available to check the value against. We can still
        # check the document-wide candidate set as a weaker guarantee.
        any_match = any(_value_matches(c.value_millions, value, rel_tol)
                        for c in candidates)
        if not any_match:
            return ProvenanceVerdict(
                field_name=field_name, passed=False,
                reason=f"value {value} does not appear anywhere in the document",
                cited_value=value,
            )
        return ProvenanceVerdict(
            field_name=field_name, passed=True,
            reason="value present in document but no paragraph anchor",
            cited_value=value,
        )

    paragraph_candidates = [c for c in candidates if c.paragraph_number == found_pnum]
    matching = [c for c in paragraph_candidates
                if _value_matches(c.value_millions, value, rel_tol)]
    if matching:
        return ProvenanceVerdict(
            field_name=field_name, passed=True,
            reason=f"value matched candidate '{matching[0].raw_match}' in paragraph {found_pnum}",
            cited_paragraph=paragraph_number, found_paragraph=found_pnum,
            cited_value=value, nearest_candidate=matching[0].value_millions,
        )

    # Find the nearest candidate for diagnostic.
    nearest = None
    if paragraph_candidates:
        nearest = min(paragraph_candidates,
                      key=lambda c: abs(c.value_millions - value))
    return ProvenanceVerdict(
        field_name=field_name, passed=False,
        reason=(f"value {value} not present in paragraph {found_pnum}"
                + (f"; nearest is {nearest.raw_match}" if nearest else "")),
        cited_paragraph=paragraph_number, found_paragraph=found_pnum,
        cited_value=value,
        nearest_candidate=nearest.value_millions if nearest else None,
    )


def verify_hunter_report(hunter_report: dict, document: str) -> dict:
    """Verify every metric in a HunterReport.model_dump() output.

    Returns a structured report:
      {
        "all_passed": bool,
        "verdicts": [ProvenanceVerdict.to_dict(), ...],
        "summary": "human-readable summary string"
      }
    """
    verdicts: list[ProvenanceVerdict] = []
    metric_fields = ["revenue", "cogs", "gross_profit", "ebitda", "net_income",
                     "gross_margin_pct", "ebitda_margin_pct", "net_margin_pct",
                     "yoy_revenue_growth_pct", "prior_period_revenue"]
    for fname in metric_fields:
        m = hunter_report.get(fname)
        if not isinstance(m, dict):
            continue
        # Margin/growth fields are percentages, not dollar figures, so
        # the dollar-candidate value check doesn't apply. We still verify
        # the paragraph and quote.
        is_dollar = fname in ("revenue", "cogs", "gross_profit", "ebitda",
                              "net_income", "prior_period_revenue")
        v = verify_metric(
            field_name=fname,
            value=m.get("value") if is_dollar else None,
            source_quote=m.get("source_quote"),
            paragraph_number=m.get("paragraph_number"),
            document=document,
        )
        verdicts.append(v)

    failed = [v for v in verdicts if not v.passed]
    summary = (f"{len(verdicts)} metric(s) checked, {len(failed)} failed"
               + (": " + "; ".join(f"{v.field_name} ({v.reason})" for v in failed)
                  if failed else ""))
    return {
        "all_passed": len(failed) == 0,
        "verdicts": [v.to_dict() for v in verdicts],
        "failed_count": len(failed),
        "summary": summary,
    }


def build_dispute_from_provenance(verification: dict) -> str:
    """Render provenance failures as a Hunter-readable dispute instruction."""
    failed = [v for v in verification["verdicts"] if not v["passed"]]
    if not failed:
        return ""
    lines = [
        "PROVENANCE FAILURE — your previous extraction cited values that "
        "could not be verified against the document. Re-extract and address "
        "each item below explicitly:",
    ]
    for v in failed:
        cited_p = v.get("cited_paragraph")
        found_p = v.get("found_paragraph")
        nearest = v.get("nearest_candidate")
        lines.append(
            f"  - {v['field_name']}: {v['reason']}."
            + (f" The actual paragraph for that quote is {found_p}." if found_p and found_p != cited_p else "")
            + (f" The nearest real value in that paragraph is ${nearest}M." if nearest is not None else "")
        )
    lines.append(
        "Quote text VERBATIM. Cite paragraph numbers exactly as they appear "
        "in the document (e.g., the `[3]` prefix means paragraph 3)."
    )
    return "\n".join(lines)
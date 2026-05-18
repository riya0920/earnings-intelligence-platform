"""
src/reconciliation/agent.py
===========================

The Reconciliation Agent.

For each structured fact the verified pipeline extracts, this agent:
  1. Looks up the corresponding XBRL value at runtime.
  2. Computes the delta between prose-extracted and XBRL values.
  3. If the delta exceeds tolerance, classifies its cause into one
     of five well-known reporting patterns (or UNEXPLAINED).

The classifier is a structured-output gpt-4o-mini call with a
focused prompt that includes the source quote, both values, and the
XBRL tag. We deliberately keep the surface area narrow: the agent
does not try to be a financial-accounting expert, only a delta
*classifier* over five patterns. UNEXPLAINED is a first-class
output, not a failure.

Design points
-------------
â€¢ NO_DELTA is detected deterministically (no LLM call) for facts
  within tolerance. This is most facts on most queries, so we save
  the LLM round-trip when it isn't needed.
â€¢ LOOKUP_FAILED is returned when XBRL has no matching fact (most
  often because the metric isn't us-gaap-tagged, like Adjusted EBITDA).
â€¢ The agent is fail-soft: classification errors return UNEXPLAINED
  with a rationale that names the underlying error.
â€¢ Tolerance defaults to 0.5% (matches the eval scorer); override
  per-call if needed.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

from src.reconciliation.schema import (
    DeltaPattern,
    ReconciliationFinding,
    ReconciliationReport,
)
from src.reconciliation.xbrl import Fact, XBRLLookup

logger = logging.getLogger(__name__)

# Tolerance: |delta| <= 0.5% is treated as a match. Same threshold the
# eval scorer uses for the "correct" verdict, so reconciliation and
# eval agree on what "matches XBRL" means.
DEFAULT_TOLERANCE_PCT = 0.5


# ---------------------------------------------------------------------------
# Input shape (duck-typed to avoid a circular import with verified_generator).
# ---------------------------------------------------------------------------


@dataclass
class ProseFactInput:
    """Minimal shape the agent needs from a StructuredFact.

    Defined here rather than importing StructuredFact from
    src.generation.verified_generator to keep this module standalone
    and unit-testable in isolation.
    """
    field_name: str
    value: float                    # USD millions
    unit: str                       # "USD_millions" or "percent"
    source_quote: str
    chunk_metadata: dict            # company, filing_type, filing_date, section


# ---------------------------------------------------------------------------
# Classifier prompt and schema description.
# ---------------------------------------------------------------------------


_CLASSIFIER_SYSTEM = """You are a financial-reporting reconciliation classifier.

For each input you receive:
- prose_value:  a number extracted from a 10-Q/10-K's narrative or tables
- xbrl_value:   the corresponding canonical us-gaap XBRL value for the same period
- xbrl_tag:     the us-gaap tag name (e.g. "Revenues", "NetIncomeLoss")
- source_quote: the verbatim sentence the prose value was extracted from
- field_name:   our canonical schema name (revenue / cogs / gross_profit / etc.)
- filing_meta:  company, filing_type, filing_date, section

Your job: classify the difference between prose_value and xbrl_value into
exactly ONE of these patterns. Be conservative â€” when in doubt, UNEXPLAINED.

PATTERNS
========

1. NON_GAAP_VS_GAAP
   The source quote is reporting an Adjusted / Non-GAAP / pro-forma /
   "ex-items" version of the metric. Common signals:
     â€¢ The phrase "adjusted", "non-GAAP", "ex-charges", "pro forma"
     â€¢ The number is described as excluding specific items
     â€¢ For EBITDA-like metrics, "Adjusted EBITDA" is almost always non-GAAP

2. PERIODICITY
   The numbers refer to different time windows that share a period_end.
   Common signals:
     â€¢ Source quote says "trailing twelve months", "year-to-date",
       "first nine months", "first six months"
     â€¢ Cumulative vs single-quarter mismatch

3. HIERARCHY
   The prose value reports a segment / business unit / geography total,
   while XBRL reports the consolidated total (or vice versa).
   Common signals:
     â€¢ Source quote names a segment ("Services revenue", "Cloud revenue",
       "Data Center revenue", "Family of Apps")
     â€¢ Source mentions geography ("Americas", "EMEA", "China")

4. VERSIONING
   The prose value is from an originally-reported figure that XBRL has
   since restated (or vice versa).
   Common signals:
     â€¢ Source quote mentions "restated", "amended", "as previously
       reported", "revised"
     â€¢ Large delta on a historical period when current periods agree

5. METRIC_MAPPING
   The prose value is a related-but-different metric that the upstream
   extraction mapped to the wrong canonical field.
   Common signals:
     â€¢ Operating income reported as EBITDA
     â€¢ Total revenue reported as product/service revenue (or vice versa)
     â€¢ Gross profit reported as operating income

6. UNEXPLAINED
   You cannot identify any of the above patterns with confidence.
   This is the right answer when:
     â€¢ The source quote doesn't contain enough context to choose
     â€¢ The delta is large and none of the five patterns fit
     â€¢ You're unsure â€” we prefer human review over a wrong classification

OUTPUT FORMAT
=============
Return ONLY a JSON object with this exact shape:

{
  "pattern": "<one of: non_gaap_vs_gaap | periodicity | hierarchy | versioning | metric_mapping | unexplained>",
  "rationale": "<one sentence, max 200 chars, explaining your choice in plain English>"
}

No prose before or after. No code fences. Just the JSON object.
"""


_VALID_PATTERN_VALUES = {p.value for p in DeltaPattern
                         if p not in (DeltaPattern.NO_DELTA,
                                      DeltaPattern.LOOKUP_FAILED)}


# ---------------------------------------------------------------------------
# Agent.
# ---------------------------------------------------------------------------


class ReconciliationAgent:
    """Classifies prose-vs-XBRL deltas for the verified pipeline.

    Lifecycle:
      reconcile() takes the list of prose-extracted facts and a ticker
      (inferred from filing metadata when possible), looks each fact up
      in XBRL, and returns a ReconciliationReport.
    """

    def __init__(
        self,
        classifier_model: str = "gpt-4o-mini",
        tolerance_pct: float = DEFAULT_TOLERANCE_PCT,
        xbrl_lookup: Optional[XBRLLookup] = None,
    ):
        self.classifier_model = classifier_model
        self.tolerance_pct = tolerance_pct
        self.xbrl = xbrl_lookup or XBRLLookup()
        # Lazy-init the OpenAI client so importing this module never
        # requires an API key (useful for tests and offline tooling).
        self._client = None

    # ------------------------------------------------------------------
    # Public entry point.
    # ------------------------------------------------------------------

    def reconcile(
        self,
        facts: list[ProseFactInput],
        ticker: Optional[str] = None,
        period_end: Optional[str] = None,
    ) -> ReconciliationReport:
        """Reconcile a batch of prose-extracted facts against XBRL.

        Args:
            facts: prose-extracted structured facts to reconcile.
            ticker: 5-letter ticker. If None, inferred from the most
                common ticker in the facts' chunk_metadata.
            period_end: YYYY-MM-DD anchor for the lookup. If None, uses
                each fact's filing_date as a rough proxy.

        Returns:
            A ReconciliationReport with one finding per input fact.
        """
        if ticker is None:
            ticker = self._infer_ticker(facts)

        findings: list[ReconciliationFinding] = []
        for f in facts:
            # Skip non-monetary facts; reconciliation against XBRL only
            # makes sense for dollar fields right now.
            if f.unit != "USD_millions":
                continue
            findings.append(self._reconcile_one(f, ticker=ticker,
                                                 period_end=period_end))

        return ReconciliationReport(
            findings=findings,
            ticker=ticker,
            period_end_used=period_end,
        )

    # ------------------------------------------------------------------
    # Per-fact reconciliation.
    # ------------------------------------------------------------------

    def _reconcile_one(
        self,
        fact: ProseFactInput,
        *,
        ticker: Optional[str],
        period_end: Optional[str],
    ) -> ReconciliationFinding:
        if ticker is None:
            return self._lookup_failed(fact, reason="no ticker available")

        # If the caller didn't pin a period, fall back to the filing_date
        # from the chunk's metadata as a hint about which quarter the
        # prose number is talking about. The XBRL lookup will still try
        # an exact-period match first; with no period at all it picks
        # the most recent quarter.
        effective_period = period_end or self._filing_date_to_period(
            fact.chunk_metadata.get("filing_date"))

        xbrl_fact = self.xbrl.lookup(ticker, fact.field_name,
                                     period_end=effective_period)
        if xbrl_fact is None and effective_period:
            # Period-specific lookup failed; try a "best available"
            # most-recent-quarter fallback before giving up.
            xbrl_fact = self.xbrl.lookup(ticker, fact.field_name,
                                         period_end=None)

        if xbrl_fact is None:
            return self._lookup_failed(fact, reason=(
                f"no us-gaap fact for {fact.field_name} ({ticker})"
            ))

        delta_pct = self._delta_pct(fact.value, xbrl_fact.value_usd_millions)

        if abs(delta_pct) <= self.tolerance_pct:
            return ReconciliationFinding(
                field_name=fact.field_name,
                prose_value=fact.value,
                xbrl_value=xbrl_fact.value_usd_millions,
                xbrl_tag=xbrl_fact.xbrl_tag,
                xbrl_period_end=xbrl_fact.key.period_end,
                delta_pct=delta_pct,
                pattern=DeltaPattern.NO_DELTA,
                rationale=(
                    f"prose ${fact.value:,.0f}M vs XBRL "
                    f"${xbrl_fact.value_usd_millions:,.0f}M "
                    f"({delta_pct:+.2f}% within {self.tolerance_pct}% tolerance)"
                ),
            )

        # Delta exceeds tolerance â€” classify the cause.
        pattern, rationale = self._classify_delta(
            fact=fact, xbrl_fact=xbrl_fact, delta_pct=delta_pct,
        )
        return ReconciliationFinding(
            field_name=fact.field_name,
            prose_value=fact.value,
            xbrl_value=xbrl_fact.value_usd_millions,
            xbrl_tag=xbrl_fact.xbrl_tag,
            xbrl_period_end=xbrl_fact.key.period_end,
            delta_pct=delta_pct,
            pattern=pattern,
            rationale=rationale,
            human_review_recommended=(pattern == DeltaPattern.UNEXPLAINED),
        )

    # ------------------------------------------------------------------
    # LLM classification.
    # ------------------------------------------------------------------

    def _classify_delta(
        self,
        *,
        fact: ProseFactInput,
        xbrl_fact: Fact,
        delta_pct: float,
    ) -> tuple[DeltaPattern, str]:
        """Call the classifier LLM and return (pattern, rationale).

        Falls back to UNEXPLAINED with the error message on any failure.
        Validates the LLM's pattern against the allowed enum values to
        avoid hallucinated patterns leaking into the report.
        """
        try:
            client = self._get_client()
        except Exception as e:  # noqa: BLE001
            logger.warning("Reconciliation classifier unavailable: %s", e)
            return (DeltaPattern.UNEXPLAINED,
                    f"classifier unavailable: {e}")

        user_payload = {
            "field_name": fact.field_name,
            "prose_value": fact.value,
            "xbrl_value": xbrl_fact.value_usd_millions,
            "xbrl_tag": xbrl_fact.xbrl_tag,
            "xbrl_period_end": xbrl_fact.key.period_end,
            "delta_pct": round(delta_pct, 2),
            "source_quote": fact.source_quote,
            "filing_meta": {
                "company": fact.chunk_metadata.get("company"),
                "filing_type": fact.chunk_metadata.get("filing_type"),
                "filing_date": fact.chunk_metadata.get("filing_date"),
                "section": fact.chunk_metadata.get("section"),
            },
        }

        try:
            resp = client.chat.completions.create(
                model=self.classifier_model,
                temperature=0,
                max_tokens=300,
                messages=[
                    {"role": "system", "content": _CLASSIFIER_SYSTEM},
                    {"role": "user", "content": json.dumps(user_payload)},
                ],
            )
            text = (resp.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001
            logger.warning("Reconciliation classifier API call failed: %s", e)
            return (DeltaPattern.UNEXPLAINED,
                    f"classifier API error: {type(e).__name__}")

        return self._parse_classifier_output(text)

    @staticmethod
    def _parse_classifier_output(text: str) -> tuple[DeltaPattern, str]:
        """Extract (pattern, rationale) from the classifier's JSON response.

        Defensive: handles code fences, leading prose, and unknown
        pattern names. Always returns a valid DeltaPattern.
        """
        # Strip code fences if the model added them despite the prompt.
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "",
                          text.strip(), flags=re.MULTILINE)
        # If there's leading text before the JSON, isolate the {...} block.
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        candidate = m.group(0) if m else cleaned

        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as e:
            return (DeltaPattern.UNEXPLAINED,
                    f"classifier returned invalid JSON: {e}")

        pattern_raw = str(parsed.get("pattern", "")).strip().lower()
        rationale = str(parsed.get("rationale", "")).strip()[:240]

        if pattern_raw not in _VALID_PATTERN_VALUES:
            return (DeltaPattern.UNEXPLAINED,
                    f"classifier returned unrecognized pattern '{pattern_raw}'")
        return (DeltaPattern(pattern_raw),
                rationale or "no rationale provided")

    # ------------------------------------------------------------------
    # Helpers.
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._client is not None:
            return self._client
        # Imported lazily so this module can be imported in environments
        # without the openai package installed.
        from openai import OpenAI
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set")
        self._client = OpenAI()
        return self._client

    @staticmethod
    def _delta_pct(prose: float, xbrl: float) -> float:
        """Signed percentage delta from XBRL baseline; safe when xbrl is 0."""
        if xbrl == 0:
            return float("inf") if prose != 0 else 0.0
        return (prose - xbrl) / xbrl * 100.0

    @staticmethod
    def _lookup_failed(
        fact: ProseFactInput,
        *,
        reason: str,
    ) -> ReconciliationFinding:
        """Build a LOOKUP_FAILED finding for cases where XBRL has no match.

        Reasons we end up here:
          â€¢ Ticker not in our CIK map (e.g. unknown company in metadata)
          â€¢ XBRL has no us-gaap fact for this field (e.g. EBITDA, which
            isn't a tagged metric for most filers)
          â€¢ Period-specific lookup miss + fallback lookup also returned None

        LOOKUP_FAILED is a pre-classification failure: we never asked the
        LLM because there was nothing to compare against. The eval treats
        these separately from UNEXPLAINED, which is when the LLM punted.
        """
        return ReconciliationFinding(
            field_name=fact.field_name,
            prose_value=fact.value,
            xbrl_value=None,
            xbrl_tag=None,
            xbrl_period_end=None,
            delta_pct=None,
            pattern=DeltaPattern.LOOKUP_FAILED,
            rationale=reason,
            human_review_recommended=False,
        )

    @staticmethod
    def _infer_ticker(facts: list[ProseFactInput]) -> Optional[str]:
        """Pick the most common ticker across the facts' chunk metadata."""
        ticker_map = {
            "Apple Inc.": "AAPL",
            "Microsoft Corporation": "MSFT",
            "Alphabet Inc.": "GOOGL",
            "NVIDIA Corporation": "NVDA",
            "Meta Platforms Inc.": "META",
        }
        counts: dict[str, int] = {}
        for f in facts:
            company = f.chunk_metadata.get("company")
            tk = (f.chunk_metadata.get("ticker")
                  or ticker_map.get(company or ""))
            if tk:
                counts[tk] = counts.get(tk, 0) + 1
        if not counts:
            return None
        return max(counts.items(), key=lambda kv: kv[1])[0]

    @staticmethod
    def _filing_date_to_period(filing_date: Optional[str]) -> Optional[str]:
        """Best-effort guess at the period_end covered by a filing.

        A 10-Q typically covers a period that ended ~30-60 days before
        the filing date. We don't try to be precise here â€” we just
        return None so the lookup uses "most recent quarter on file".
        The XBRL lookup's most-recent fallback is more reliable than
        any heuristic we could write.
        """
        return None
"""
Cross-Company Risk Comparison & Temporal Analysis Module

Two analytical capabilities that turn this from a RAG demo into
a financial intelligence platform:

1. Cross-Company Comparison:
   Compare risk disclosures across companies — "How does NVIDIA's
   AI risk language differ from Microsoft's?" Extracts risk signals
   per company, aligns them by category, and generates comparative
   analysis.

2. Temporal Analysis:
   Track how risk language evolves across filing years for the same
   company — "What new risks did Meta add in their latest 10-K?"
   Detects new risks, removed risks, and escalation/de-escalation
   patterns over time.

Both features use structured extraction via GPT-4o-mini and
operate on the pre-ingested filing sections.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────


@dataclass
class CompanyRiskProfile:
    """Risk profile for a single company from a single filing."""

    company: str
    ticker: str
    filing_date: str
    risks: list[dict] = field(default_factory=list)
    # Each risk: {"category": str, "summary": str, "severity": str, "keywords": list[str]}


@dataclass
class RiskComparison:
    """Head-to-head risk comparison between two companies."""

    company_a: str
    company_b: str
    shared_risks: list[dict] = field(default_factory=list)
    unique_to_a: list[dict] = field(default_factory=list)
    unique_to_b: list[dict] = field(default_factory=list)
    analysis: str = ""


@dataclass
class TemporalRiskChange:
    """How a company's risk profile changed between two filings."""

    company: str
    ticker: str
    earlier_date: str
    later_date: str
    new_risks: list[dict] = field(default_factory=list)
    removed_risks: list[dict] = field(default_factory=list)
    escalated_risks: list[dict] = field(default_factory=list)
    de_escalated_risks: list[dict] = field(default_factory=list)
    analysis: str = ""


# ─────────────────────────────────────────────────────────
# Risk Profile Extractor
# ─────────────────────────────────────────────────────────


class RiskProfileExtractor:
    """
    Extract structured risk profiles from filing sections.
    Used by both cross-company comparison and temporal analysis.
    """

    EXTRACTION_PROMPT = """Analyze this SEC filing risk factors section and extract a structured risk profile.

Company: {company} ({ticker})
Filing Date: {filing_date}

TEXT:
{text}

Extract 5-10 key risk categories. For each risk, provide:
- category: one of [litigation, regulatory, supply_chain, macroeconomic, cybersecurity, competitive, technology, operational, financial, geopolitical]
- summary: one sentence describing the specific risk (max 30 words)
- severity: low, medium, high, or critical
- keywords: 3-5 key terms from the text related to this risk

Return ONLY a JSON array:
[{{"category": "regulatory", "summary": "Antitrust investigations from DOJ targeting search advertising practices", "severity": "high", "keywords": ["antitrust", "DOJ", "advertising", "investigation"]}}]"""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model

    def extract_profile(
        self,
        content: str,
        company: str,
        ticker: str,
        filing_date: str,
    ) -> CompanyRiskProfile:
        """Extract a risk profile from a single filing section."""
        # Truncate long sections
        if len(content) > 15000:
            content = content[:15000] + "\n[...truncated...]"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": self.EXTRACTION_PROMPT.format(
                            company=company,
                            ticker=ticker,
                            filing_date=filing_date,
                            text=content,
                        ),
                    }
                ],
                temperature=0.0,
                max_tokens=2000,
            )

            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            risks = json.loads(raw)

            return CompanyRiskProfile(
                company=company,
                ticker=ticker,
                filing_date=filing_date,
                risks=risks,
            )
        except Exception as e:
            logger.warning(f"Risk extraction failed for {ticker} {filing_date}: {e}")
            return CompanyRiskProfile(
                company=company, ticker=ticker, filing_date=filing_date
            )


# ─────────────────────────────────────────────────────────
# Cross-Company Comparison
# ─────────────────────────────────────────────────────────


class CrossCompanyAnalyzer:
    """
    Compare risk disclosures across companies.

    Extracts risk profiles for each company, aligns them by category,
    identifies shared vs unique risks, and generates a comparative
    analysis narrative.
    """

    COMPARISON_PROMPT = """You are a senior financial analyst comparing risk disclosures between two companies.

COMPANY A: {company_a} ({ticker_a}) — Filing Date: {date_a}
Risks:
{risks_a}

COMPANY B: {company_b} ({ticker_b}) — Filing Date: {date_b}
Risks:
{risks_b}

Provide a structured comparison:

1. SHARED RISKS: Risk categories both companies mention. For each, note how their framing differs.
2. UNIQUE TO {ticker_a}: Risks only {ticker_a} discloses.
3. UNIQUE TO {ticker_b}: Risks only {ticker_b} discloses.
4. KEY INSIGHT: One paragraph summarizing the most important difference in their risk profiles.

Return ONLY a JSON object:
{{
  "shared_risks": [{{"category": "...", "company_a_framing": "...", "company_b_framing": "..."}}],
  "unique_to_a": [{{"category": "...", "summary": "..."}}],
  "unique_to_b": [{{"category": "...", "summary": "..."}}],
  "analysis": "One paragraph key insight..."
}}"""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model
        self.extractor = RiskProfileExtractor(model=model)

    def compare(
        self,
        sections_a: list[dict],
        sections_b: list[dict],
        company_a_info: dict,
        company_b_info: dict,
    ) -> RiskComparison:
        """
        Compare risk profiles between two companies.

        Args:
            sections_a: Risk factor sections for company A
            sections_b: Risk factor sections for company B
            company_a_info: {"company": str, "ticker": str}
            company_b_info: {"company": str, "ticker": str}
        """
        # Get most recent risk factors section for each company
        section_a = self._get_latest_risk_section(sections_a)
        section_b = self._get_latest_risk_section(sections_b)

        if not section_a or not section_b:
            logger.warning("Missing risk factors sections for comparison")
            return RiskComparison(
                company_a=company_a_info.get("company", ""),
                company_b=company_b_info.get("company", ""),
                analysis="Insufficient data for comparison.",
            )

        # Extract risk profiles
        profile_a = self.extractor.extract_profile(
            content=section_a["content"],
            company=company_a_info["company"],
            ticker=company_a_info["ticker"],
            filing_date=section_a.get("filing_date", "unknown"),
        )

        profile_b = self.extractor.extract_profile(
            content=section_b["content"],
            company=company_b_info["company"],
            ticker=company_b_info["ticker"],
            filing_date=section_b.get("filing_date", "unknown"),
        )

        # Generate comparison
        risks_a_text = "\n".join(
            f"- [{r['category']}] {r['summary']} (severity: {r['severity']})"
            for r in profile_a.risks
        )
        risks_b_text = "\n".join(
            f"- [{r['category']}] {r['summary']} (severity: {r['severity']})"
            for r in profile_b.risks
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": self.COMPARISON_PROMPT.format(
                            company_a=profile_a.company,
                            ticker_a=profile_a.ticker,
                            date_a=profile_a.filing_date,
                            risks_a=risks_a_text,
                            company_b=profile_b.company,
                            ticker_b=profile_b.ticker,
                            date_b=profile_b.filing_date,
                            risks_b=risks_b_text,
                        ),
                    }
                ],
                temperature=0.1,
                max_tokens=2000,
            )

            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(raw)

            return RiskComparison(
                company_a=profile_a.company,
                company_b=profile_b.company,
                shared_risks=data.get("shared_risks", []),
                unique_to_a=data.get("unique_to_a", []),
                unique_to_b=data.get("unique_to_b", []),
                analysis=data.get("analysis", ""),
            )
        except Exception as e:
            logger.error(f"Comparison generation failed: {e}")
            return RiskComparison(
                company_a=profile_a.company,
                company_b=profile_b.company,
                analysis=f"Comparison failed: {e}",
            )

    @staticmethod
    def _get_latest_risk_section(sections: list[dict]) -> dict | None:
        """Get the most recent risk_factors section."""
        risk_sections = [s for s in sections if s.get("section_name") == "risk_factors"]
        if not risk_sections:
            return None
        return sorted(
            risk_sections, key=lambda s: s.get("filing_date", ""), reverse=True
        )[0]


# ─────────────────────────────────────────────────────────
# Temporal Analysis
# ─────────────────────────────────────────────────────────


class TemporalAnalyzer:
    """
    Track how a company's risk language evolves across filing years.

    Compares risk profiles from two different filing dates to detect:
    - New risks added
    - Risks removed
    - Risks that escalated in severity
    - Risks that de-escalated
    """

    TEMPORAL_PROMPT = """You are analyzing how a company's risk disclosures changed between two filings.

COMPANY: {company} ({ticker})

EARLIER FILING ({earlier_date}):
{earlier_risks}

LATER FILING ({later_date}):
{later_risks}

Identify:
1. NEW RISKS: Risks in the later filing not present in the earlier one
2. REMOVED RISKS: Risks in the earlier filing no longer mentioned
3. ESCALATED: Risks present in both but with increased severity or emphasis
4. DE-ESCALATED: Risks present in both but with decreased severity or emphasis
5. KEY INSIGHT: One paragraph on the most significant shift in risk profile

Return ONLY a JSON object:
{{
  "new_risks": [{{"category": "...", "summary": "..."}}],
  "removed_risks": [{{"category": "...", "summary": "..."}}],
  "escalated_risks": [{{"category": "...", "from_severity": "...", "to_severity": "...", "summary": "..."}}],
  "de_escalated_risks": [{{"category": "...", "from_severity": "...", "to_severity": "...", "summary": "..."}}],
  "analysis": "One paragraph key insight..."
}}"""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model
        self.extractor = RiskProfileExtractor(model=model)

    def analyze_evolution(
        self,
        sections: list[dict],
        company: str,
        ticker: str,
    ) -> list[TemporalRiskChange]:
        """
        Analyze risk evolution across all available filing dates for a company.

        Returns a list of TemporalRiskChange objects, one for each
        consecutive pair of filing dates.
        """
        # Get risk_factors sections sorted by date
        risk_sections = sorted(
            [s for s in sections if s.get("section_name") == "risk_factors"],
            key=lambda s: s.get("filing_date", ""),
        )

        if len(risk_sections) < 2:
            logger.warning(
                f"Need at least 2 risk factor filings for {ticker}, found {len(risk_sections)}"
            )
            return []

        # Extract risk profiles for each filing
        profiles = []
        for section in risk_sections:
            profile = self.extractor.extract_profile(
                content=section["content"],
                company=company,
                ticker=ticker,
                filing_date=section.get("filing_date", "unknown"),
            )
            profiles.append(profile)

        # Compare consecutive pairs
        changes = []
        for i in range(len(profiles) - 1):
            earlier = profiles[i]
            later = profiles[i + 1]

            change = self._compare_profiles(earlier, later)
            changes.append(change)

        return changes

    def _compare_profiles(
        self,
        earlier: CompanyRiskProfile,
        later: CompanyRiskProfile,
    ) -> TemporalRiskChange:
        """Compare two risk profiles and identify changes."""
        earlier_risks_text = "\n".join(
            f"- [{r['category']}] {r['summary']} (severity: {r['severity']})"
            for r in earlier.risks
        )
        later_risks_text = "\n".join(
            f"- [{r['category']}] {r['summary']} (severity: {r['severity']})"
            for r in later.risks
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": self.TEMPORAL_PROMPT.format(
                            company=earlier.company,
                            ticker=earlier.ticker,
                            earlier_date=earlier.filing_date,
                            later_date=later.filing_date,
                            earlier_risks=earlier_risks_text,
                            later_risks=later_risks_text,
                        ),
                    }
                ],
                temperature=0.1,
                max_tokens=2000,
            )

            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(raw)

            return TemporalRiskChange(
                company=earlier.company,
                ticker=earlier.ticker,
                earlier_date=earlier.filing_date,
                later_date=later.filing_date,
                new_risks=data.get("new_risks", []),
                removed_risks=data.get("removed_risks", []),
                escalated_risks=data.get("escalated_risks", []),
                de_escalated_risks=data.get("de_escalated_risks", []),
                analysis=data.get("analysis", ""),
            )
        except Exception as e:
            logger.error(f"Temporal analysis failed: {e}")
            return TemporalRiskChange(
                company=earlier.company,
                ticker=earlier.ticker,
                earlier_date=earlier.filing_date,
                later_date=later.filing_date,
                analysis=f"Analysis failed: {e}",
            )


# ─────────────────────────────────────────────────────────
# CLI-Friendly Functions
# ─────────────────────────────────────────────────────────


def run_cross_company_comparison(
    all_sections: list[dict],
    ticker_a: str,
    ticker_b: str,
) -> RiskComparison:
    """Run a cross-company risk comparison from CLI."""
    sections_a = [s for s in all_sections if s.get("ticker") == ticker_a]
    sections_b = [s for s in all_sections if s.get("ticker") == ticker_b]

    if not sections_a:
        raise ValueError(f"No sections found for {ticker_a}")
    if not sections_b:
        raise ValueError(f"No sections found for {ticker_b}")

    company_a_info = {
        "company": sections_a[0].get("company", ticker_a),
        "ticker": ticker_a,
    }
    company_b_info = {
        "company": sections_b[0].get("company", ticker_b),
        "ticker": ticker_b,
    }

    analyzer = CrossCompanyAnalyzer()
    result = analyzer.compare(sections_a, sections_b, company_a_info, company_b_info)

    # Pretty print
    print(f"\n{'=' * 70}")
    print(f"RISK COMPARISON: {result.company_a} vs {result.company_b}")
    print(f"{'=' * 70}")

    if result.shared_risks:
        print(f"\nShared Risks ({len(result.shared_risks)}):")
        for r in result.shared_risks:
            print(f"  [{r.get('category', '?')}]")
            print(f"    {ticker_a}: {r.get('company_a_framing', 'N/A')}")
            print(f"    {ticker_b}: {r.get('company_b_framing', 'N/A')}")

    if result.unique_to_a:
        print(f"\nUnique to {ticker_a} ({len(result.unique_to_a)}):")
        for r in result.unique_to_a:
            print(f"  [{r.get('category', '?')}] {r.get('summary', '')}")

    if result.unique_to_b:
        print(f"\nUnique to {ticker_b} ({len(result.unique_to_b)}):")
        for r in result.unique_to_b:
            print(f"  [{r.get('category', '?')}] {r.get('summary', '')}")

    print(f"\nKey Insight:\n  {result.analysis}")
    print(f"{'=' * 70}")

    return result


def run_temporal_analysis(
    all_sections: list[dict],
    ticker: str,
) -> list[TemporalRiskChange]:
    """Run temporal risk analysis for a single company from CLI."""
    sections = [s for s in all_sections if s.get("ticker") == ticker]

    if not sections:
        raise ValueError(f"No sections found for {ticker}")

    company = sections[0].get("company", ticker)
    analyzer = TemporalAnalyzer()
    changes = analyzer.analyze_evolution(sections, company, ticker)

    # Pretty print
    print(f"\n{'=' * 70}")
    print(f"TEMPORAL RISK ANALYSIS: {company} ({ticker})")
    print(f"{'=' * 70}")

    for change in changes:
        print(f"\n{change.earlier_date} → {change.later_date}")
        print(f"{'─' * 50}")

        if change.new_risks:
            print(f"  New risks ({len(change.new_risks)}):")
            for r in change.new_risks:
                print(f"    + [{r.get('category', '?')}] {r.get('summary', '')}")

        if change.removed_risks:
            print(f"  Removed risks ({len(change.removed_risks)}):")
            for r in change.removed_risks:
                print(f"    - [{r.get('category', '?')}] {r.get('summary', '')}")

        if change.escalated_risks:
            print(f"  Escalated ({len(change.escalated_risks)}):")
            for r in change.escalated_risks:
                print(
                    f"    ↑ [{r.get('category', '?')}] {r.get('summary', '')} "
                    f"({r.get('from_severity', '?')} → {r.get('to_severity', '?')})"
                )

        if change.de_escalated_risks:
            print(f"  De-escalated ({len(change.de_escalated_risks)}):")
            for r in change.de_escalated_risks:
                print(
                    f"    ↓ [{r.get('category', '?')}] {r.get('summary', '')} "
                    f"({r.get('from_severity', '?')} → {r.get('to_severity', '?')})"
                )

        print(f"\n  Insight: {change.analysis}")

    print(f"\n{'=' * 70}")
    return changes

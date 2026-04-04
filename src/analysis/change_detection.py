"""
Filing Change Detection Module

Compares a company's 10-K filing sections year-over-year to detect:
  - NEW paragraphs/disclosures added (potential emerging risks)
  - REMOVED paragraphs (risks that were resolved or de-emphasized)
  - MATERIALLY MODIFIED language (subtle shifts in tone/severity)

Uses a combination of:
  1. Sentence-level embedding similarity to align paragraphs across years
  2. GPT-4o-mini to classify and explain material changes
  3. Structured output with severity scoring

This is what SEC analysts do manually — automating it is the kind
of signal hedge funds pay for.

Usage:
    from src.analysis.change_detection import FilingChangeDetector

    detector = FilingChangeDetector(config)
    report = detector.detect_changes("NVDA", sections)
    for change in report.changes:
        print(change)

CLI:
    python -m src.main diff NVDA
    python -m src.main diff AAPL --section "Risk Factors"
"""

import json
import logging
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class FilingChange:
    """A single detected change between two filing years."""
    change_type: str  # "added" | "removed" | "modified"
    section: str
    severity: str  # "low" | "medium" | "high" | "critical"
    summary: str
    old_text: Optional[str] = None  # For modified/removed
    new_text: Optional[str] = None  # For modified/added
    similarity_score: Optional[float] = None  # For modified
    category: str = ""  # litigation, regulatory, etc.
    analyst_note: str = ""  # GPT-generated explanation of why this matters


@dataclass
class ChangeReport:
    """Full change detection report for a company."""
    ticker: str
    old_filing_date: str
    new_filing_date: str
    section_filter: Optional[str]
    changes: list[FilingChange]
    summary: str  # GPT-generated executive summary
    stats: dict = field(default_factory=dict)
    latency_seconds: float = 0.0


CHANGE_CLASSIFICATION_PROMPT = """You are a financial analyst reviewing changes between two years of a company's SEC 10-K filing.

I will provide you with detected changes (additions, removals, and modifications) in the filing text.

For each change, provide:
1. severity: How material is this change? (low / medium / high / critical)
2. category: What type of risk/topic? (litigation, regulatory, supply_chain, macroeconomic, cybersecurity, competitive, financial, operational, strategic, other)
3. analyst_note: A 1-2 sentence explanation of WHY this change matters to an investor. Be specific.

Then provide an executive_summary (3-5 sentences) of the most important changes overall.

Respond in this JSON format:
{
    "classifications": [
        {
            "index": 0,
            "severity": "high",
            "category": "regulatory",
            "analyst_note": "Company added new disclosure about EU AI Act compliance requirements, suggesting increased regulatory exposure in their core AI business."
        }
    ],
    "executive_summary": "The most significant changes in this filing period include..."
}

Return ONLY valid JSON, no markdown fences."""


class FilingChangeDetector:
    """
    Detects and classifies changes between consecutive 10-K filings.

    Pipeline:
    1. Extract and align paragraphs from two filing years
    2. Use SequenceMatcher for fast text similarity
    3. Classify unmatched/modified paragraphs as additions/removals/modifications
    4. Use GPT-4o-mini to score severity and explain significance
    """

    def __init__(self, config: dict):
        self.config = config
        self.client = OpenAI()
        self.model = config.get("generation", {}).get("model", "gpt-4o-mini")
        self.similarity_threshold = 0.6  # Below this = different paragraph
        self.modification_threshold = 0.85  # Below this but above similarity = modified

    def detect_changes(
        self,
        ticker: str,
        sections: list[dict],
        section_filter: Optional[str] = None,
    ) -> ChangeReport:
        """
        Detect changes between the two most recent filings for a company.

        Args:
            ticker: Company ticker (e.g., "NVDA")
            sections: All ingested filing sections
            section_filter: Optional section name to focus on (e.g., "Risk Factors")

        Returns:
            ChangeReport with classified changes and executive summary
        """
        start_time = time.time()

        # Step 1: Get this company's sections, sorted by filing date
        company_sections = [
            s for s in sections
            if s.get("ticker", "").upper() == ticker.upper()
        ]

        if not company_sections:
            return ChangeReport(
                ticker=ticker, old_filing_date="", new_filing_date="",
                section_filter=section_filter, changes=[], summary="No filings found.",
            )

        # Group by filing date
        filings_by_date = self._group_by_filing_date(company_sections)
        dates = sorted(filings_by_date.keys())

        if len(dates) < 2:
            return ChangeReport(
                ticker=ticker, old_filing_date=dates[0] if dates else "",
                new_filing_date="", section_filter=section_filter,
                changes=[], summary="Only one filing available — need at least two for comparison.",
            )

        old_date = dates[-2]
        new_date = dates[-1]
        old_sections = filings_by_date[old_date]
        new_sections = filings_by_date[new_date]

        logger.info(f"Comparing {ticker} filings: {old_date} vs {new_date}")

        # Step 2: Filter by section if requested
        if section_filter:
            old_sections = [
                s for s in old_sections
                if section_filter.lower() in s.get("section", "").lower()
            ]
            new_sections = [
                s for s in new_sections
                if section_filter.lower() in s.get("section", "").lower()
            ]

        # Step 3: Extract paragraphs and detect changes
        old_paragraphs = self._extract_paragraphs(old_sections)
        new_paragraphs = self._extract_paragraphs(new_sections)

        raw_changes = self._diff_paragraphs(old_paragraphs, new_paragraphs)

        if not raw_changes:
            return ChangeReport(
                ticker=ticker, old_filing_date=old_date, new_filing_date=new_date,
                section_filter=section_filter, changes=[],
                summary=f"No material changes detected between {old_date} and {new_date} filings.",
                latency_seconds=round(time.time() - start_time, 2),
            )

        # Step 4: Classify changes with GPT-4o-mini
        classified = self._classify_changes(raw_changes, ticker, old_date, new_date)

        latency = time.time() - start_time

        stats = {
            "total_changes": len(classified),
            "added": sum(1 for c in classified if c.change_type == "added"),
            "removed": sum(1 for c in classified if c.change_type == "removed"),
            "modified": sum(1 for c in classified if c.change_type == "modified"),
            "critical": sum(1 for c in classified if c.severity == "critical"),
            "high": sum(1 for c in classified if c.severity == "high"),
            "old_paragraphs": len(old_paragraphs),
            "new_paragraphs": len(new_paragraphs),
        }

        return ChangeReport(
            ticker=ticker,
            old_filing_date=old_date,
            new_filing_date=new_date,
            section_filter=section_filter,
            changes=classified,
            summary=classified[0].analyst_note if classified else "",
            stats=stats,
            latency_seconds=round(latency, 2),
        )

    def _group_by_filing_date(
        self, sections: list[dict]
    ) -> dict[str, list[dict]]:
        """Group sections by filing date."""
        by_date: dict[str, list[dict]] = {}
        for s in sections:
            date = s.get("filing_date", "unknown")
            if date not in by_date:
                by_date[date] = []
            by_date[date].append(s)
        return by_date

    def _extract_paragraphs(
        self, sections: list[dict]
    ) -> list[dict]:
        """
        Break sections into individual paragraphs for granular comparison.

        Returns list of {text, section, index} dicts.
        """
        paragraphs = []
        for section in sections:
            content = section.get("content", "")
            section_name = section.get("section", "unknown")

            # Split on double newlines or significant whitespace
            raw_paras = [
                p.strip() for p in content.split("\n\n")
                if p.strip() and len(p.strip()) > 50  # Skip short fragments
            ]

            for i, para in enumerate(raw_paras):
                paragraphs.append({
                    "text": para,
                    "section": section_name,
                    "index": len(paragraphs),
                })

        return paragraphs

    def _diff_paragraphs(
        self,
        old_paragraphs: list[dict],
        new_paragraphs: list[dict],
    ) -> list[FilingChange]:
        """
        Compare old and new paragraphs using text similarity.

        Algorithm:
        1. For each new paragraph, find best match in old paragraphs
        2. If best match > modification_threshold: no change (identical)
        3. If best match between similarity and modification thresholds: modified
        4. If best match < similarity_threshold: added (new content)
        5. Any old paragraphs with no match in new: removed
        """
        changes = []
        matched_old_indices = set()

        for new_para in new_paragraphs:
            best_score = 0.0
            best_old_idx = -1
            best_old_para = None

            for i, old_para in enumerate(old_paragraphs):
                if i in matched_old_indices:
                    continue
                score = SequenceMatcher(
                    None, old_para["text"], new_para["text"]
                ).ratio()
                if score > best_score:
                    best_score = score
                    best_old_idx = i
                    best_old_para = old_para

            if best_score >= self.modification_threshold:
                # Nearly identical — no material change
                matched_old_indices.add(best_old_idx)
            elif best_score >= self.similarity_threshold:
                # Modified — same topic but different language
                matched_old_indices.add(best_old_idx)
                changes.append(FilingChange(
                    change_type="modified",
                    section=new_para["section"],
                    severity="",  # Will be classified by GPT
                    summary="",
                    old_text=best_old_para["text"][:800] if best_old_para else None,
                    new_text=new_para["text"][:800],
                    similarity_score=round(best_score, 3),
                ))
            else:
                # New content — no good match found
                changes.append(FilingChange(
                    change_type="added",
                    section=new_para["section"],
                    severity="",
                    summary="",
                    new_text=new_para["text"][:800],
                ))

        # Find removed paragraphs (old content with no match in new)
        for i, old_para in enumerate(old_paragraphs):
            if i not in matched_old_indices:
                changes.append(FilingChange(
                    change_type="removed",
                    section=old_para["section"],
                    severity="",
                    summary="",
                    old_text=old_para["text"][:800],
                ))

        logger.info(
            f"Diff results: {sum(1 for c in changes if c.change_type == 'added')} added, "
            f"{sum(1 for c in changes if c.change_type == 'removed')} removed, "
            f"{sum(1 for c in changes if c.change_type == 'modified')} modified"
        )

        return changes

    def _classify_changes(
        self,
        changes: list[FilingChange],
        ticker: str,
        old_date: str,
        new_date: str,
    ) -> list[FilingChange]:
        """Use GPT-4o-mini to classify severity and explain each change."""

        # Cap at 20 changes to stay within token limits
        changes_to_classify = changes[:20]

        changes_for_prompt = []
        for i, c in enumerate(changes_to_classify):
            entry = {
                "index": i,
                "type": c.change_type,
                "section": c.section,
            }
            if c.old_text:
                entry["old_text"] = c.old_text[:500]
            if c.new_text:
                entry["new_text"] = c.new_text[:500]
            if c.similarity_score is not None:
                entry["similarity"] = c.similarity_score
            changes_for_prompt.append(entry)

        user_prompt = (
            f"Company: {ticker}\n"
            f"Comparing filings: {old_date} → {new_date}\n\n"
            f"Detected changes:\n{json.dumps(changes_for_prompt, indent=2)}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": CHANGE_CLASSIFICATION_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )

            parsed = json.loads(response.choices[0].message.content)
            classifications = parsed.get("classifications", [])
            executive_summary = parsed.get("executive_summary", "")

            # Apply classifications back to changes
            for cls in classifications:
                idx = cls.get("index", -1)
                if 0 <= idx < len(changes_to_classify):
                    changes_to_classify[idx].severity = cls.get("severity", "medium")
                    changes_to_classify[idx].category = cls.get("category", "other")
                    changes_to_classify[idx].analyst_note = cls.get("analyst_note", "")

            # Set summary on the first change as a hack to pass it through
            if changes_to_classify and executive_summary:
                changes_to_classify[0].analyst_note = executive_summary

            # Fill in any unclassified
            for c in changes_to_classify:
                if not c.severity:
                    c.severity = "medium"
                if not c.summary:
                    c.summary = f"{c.change_type.title()} in {c.section}"

        except Exception as e:
            logger.error(f"Classification failed: {e}")
            for c in changes_to_classify:
                c.severity = "medium"
                c.summary = f"{c.change_type.title()} in {c.section}"
                c.analyst_note = "Classification unavailable"

        return changes_to_classify


def run_change_detection(
    ticker: str,
    config: dict,
    sections: list[dict],
    section_filter: Optional[str] = None,
) -> ChangeReport:
    """
    Convenience function to run filing change detection.

    Args:
        ticker: Company ticker
        config: Pipeline config
        sections: All ingested sections
        section_filter: Optional section to focus on

    Returns:
        ChangeReport with all detected and classified changes
    """
    detector = FilingChangeDetector(config)
    report = detector.detect_changes(ticker, sections, section_filter)

    # Pretty-print
    print(f"\n{'=' * 70}")
    print(f"FILING CHANGE DETECTION: {ticker}")
    print(f"{'=' * 70}")
    print(f"Comparing: {report.old_filing_date} → {report.new_filing_date}")
    if report.section_filter:
        print(f"Section filter: {report.section_filter}")
    print(f"Latency: {report.latency_seconds}s")

    if report.stats:
        print(f"\nStats: {report.stats.get('added', 0)} added | "
              f"{report.stats.get('removed', 0)} removed | "
              f"{report.stats.get('modified', 0)} modified | "
              f"{report.stats.get('critical', 0)} critical | "
              f"{report.stats.get('high', 0)} high severity")

    if report.changes:
        # Print executive summary (stored in first change's analyst_note)
        print(f"\n{'─' * 70}")
        print(f"EXECUTIVE SUMMARY:")
        print(f"{report.changes[0].analyst_note}")

        print(f"\n{'─' * 70}")
        print(f"DETAILED CHANGES:")

        # Sort: critical first, then high, etc.
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        sorted_changes = sorted(
            report.changes,
            key=lambda c: severity_order.get(c.severity, 4),
        )

        for i, change in enumerate(sorted_changes, 1):
            icon = {"added": "🟢", "removed": "🔴", "modified": "🟡"}.get(
                change.change_type, "⚪"
            )
            sev_icon = {"critical": "🚨", "high": "⚠️", "medium": "📋", "low": "ℹ️"}.get(
                change.severity, ""
            )

            print(f"\n  {i}. {icon} {change.change_type.upper()} {sev_icon} [{change.severity}] [{change.category}]")
            print(f"     Section: {change.section}")

            if change.analyst_note and i > 1:  # Skip first (has executive summary)
                print(f"     Note: {change.analyst_note}")

            if change.change_type == "added" and change.new_text:
                preview = change.new_text[:200].replace("\n", " ")
                print(f"     Added: \"{preview}...\"")
            elif change.change_type == "removed" and change.old_text:
                preview = change.old_text[:200].replace("\n", " ")
                print(f"     Removed: \"{preview}...\"")
            elif change.change_type == "modified":
                if change.old_text:
                    preview = change.old_text[:150].replace("\n", " ")
                    print(f"     Before: \"{preview}...\"")
                if change.new_text:
                    preview = change.new_text[:150].replace("\n", " ")
                    print(f"     After:  \"{preview}...\"")
                if change.similarity_score is not None:
                    print(f"     Similarity: {change.similarity_score:.1%}")

    else:
        print("\nNo material changes detected.")

    print(f"\n{'=' * 70}")

    # Save report to JSON
    output_path = Path("data/processed") / f"changes_{ticker}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report_dict = {
        "ticker": report.ticker,
        "old_filing_date": report.old_filing_date,
        "new_filing_date": report.new_filing_date,
        "section_filter": report.section_filter,
        "stats": report.stats,
        "latency_seconds": report.latency_seconds,
        "changes": [
            {
                "change_type": c.change_type,
                "section": c.section,
                "severity": c.severity,
                "category": c.category,
                "analyst_note": c.analyst_note,
                "similarity_score": c.similarity_score,
                "old_text_preview": (c.old_text[:300] if c.old_text else None),
                "new_text_preview": (c.new_text[:300] if c.new_text else None),
            }
            for c in report.changes
        ],
    }

    with open(output_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"Report saved to {output_path}")

    return report
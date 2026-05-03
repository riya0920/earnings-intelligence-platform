"""
SEC EDGAR Filing Ingestion Module

Fetches 10-K/10-Q filings from SEC EDGAR, parses them into structured
sections (Risk Factors, MD&A, Business Overview), and stores them as
clean text with metadata for downstream chunking.

SEC EDGAR API docs: https://efts.sec.gov/LATEST/
Rate limit: 10 requests/second with proper User-Agent header.
"""

import os
import re
import json
import time
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class FilingSection:
    """A parsed section from a SEC filing."""

    company: str
    ticker: str
    cik: str
    filing_type: str
    filing_date: str
    accession_number: str
    section_name: str
    section_title: str
    content: str
    word_count: int = 0
    url: str = ""

    def __post_init__(self):
        self.word_count = len(self.content.split())


@dataclass
class Filing:
    """A complete SEC filing with all extracted sections."""

    company: str
    ticker: str
    cik: str
    filing_type: str
    filing_date: str
    accession_number: str
    url: str
    sections: list[FilingSection] = field(default_factory=list)

    @property
    def total_words(self) -> int:
        return sum(s.word_count for s in self.sections)


class SECEdgarClient:
    """Client for SEC EDGAR full-text search and filing retrieval."""

    BASE_URL = "https://efts.sec.gov/LATEST"
    FILING_URL = "https://www.sec.gov/Archives/edgar/data"

    # Section header patterns for 10-K parsing
    # SEC filings use wildly inconsistent formatting: "ITEM 1A.", "Item 1A—",
    # "Item\xa01A -", unicode dashes, non-breaking spaces, ALL CAPS, etc.
    # These patterns are intentionally broad to handle real-world variation.
    SECTION_PATTERNS = {
        "risk_factors": [
            r"item[\s\xa0]*1[\s\xa0]*a[\s\.\-—–:]*[\s\xa0]*risk[\s\xa0]+factors",
            r"ITEM[\s\xa0]*1A[\s\.\-—–:]*[\s\xa0]*RISK[\s\xa0]+FACTORS",
            r"risk[\s\xa0]+factors\s*\n",
        ],
        "mda": [
            r"item[\s\xa0]*7[\s\.\-—–:]*[\s\xa0]*management[\s\xa0''].{0,15}discussion",
            r"ITEM[\s\xa0]*7[\s\.\-—–:]*[\s\xa0]*MANAGEMENT",
            r"management.{0,5}s?\s+discussion\s+and\s+analysis",
        ],
        "business_overview": [
            r"item[\s\xa0]*1[\s\.\-—–:]*[\s\xa0]*business\b(?![\s\xa0]*combination)",
            r"ITEM[\s\xa0]*1[\s\.\-—–:]*[\s\xa0]*BUSINESS\b",
        ],
        "financial_statements": [
            r"item[\s\xa0]*8[\s\.\-—–:]*[\s\xa0]*financial[\s\xa0]+statements",
            r"ITEM[\s\xa0]*8[\s\.\-—–:]*[\s\xa0]*FINANCIAL[\s\xa0]+STATEMENTS",
            r"consolidated\s+statements?\s+of\s+(?:operations|income|comprehensive)",
        ],
    }

    # Patterns for the START of the next section (to know where current ends)
    # Must handle: "Item 1A.", "ITEM 2.", "Item\xa07.", etc.
    NEXT_SECTION_PATTERN = re.compile(
        r"\n\s*(?:item|ITEM)[\s\xa0]*\d+[a-zA-Z]?[\s\.\-—–:]+",
        re.MULTILINE,
    )

    def __init__(self, user_agent: str, rate_limit_delay: float = 0.12):
        """
        Args:
            user_agent: Required by SEC — format "Company/App contact@email.com"
            rate_limit_delay: Seconds between requests (SEC limit: 10/sec)
        """
        self.user_agent = user_agent
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept-Encoding": "gzip, deflate",
            }
        )

    def _request(self, url: str, params: Optional[dict] = None) -> requests.Response:
        """Make a rate-limited request to SEC EDGAR."""
        time.sleep(self.rate_limit_delay)
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response

    def search_filings(
        self,
        cik: str,
        filing_type: str = "10-K",
        max_results: int = 4,
    ) -> list[dict]:
        """
        Search for filings by CIK number using EDGAR full-text search.

        Returns list of filing metadata dicts with accession numbers and dates.
        """
        # Use the EDGAR submissions API for reliable results
        cik_padded = cik.lstrip("0").zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"

        try:
            response = self._request(url)
            data = response.json()
        except Exception as e:
            logger.error(f"Failed to fetch submissions for CIK {cik}: {e}")
            return []

        recent = data.get("filings", {}).get("recent", {})
        if not recent:
            return []

        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        filings = []
        for i, form in enumerate(forms):
            if form == filing_type and len(filings) < max_results:
                accession = accessions[i].replace("-", "")
                filings.append(
                    {
                        "filing_type": form,
                        "filing_date": dates[i],
                        "accession_number": accessions[i],
                        "primary_document": primary_docs[i],
                        "url": (
                            f"{self.FILING_URL}/{cik_padded}/{accession}"
                            f"/{primary_docs[i]}"
                        ),
                    }
                )

        logger.info(f"Found {len(filings)} {filing_type} filings for CIK {cik}")
        return filings

    def fetch_filing_html(self, url: str) -> str:
        """Fetch the raw HTML content of a filing."""
        response = self._request(url)
        return response.text

    def _clean_text(self, text: str) -> str:
        """Clean extracted text: normalize whitespace, remove artifacts."""
        # Remove multiple newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove excessive spaces
        text = re.sub(r"[ \t]{2,}", " ", text)
        # Remove page numbers and headers/footers
        text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
        # Remove table of contents references
        text = re.sub(
            r"(?:table of contents|index)\s*\n", "", text, flags=re.IGNORECASE
        )
        return text.strip()

    def parse_sections(
        self, html: str, filing_meta: dict, company_info: dict
    ) -> list[FilingSection]:
        """
        Parse a 10-K filing HTML into structured sections.

        Uses regex patterns to identify section boundaries and extracts
        clean text content for each section.
        """
        soup = BeautifulSoup(html, "lxml")

        # Remove script/style elements
        for tag in soup(["script", "style", "meta", "link"]):
            tag.decompose()

        full_text = soup.get_text(separator="\n")
        full_text = self._clean_text(full_text)

        sections = []

        for section_name, patterns in self.SECTION_PATTERNS.items():
            section_content = self._extract_section(full_text, patterns)

            if section_content and len(section_content.split()) > 50:
                section = FilingSection(
                    company=company_info["name"],
                    ticker=company_info["ticker"],
                    cik=company_info["cik"],
                    filing_type=filing_meta["filing_type"],
                    filing_date=filing_meta["filing_date"],
                    accession_number=filing_meta["accession_number"],
                    section_name=section_name,
                    section_title=section_name.replace("_", " ").title(),
                    content=section_content,
                    url=filing_meta["url"],
                )
                sections.append(section)
                logger.debug(f"  Extracted {section_name}: {section.word_count} words")
            else:
                logger.debug(f"  Section {section_name} not found or too short")

        return sections

    def _extract_section(self, text: str, patterns: list[str]) -> str:
        """
        Extract a section from filing text using regex patterns.

        Finds the section header, then captures text until the next
        Item header is found. Tries each pattern in order, using both
        the raw pattern and a case-insensitive fallback.
        """
        for pattern in patterns:
            # Try case-insensitive match
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                start = match.end()
                remaining = text[start:]

                # Find the next "Item X" header to mark the end
                next_item = self.NEXT_SECTION_PATTERN.search(remaining)

                if next_item and next_item.start() > 200:
                    # Only use next_item boundary if we got meaningful content
                    content = remaining[: next_item.start()]
                elif next_item and next_item.start() <= 200:
                    # Too short — the match might be a ToC entry, skip it
                    # Try finding the SECOND occurrence of this pattern
                    second_match = re.search(
                        pattern, text[match.end() :], re.IGNORECASE | re.MULTILINE
                    )
                    if second_match:
                        start2 = match.end() + second_match.end()
                        remaining2 = text[start2:]
                        next_item2 = self.NEXT_SECTION_PATTERN.search(remaining2)
                        if next_item2:
                            content = remaining2[: next_item2.start()]
                        else:
                            content = remaining2[:50000]
                    else:
                        content = remaining[:50000]
                else:
                    content = remaining[:50000]

                content = self._clean_text(content)
                if len(content.split()) > 50:
                    return content

        return ""


class FilingIngestionPipeline:
    """
    End-to-end pipeline for ingesting SEC filings.

    Orchestrates fetching, parsing, and storing filings with metadata.
    """

    def __init__(self, config: dict, output_dir: str = "data/raw"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        user_agent = config.get(
            "user_agent",
            os.getenv("SEC_EDGAR_USER_AGENT", "EarningsIntel research@example.com"),
        )
        self.client = SECEdgarClient(user_agent=user_agent)

    def ingest_company(self, company: dict) -> list[Filing]:
        """Ingest all filings for a single company."""
        logger.info(f"Ingesting filings for {company['name']} ({company['ticker']})")

        filings = []
        for filing_type in self.config.get("filing_types", ["10-K"]):
            filing_metas = self.client.search_filings(
                cik=company["cik"],
                filing_type=filing_type,
                max_results=self.config.get("max_filings_per_company", 4),
            )

            for meta in tqdm(
                filing_metas,
                desc=f"  {company['ticker']} {filing_type}",
                leave=False,
            ):
                try:
                    html = self.client.fetch_filing_html(meta["url"])
                    sections = self.client.parse_sections(html, meta, company)

                    filing = Filing(
                        company=company["name"],
                        ticker=company["ticker"],
                        cik=company["cik"],
                        filing_type=filing_type,
                        filing_date=meta["filing_date"],
                        accession_number=meta["accession_number"],
                        url=meta["url"],
                        sections=sections,
                    )
                    filings.append(filing)

                    logger.info(
                        f"  {filing_type} {meta['filing_date']}: "
                        f"{len(sections)} sections, {filing.total_words} words"
                    )
                except Exception as e:
                    logger.error(
                        f"  Failed to process {filing_type} "
                        f"{meta.get('filing_date', '?')}: {e}"
                    )

        return filings

    def ingest_all(self) -> list[Filing]:
        """Ingest filings for all configured companies."""
        all_filings = []

        companies = self.config.get("companies", [])
        for company in companies:
            filings = self.ingest_company(company)
            all_filings.extend(filings)

            # Save incrementally
            self._save_filings(filings, company["ticker"])

        logger.info(
            f"Total: {len(all_filings)} filings, "
            f"{sum(f.total_words for f in all_filings)} words"
        )
        return all_filings

    def _save_filings(self, filings: list[Filing], ticker: str):
        """Save parsed filings as JSON for reproducibility."""
        output_path = self.output_dir / f"{ticker}_filings.json"

        serializable = []
        for filing in filings:
            filing_dict = {
                "company": filing.company,
                "ticker": filing.ticker,
                "cik": filing.cik,
                "filing_type": filing.filing_type,
                "filing_date": filing.filing_date,
                "accession_number": filing.accession_number,
                "url": filing.url,
                "total_words": filing.total_words,
                "sections": [asdict(s) for s in filing.sections],
            }
            serializable.append(filing_dict)

        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)

        logger.info(f"Saved {len(filings)} filings to {output_path}")

    @staticmethod
    def load_filings(filepath: str) -> list[dict]:
        """Load previously saved filings from JSON."""
        with open(filepath, "r") as f:
            return json.load(f)

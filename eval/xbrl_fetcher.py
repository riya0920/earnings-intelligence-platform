"""
eval/xbrl_fetcher.py
====================

Fetch canonical financial facts from SEC EDGAR's XBRL Company Facts API.

Why XBRL
--------
Public filings are submitted in XBRL format with structured tags
(`us-gaap:Revenues`, `us-gaap:CostOfRevenue`, etc.). The SEC exposes
these as JSON via the company-facts endpoint. Using XBRL as ground
truth means our eval is anchored to what companies actually filed,
independent of any LLM, retriever, or prompt — the only "answer key"
we can fully trust.

Endpoint
--------
    https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json

Returns: {
    "cik": int,
    "entityName": "APPLE INC",
    "facts": {
        "us-gaap": {
            "Revenues": {
                "label": "Revenues",
                "units": {
                    "USD": [
                        {"end": "2024-09-28", "val": 391035000000,
                         "form": "10-K", "fy": 2024, "fp": "FY", ...},
                        ...
                    ]
                }
            },
            ...
        }
    }
}

A "fact" is one period's reported value for one tag. We index facts by
(ticker, tag, period_end) for fast lookup during scoring. The API requires
a User-Agent header; SEC asks for company/contact info.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests


# CIK numbers for our 5 tickers. Sourced from EDGAR's company-tickers.json.
TICKER_TO_CIK: dict[str, int] = {
    "AAPL": 320193,
    "MSFT": 789019,
    "GOOGL": 1652044,
    "NVDA": 1045810,
    "META": 1326801,
}

# XBRL tag → our canonical schema field name. The us-gaap taxonomy has
# multiple synonyms for the same concept across companies and years; we
# accept any tag that maps and prefer the first one found per period.
XBRL_TAG_MAP: dict[str, str] = {
    # Revenue (companies use slightly different tags)
    "Revenues": "revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue",
    "RevenueFromContractWithCustomerIncludingAssessedTax": "revenue",
    "SalesRevenueNet": "revenue",
    # Cost of revenue / cost of goods sold
    "CostOfRevenue": "cogs",
    "CostOfGoodsAndServicesSold": "cogs",
    "CostOfGoodsSold": "cogs",
    # Gross profit (when explicitly tagged — some filers don't tag it)
    "GrossProfit": "gross_profit",
    # Operating income (closest to EBITDA in many filings)
    "OperatingIncomeLoss": "operating_income",
    # Net income
    "NetIncomeLoss": "net_income",
    "ProfitLoss": "net_income",
}

SEC_BASE = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
CACHE_DIR = Path(__file__).parent / ".xbrl_cache"


def _build_user_agent() -> str:
    """Build the User-Agent string SEC requires.

    SEC's policy (https://www.sec.gov/os/accessing-edgar-data): every
    automated request must include a User-Agent that identifies the
    requester with a real contact email. Generic strings like
    "Mozilla/5.0" or placeholder emails get rate-limited to 0 (HTTP 403)
    and the IP is blocked for ~10 minutes.

    We read SEC_USER_AGENT from the environment so each user supplies
    their OWN email. Format expected:
        "CompanyOrAppName YourName your.email@example.com"
    """
    ua = os.getenv("SEC_USER_AGENT", "").strip()
    if ua and "@" in ua:
        return ua
    raise RuntimeError(
        "SEC EDGAR requires a real contact email in the User-Agent header.\n"
        "Set the SEC_USER_AGENT environment variable to identify yourself, e.g.:\n\n"
        "    PowerShell:  $env:SEC_USER_AGENT = "
        "'EIP-Eval Riya Soni rsoni6@stevens.edu'\n"
        "    bash:        export SEC_USER_AGENT="
        "'EIP-Eval Riya Soni rsoni6@stevens.edu'\n\n"
        "Without this, SEC returns 403 Forbidden and may block your IP "
        "for ~10 minutes.\n"
        "See https://www.sec.gov/os/accessing-edgar-data for SEC's policy."
    )


@dataclass(frozen=True)
class FactKey:
    """Identifies one fact: which company, which metric, which period."""
    ticker: str
    field: str               # canonical field name (revenue / cogs / etc.)
    period_end: str          # YYYY-MM-DD
    fiscal_period: str       # "Q1" / "Q2" / "Q3" / "FY"
    fiscal_year: int


@dataclass(frozen=True)
class Fact:
    key: FactKey
    value_usd_millions: float
    form: str                # "10-Q" / "10-K"
    filed_date: str          # YYYY-MM-DD
    xbrl_tag: str            # original tag name
    duration_days: int = 0   # length of the reporting period (90 ≈ quarter,
                             # 180 ≈ H1, 270 ≈ Q1+Q2+Q3, 365 ≈ FY)


def fetch_company_facts(ticker: str, *, force_refresh: bool = False) -> dict:
    """Download (or load cached) XBRL facts for a ticker.

    SEC asks for a polite User-Agent and rate-limits to ~10 req/sec.
    We cache aggressively because the JSON is large (multi-MB) and
    facts only change when new filings post.
    """
    cik = TICKER_TO_CIK[ticker.upper()]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"CIK{cik:010d}.json"

    if cache.exists() and not force_refresh:
        return json.loads(cache.read_text())

    url = SEC_BASE.format(cik=cik)
    headers = {
        "User-Agent": _build_user_agent(),
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    }
    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code == 403:
        raise RuntimeError(
            f"SEC returned 403 Forbidden for {url}.\n"
            f"This usually means your IP is rate-limit-blocked for ~10 min, "
            f"or your User-Agent doesn't include a real email.\n"
            f"Check SEC_USER_AGENT and wait 10 min if you've made many "
            f"recent requests."
        )
    resp.raise_for_status()
    data = resp.json()
    cache.write_text(json.dumps(data))
    # Be polite to SEC: 10 req/sec hard limit; we do far fewer.
    time.sleep(0.15)
    return data


def _normalize_value(raw_value: float, units_key: str) -> Optional[float]:
    """Convert XBRL raw value to USD millions.

    XBRL reports in absolute USD (or cents, shares, etc.). Most income-
    statement tags use `USD`; our schema is millions of USD. We reject
    non-USD units rather than guessing.
    """
    if units_key != "USD":
        return None
    return raw_value / 1_000_000.0


def _classify_period(form: str, fp: str) -> str:
    """Classify a reported period as Q1/Q2/Q3/Q4/FY for question matching."""
    fp = (fp or "").upper().strip()
    if fp == "FY":
        return "FY"
    if fp in ("Q1", "Q2", "Q3"):
        return fp
    # 10-K filings sometimes report Q4 as FY but the value reflects full year.
    if form == "10-K" and fp == "Q4":
        return "Q4"
    if form == "10-K":
        return "FY"
    return fp or "?"


def index_facts(ticker: str) -> dict[FactKey, Fact]:
    """Walk a company's XBRL JSON and return a {FactKey: Fact} index.

    When multiple XBRL tags map to the same canonical field (e.g.,
    Revenues vs RevenueFromContractWithCustomer for revenue), the first
    tag encountered wins for a given period. This mirrors how a human
    reading the filing would resolve "the revenue line" — there is one
    canonical value per period.

    Critical detail — duration filtering
    ------------------------------------
    Income-statement XBRL facts come with `start` and `end` dates. SEC
    filings emit the same metric multiple times for the same period_end:

        Apple Q2 FY2026 10-Q reports Revenues with:
          start=2025-12-28, end=2026-03-28  (3 months  ≈ 90 days)  -> $111,184M
          start=2025-09-28, end=2026-03-28  (6 months  ≈ 180 days) -> $254,940M  (cumulative)

    A naive index treats these as two facts with the same key. Last-write
    wins gives the wrong answer half the time. We use duration to
    distinguish:
      * For quarterly fields (Q1/Q2/Q3): keep only ~90-day spans.
      * For FY fields: keep only ~365-day spans.

    This makes "revenue for Q2 FY2026" return the *quarterly* value, not
    the year-to-date cumulative.
    """
    from datetime import date

    raw = fetch_company_facts(ticker)
    facts_root = raw.get("facts", {}).get("us-gaap", {})
    out: dict[FactKey, Fact] = {}

    for tag, body in facts_root.items():
        canonical = XBRL_TAG_MAP.get(tag)
        if canonical is None:
            continue
        for unit_key, entries in (body.get("units") or {}).items():
            for entry in entries:
                form = entry.get("form", "")
                if form not in ("10-K", "10-Q"):
                    continue
                fy = entry.get("fy")
                fp = _classify_period(form, entry.get("fp", ""))
                period_end = entry.get("end")
                if fy is None or period_end is None:
                    continue
                # Compute the duration. Balance-sheet items (which we don't
                # currently map) have no `start`; income-statement items do.
                start = entry.get("start")
                duration = 0
                if start:
                    try:
                        duration = (date.fromisoformat(period_end)
                                    - date.fromisoformat(start)).days
                    except (ValueError, TypeError):
                        duration = 0
                # Filter to the duration that matches the fiscal period.
                # Quarterly items: ~90 days (allow 80-100 for variation).
                # FY items: ~365 days (allow 350-380).
                if fp in ("Q1", "Q2", "Q3", "Q4"):
                    if not (80 <= duration <= 100):
                        continue
                elif fp == "FY":
                    if not (350 <= duration <= 380):
                        continue
                # Else: unknown classification, accept any.
                value_m = _normalize_value(entry.get("val", 0), unit_key)
                if value_m is None:
                    continue
                key = FactKey(
                    ticker=ticker.upper(), field=canonical,
                    period_end=period_end, fiscal_period=fp, fiscal_year=int(fy),
                )
                # First tag wins per period — don't overwrite.
                if key in out:
                    continue
                out[key] = Fact(
                    key=key, value_usd_millions=round(value_m, 1),
                    form=form, filed_date=entry.get("filed", ""),
                    xbrl_tag=tag, duration_days=duration,
                )
    return out


def latest_period_facts(ticker: str, fiscal_period: str = "Q1") -> dict[str, Fact]:
    """Return the most recent reported facts for a given fiscal period.

    Returns a {field: Fact} dict for the most recent fiscal_year where
    that ticker reported a value for the period. Useful for "what was
    the most recent Q1 revenue" style questions.
    """
    idx = index_facts(ticker)
    by_year_field: dict[tuple[int, str], Fact] = {}
    for key, fact in idx.items():
        if key.fiscal_period == fiscal_period:
            by_year_field[(key.fiscal_year, key.field)] = fact
    if not by_year_field:
        return {}
    latest_year = max(y for (y, _) in by_year_field.keys())
    return {field: fact for (year, field), fact in by_year_field.items()
            if year == latest_year}


if __name__ == "__main__":
    # CLI: pre-warm cache for all 5 tickers and print a summary.
    import sys
    tickers = sys.argv[1:] or list(TICKER_TO_CIK.keys())
    for tk in tickers:
        try:
            idx = index_facts(tk)
            fields = sorted({k.field for k in idx.keys()})
            years = sorted({k.fiscal_year for k in idx.keys()})
            print(f"{tk:<6} {len(idx):>5} facts  fields={fields}  "
                  f"years={years[:3]}...{years[-3:] if len(years) > 3 else ''}")
        except Exception as e:
            print(f"{tk:<6} ERROR: {type(e).__name__}: {e}")
"""
src/reconciliation/xbrl.py
==========================

Runtime XBRL lookup, promoted from `eval/xbrl_fetcher.py`.

This module is the runtime-side twin of the eval-time fetcher. The
eval uses XBRL as offline ground truth; this module uses XBRL as a
real-time anchor for the verified pipeline. The fetching, caching,
and indexing logic is identical â€” we add a higher-level `XBRLLookup`
class that:

  1. Caches indexed facts per ticker in-process (so successive
     reconciliations for the same ticker don't re-walk the JSON).
  2. Exposes a focused lookup interface: given (ticker, field,
     period_end), return the matching `Fact` or None.
  3. Fails soft: lookup errors return None rather than raising,
     so a transient SEC outage degrades the system to "no
     reconciliation available" rather than breaking every query.

The eval framework imports `eval/xbrl_fetcher.py` for backwards
compatibility; that file is now a thin re-export of the symbols
defined here.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)


# CIK numbers for our 5 tickers. Sourced from EDGAR's company-tickers.json.
TICKER_TO_CIK: dict[str, int] = {
    "AAPL": 320193,
    "MSFT": 789019,
    "GOOGL": 1652044,
    "NVDA": 1045810,
    "META": 1326801,
}

# XBRL tag -> our canonical schema field. The us-gaap taxonomy has
# multiple synonyms for the same concept across companies and years; we
# accept any tag that maps and prefer the first one found per period.
XBRL_TAG_MAP: dict[str, str] = {
    # Revenue
    "Revenues": "revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue",
    "RevenueFromContractWithCustomerIncludingAssessedTax": "revenue",
    "SalesRevenueNet": "revenue",
    # Cost of revenue / cost of goods sold
    "CostOfRevenue": "cogs",
    "CostOfGoodsAndServicesSold": "cogs",
    "CostOfGoodsSold": "cogs",
    # Gross profit
    "GrossProfit": "gross_profit",
    # Operating income (closest tagged proxy for EBITDA; many filers
    # don't tag EBITDA directly â€” reconciliation will flag this gap).
    "OperatingIncomeLoss": "operating_income",
    # Net income
    "NetIncomeLoss": "net_income",
    "ProfitLoss": "net_income",
}

SEC_BASE = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
# Cache lives next to the eval cache so a refresh in either place
# warms the other. Override with EIP_XBRL_CACHE_DIR for tests.
CACHE_DIR = Path(os.getenv(
    "EIP_XBRL_CACHE_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "eval" / ".xbrl_cache"),
))


def _build_user_agent() -> str:
    """SEC requires a real contact email; we read from SEC_USER_AGENT env.

    Generic User-Agents or placeholder emails get the IP blocked for
    ~10 minutes by SEC's rate limiter. See:
    https://www.sec.gov/os/accessing-edgar-data
    """
    ua = os.getenv("SEC_USER_AGENT", "").strip()
    if ua and "@" in ua:
        return ua
    raise RuntimeError(
        "SEC EDGAR requires a real contact email in the User-Agent header.\n"
        "Set the SEC_USER_AGENT environment variable, e.g.:\n"
        "    $env:SEC_USER_AGENT = 'YourApp YourName your@email.com'"
    )


@dataclass(frozen=True)
class FactKey:
    """Identifies one fact: which company, which metric, which period."""
    ticker: str
    field: str
    period_end: str          # YYYY-MM-DD
    fiscal_period: str       # "Q1" / "Q2" / "Q3" / "FY"
    fiscal_year: int


@dataclass(frozen=True)
class Fact:
    """One reported value at one period, normalized to USD millions."""
    key: FactKey
    value_usd_millions: float
    form: str                # "10-Q" / "10-K"
    filed_date: str          # YYYY-MM-DD
    xbrl_tag: str
    duration_days: int = 0


# ---------------------------------------------------------------------------
# Low-level fetch + index (unchanged behavior from eval/xbrl_fetcher.py).
# ---------------------------------------------------------------------------


def fetch_company_facts(ticker: str, *, force_refresh: bool = False) -> dict:
    """Download (or load cached) XBRL facts JSON for a ticker."""
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
            f"SEC returned 403 for {url}. Check SEC_USER_AGENT or wait "
            f"~10 min if you've been rate-limited."
        )
    resp.raise_for_status()
    data = resp.json()
    cache.write_text(json.dumps(data))
    time.sleep(0.15)  # be polite to SEC
    return data


def _normalize_value(raw_value: float, units_key: str) -> Optional[float]:
    """Convert raw XBRL value to USD millions; reject non-USD."""
    if units_key != "USD":
        return None
    return raw_value / 1_000_000.0


def _classify_period(form: str, fp: str) -> str:
    fp = (fp or "").upper().strip()
    if fp == "FY":
        return "FY"
    if fp in ("Q1", "Q2", "Q3"):
        return fp
    if form == "10-K" and fp == "Q4":
        return "Q4"
    if form == "10-K":
        return "FY"
    return fp or "?"


def index_facts(ticker: str) -> dict[FactKey, Fact]:
    """Walk a company's XBRL JSON and return {FactKey: Fact}.

    Critical: duration filtering distinguishes quarterly values from
    cumulative YTD values that share the same period_end. See the
    docstring in eval/xbrl_fetcher.py for the full explanation.
    """
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
                start = entry.get("start")
                duration = 0
                if start:
                    try:
                        duration = (date.fromisoformat(period_end)
                                    - date.fromisoformat(start)).days
                    except (ValueError, TypeError):
                        duration = 0
                if fp in ("Q1", "Q2", "Q3", "Q4"):
                    if not (80 <= duration <= 100):
                        continue
                elif fp == "FY":
                    if not (350 <= duration <= 380):
                        continue
                value_m = _normalize_value(entry.get("val", 0), unit_key)
                if value_m is None:
                    continue
                key = FactKey(
                    ticker=ticker.upper(), field=canonical,
                    period_end=period_end, fiscal_period=fp,
                    fiscal_year=int(fy),
                )
                if key in out:
                    continue  # first tag wins per period
                out[key] = Fact(
                    key=key, value_usd_millions=round(value_m, 1),
                    form=form, filed_date=entry.get("filed", ""),
                    xbrl_tag=tag, duration_days=duration,
                )
    return out


def latest_period_facts(ticker: str, fiscal_period: str = "Q1") -> dict[str, Fact]:
    """Most recent reported facts for a given fiscal period."""
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


# ---------------------------------------------------------------------------
# Runtime lookup wrapper.
# ---------------------------------------------------------------------------


class XBRLLookup:
    """Cache-friendly, fail-soft XBRL lookup for the runtime reconciliation step.

    Usage:
        lookup = XBRLLookup()
        fact = lookup.lookup("AAPL", "revenue", period_end="2025-12-27")
        if fact:
            print(fact.value_usd_millions, fact.xbrl_tag)

    Failure modes (all return None instead of raising):
      â€¢ Ticker not in our CIK map
      â€¢ SEC fetch fails (network, 403, etc.)
      â€¢ No fact for the requested (ticker, field, period_end)

    The first call per ticker walks the indexed JSON; subsequent calls
    are dict lookups.
    """

    def __init__(self):
        # Lazy per-ticker index cache; populated on first lookup.
        self._index_cache: dict[str, dict[FactKey, Fact]] = {}

    def _get_index(self, ticker: str) -> Optional[dict[FactKey, Fact]]:
        tk = ticker.upper()
        if tk in self._index_cache:
            return self._index_cache[tk]
        if tk not in TICKER_TO_CIK:
            logger.info("XBRLLookup: ticker %s not in CIK map", tk)
            return None
        try:
            idx = index_facts(tk)
        except Exception as e:  # noqa: BLE001 - intentional broad catch
            logger.warning("XBRLLookup: failed to index %s: %s", tk, e)
            return None
        self._index_cache[tk] = idx
        return idx

    def lookup(
        self,
        ticker: str,
        field: str,
        period_end: Optional[str] = None,
    ) -> Optional[Fact]:
        """Return the matching Fact or None.

        If period_end is given, requires an exact match on the period.
        If period_end is None, returns the most recent quarterly fact
        for the requested field (useful when the verified pipeline
        couldn't pin a period from the user's question).
        """
        idx = self._get_index(ticker)
        if idx is None:
            return None

        if period_end:
            for key, fact in idx.items():
                if key.field == field and key.period_end == period_end:
                    return fact
            return None

        # No period given: pick the most recent quarterly value.
        candidates = [
            (k, f) for k, f in idx.items()
            if k.field == field and k.fiscal_period in ("Q1", "Q2", "Q3")
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda kf: kf[0].period_end)[1]

    def latest_period_end(self, ticker: str) -> Optional[str]:
        """Most recent quarterly period_end on file for this ticker."""
        idx = self._get_index(ticker)
        if idx is None:
            return None
        quarterly = [k for k in idx if k.fiscal_period in ("Q1", "Q2", "Q3")]
        if not quarterly:
            return None
        return max(k.period_end for k in quarterly)
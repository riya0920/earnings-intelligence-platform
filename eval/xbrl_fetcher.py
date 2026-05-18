"""
eval/xbrl_fetcher.py
====================

Backwards-compat shim. The implementation moved to
``src/reconciliation/xbrl.py`` when XBRL lookup was promoted from an
eval-time utility to a runtime component. The eval framework continues
to import from this path; this file re-exports the same names so no
eval code had to change.

For the canonical implementation, edit ``src/reconciliation/xbrl.py``.
"""
from src.reconciliation.xbrl import (  # noqa: F401
    CACHE_DIR,
    Fact,
    FactKey,
    SEC_BASE,
    TICKER_TO_CIK,
    XBRL_TAG_MAP,
    XBRLLookup,
    fetch_company_facts,
    index_facts,
    latest_period_facts,
)

# Internal helpers some eval scripts may import directly.
from src.reconciliation.xbrl import (  # noqa: F401
    _build_user_agent,
    _classify_period,
    _normalize_value,
)

__all__ = [
    "TICKER_TO_CIK",
    "XBRL_TAG_MAP",
    "SEC_BASE",
    "CACHE_DIR",
    "FactKey",
    "Fact",
    "XBRLLookup",
    "fetch_company_facts",
    "index_facts",
    "latest_period_facts",
]
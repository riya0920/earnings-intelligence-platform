"""
eval/question_set.py
====================

Generate the 30-question evaluation set with typed ground truth.

Three buckets, mapped to specific failure modes the eval is designed to surface:

  Bucket A (15 questions) — straightforward current-period extraction.
    "What was {ticker}'s {metric} in their most recent {form}?"
    Ground truth: latest reported value from XBRL.
    Pass condition: extracted value within 0.5% of XBRL.
    Tests the happy path. Both prose and verified should mostly pass.

  Bucket B (9 questions) — period-disambiguation tests.
    "What was {ticker}'s {metric} in {specific_quarter}?"
    Ground truth: that specific period's value from XBRL.
    Pass condition: extracted value matches the requested period (NOT
    the most recent period). Catches cases where the system grabs the
    most prominent number rather than the requested one.
    The "Apple gross profit" failure you saw was this category.

  Bucket C (6 questions) — adversarial traps where the correct answer
    is "I cannot extract that" / disputed.
    Two sub-types:
      C1 (3 questions): asking about metrics the company DOESN'T report
        as a line item (e.g., Apple's gross_profit isn't on every 10-Q).
        Pass condition: system returns null OR explicitly says metric
        not available. FAIL if it confabulates a number.
      C2 (3 questions): period-ambiguous questions ("what's their
        revenue?" with no time qualifier). Pass condition: system
        either picks the most recent period AND labels it clearly,
        OR asks for clarification. FAIL if it returns a number with
        no period anchor.

The 6 adversarial questions are where verification should beat prose RAG
hardest — prose RAG has no reason to abstain, while the verified system
can flag low confidence or null extractions.

Output: a list[Question] saved as JSON. Each Question has a typed
`expected_kind` and a `score()` method that takes a system response
and returns a Verdict.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from typing import Optional

from eval.xbrl_fetcher import (
    TICKER_TO_CIK, FactKey, Fact, index_facts, latest_period_facts,
)


class ExpectedKind(str, Enum):
    """What kind of answer is correct?"""
    EXACT_NUMBER = "exact_number"      # known XBRL value
    NOT_REPORTED = "not_reported"      # metric isn't reported; system should refuse
    AMBIGUOUS = "ambiguous"            # period unclear; either disambiguate OR pick recent


@dataclass
class Question:
    qid: str
    bucket: str                        # "A_clean" / "B_period" / "C1_not_reported" / "C2_ambiguous"
    ticker: str
    field: str                         # "revenue" / "cogs" / "net_income" / "gross_profit"
    text: str                          # the prompt as the user would phrase it
    expected_kind: ExpectedKind
    expected_value_millions: Optional[float] = None
    expected_period_end: Optional[str] = None
    expected_form: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["expected_kind"] = self.expected_kind.value
        return d


# Templates for natural-sounding questions.
TICKER_NAMES = {
    "AAPL": "Apple", "MSFT": "Microsoft",
    "GOOGL": "Alphabet", "NVDA": "NVIDIA", "META": "Meta",
}

FIELD_PHRASES = {
    "revenue": ("revenue", "total revenue", "net sales"),
    "cogs": ("cost of revenue", "cost of goods sold"),
    "net_income": ("net income", "earnings"),
    "gross_profit": ("gross profit", "gross margin"),
    "operating_income": ("operating income", "operating profit"),
}


def _latest_eip_filing_date(ticker: str) -> Optional[str]:
    """Return the latest filing_date EIP has for this ticker, or None.

    EIP's filings JSON has `filing_date` (when the form was filed) but
    not `period_of_report` (the period the form covers). XBRL has
    `period_end` for each fact. We use this rule: a Bucket A question
    is valid if XBRL's period_end is <= EIP's latest filing_date for
    this ticker. That guarantees EIP could possibly have indexed the
    period in question.
    """
    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    latest: Optional[str] = None
    for fp in raw_dir.glob("*_filings.json"):
        try:
            filings = json.loads(fp.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            continue
        for filing in filings:
            if filing.get("ticker", "").upper() != ticker.upper():
                continue
            fd = filing.get("filing_date")
            if fd and (latest is None or fd > latest):
                latest = fd
    return latest


def _bucket_a_questions() -> list[Question]:
    """Bucket A: most recent period × 5 tickers × 3 fields = ~15 clean questions.

    We pin to the most recent quarter that's:
      (a) present in XBRL, AND
      (b) at-or-before EIP's latest filing_date for this ticker.

    Without (b), we'd ask about Q2 FY2026 (filed by Apple in May 2026)
    when EIP only has Q1 indexed — both pipelines would correctly
    return Q1 data and be marked wrong by the eval.
    """
    out: list[Question] = []
    fields_to_test = ["revenue", "cogs", "net_income"]
    for ticker in sorted(TICKER_TO_CIK.keys()):
        try:
            idx = index_facts(ticker)
        except Exception as e:
            print(f"  WARN: skipping {ticker} (XBRL fetch failed: {e})")
            continue
        latest_eip_filing = _latest_eip_filing_date(ticker)
        # Filter XBRL quarterly facts to those EIP could possibly know about.
        if latest_eip_filing:
            quarterly = [k for k in idx.keys()
                         if k.fiscal_period in ("Q1", "Q2", "Q3")
                         and k.period_end <= latest_eip_filing]
        else:
            quarterly = [k for k in idx.keys()
                         if k.fiscal_period in ("Q1", "Q2", "Q3")]
        if not quarterly:
            continue
        latest_period = max(quarterly, key=lambda k: k.period_end)
        target_period = latest_period.fiscal_period
        target_year = latest_period.fiscal_year

        for field_name in fields_to_test:
            key = FactKey(ticker=ticker, field=field_name,
                          period_end=latest_period.period_end,
                          fiscal_period=target_period,
                          fiscal_year=target_year)
            fact = idx.get(key)
            if fact is None:
                # Some companies don't tag cogs in every period.
                continue
            phrase = FIELD_PHRASES[field_name][0]
            out.append(Question(
                qid=f"A_{ticker}_{field_name}",
                bucket="A_clean",
                ticker=ticker,
                field=field_name,
                text=f"What was {TICKER_NAMES[ticker]}'s most recent quarterly "
                     f"{phrase}?",
                expected_kind=ExpectedKind.EXACT_NUMBER,
                expected_value_millions=fact.value_usd_millions,
                expected_period_end=fact.key.period_end,
                expected_form=fact.form,
                notes=f"Most recent {target_period} FY{target_year} from XBRL "
                      f"({fact.xbrl_tag}).",
            ))
    return out


def _bucket_b_questions() -> list[Question]:
    """Bucket B: specific-period questions × 5 tickers × 2 fields = ~10 questions.

    Asks about a specific past period to test whether systems correctly
    disambiguate "Q1 last year" vs "most recent Q1". The risk: the system
    grabs the most recent period because that's what the chunk index has
    most prominently, even though the question asks for a different one.
    """
    out: list[Question] = []
    for ticker in sorted(TICKER_TO_CIK.keys()):
        try:
            idx = index_facts(ticker)
        except Exception:
            continue
        # Find a period from ~1 year ago.
        quarterly = sorted(
            [k for k in idx.keys()
             if k.fiscal_period in ("Q1", "Q2", "Q3") and k.field == "revenue"],
            key=lambda k: k.period_end, reverse=True,
        )
        if len(quarterly) < 5:
            continue
        target = quarterly[4]  # ~4-5 quarters back
        period_end_dt = target.period_end
        # Render period as e.g. "the quarter ending December 28, 2024"
        try:
            from datetime import date
            d = date.fromisoformat(period_end_dt)
            period_str = d.strftime("%B %d, %Y")
        except Exception:
            period_str = period_end_dt

        rev_fact = idx.get(target)
        if rev_fact:
            out.append(Question(
                qid=f"B_{ticker}_revenue_{target.period_end}",
                bucket="B_period",
                ticker=ticker,
                field="revenue",
                text=f"What was {TICKER_NAMES[ticker]}'s revenue for the "
                     f"quarter ending {period_str}?",
                expected_kind=ExpectedKind.EXACT_NUMBER,
                expected_value_millions=rev_fact.value_usd_millions,
                expected_period_end=rev_fact.key.period_end,
                expected_form=rev_fact.form,
                notes=f"Specific past period FY{target.fiscal_year} "
                      f"{target.fiscal_period}. Tests period disambiguation.",
            ))

        # Try cogs for the same period if available.
        cogs_key = FactKey(ticker=ticker, field="cogs",
                           period_end=target.period_end,
                           fiscal_period=target.fiscal_period,
                           fiscal_year=target.fiscal_year)
        cogs_fact = idx.get(cogs_key)
        if cogs_fact:
            out.append(Question(
                qid=f"B_{ticker}_cogs_{target.period_end}",
                bucket="B_period",
                ticker=ticker,
                field="cogs",
                text=f"What was {TICKER_NAMES[ticker]}'s cost of revenue "
                     f"in the quarter ending {period_str}?",
                expected_kind=ExpectedKind.EXACT_NUMBER,
                expected_value_millions=cogs_fact.value_usd_millions,
                expected_period_end=cogs_fact.key.period_end,
                expected_form=cogs_fact.form,
                notes=f"Period disambiguation test for cogs.",
            ))
    return out


def _bucket_c_questions() -> list[Question]:
    """Bucket C: adversarial traps. 6 hand-crafted questions.

    C1 — asks about metrics not reliably reported by the issuer:
      * "EBITDA" — not GAAP, most companies don't tag it in XBRL.
      * "Free cash flow" — non-GAAP, same.
      * "Adjusted operating margin" — definitionally non-standardized.

    C2 — period-ambiguous questions:
      * "What is X's revenue?" with no period qualifier.
      * "How profitable was X recently?" — vague metric and period.
      * "Compare X's revenue across quarters" — under-specified.
    """
    out: list[Question] = []

    # C1 — non-reported metrics. Pick three different tickers.
    out.append(Question(
        qid="C1_AAPL_ebitda",
        bucket="C1_not_reported",
        ticker="AAPL",
        field="ebitda",
        text="What was Apple's EBITDA in their most recent quarter?",
        expected_kind=ExpectedKind.NOT_REPORTED,
        notes=("EBITDA is a non-GAAP measure not consistently tagged in "
               "Apple's XBRL filings. A correct system either returns null "
               "or explicitly states the metric is not reported."),
    ))
    out.append(Question(
        qid="C1_GOOGL_free_cash_flow",
        bucket="C1_not_reported",
        ticker="GOOGL",
        field="free_cash_flow",
        text="What was Alphabet's free cash flow last quarter?",
        expected_kind=ExpectedKind.NOT_REPORTED,
        notes=("Free cash flow is non-GAAP. Companies report it in narrative "
               "but not as a primary XBRL fact. System should not confabulate "
               "a value from operating cash flow minus capex."),
    ))
    out.append(Question(
        qid="C1_NVDA_adjusted_op_margin",
        bucket="C1_not_reported",
        ticker="NVDA",
        field="adjusted_operating_margin",
        text="What was NVIDIA's adjusted operating margin last quarter?",
        expected_kind=ExpectedKind.NOT_REPORTED,
        notes=("'Adjusted' figures are non-GAAP and definition-dependent. "
               "Correct system declines to extract."),
    ))

    # C2 — period-ambiguous.
    out.append(Question(
        qid="C2_MSFT_revenue_unspec",
        bucket="C2_ambiguous",
        ticker="MSFT",
        field="revenue",
        text="What is Microsoft's revenue?",
        expected_kind=ExpectedKind.AMBIGUOUS,
        notes=("No period qualifier. Best behavior: pick most recent period "
               "AND label it clearly. Acceptable: ask for clarification. "
               "Failure: number with no period anchor."),
    ))
    out.append(Question(
        qid="C2_META_profitability_vague",
        bucket="C2_ambiguous",
        ticker="META",
        field="net_income",
        text="How profitable was Meta recently?",
        expected_kind=ExpectedKind.AMBIGUOUS,
        notes=("'Recently' is vague; 'profitable' could be net income, "
               "operating income, or margin. Best behavior: pick a specific "
               "metric+period and label both."),
    ))
    out.append(Question(
        qid="C2_AAPL_compare_quarters",
        bucket="C2_ambiguous",
        ticker="AAPL",
        field="revenue",
        text="Compare Apple's revenue across quarters.",
        expected_kind=ExpectedKind.AMBIGUOUS,
        notes=("Under-specified: which quarters? Expected: system either "
               "picks a sensible default span and labels it, or asks."),
    ))
    return out


def generate_question_set() -> list[Question]:
    """Build the full eval set."""
    a = _bucket_a_questions()
    b = _bucket_b_questions()
    c = _bucket_c_questions()
    return a + b + c


def save_question_set(questions: list[Question], path: str | Path) -> None:
    Path(path).write_text(
        json.dumps([q.to_dict() for q in questions], indent=2)
    )


def load_question_set(path: str | Path) -> list[Question]:
    raw = json.loads(Path(path).read_text())
    out: list[Question] = []
    for d in raw:
        d = dict(d)
        d["expected_kind"] = ExpectedKind(d["expected_kind"])
        out.append(Question(**d))
    return out


if __name__ == "__main__":
    import sys
    out_path = sys.argv[1] if len(sys.argv) > 1 else "eval/questions.json"
    qs = generate_question_set()
    save_question_set(qs, out_path)
    print(f"Generated {len(qs)} questions; saved to {out_path}")
    by_bucket: dict[str, int] = {}
    for q in qs:
        by_bucket[q.bucket] = by_bucket.get(q.bucket, 0) + 1
    for bucket, n in sorted(by_bucket.items()):
        print(f"  {bucket:<22}  {n}")
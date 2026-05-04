# Combined Evaluation: Verified RAG vs Prose RAG

End-to-end empirical comparison of EIP's prose-only `query` pipeline against the verified `verify` pipeline (RAG + adversarial auditor).

## What it measures

For 30 questions across all 5 ingested companies (AAPL, MSFT, GOOGL, NVDA, META), it runs each question through both pipelines and scores each response against ground truth pulled from SEC EDGAR's XBRL Company Facts API. The headline metric is **confident-wrong rate** — how often each pipeline confidently returns a number that doesn't match the canonical filed value.

## Question set

| Bucket | Count | What it tests |
|---|---|---|
| **A — clean** | 15 | Most-recent-period revenue / cogs / net_income across the 5 tickers. Both pipelines should mostly pass. |
| **B — period disambiguation** | ~9 | Specific past quarter, e.g. "What was Apple's revenue for the quarter ending December 28, 2024?" Catches systems that grab the most prominent number rather than the requested one. |
| **C — adversarial** | 6 | C1 (3): metrics not reported by the company (EBITDA, free cash flow, "adjusted operating margin"). C2 (3): period-ambiguous questions. Pass condition is *correctly refusing*, not extracting a number. |

Total ≈ 30 (Bucket B varies slightly by ticker availability in XBRL).

## Ground truth

`xbrl_fetcher.py` pulls each company's facts from `data.sec.gov/api/xbrl/companyfacts/CIK*.json` — the same XBRL data the SEC ingests from filings. We map XBRL tags (`us-gaap:Revenues`, `us-gaap:CostOfRevenue`, `us-gaap:NetIncomeLoss`, etc.) to our canonical schema. This makes ground truth fully objective and independent of any LLM.

## Workflow

```bash
# 1. Pre-warm the XBRL cache (one-time, ~30 sec)
python -m eval.xbrl_fetcher

# 2. Generate the question set
python -m eval.question_set eval/questions.json

# 3. Run both pipelines (longest step — ~30-60 min for 30 questions × 2 pipelines)
$env:HUNTER_MODEL = "gemini-2.5-flash"   # PowerShell
$env:AUDITOR_MODEL = "gpt-4o-mini"
python -m eval.runner --questions eval/questions.json --out eval/results.jsonl

# 3a. Smoke-test on first 3 questions before the full run:
python -m eval.runner --limit 3 --out eval/smoke.jsonl

# 3b. Resume after interruption (skips already-done qid+pipeline pairs):
python -m eval.runner --resume

# 4. Aggregate the report
python -m eval.report --in eval/results.jsonl
python -m eval.report --in eval/results.jsonl --json eval/metrics.json
```

## Cost & time

Per-question cost (HUNTER=gemini-2.5-flash, AUDITOR=gpt-4o-mini, prose=gpt-4o-mini):

- Prose pipeline: ~$0.001 / question, ~5s wall time after retriever is warm
- Verified pipeline: ~$0.005 / question, ~30-60s wall time (3 LLM calls + provenance + arbiter)

Full sweep: ~$0.20 and 30-60 minutes of wall time.

## Reading the report

The `--json` flag emits machine-readable metrics for inclusion in the EIP README:

```json
{
  "per_pipeline": {
    "prose":    {"pass_rate": {"passed": 14, "n": 30, "rate": 0.467, "ci_low": 0.302, "ci_high": 0.639},
                 "confident_wrong_rate": {"confident_wrong": 11, "n": 30, "rate": 0.367}},
    "verified": {"pass_rate": {"passed": 24, "n": 30, "rate": 0.800, "ci_low": 0.627, "ci_high": 0.905},
                 "confident_wrong_rate": {"confident_wrong": 2, "n": 30, "rate": 0.067}}
  },
  "paired_mcnemar": {"n_paired": 30, "verified_only_pass": 12, "prose_only_pass": 2,
                     "p_value": 0.013}
}
```

The headline interview line is the first row of `confident_wrong_rate`: *"prose RAG returned a confidently-wrong number on X of 30; verified caught Y of those, p=Z by McNemar's exact test."*

## Caveats

- **n=30 is small.** Wilson CIs are wide — we report them so the limitation is visible, not hidden.
- **XBRL coverage is imperfect.** Some companies tag "Revenues" while others use "RevenueFromContractWithCustomer". `xbrl_fetcher.XBRL_TAG_MAP` accepts both; if a Bucket-A question has no GT in XBRL it's silently skipped at generation time.
- **Adversarial questions are designed to fail.** They test for *appropriate refusal*, which prose RAG generally won't do (it'll confabulate). The verified pipeline's structured-facts emptiness or DISPUTED status is what passes.
- **The eval set is small enough to overfit.** If you tweak prompts after seeing results, you're tuning to this set. Treat it as a regression check, not a leaderboard.
"""
tests/test_reconciliation.py
============================

Unit tests for src/reconciliation/.

Coverage:
  â€¢ Schema: enum semantics, ReconciliationFinding/Report serialization,
    histogram correctness, human-review flagging
  â€¢ XBRLLookup: caching, fail-soft semantics, period_end matching,
    latest-quarter fallback when period_end is None
  â€¢ ReconciliationAgent: delta_pct math (including division-by-zero),
    classifier output parsing (clean, code-fenced, malformed, unknown
    pattern), NO_DELTA fast-path (no LLM call), tolerance boundary,
    ticker inference from chunk metadata

Network is mocked throughout: no real SEC EDGAR or OpenAI calls.
Run with: pytest tests/test_reconciliation.py -v
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from src.reconciliation.agent import (
    ProseFactInput,
    ReconciliationAgent,
)
from src.reconciliation.schema import (
    DeltaPattern,
    ReconciliationFinding,
    ReconciliationReport,
)
from src.reconciliation.xbrl import (
    Fact,
    FactKey,
    TICKER_TO_CIK,
    XBRLLookup,
    XBRL_TAG_MAP,
)


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_facts_index():
    """A small fake XBRL index for AAPL with two periods of revenue."""
    return {
        FactKey("AAPL", "revenue", "2025-12-27", "Q1", 2026): Fact(
            key=FactKey("AAPL", "revenue", "2025-12-27", "Q1", 2026),
            value_usd_millions=143756.0,
            form="10-Q", filed_date="2026-01-30",
            xbrl_tag="Revenues", duration_days=91,
        ),
        FactKey("AAPL", "revenue", "2025-09-28", "FY", 2025): Fact(
            key=FactKey("AAPL", "revenue", "2025-09-28", "FY", 2025),
            value_usd_millions=391035.0,
            form="10-K", filed_date="2025-10-31",
            xbrl_tag="Revenues", duration_days=365,
        ),
        FactKey("AAPL", "net_income", "2025-12-27", "Q1", 2026): Fact(
            key=FactKey("AAPL", "net_income", "2025-12-27", "Q1", 2026),
            value_usd_millions=42097.0,
            form="10-Q", filed_date="2026-01-30",
            xbrl_tag="NetIncomeLoss", duration_days=91,
        ),
    }


@pytest.fixture
def aapl_revenue_fact(sample_facts_index):
    """One specific fact for direct comparisons."""
    return sample_facts_index[
        FactKey("AAPL", "revenue", "2025-12-27", "Q1", 2026)
    ]


def make_prose_fact(field_name="revenue", value=143756.0, **overrides):
    """Convenience constructor for ProseFactInput used in many tests."""
    defaults = dict(
        field_name=field_name,
        value=value,
        unit="USD_millions",
        source_quote="Net sales were $143,756 million.",
        chunk_metadata={
            "company": "Apple Inc.",
            "filing_type": "10-Q",
            "filing_date": "2026-01-30",
            "section": "financial_statements",
        },
    )
    defaults.update(overrides)
    return ProseFactInput(**defaults)


# ---------------------------------------------------------------------------
# Schema tests.
# ---------------------------------------------------------------------------


class TestDeltaPattern:

    def test_legitimate_patterns(self):
        legit = {
            DeltaPattern.NO_DELTA,
            DeltaPattern.NON_GAAP_VS_GAAP,
            DeltaPattern.PERIODICITY,
            DeltaPattern.HIERARCHY,
            DeltaPattern.VERSIONING,
            DeltaPattern.METRIC_MAPPING,
        }
        for p in legit:
            assert p.is_legitimate, f"{p} should be legitimate"

    def test_non_legitimate_patterns(self):
        for p in (DeltaPattern.UNEXPLAINED, DeltaPattern.LOOKUP_FAILED):
            assert not p.is_legitimate, f"{p} should NOT be legitimate"

    def test_string_values_are_lowercase_snake(self):
        for p in DeltaPattern:
            assert p.value.islower()
            assert " " not in p.value


class TestReconciliationFinding:

    def test_matches_property(self):
        f = ReconciliationFinding(
            field_name="revenue", prose_value=100, xbrl_value=100,
            xbrl_tag="Revenues", xbrl_period_end="2025-12-27",
            delta_pct=0.0, pattern=DeltaPattern.NO_DELTA, rationale="ok",
        )
        assert f.matches is True

        f2 = ReconciliationFinding(
            field_name="revenue", prose_value=100, xbrl_value=110,
            xbrl_tag="Revenues", xbrl_period_end="2025-12-27",
            delta_pct=-9.1, pattern=DeltaPattern.NON_GAAP_VS_GAAP,
            rationale="adj",
        )
        assert f2.matches is False

    def test_to_dict_includes_pattern_value(self):
        f = ReconciliationFinding(
            field_name="revenue", prose_value=100, xbrl_value=110,
            xbrl_tag="Revenues", xbrl_period_end="2025-12-27",
            delta_pct=-9.1, pattern=DeltaPattern.PERIODICITY, rationale="TTM",
        )
        d = f.to_dict()
        assert d["pattern"] == "periodicity"
        assert d["delta_pct"] == -9.1


class TestReconciliationReport:

    def test_empty_report(self):
        r = ReconciliationReport()
        assert r.total == 0
        assert r.matches == 0
        assert r.pattern_counts == {}

    def test_aggregations(self):
        findings = [
            ReconciliationFinding(
                field_name="revenue", prose_value=100, xbrl_value=100,
                xbrl_tag="Revenues", xbrl_period_end="2025-12-27",
                delta_pct=0.0, pattern=DeltaPattern.NO_DELTA, rationale="ok",
            ),
            ReconciliationFinding(
                field_name="ebitda", prose_value=30, xbrl_value=25,
                xbrl_tag="OperatingIncomeLoss", xbrl_period_end="2025-12-27",
                delta_pct=20.0, pattern=DeltaPattern.NON_GAAP_VS_GAAP,
                rationale="adj",
            ),
            ReconciliationFinding(
                field_name="cogs", prose_value=50, xbrl_value=70,
                xbrl_tag="CostOfRevenue", xbrl_period_end="2025-12-27",
                delta_pct=-28.6, pattern=DeltaPattern.UNEXPLAINED,
                rationale="?", human_review_recommended=True,
            ),
            ReconciliationFinding(
                field_name="gross_profit", prose_value=50, xbrl_value=None,
                xbrl_tag=None, xbrl_period_end=None,
                delta_pct=None, pattern=DeltaPattern.LOOKUP_FAILED,
                rationale="no fact",
            ),
        ]
        r = ReconciliationReport(findings=findings, ticker="AAPL")
        assert r.total == 4
        assert r.matches == 1
        assert r.legitimate_deltas == 1  # only NON_GAAP_VS_GAAP
        assert r.human_review_count == 1
        assert r.lookup_failed_count == 1

        counts = r.pattern_counts
        assert counts["no_delta"] == 1
        assert counts["non_gaap_vs_gaap"] == 1
        assert counts["unexplained"] == 1
        assert counts["lookup_failed"] == 1

    def test_to_dict_round_trip_shape(self):
        r = ReconciliationReport(
            findings=[ReconciliationFinding(
                field_name="revenue", prose_value=100, xbrl_value=100,
                xbrl_tag="Revenues", xbrl_period_end="2025-12-27",
                delta_pct=0.0, pattern=DeltaPattern.NO_DELTA, rationale="ok",
            )],
            ticker="AAPL", period_end_used="2025-12-27",
        )
        d = r.to_dict()
        for key in ("ticker", "period_end_used", "total", "matches",
                    "legitimate_deltas", "human_review_count",
                    "lookup_failed_count", "pattern_counts", "findings"):
            assert key in d


# ---------------------------------------------------------------------------
# XBRLLookup tests.
# ---------------------------------------------------------------------------


class TestXBRLLookup:

    def test_unknown_ticker_returns_none(self):
        lookup = XBRLLookup()
        assert lookup.lookup("UNKNOWN_TICKER", "revenue") is None

    def test_period_specific_lookup_matches(self, sample_facts_index):
        lookup = XBRLLookup()
        # Inject fake index into cache so we don't hit network.
        lookup._index_cache["AAPL"] = sample_facts_index
        fact = lookup.lookup("AAPL", "revenue", period_end="2025-12-27")
        assert fact is not None
        assert fact.value_usd_millions == 143756.0
        assert fact.xbrl_tag == "Revenues"

    def test_period_specific_lookup_no_match(self, sample_facts_index):
        lookup = XBRLLookup()
        lookup._index_cache["AAPL"] = sample_facts_index
        # Period we don't have facts for.
        fact = lookup.lookup("AAPL", "revenue", period_end="2099-12-31")
        assert fact is None

    def test_no_period_picks_most_recent_quarterly(self, sample_facts_index):
        lookup = XBRLLookup()
        lookup._index_cache["AAPL"] = sample_facts_index
        # Should pick Q1 2026 (the only quarterly), not the FY 2025 fact.
        fact = lookup.lookup("AAPL", "revenue", period_end=None)
        assert fact is not None
        assert fact.key.fiscal_period == "Q1"
        assert fact.key.period_end == "2025-12-27"

    def test_latest_period_end_returns_most_recent_quarter(self, sample_facts_index):
        lookup = XBRLLookup()
        lookup._index_cache["AAPL"] = sample_facts_index
        # FY fact has later period_end (2025-09-28 vs 2025-12-27)... wait,
        # Q1 ends 2025-12-27 which IS more recent. Verify.
        assert lookup.latest_period_end("AAPL") == "2025-12-27"

    def test_failed_fetch_returns_none(self):
        """If index_facts raises (network down, 403, etc.), lookup returns None."""
        lookup = XBRLLookup()
        with patch(
            "src.reconciliation.xbrl.index_facts",
            side_effect=RuntimeError("network down"),
        ):
            assert lookup.lookup("AAPL", "revenue") is None

    def test_ticker_cik_map_includes_all_five(self):
        """Sanity check: all 5 eval tickers are present."""
        for tk in ("AAPL", "MSFT", "GOOGL", "NVDA", "META"):
            assert tk in TICKER_TO_CIK


class TestXBRLTagMap:

    def test_revenue_synonyms(self):
        """All revenue-tag synonyms map to the same canonical field."""
        synonyms = [
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet",
        ]
        for tag in synonyms:
            assert XBRL_TAG_MAP[tag] == "revenue", \
                f"{tag} should map to 'revenue'"

    def test_cogs_synonyms(self):
        for tag in ("CostOfRevenue", "CostOfGoodsAndServicesSold",
                    "CostOfGoodsSold"):
            assert XBRL_TAG_MAP[tag] == "cogs"

    def test_net_income_synonyms(self):
        for tag in ("NetIncomeLoss", "ProfitLoss"):
            assert XBRL_TAG_MAP[tag] == "net_income"


# ---------------------------------------------------------------------------
# ReconciliationAgent tests.
# ---------------------------------------------------------------------------


class TestDeltaMath:

    def test_zero_delta(self):
        assert ReconciliationAgent._delta_pct(100, 100) == 0.0

    def test_positive_delta(self):
        assert ReconciliationAgent._delta_pct(110, 100) == 10.0

    def test_negative_delta(self):
        assert ReconciliationAgent._delta_pct(95, 100) == -5.0

    def test_xbrl_zero_prose_nonzero_is_infinity(self):
        result = ReconciliationAgent._delta_pct(50, 0)
        assert result == float("inf")

    def test_xbrl_zero_prose_zero_is_zero(self):
        assert ReconciliationAgent._delta_pct(0, 0) == 0.0


class TestClassifierOutputParsing:

    def test_clean_json(self):
        text = '{"pattern": "non_gaap_vs_gaap", "rationale": "Adjusted EBITDA"}'
        pattern, rationale = ReconciliationAgent._parse_classifier_output(text)
        assert pattern == DeltaPattern.NON_GAAP_VS_GAAP
        assert rationale == "Adjusted EBITDA"

    def test_code_fenced_json(self):
        text = '```json\n{"pattern": "periodicity", "rationale": "TTM"}\n```'
        pattern, rationale = ReconciliationAgent._parse_classifier_output(text)
        assert pattern == DeltaPattern.PERIODICITY
        assert rationale == "TTM"

    def test_unfenced_with_leading_prose(self):
        """Model adds chatter before the JSON â€” we should still extract it."""
        text = (
            'Here is the classification:\n'
            '{"pattern": "hierarchy", "rationale": "Services segment"}'
        )
        pattern, _ = ReconciliationAgent._parse_classifier_output(text)
        assert pattern == DeltaPattern.HIERARCHY

    def test_unknown_pattern_falls_to_unexplained(self):
        text = '{"pattern": "bogus_pattern", "rationale": "x"}'
        pattern, rationale = ReconciliationAgent._parse_classifier_output(text)
        assert pattern == DeltaPattern.UNEXPLAINED
        assert "bogus_pattern" in rationale

    def test_invalid_json_falls_to_unexplained(self):
        text = "not json at all"
        pattern, rationale = ReconciliationAgent._parse_classifier_output(text)
        assert pattern == DeltaPattern.UNEXPLAINED
        # Rationale should explain WHY it failed.
        assert "json" in rationale.lower() or "invalid" in rationale.lower()

    def test_rationale_truncated_to_240_chars(self):
        long_rationale = "x" * 500
        text = f'{{"pattern": "non_gaap_vs_gaap", "rationale": "{long_rationale}"}}'
        _, rationale = ReconciliationAgent._parse_classifier_output(text)
        assert len(rationale) <= 240

    def test_missing_rationale_provides_fallback(self):
        text = '{"pattern": "metric_mapping"}'
        pattern, rationale = ReconciliationAgent._parse_classifier_output(text)
        assert pattern == DeltaPattern.METRIC_MAPPING
        assert rationale  # not empty


class TestTickerInference:

    def test_infers_from_company_name(self):
        facts = [
            make_prose_fact(),  # company = Apple Inc.
            make_prose_fact(field_name="cogs", value=74525.0),
        ]
        assert ReconciliationAgent._infer_ticker(facts) == "AAPL"

    def test_infers_from_explicit_ticker_metadata(self):
        f = make_prose_fact()
        f.chunk_metadata = {"ticker": "MSFT", "filing_type": "10-Q"}
        assert ReconciliationAgent._infer_ticker([f]) == "MSFT"

    def test_returns_none_for_unknown_company(self):
        f = make_prose_fact()
        f.chunk_metadata = {"company": "Some Random Inc."}
        assert ReconciliationAgent._infer_ticker([f]) is None

    def test_returns_none_for_empty_facts(self):
        assert ReconciliationAgent._infer_ticker([]) is None


class TestReconciliationAgentNoDeltaFastPath:
    """When the prose value matches XBRL within tolerance, no LLM call is made."""

    def test_within_tolerance_no_delta(self, sample_facts_index):
        agent = ReconciliationAgent(tolerance_pct=0.5)
        agent.xbrl._index_cache["AAPL"] = sample_facts_index

        # prose = 143,750 vs XBRL = 143,756  -> -0.004% delta
        prose = make_prose_fact(value=143750.0)
        report = agent.reconcile([prose], ticker="AAPL",
                                  period_end="2025-12-27")

        assert report.total == 1
        finding = report.findings[0]
        assert finding.pattern == DeltaPattern.NO_DELTA
        assert finding.matches is True
        assert abs(finding.delta_pct) <= 0.5

    def test_outside_tolerance_triggers_classifier(self, sample_facts_index):
        """Above-tolerance delta should NOT short-circuit to NO_DELTA."""
        agent = ReconciliationAgent(tolerance_pct=0.5)
        agent.xbrl._index_cache["AAPL"] = sample_facts_index

        # Mock the classifier so we don't need OpenAI.
        with patch.object(
            ReconciliationAgent,
            "_classify_delta",
            return_value=(DeltaPattern.NON_GAAP_VS_GAAP, "mocked adj"),
        ):
            # prose = 150,000 vs XBRL = 143,756  -> ~4.3% delta
            prose = make_prose_fact(value=150000.0)
            report = agent.reconcile([prose], ticker="AAPL",
                                      period_end="2025-12-27")

        finding = report.findings[0]
        assert finding.pattern == DeltaPattern.NON_GAAP_VS_GAAP
        assert finding.rationale == "mocked adj"

    def test_lookup_failed_when_no_xbrl_fact(self, sample_facts_index):
        agent = ReconciliationAgent()
        agent.xbrl._index_cache["AAPL"] = sample_facts_index

        # Ask about a field that has no matching XBRL fact.
        prose = make_prose_fact(field_name="ebitda", value=50000.0)
        report = agent.reconcile([prose], ticker="AAPL",
                                  period_end="2025-12-27")

        finding = report.findings[0]
        assert finding.pattern == DeltaPattern.LOOKUP_FAILED
        assert finding.xbrl_value is None

    def test_unknown_ticker_yields_lookup_failed(self):
        agent = ReconciliationAgent()
        prose = make_prose_fact()
        report = agent.reconcile([prose], ticker="UNKNOWN_TICKER")
        assert report.findings[0].pattern == DeltaPattern.LOOKUP_FAILED

    def test_non_usd_facts_are_skipped(self, sample_facts_index):
        """Percentage facts shouldn't be reconciled (no XBRL anchor)."""
        agent = ReconciliationAgent()
        agent.xbrl._index_cache["AAPL"] = sample_facts_index

        prose = make_prose_fact(
            field_name="gross_margin_pct", value=37.5, unit="percent",
        )
        report = agent.reconcile([prose], ticker="AAPL")
        # The fact was skipped, so no findings.
        assert report.total == 0


class TestReconciliationAgentClassifierFailures:

    def test_classifier_api_failure_returns_unexplained(self, sample_facts_index):
        """If the LLM call raises, agent emits UNEXPLAINED, doesn't crash."""
        agent = ReconciliationAgent()
        agent.xbrl._index_cache["AAPL"] = sample_facts_index

        with patch.object(
            ReconciliationAgent,
            "_get_client",
            side_effect=RuntimeError("OPENAI_API_KEY not set"),
        ):
            prose = make_prose_fact(value=200000.0)  # well over tolerance
            report = agent.reconcile([prose], ticker="AAPL",
                                      period_end="2025-12-27")

        finding = report.findings[0]
        assert finding.pattern == DeltaPattern.UNEXPLAINED
        assert finding.human_review_recommended is True


# ---------------------------------------------------------------------------
# Bucket D scoring tests.
# ---------------------------------------------------------------------------


class TestBucketDScoring:
    """Verify the partial-credit scoring rules for Bucket D."""

    def setup_method(self):
        # Lazy import to keep the scorer optional in this test module.
        from eval.bucket_d import _bucket_d_questions
        from eval.reconciliation_scorer import score_reconciliation
        self.questions = {q.qid: q for q in _bucket_d_questions()}
        self.score = score_reconciliation

    def _report_with_pattern(self, field, pattern):
        return {
            "findings": [{
                "field_name": field,
                "pattern": pattern,
                "rationale": "test rationale",
            }],
        }

    def test_exact_match_full_credit(self):
        from eval.reconciliation_scorer import ReconciliationVerdict
        q = self.questions["D_AAPL_services_revenue"]
        report = self._report_with_pattern(q.field, "hierarchy")
        s = self.score(q, report)
        assert s.verdict == ReconciliationVerdict.PATTERN_MATCH
        assert s.credit == 1.0

    def test_legitimate_alternative_half_credit(self):
        from eval.reconciliation_scorer import ReconciliationVerdict
        # Question expects HIERARCHY but agent said NON_GAAP_VS_GAAP.
        q = self.questions["D_AAPL_services_revenue"]
        report = self._report_with_pattern(q.field, "non_gaap_vs_gaap")
        s = self.score(q, report)
        assert s.verdict == ReconciliationVerdict.PATTERN_LEGITIMATE_ALTERNATIVE
        assert s.credit == 0.5

    def test_unexplained_zero_credit(self):
        from eval.reconciliation_scorer import ReconciliationVerdict
        q = self.questions["D_AAPL_services_revenue"]
        report = self._report_with_pattern(q.field, "unexplained")
        s = self.score(q, report)
        assert s.verdict == ReconciliationVerdict.INCORRECTLY_FLAGGED_UNEXPLAINED
        assert s.credit == 0.0

    def test_no_delta_when_expected_zero_credit(self):
        from eval.reconciliation_scorer import ReconciliationVerdict
        q = self.questions["D_AAPL_services_revenue"]
        report = self._report_with_pattern(q.field, "no_delta")
        s = self.score(q, report)
        assert s.verdict == ReconciliationVerdict.NO_DELTA_WHEN_EXPECTED
        assert s.credit == 0.0

    def test_lookup_failed_zero_credit(self):
        from eval.reconciliation_scorer import ReconciliationVerdict
        q = self.questions["D_NVDA_ebitda"]
        report = self._report_with_pattern(q.field, "lookup_failed")
        s = self.score(q, report)
        assert s.verdict == ReconciliationVerdict.LOOKUP_FAILED
        assert s.credit == 0.0

    def test_no_report_zero_credit(self):
        from eval.reconciliation_scorer import ReconciliationVerdict
        q = self.questions["D_AAPL_services_revenue"]
        s = self.score(q, None)
        assert s.verdict == ReconciliationVerdict.NO_REPORT
        assert s.credit == 0.0

    def test_non_bucket_d_question(self):
        from eval.reconciliation_scorer import ReconciliationVerdict
        from eval.question_set import Question, ExpectedKind
        non_d = Question(
            qid="A_AAPL_revenue", bucket="A_clean", ticker="AAPL",
            field="revenue", text="?", expected_kind=ExpectedKind.EXACT_NUMBER,
        )
        s = self.score(non_d, {"findings": []})
        assert s.verdict == ReconciliationVerdict.NOT_BUCKET_D
        assert s.credit == 0.0

    def test_aggregate_scores(self):
        from eval.reconciliation_scorer import (
            aggregate_reconciliation_scores,
            ReconciliationScore, ReconciliationVerdict,
        )
        scores = [
            ReconciliationScore(qid="q1", expected_pattern="hierarchy",
                                actual_pattern="hierarchy",
                                verdict=ReconciliationVerdict.PATTERN_MATCH,
                                credit=1.0),
            ReconciliationScore(qid="q2", expected_pattern="hierarchy",
                                actual_pattern="non_gaap_vs_gaap",
                                verdict=ReconciliationVerdict.PATTERN_LEGITIMATE_ALTERNATIVE,
                                credit=0.5),
            ReconciliationScore(qid="q3", expected_pattern="periodicity",
                                actual_pattern="unexplained",
                                verdict=ReconciliationVerdict.INCORRECTLY_FLAGGED_UNEXPLAINED,
                                credit=0.0),
        ]
        agg = aggregate_reconciliation_scores(scores)
        assert agg["total"] == 3
        assert agg["credit_sum"] == 1.5
        assert agg["accuracy_pct"] == 50.0
        assert agg["exact_matches"] == 1
        assert agg["legitimate_alternatives"] == 1
        assert agg["unexplained"] == 1


# ---------------------------------------------------------------------------
# Bucket D question structure tests.
# ---------------------------------------------------------------------------


class TestBucketDQuestions:
    """Verify the questions themselves have the right structure."""

    def setup_method(self):
        from eval.bucket_d import _bucket_d_questions, extract_expected_pattern
        self.questions = _bucket_d_questions()
        self.extract = extract_expected_pattern

    def test_count_matches_design(self):
        # 2 non-gaap + 2 periodicity + 2 hierarchy + 2 metric + 1 versioning = 9
        assert len(self.questions) == 9

    def test_every_question_has_valid_pattern(self):
        from eval.bucket_d import LEGITIMATE_PATTERNS
        for q in self.questions:
            pattern = self.extract(q)
            assert pattern is not None, f"{q.qid} has no expected_pattern"
            assert pattern in LEGITIMATE_PATTERNS, \
                f"{q.qid} has invalid pattern {pattern}"

    def test_all_five_patterns_covered(self):
        patterns_in_set = {self.extract(q) for q in self.questions}
        expected = {"non_gaap_vs_gaap", "periodicity", "hierarchy",
                    "metric_mapping", "versioning"}
        assert patterns_in_set == expected

    def test_all_in_bucket_d(self):
        for q in self.questions:
            assert q.bucket == "D_reconciliation"

    def test_qids_are_unique(self):
        qids = [q.qid for q in self.questions]
        assert len(qids) == len(set(qids))

    def test_synthetic_question_is_labeled(self):
        synthetic = [q for q in self.questions if "SYNTHETIC" in q.qid]
        assert len(synthetic) == 1
        # Synthetic flag should also appear in notes.
        assert "SYNTHETIC" in synthetic[0].notes
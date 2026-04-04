"""
Four-Layer Evaluation Framework for RAG Systems

Production RAG systems can't rely on a single evaluation method.
This module implements a layered evaluation strategy where each
layer catches what the others miss:

Layer 1 — Retrieval Quality (no LLM needed, instant):
    Did the retriever pull relevant chunks? Checks entity coverage,
    topic overlap, and source diversity. Catches retrieval failures
    before wasting money on generation.

Layer 2 — LLM-as-Judge with Rubric (automated, scalable):
    Scores every generated answer on a structured 5-dimension rubric:
    groundedness, completeness, citation quality, financial precision,
    and coherence. No ground truth required.

Layer 3 — Pairwise Comparison (automated, more reliable):
    For each query, compares answers from two configs head-to-head.
    Produces more stable rankings than absolute scoring.

Layer 4 — Gold Set Calibration (manual, high-trust):
    A small set of 5 queries with hand-written reference answers.
    Used to calibrate and validate the automated metrics.

The BenchmarkRunner orchestrates all four layers and logs
everything to MLflow for reproducible comparison.
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from itertools import combinations

import numpy as np
import mlflow
from openai import OpenAI

from src.chunking.strategies import get_chunker
from src.retrieval.retrievers import build_retriever, RetrievalResult
from src.generation.generator import RAGGenerator, GeneratedAnswer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# Evaluation Queries
# ─────────────────────────────────────────────────────────

EVAL_QUERIES = [
    {
        "question": "What are the main risk factors Apple disclosed in their most recent 10-K?",
        "expected_entities": ["Apple", "risk", "competition", "supply chain", "regulatory"],
        "expected_sections": ["risk_factors"],
    },
    {
        "question": "How has Microsoft's revenue from cloud services changed according to their filings?",
        "expected_entities": ["Microsoft", "cloud", "Azure", "revenue"],
        "expected_sections": ["mda", "business_overview"],
    },
    {
        "question": "What supply chain risks does NVIDIA discuss in their SEC filings?",
        "expected_entities": ["NVIDIA", "supply chain", "semiconductor", "manufacturing"],
        "expected_sections": ["risk_factors"],
    },
    {
        "question": "Compare the cybersecurity risk disclosures between Meta and Alphabet.",
        "expected_entities": ["Meta", "Alphabet", "cybersecurity", "data breach", "security"],
        "expected_sections": ["risk_factors"],
    },
    {
        "question": "What regulatory risks does Alphabet face according to their 10-K filings?",
        "expected_entities": ["Alphabet", "Google", "regulatory", "antitrust", "privacy"],
        "expected_sections": ["risk_factors"],
    },
    {
        "question": "How do Apple and Microsoft compare in their discussion of competitive pressures?",
        "expected_entities": ["Apple", "Microsoft", "competition", "competitive"],
        "expected_sections": ["risk_factors", "mda"],
    },
    {
        "question": "What macroeconomic risks are common across all five companies' filings?",
        "expected_entities": ["macroeconomic", "inflation", "interest rate", "currency"],
        "expected_sections": ["risk_factors"],
    },
    {
        "question": "What litigation risks does Meta disclose in their recent filings?",
        "expected_entities": ["Meta", "litigation", "lawsuit", "legal", "proceedings"],
        "expected_sections": ["risk_factors"],
    },
    {
        "question": "How has NVIDIA discussed AI-related opportunities and risks in their filings?",
        "expected_entities": ["NVIDIA", "AI", "artificial intelligence", "data center", "GPU"],
        "expected_sections": ["risk_factors", "mda", "business_overview"],
    },
    {
        "question": "What are the key differences in how these tech companies discuss data privacy risks?",
        "expected_entities": ["privacy", "data", "GDPR", "regulation", "user"],
        "expected_sections": ["risk_factors"],
    },
]

# Layer 4: Gold set — hand-written reference answers for calibration
GOLD_SET = [
    {
        "question": "What are the main risk factors Apple disclosed in their most recent 10-K?",
        "reference_answer": (
            "Apple's 10-K identifies several key risk categories: macroeconomic conditions "
            "including inflation and currency fluctuations that affect consumer spending; "
            "intense competition in all product categories; supply chain concentration risks "
            "with key components sourced from limited suppliers; regulatory and legal risks "
            "across multiple jurisdictions including antitrust scrutiny; and cybersecurity "
            "threats to both their infrastructure and customer data."
        ),
        "key_claims": [
            "macroeconomic risks including inflation and currency",
            "competition across product categories",
            "supply chain concentration with limited suppliers",
            "regulatory and legal risks across jurisdictions",
            "cybersecurity threats to infrastructure and data",
        ],
    },
    {
        "question": "What supply chain risks does NVIDIA discuss in their SEC filings?",
        "reference_answer": (
            "NVIDIA's filings highlight dependence on third-party foundries, particularly "
            "TSMC, for manufacturing their GPUs. They identify geopolitical risks including "
            "US-China trade restrictions and export controls on advanced semiconductors. "
            "The company notes long manufacturing lead times and acknowledges that "
            "disruptions at key facilities could significantly impact production."
        ),
        "key_claims": [
            "dependence on third-party foundries for manufacturing",
            "geopolitical risks and export controls",
            "long manufacturing lead times",
            "vulnerability to disruptions at key facilities",
        ],
    },
    {
        "question": "What litigation risks does Meta disclose in their recent filings?",
        "reference_answer": (
            "Meta's 10-K discloses significant litigation exposure: ongoing privacy-related "
            "lawsuits tied to data collection practices; antitrust investigations from the "
            "FTC and state attorneys general; content moderation litigation; securities "
            "fraud class actions; and patent infringement claims."
        ),
        "key_claims": [
            "privacy-related lawsuits about data collection",
            "antitrust investigations from FTC",
            "content moderation litigation",
            "securities fraud class actions",
            "potential for significant financial penalties",
        ],
    },
    {
        "question": "How has NVIDIA discussed AI-related opportunities and risks in their filings?",
        "reference_answer": (
            "NVIDIA frames AI as both their primary growth driver and a source of risk. "
            "On the opportunity side, they highlight surging demand for GPU computing in "
            "data centers driven by generative AI. On the risk side, they note potential "
            "regulatory restrictions on AI technology, export controls, rapid technological "
            "change, and concentration of revenue among few customers."
        ),
        "key_claims": [
            "AI as primary growth driver",
            "data center GPU demand from generative AI",
            "regulatory restrictions on AI technology",
            "export controls limiting sales",
            "revenue concentration among few customers",
        ],
    },
    {
        "question": "What regulatory risks does Alphabet face according to their 10-K filings?",
        "reference_answer": (
            "Alphabet faces extensive regulatory risks: antitrust investigations from the "
            "DOJ and European Commission; data privacy regulations including GDPR; content "
            "moderation requirements; advertising regulation that could restrict their core "
            "business model; and increasing scrutiny of AI systems."
        ),
        "key_claims": [
            "antitrust from DOJ and European Commission",
            "GDPR and data privacy regulations",
            "content moderation and platform liability",
            "advertising regulation affecting core business",
            "AI regulation and scrutiny",
        ],
    },
]


# ─────────────────────────────────────────────────────────
# Layer 1: Retrieval Quality Evaluator (No LLM needed)
# ─────────────────────────────────────────────────────────

@dataclass
class RetrievalQualityScore:
    """Scores from retrieval-only evaluation."""
    entity_coverage: float
    section_accuracy: float
    source_diversity: float
    avg_relevance_score: float
    num_chunks_retrieved: int


class RetrievalQualityEvaluator:
    """
    Layer 1: Evaluate retrieval quality without any LLM calls.

    Checks whether the retriever pulled chunks that contain the
    expected entities, come from the right sections, and cover
    diverse sources. Catches retrieval failures instantly and for free.
    """

    def evaluate(
        self,
        query_info: dict,
        retrieval_results: list[RetrievalResult],
    ) -> RetrievalQualityScore:
        if not retrieval_results:
            return RetrievalQualityScore(0, 0, 0, 0, 0)

        all_text = " ".join(r.content.lower() for r in retrieval_results)

        # Entity coverage
        expected_entities = query_info.get("expected_entities", [])
        if expected_entities:
            found = sum(1 for e in expected_entities if e.lower() in all_text)
            entity_coverage = found / len(expected_entities)
        else:
            entity_coverage = 1.0

        # Section accuracy
        expected_sections = set(query_info.get("expected_sections", []))
        if expected_sections:
            retrieved_sections = set(r.metadata.get("section", "") for r in retrieval_results)
            section_accuracy = len(expected_sections & retrieved_sections) / len(expected_sections)
        else:
            section_accuracy = 1.0

        # Source diversity
        sources = set()
        for r in retrieval_results:
            sources.add((r.metadata.get("ticker", "?"), r.metadata.get("filing_date", "?")))
        source_diversity = min(len(sources) / 5.0, 1.0)

        # Average retriever confidence
        avg_score = float(np.mean([r.score for r in retrieval_results]))

        return RetrievalQualityScore(
            entity_coverage=round(entity_coverage, 3),
            section_accuracy=round(section_accuracy, 3),
            source_diversity=round(source_diversity, 3),
            avg_relevance_score=round(avg_score, 3),
            num_chunks_retrieved=len(retrieval_results),
        )


# ─────────────────────────────────────────────────────────
# Layer 2: LLM-as-Judge with Structured Rubric
# ─────────────────────────────────────────────────────────

@dataclass
class RubricScore:
    """Detailed rubric scores from LLM judge."""
    groundedness: float
    completeness: float
    citation_quality: float
    financial_precision: float
    coherence: float
    overall: float

    @staticmethod
    def from_dict(d: dict) -> "RubricScore":
        g = float(d.get("groundedness", 3))
        c = float(d.get("completeness", 3))
        cq = float(d.get("citation_quality", 3))
        fp = float(d.get("financial_precision", 3))
        co = float(d.get("coherence", 3))
        overall = 0.25 * g + 0.20 * c + 0.20 * cq + 0.20 * fp + 0.15 * co
        return RubricScore(g, c, cq, fp, co, round(overall, 2))


class RubricJudge:
    """
    Layer 2: LLM-as-Judge with a structured 5-dimension rubric.

    No ground truth needed — evaluates answer quality purely against
    the retrieved context and the question.
    """

    RUBRIC_PROMPT = """You are an expert evaluator for a financial RAG system. 
Score this answer on 5 dimensions using a 1-5 scale.

QUESTION: {question}

RETRIEVED CONTEXT:
{context}

AI ANSWER:
{answer}

SCORING RUBRIC:
1. GROUNDEDNESS (1-5): Is EVERY claim in the answer directly supported by the retrieved context? 
   5 = every claim traceable to context, 1 = significant unsupported claims
2. COMPLETENESS (1-5): Does the answer address all aspects of the question?
   5 = thorough coverage, 1 = major aspects missing
3. CITATION_QUALITY (1-5): Does the answer reference specific companies, filing types, dates, or sections?
   5 = precise citations throughout, 1 = no specific references
4. FINANCIAL_PRECISION (1-5): Does it use correct financial terminology and concepts?
   5 = precise professional language, 1 = vague or incorrect terminology
5. COHERENCE (1-5): Is the answer well-structured, clear, and easy to follow?
   5 = excellent structure, 1 = disorganized or confusing

Return ONLY a JSON object with these exact keys:
{{"groundedness": 4, "completeness": 3, "citation_quality": 4, "financial_precision": 3, "coherence": 4}}"""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model

    def score(self, question: str, answer: str, contexts: list[str]) -> RubricScore:
        context_text = "\n\n---\n\n".join(contexts[:5])
        if len(context_text) > 6000:
            context_text = context_text[:6000] + "\n[...truncated...]"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": self.RUBRIC_PROMPT.format(
                        question=question, answer=answer, context=context_text,
                    ),
                }],
                temperature=0.0,
                max_tokens=100,
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            return RubricScore.from_dict(json.loads(raw))
        except Exception as e:
            logger.warning(f"Rubric judge failed: {e}")
            return RubricScore(3, 3, 3, 3, 3, 3.0)


# ─────────────────────────────────────────────────────────
# Layer 3: Pairwise Comparison Evaluator
# ─────────────────────────────────────────────────────────

@dataclass
class PairwiseResult:
    """Result of a head-to-head comparison between two configs."""
    config_a: str
    config_b: str
    question: str
    winner: str
    reasoning: str
    confidence: float


class PairwiseEvaluator:
    """
    Layer 3: Compare answers from two configs head-to-head.

    "Which is better?" is easier for an LLM to judge than absolute
    scoring, producing more stable and reliable rankings.
    """

    COMPARISON_PROMPT = """You are comparing two answers to a financial analysis question.

QUESTION: {question}

ANSWER A ({config_a}):
{answer_a}

ANSWER B ({config_b}):
{answer_b}

Compare on: accuracy, completeness, citation quality, and usefulness.

Return ONLY a JSON object:
{{"winner": "A", "reasoning": "Answer A provides more specific citations and covers all risk categories", "confidence": 4}}

- winner: "A", "B", or "tie"
- reasoning: one sentence explaining why
- confidence: 1-5 (5 = very obvious winner, 1 = nearly identical)"""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model

    def compare(
        self, question: str, answer_a: str, answer_b: str,
        config_a: str, config_b: str,
    ) -> PairwiseResult:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": self.COMPARISON_PROMPT.format(
                        question=question, answer_a=answer_a, answer_b=answer_b,
                        config_a=config_a, config_b=config_b,
                    ),
                }],
                temperature=0.0,
                max_tokens=150,
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(raw)
            return PairwiseResult(
                config_a=config_a, config_b=config_b, question=question,
                winner=data.get("winner", "tie"),
                reasoning=data.get("reasoning", ""),
                confidence=float(data.get("confidence", 3)),
            )
        except Exception as e:
            logger.warning(f"Pairwise comparison failed: {e}")
            return PairwiseResult(config_a, config_b, question, "tie", str(e), 1.0)


# ─────────────────────────────────────────────────────────
# Layer 4: Gold Set Calibration Evaluator
# ─────────────────────────────────────────────────────────

@dataclass
class GoldSetScore:
    """Score against a hand-written reference answer."""
    question: str
    claims_covered: int
    claims_total: int
    claim_coverage: float
    factual_errors: int
    semantic_similarity: float


class GoldSetEvaluator:
    """
    Layer 4: Evaluate against hand-written reference answers.

    Uses a small gold set (5 queries) with human-written reference
    answers and key claims. This is the calibration anchor.
    """

    CLAIM_CHECK_PROMPT = """Given a reference answer and an AI-generated answer to a financial question, check which key claims from the reference are covered.

QUESTION: {question}

REFERENCE ANSWER:
{reference}

AI ANSWER:
{answer}

KEY CLAIMS TO CHECK:
{claims_list}

For each claim: "COVERED" if addressed (even if worded differently), "MISSING" if not, "WRONG" if contradicted.

Return ONLY a JSON object:
{{"results": ["COVERED", "MISSING", "COVERED", "WRONG", "COVERED"]}}"""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model

    def evaluate(self, question: str, answer: str, gold_entry: dict) -> GoldSetScore:
        key_claims = gold_entry.get("key_claims", [])
        reference = gold_entry.get("reference_answer", "")

        if not key_claims:
            return GoldSetScore(question, 0, 0, 0.0, 0, 0.0)

        claims_list = "\n".join(f"{i+1}. {c}" for i, c in enumerate(key_claims))

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": self.CLAIM_CHECK_PROMPT.format(
                        question=question, reference=reference,
                        answer=answer, claims_list=claims_list,
                    ),
                }],
                temperature=0.0,
                max_tokens=200,
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(raw)
            results = data.get("results", [])

            covered = sum(1 for r in results if r == "COVERED")
            wrong = sum(1 for r in results if r == "WRONG")
            coverage = covered / len(key_claims) if key_claims else 0.0

            # Word overlap as semantic similarity proxy
            ref_words = set(reference.lower().split())
            ans_words = set(answer.lower().split())
            similarity = len(ref_words & ans_words) / len(ref_words) if ref_words else 0.0

            return GoldSetScore(
                question=question, claims_covered=covered,
                claims_total=len(key_claims), claim_coverage=round(coverage, 3),
                factual_errors=wrong, semantic_similarity=round(similarity, 3),
            )
        except Exception as e:
            logger.warning(f"Gold set evaluation failed: {e}")
            return GoldSetScore(question, 0, len(key_claims), 0.0, 0, 0.0)


# ─────────────────────────────────────────────────────────
# Unified Evaluation Result
# ─────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    """Complete evaluation result for a single query across all layers."""
    question: str
    answer: str
    contexts: list[str]
    # Layer 1
    retrieval_entity_coverage: float = 0.0
    retrieval_section_accuracy: float = 0.0
    retrieval_source_diversity: float = 0.0
    # Layer 2
    rubric_groundedness: float = 0.0
    rubric_completeness: float = 0.0
    rubric_citation_quality: float = 0.0
    rubric_financial_precision: float = 0.0
    rubric_coherence: float = 0.0
    rubric_overall: float = 0.0
    # Layer 4
    gold_claim_coverage: float | None = None
    gold_factual_errors: int | None = None
    # Timing
    latency_seconds: float = 0.0


@dataclass
class BenchmarkResult:
    """Aggregated results for a single (chunking x retrieval) configuration."""
    chunking_strategy: str
    retrieval_strategy: str
    num_queries: int
    num_chunks: int
    # Layer 1
    avg_entity_coverage: float
    avg_section_accuracy: float
    avg_source_diversity: float
    # Layer 2
    avg_rubric_groundedness: float
    avg_rubric_completeness: float
    avg_rubric_citation_quality: float
    avg_rubric_financial_precision: float
    avg_rubric_coherence: float
    avg_rubric_overall: float
    # Layer 4
    avg_gold_claim_coverage: float
    avg_gold_factual_errors: float
    # Performance
    avg_latency_seconds: float
    individual_results: list[EvalResult] = field(default_factory=list)

    @property
    def composite_score(self) -> float:
        """Weighted composite: L1 25%, L2 50%, L4 25%."""
        retrieval = (
            0.5 * self.avg_entity_coverage
            + 0.3 * self.avg_section_accuracy
            + 0.2 * self.avg_source_diversity
        )
        rubric = self.avg_rubric_overall / 5.0
        gold = self.avg_gold_claim_coverage
        return round(0.25 * retrieval + 0.50 * rubric + 0.25 * gold, 3)


# ─────────────────────────────────────────────────────────
# Benchmark Runner
# ─────────────────────────────────────────────────────────

class BenchmarkRunner:
    """Orchestrates the full four-layer evaluation benchmark."""

    def __init__(
        self,
        sections: list[dict],
        chunking_configs: dict,
        retrieval_configs: dict,
        generation_config: dict,
        tracking_config: dict,
    ):
        self.sections = sections
        self.chunking_configs = chunking_configs
        self.retrieval_configs = retrieval_configs
        self.generator = RAGGenerator(
            model=generation_config.get("model", "gpt-4o-mini"),
            temperature=generation_config.get("temperature", 0.1),
        )

        # All four layers
        self.retrieval_evaluator = RetrievalQualityEvaluator()
        self.rubric_judge = RubricJudge()
        self.pairwise_evaluator = PairwiseEvaluator()
        self.gold_evaluator = GoldSetEvaluator()

        self.gold_lookup = {g["question"]: g for g in GOLD_SET}

        # MLflow
        mlflow.set_tracking_uri(tracking_config.get("tracking_uri", "mlruns"))
        mlflow.set_experiment(
            tracking_config.get("experiment_name", "earnings-intelligence-rag")
        )

    def run_single_config(
        self,
        chunking_strategy: str,
        retrieval_strategy: str,
        eval_queries: list[dict] | None = None,
    ) -> tuple:
        """Run all four evaluation layers for a single configuration."""
        if eval_queries is None:
            eval_queries = EVAL_QUERIES

        config_name = f"{chunking_strategy} × {retrieval_strategy}"
        logger.info(f"Running benchmark: {config_name}")

        # Chunk
        chunk_config = self.chunking_configs.get(chunking_strategy, {})
        chunker = get_chunker(chunking_strategy, chunk_config)
        documents = chunker.chunk_sections(self.sections)

        # Index
        retrieval_config = self.retrieval_configs.get(retrieval_strategy, {})
        retrieval_config["collection_suffix"] = f"{chunking_strategy}_{retrieval_strategy}"
        retriever = build_retriever(retrieval_strategy, retrieval_config)
        retriever.index(documents)

        # Evaluate each query
        individual_results = []
        answers_by_question = {}

        for eq in eval_queries:
            start_time = time.time()

            # Retrieve
            retrieval_results = retriever.retrieve(eq["question"], top_k=10)

            # Generate
            gen_answer = self.generator.generate(eq["question"], retrieval_results)
            latency = time.time() - start_time

            # Layer 1: Retrieval quality (free, instant)
            ret_score = self.retrieval_evaluator.evaluate(eq, retrieval_results)

            # Layer 2: Rubric judge
            rubric = self.rubric_judge.score(
                question=eq["question"],
                answer=gen_answer.answer,
                contexts=gen_answer.contexts,
            )

            # Layer 4: Gold set (if available for this query)
            gold_coverage = None
            gold_errors = None
            gold_entry = self.gold_lookup.get(eq["question"])
            if gold_entry:
                gold = self.gold_evaluator.evaluate(
                    question=eq["question"],
                    answer=gen_answer.answer,
                    gold_entry=gold_entry,
                )
                gold_coverage = gold.claim_coverage
                gold_errors = gold.factual_errors

            individual_results.append(EvalResult(
                question=eq["question"],
                answer=gen_answer.answer,
                contexts=gen_answer.contexts[:3],
                retrieval_entity_coverage=ret_score.entity_coverage,
                retrieval_section_accuracy=ret_score.section_accuracy,
                retrieval_source_diversity=ret_score.source_diversity,
                rubric_groundedness=rubric.groundedness,
                rubric_completeness=rubric.completeness,
                rubric_citation_quality=rubric.citation_quality,
                rubric_financial_precision=rubric.financial_precision,
                rubric_coherence=rubric.coherence,
                rubric_overall=rubric.overall,
                gold_claim_coverage=gold_coverage,
                gold_factual_errors=gold_errors,
                latency_seconds=latency,
            ))
            answers_by_question[eq["question"]] = gen_answer.answer

        # Aggregate
        gold_results = [r for r in individual_results if r.gold_claim_coverage is not None]

        benchmark = BenchmarkResult(
            chunking_strategy=chunking_strategy,
            retrieval_strategy=retrieval_strategy,
            num_queries=len(eval_queries),
            num_chunks=len(documents),
            avg_entity_coverage=float(np.mean([r.retrieval_entity_coverage for r in individual_results])),
            avg_section_accuracy=float(np.mean([r.retrieval_section_accuracy for r in individual_results])),
            avg_source_diversity=float(np.mean([r.retrieval_source_diversity for r in individual_results])),
            avg_rubric_groundedness=float(np.mean([r.rubric_groundedness for r in individual_results])),
            avg_rubric_completeness=float(np.mean([r.rubric_completeness for r in individual_results])),
            avg_rubric_citation_quality=float(np.mean([r.rubric_citation_quality for r in individual_results])),
            avg_rubric_financial_precision=float(np.mean([r.rubric_financial_precision for r in individual_results])),
            avg_rubric_coherence=float(np.mean([r.rubric_coherence for r in individual_results])),
            avg_rubric_overall=float(np.mean([r.rubric_overall for r in individual_results])),
            avg_gold_claim_coverage=float(np.mean([r.gold_claim_coverage for r in gold_results])) if gold_results else 0.0,
            avg_gold_factual_errors=float(np.mean([r.gold_factual_errors for r in gold_results])) if gold_results else 0.0,
            avg_latency_seconds=float(np.mean([r.latency_seconds for r in individual_results])),
            individual_results=individual_results,
        )

        self._log_to_mlflow(benchmark)
        return benchmark, answers_by_question

    def run_all(self, eval_queries: list[dict] | None = None) -> list[BenchmarkResult]:
        """Run benchmark for all configs, then pairwise comparisons on top configs."""
        results = []
        all_answers = {}

        # Phase 1: All configs
        for chunk_strategy in self.chunking_configs:
            for retrieval_strategy in self.retrieval_configs:
                try:
                    benchmark, answers = self.run_single_config(
                        chunk_strategy, retrieval_strategy, eval_queries
                    )
                    results.append(benchmark)
                    config_name = f"{chunk_strategy} × {retrieval_strategy}"
                    all_answers[config_name] = answers
                except Exception as e:
                    logger.error(f"Failed {chunk_strategy} × {retrieval_strategy}: {e}")

        # Phase 2: Pairwise on top 4
        if len(results) >= 2:
            top_configs = sorted(results, key=lambda x: x.avg_rubric_overall, reverse=True)[:4]
            pairwise_results = self._run_pairwise(top_configs, all_answers, eval_queries)
            self._print_pairwise_summary(pairwise_results)

            # Save pairwise results
            pairwise_path = Path("data/processed/pairwise_results.json")
            pairwise_path.parent.mkdir(parents=True, exist_ok=True)
            with open(pairwise_path, "w") as f:
                json.dump([asdict(r) for r in pairwise_results], f, indent=2)

        self._print_summary(results)
        return results

    def _run_pairwise(
        self, top_configs: list[BenchmarkResult],
        all_answers: dict, eval_queries: list[dict] | None = None,
    ) -> list[PairwiseResult]:
        """Layer 3: Pairwise comparisons between top configs."""
        if eval_queries is None:
            eval_queries = EVAL_QUERIES

        logger.info(f"Running pairwise comparisons on top {len(top_configs)} configs...")
        pairwise_results = []

        config_names = [
            f"{c.chunking_strategy} × {c.retrieval_strategy}" for c in top_configs
        ]

        for config_a, config_b in combinations(config_names, 2):
            for eq in eval_queries[:5]:  # First 5 to save cost
                answer_a = all_answers.get(config_a, {}).get(eq["question"], "")
                answer_b = all_answers.get(config_b, {}).get(eq["question"], "")

                if answer_a and answer_b:
                    result = self.pairwise_evaluator.compare(
                        question=eq["question"],
                        answer_a=answer_a, answer_b=answer_b,
                        config_a=config_a, config_b=config_b,
                    )
                    pairwise_results.append(result)

        return pairwise_results

    def _log_to_mlflow(self, benchmark: BenchmarkResult):
        """Log to MLflow with all four layers."""
        with mlflow.start_run(
            run_name=f"{benchmark.chunking_strategy}_{benchmark.retrieval_strategy}",
        ):
            mlflow.log_param("chunking_strategy", benchmark.chunking_strategy)
            mlflow.log_param("retrieval_strategy", benchmark.retrieval_strategy)
            mlflow.log_param("num_chunks", benchmark.num_chunks)
            mlflow.log_param("num_eval_queries", benchmark.num_queries)

            # Layer 1
            mlflow.log_metric("L1_entity_coverage", benchmark.avg_entity_coverage)
            mlflow.log_metric("L1_section_accuracy", benchmark.avg_section_accuracy)
            mlflow.log_metric("L1_source_diversity", benchmark.avg_source_diversity)

            # Layer 2
            mlflow.log_metric("L2_groundedness", benchmark.avg_rubric_groundedness)
            mlflow.log_metric("L2_completeness", benchmark.avg_rubric_completeness)
            mlflow.log_metric("L2_citation_quality", benchmark.avg_rubric_citation_quality)
            mlflow.log_metric("L2_financial_precision", benchmark.avg_rubric_financial_precision)
            mlflow.log_metric("L2_coherence", benchmark.avg_rubric_coherence)
            mlflow.log_metric("L2_rubric_overall", benchmark.avg_rubric_overall)

            # Layer 4
            mlflow.log_metric("L4_gold_claim_coverage", benchmark.avg_gold_claim_coverage)
            mlflow.log_metric("L4_gold_factual_errors", benchmark.avg_gold_factual_errors)

            # Composite
            mlflow.log_metric("composite_score", benchmark.composite_score)
            mlflow.log_metric("latency_seconds", benchmark.avg_latency_seconds)

            # Artifact
            results_json = json.dumps(
                [asdict(r) for r in benchmark.individual_results], indent=2, default=str,
            )
            artifact_path = Path("mlflow_artifacts")
            artifact_path.mkdir(exist_ok=True)
            result_file = artifact_path / f"{benchmark.chunking_strategy}_{benchmark.retrieval_strategy}.json"
            result_file.write_text(results_json)
            mlflow.log_artifact(str(result_file))

        logger.info(
            f"Logged: {benchmark.chunking_strategy} × {benchmark.retrieval_strategy} | "
            f"composite={benchmark.composite_score:.3f}"
        )

    @staticmethod
    def _print_summary(results: list[BenchmarkResult]):
        """Print the full four-layer benchmark summary."""
        print("\n" + "=" * 130)
        print("FOUR-LAYER BENCHMARK RESULTS")
        print("=" * 130)

        print(
            f"{'Config':<32} │ "
            f"{'EntCov':>6} {'SecAcc':>6} │ "
            f"{'Grnd':>4} {'Comp':>4} {'Cite':>4} {'FinP':>4} {'Rubr':>5} │ "
            f"{'GoldCov':>7} {'Errs':>4} │ "
            f"{'Score':>6} {'Lat':>5}"
        )
        print("─" * 130)

        for r in sorted(results, key=lambda x: x.composite_score, reverse=True):
            config = f"{r.chunking_strategy} × {r.retrieval_strategy}"
            print(
                f"{config:<32} │ "
                f"{r.avg_entity_coverage:>6.2f} {r.avg_section_accuracy:>6.2f} │ "
                f"{r.avg_rubric_groundedness:>4.1f} {r.avg_rubric_completeness:>4.1f} "
                f"{r.avg_rubric_citation_quality:>4.1f} {r.avg_rubric_financial_precision:>4.1f} "
                f"{r.avg_rubric_overall:>5.2f} │ "
                f"{r.avg_gold_claim_coverage:>7.2f} {r.avg_gold_factual_errors:>4.1f} │ "
                f"{r.composite_score:>6.3f} {r.avg_latency_seconds:>4.1f}s"
            )

        print("=" * 130)
        print("L1: EntCov=entity coverage, SecAcc=section accuracy (no LLM, free)")
        print("L2: Grnd=groundedness, Comp=completeness, Cite=citations, FinP=financial precision, Rubr=overall (1-5)")
        print("L4: GoldCov=claim coverage vs reference, Errs=factual errors")
        print("Score = weighted composite (L1: 25%, L2: 50%, L4: 25%)")

    @staticmethod
    def _print_pairwise_summary(results: list[PairwiseResult]):
        """Print pairwise comparison win rates."""
        print("\n" + "=" * 80)
        print("PAIRWISE COMPARISONS (Layer 3)")
        print("=" * 80)

        wins: dict[str, float] = {}
        total: dict[str, int] = {}

        for r in results:
            for config in [r.config_a, r.config_b]:
                wins.setdefault(config, 0)
                total.setdefault(config, 0)

            total[r.config_a] += 1
            total[r.config_b] += 1

            if r.winner == "A":
                wins[r.config_a] += 1
            elif r.winner == "B":
                wins[r.config_b] += 1
            else:
                wins[r.config_a] += 0.5
                wins[r.config_b] += 0.5

        print(f"{'Config':<35} {'Wins':>6} {'Total':>6} {'Win Rate':>9}")
        print("─" * 60)
        for config in sorted(wins, key=lambda c: wins[c] / max(total[c], 1), reverse=True):
            rate = wins[config] / max(total[config], 1)
            print(f"{config:<35} {wins[config]:>6.1f} {total[config]:>6} {rate:>8.1%}")

        print("=" * 80)
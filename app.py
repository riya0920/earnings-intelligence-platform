"""
Earnings Intelligence Platform — Streamlit Demo

Run with: streamlit run app.py
"""

import json
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Earnings Intelligence Platform",
    page_icon="📊",
    layout="wide",
)


def load_sections():
    """Load all ingested filing sections."""
    raw_dir = Path("data/raw")
    sections = []
    for filepath in raw_dir.glob("*_filings.json"):
        with open(filepath) as f:
            filings = json.load(f)
            for filing in filings:
                sections.extend(filing.get("sections", []))
    return sections


def load_benchmark_results():
    """Load benchmark results if available."""
    path = Path("data/processed/benchmark_results.json")
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def get_available_tickers(sections):
    """Get unique tickers from sections."""
    return sorted(set(s.get("ticker", "") for s in sections if s.get("ticker")))


# ─── Sidebar ───
st.sidebar.title("📊 Earnings Intelligence")
st.sidebar.markdown("Self-evaluating RAG over SEC 10-K filings")

sections = load_sections()
if not sections:
    st.error("No ingested data found. Run `python -m src.main ingest` first.")
    st.stop()

tickers = get_available_tickers(sections)
st.sidebar.success(f"Loaded {len(sections)} sections from {len(tickers)} companies")
st.sidebar.markdown(f"**Companies:** {', '.join(tickers)}")

page = st.sidebar.radio(
    "Navigate",
    ["Query", "Verified Query", "Risk Comparison", "Temporal Analysis", "Benchmark Results"],
)


# ─── Page: Query ───
if page == "Query":
    st.title("Ask a Question")
    st.markdown(
        "Query SEC 10-K filings using the best RAG configuration (semantic chunking + hybrid retrieval)."
    )

    query = st.text_input(
        "Your question:",
        placeholder="What are the main risk factors Apple disclosed in their most recent 10-K?",
    )

    if st.button("Search", type="primary") and query:
        with st.spinner("Chunking → Retrieving → Generating..."):
            try:
                import yaml
                from dotenv import load_dotenv

                load_dotenv()

                with open("configs/default.yaml") as f:
                    config = yaml.safe_load(f)

                from src.chunking.strategies import get_chunker
                from src.retrieval.retrievers import build_retriever
                from src.generation.generator import RAGGenerator

                chunker = get_chunker(
                    "semantic", config["chunking"]["strategies"]["semantic"]
                )
                documents = chunker.chunk_sections(sections)

                ret_config = config["retrieval"]["strategies"]["hybrid"]
                ret_config["collection_suffix"] = "streamlit_query"
                ret_config["embedding_model"] = config["retrieval"]["embedding_model"]
                ret_config["vectorstore_path"] = config["retrieval"]["vectorstore_path"]
                retriever = build_retriever("hybrid", ret_config)
                retriever.index(documents)

                results = retriever.retrieve(query, top_k=10)

                generator = RAGGenerator(
                    model=config["generation"]["model"],
                    temperature=config["generation"]["temperature"],
                )
                answer = generator.generate(query, results)

                st.markdown("### Answer")
                st.markdown(answer.answer)

                st.markdown("---")
                with st.expander(f"Retrieved {len(answer.contexts)} source chunks"):
                    for i, (ctx, meta) in enumerate(
                        zip(answer.contexts, answer.context_metadata)
                    ):
                        st.markdown(
                            f"**Source {i+1}:** {meta.get('company', '?')} | "
                            f"{meta.get('filing_type', '?')} | {meta.get('filing_date', '?')} | "
                            f"Section: {meta.get('section', '?')}"
                        )
                        st.text(ctx[:500] + "..." if len(ctx) > 500 else ctx)
                        st.markdown("---")

                st.caption(f"Tokens used: {answer.usage.get('total_tokens', 'N/A')}")

            except Exception as e:
                st.error(f"Error: {e}")


# ─── Page: Verified Query ───
elif page == "Verified Query":
    st.title("Verified Query")
    st.markdown(
        "**Quantitative questions with adversarial verification.** "
        "RAG retrieves the relevant chunks; a multi-agent auditor (Hunter + "
        "Forensic Auditor + Arbiter) extracts structured numbers and verifies "
        "every value against the source via three layers of defense — "
        "heterogeneous models, deterministic provenance, and consistency checks."
    )

    query = st.text_input(
        "Your question:",
        placeholder="What was Apple's revenue in their most recent 10-K?",
        key="verify_query",
    )

    if st.button("Verify", type="primary", key="verify_btn") and query:
        with st.spinner("Retrieving → Hunter || Auditor → verifier → arbiter..."):
            try:
                import yaml
                from dotenv import load_dotenv

                load_dotenv()

                with open("configs/default.yaml") as f:
                    config = yaml.safe_load(f)

                from src.chunking.strategies import get_chunker
                from src.retrieval.retrievers import build_retriever
                from src.generation.verified_generator import VerifiedRAGGenerator

                chunker = get_chunker(
                    "semantic", config["chunking"]["strategies"]["semantic"]
                )
                documents = chunker.chunk_sections(sections)

                ret_config = config["retrieval"]["strategies"]["hybrid_reranked"]
                ret_config["collection_suffix"] = "streamlit_verify"
                ret_config["embedding_model"] = config["retrieval"]["embedding_model"]
                ret_config["vectorstore_path"] = config["retrieval"]["vectorstore_path"]
                retriever = build_retriever("hybrid_reranked", ret_config)
                retriever.index(documents)

                results = retriever.retrieve(query, top_k=10)

                generator = VerifiedRAGGenerator(
                    prose_model=config["generation"]["model"],
                    prose_temperature=config["generation"]["temperature"],
                )
                answer = generator.generate(query, results)
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

            # ─── Verification badge ───
            v = answer.verification
            badge_text = {
                "verified": "✅ VERIFIED",
                "verified_with_warnings": "⚠️ VERIFIED WITH WARNINGS",
                "disputed": "❌ DISPUTED — values could not be reconciled",
            }.get(v.status, "⚠️ UNKNOWN")
            badge_color = {
                "verified": "#0f7b3d",
                "verified_with_warnings": "#b07b00",
                "disputed": "#a8201a",
            }.get(v.status, "#555555")

            st.markdown(
                f"<div style='padding:0.75rem 1rem;border-radius:6px;"
                f"background:{badge_color};color:white;font-weight:600;"
                f"font-size:1rem;margin-bottom:1rem;'>{badge_text}</div>",
                unsafe_allow_html=True,
            )

            cols = st.columns(4)
            cols[0].metric("Consensus", "yes" if v.consensus_met else "no")
            cols[1].metric("Iterations", f"{v.iterations}/3")
            cols[2].metric(
                "Provenance",
                "all pass" if v.provenance_all_passed else f"fail ({v.provenance_failed_count})",
            )
            cols[3].metric("Anomalies", v.consistency_anomaly_count)

            # ─── Prose answer ───
            st.markdown("### Prose Answer")
            st.markdown(answer.answer)

            # ─── Verified structured facts ───
            if answer.has_quantitative_facts:
                st.markdown("### Verified Structured Facts")
                rows = []
                for f in answer.structured_facts:
                    cm = f.chunk_metadata or {}
                    rows.append({
                        "Field": f.field_name,
                        "Value": f.value,
                        "Unit": f.unit,
                        "Verified": "✓" if f.provenance_verified else "✗",
                        "Actual?": "yes" if f.is_actual else "no (forecast/guidance)",
                        "Source": (
                            f"{cm.get('company','?')} | {cm.get('filing_type','?')} | "
                            f"{cm.get('filing_date','?')} | {cm.get('section','?')}"
                        ),
                        "Chunk": f.chunk_index + 1 if f.chunk_index >= 0 else None,
                    })
                st.dataframe(rows, use_container_width=True, hide_index=True)

                with st.expander("Source quotes for each fact"):
                    for f in answer.structured_facts:
                        verified_mark = "✓" if f.provenance_verified else "✗"
                        st.markdown(
                            f"**{verified_mark} {f.field_name}** = "
                            f"{f.value} {f.unit}"
                        )
                        if f.source_quote:
                            st.markdown(f"> {f.source_quote}")
                        st.markdown("---")
            else:
                st.info(
                    "No quantitative facts were extracted. Either this is a "
                    "qualitative question, or the retrieved chunks contained "
                    "no numeric values that the auditor could verify. The "
                    "prose answer above is the response — it just hasn't been "
                    "verified by the structured layer."
                )

            # ─── Anomalies ───
            if answer.consistency_anomalies:
                st.markdown("### Consistency Anomalies")
                st.warning(
                    "The auditor's deterministic checks found mathematical "
                    "inconsistencies in the extracted values. Each anomaly "
                    "below indicates at least one extracted primary "
                    "(revenue / cogs / gross_profit / etc.) is wrong."
                )
                for a in answer.consistency_anomalies:
                    st.markdown(
                        f"**[{a.get('check','?')}]** {a.get('explanation','')}"
                    )

            # ─── Retrieved chunks ───
            with st.expander(f"Retrieved {len(answer.contexts)} source chunks"):
                for i, (ctx, meta) in enumerate(
                    zip(answer.contexts, answer.context_metadata), start=1
                ):
                    st.markdown(
                        f"**Chunk {i}:** {meta.get('company','?')} | "
                        f"{meta.get('filing_type','?')} | "
                        f"{meta.get('filing_date','?')} | "
                        f"Section: {meta.get('section','?')}"
                    )
                    st.text(ctx[:500] + "..." if len(ctx) > 500 else ctx)
                    st.markdown("---")

            # ─── Audit log tail ───
            with st.expander("Audit log (graph trace)"):
                for line in v.audit_log_tail:
                    st.text(line)
                if v.rationale:
                    st.markdown(f"**Arbiter rationale:** {v.rationale}")

            st.caption(f"Prose tokens: {answer.usage.get('total_tokens', 'N/A')}")


# ─── Page: Risk Comparison ───
elif page == "Risk Comparison":
    st.title("Cross-Company Risk Comparison")
    st.markdown("Compare risk disclosures between two companies.")

    col1, col2 = st.columns(2)
    with col1:
        ticker_a = st.selectbox("Company A", tickers, index=0)
    with col2:
        ticker_b = st.selectbox("Company B", tickers, index=min(1, len(tickers) - 1))

    if st.button("Compare Risks", type="primary") and ticker_a != ticker_b:
        with st.spinner(f"Analyzing {ticker_a} vs {ticker_b}..."):
            try:
                from dotenv import load_dotenv

                load_dotenv()
                from src.analysis import run_cross_company_comparison

                result = run_cross_company_comparison(sections, ticker_a, ticker_b)

                st.markdown(f"### {result.company_a} vs {result.company_b}")

                if result.shared_risks:
                    st.markdown("#### Shared Risks")
                    for r in result.shared_risks:
                        with st.expander(f"🔄 {r.get('category', 'Unknown').title()}"):
                            st.markdown(
                                f"**{ticker_a}:** {r.get('company_a_framing', 'N/A')}"
                            )
                            st.markdown(
                                f"**{ticker_b}:** {r.get('company_b_framing', 'N/A')}"
                            )

                col1, col2 = st.columns(2)
                with col1:
                    if result.unique_to_a:
                        st.markdown(f"#### Unique to {ticker_a}")
                        for r in result.unique_to_a:
                            st.markdown(
                                f"- **{r.get('category', '?').title()}**: {r.get('summary', '')}"
                            )
                with col2:
                    if result.unique_to_b:
                        st.markdown(f"#### Unique to {ticker_b}")
                        for r in result.unique_to_b:
                            st.markdown(
                                f"- **{r.get('category', '?').title()}**: {r.get('summary', '')}"
                            )

                if result.analysis:
                    st.markdown("#### Key Insight")
                    st.info(result.analysis)

            except Exception as e:
                st.error(f"Error: {e}")

    elif ticker_a == ticker_b:
        st.warning("Select two different companies to compare.")


# ─── Page: Temporal Analysis ───
elif page == "Temporal Analysis":
    st.title("Risk Evolution Over Time")
    st.markdown("Track how a company's risk disclosures changed across filing years.")

    ticker = st.selectbox("Select Company", tickers)

    if st.button("Analyze Evolution", type="primary"):
        with st.spinner(f"Analyzing {ticker} risk evolution..."):
            try:
                from dotenv import load_dotenv

                load_dotenv()
                from src.analysis import run_temporal_analysis

                changes = run_temporal_analysis(sections, ticker)

                if not changes:
                    st.warning(f"Need at least 2 risk factor filings for {ticker}.")
                else:
                    for change in changes:
                        st.markdown(f"### {change.earlier_date} → {change.later_date}")

                        col1, col2 = st.columns(2)
                        with col1:
                            if change.new_risks:
                                st.markdown("**🆕 New Risks**")
                                for r in change.new_risks:
                                    st.markdown(
                                        f"- **{r.get('category', '?').title()}**: {r.get('summary', '')}"
                                    )

                            if change.escalated_risks:
                                st.markdown("**⬆️ Escalated**")
                                for r in change.escalated_risks:
                                    st.markdown(
                                        f"- **{r.get('category', '?').title()}**: {r.get('summary', '')} "
                                        f"({r.get('from_severity', '?')} → {r.get('to_severity', '?')})"
                                    )

                        with col2:
                            if change.removed_risks:
                                st.markdown("**🗑️ Removed Risks**")
                                for r in change.removed_risks:
                                    st.markdown(
                                        f"- **{r.get('category', '?').title()}**: {r.get('summary', '')}"
                                    )

                            if change.de_escalated_risks:
                                st.markdown("**⬇️ De-escalated**")
                                for r in change.de_escalated_risks:
                                    st.markdown(
                                        f"- **{r.get('category', '?').title()}**: {r.get('summary', '')} "
                                        f"({r.get('from_severity', '?')} → {r.get('to_severity', '?')})"
                                    )

                        if change.analysis:
                            st.info(f"**Insight:** {change.analysis}")
                        st.markdown("---")

            except Exception as e:
                st.error(f"Error: {e}")


# ─── Page: Benchmark Results ───
elif page == "Benchmark Results":
    st.title("Benchmark Results")
    st.markdown("Four-layer evaluation across 12 RAG configurations.")

    results = load_benchmark_results()
    if not results:
        st.warning(
            "No benchmark results found. Run `python -m src.main benchmark` first."
        )
    else:
        import pandas as pd

        df = pd.DataFrame(results)
        df = df.sort_values("composite_score", ascending=False)

        st.markdown("### Composite Scores by Configuration")
        st.bar_chart(
            df.set_index("config")["composite_score"],
            height=400,
        )

        st.markdown("### Full Results Table")
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("### Layer Breakdown")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Layer 1: Retrieval Quality**")
            if "L1_entity_coverage" in df.columns:
                st.bar_chart(
                    df.set_index("config")[
                        ["L1_entity_coverage", "L1_section_accuracy"]
                    ],
                    height=300,
                )

        with col2:
            st.markdown("**Layer 2: Rubric Judge**")
            if "L2_rubric_overall" in df.columns:
                st.bar_chart(
                    df.set_index("config")["L2_rubric_overall"],
                    height=300,
                )

        with col3:
            st.markdown("**Layer 4: Gold Set**")
            if "L4_gold_claim_coverage" in df.columns:
                st.bar_chart(
                    df.set_index("config")["L4_gold_claim_coverage"],
                    height=300,
                )

        # Load pairwise results if available
        pairwise_path = Path("data/processed/pairwise_results.json")
        if pairwise_path.exists():
            st.markdown("### Layer 3: Pairwise Comparisons")
            with open(pairwise_path) as f:
                pairwise = json.load(f)

            wins = {}
            total = {}
            for r in pairwise:
                for config in [r["config_a"], r["config_b"]]:
                    wins.setdefault(config, 0)
                    total.setdefault(config, 0)
                total[r["config_a"]] += 1
                total[r["config_b"]] += 1
                if r["winner"] == "A":
                    wins[r["config_a"]] += 1
                elif r["winner"] == "B":
                    wins[r["config_b"]] += 1
                else:
                    wins[r["config_a"]] += 0.5
                    wins[r["config_b"]] += 0.5

            pairwise_df = pd.DataFrame(
                [
                    {"Config": k, "Win Rate": wins[k] / max(total[k], 1)}
                    for k in sorted(
                        wins, key=lambda c: wins[c] / max(total[c], 1), reverse=True
                    )
                ]
            )
            st.bar_chart(pairwise_df.set_index("Config")["Win Rate"], height=300)

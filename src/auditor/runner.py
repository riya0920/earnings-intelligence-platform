"""
src/auditor/runner.py
=====================

Vendored from the standalone `adversarial_auditor` repo. Provides
`run_audit(document) -> final_state` for use by EIP's verified generator.

The CLI entry point and pretty-printer have been stripped. For the
standalone version see github.com/riya0920/adversarial-auditor.

Graph topology (unchanged from the standalone version):

                         START
                  +--------+--------+
                  |                 |
                  v                 v
               hunter            auditor      (parallel — neither sees the other)
                  +--------+--------+
                           v
                       verifier   (Layer 2: deterministic provenance check)
                  +--------+--------+
              all passed       any failed
                  v                 v
              arbiter          dispute
                  v                 v
              consensus        re-extract
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from .nodes import (
    arbiter_node,
    auditor_node,
    dispute_node,
    error_node,
    hunter_node,
    provenance_verifier_node,
    route_after_arbiter,
    route_after_verifier,
)
from .state import AgentState


def build_graph():
    """Assemble the cyclic LangGraph."""
    workflow = StateGraph(AgentState)

    workflow.add_node("hunter", hunter_node)
    workflow.add_node("auditor", auditor_node)
    workflow.add_node("verifier", provenance_verifier_node)
    workflow.add_node("arbiter", arbiter_node)
    workflow.add_node("dispute", dispute_node)
    workflow.add_node("error", error_node)

    workflow.add_edge(START, "hunter")
    workflow.add_edge(START, "auditor")
    workflow.add_edge("hunter", "verifier")
    workflow.add_edge("auditor", "verifier")
    workflow.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {"arbiter": "arbiter", "dispute": "dispute",
         "max_iterations": END, "error": "error"},
    )
    workflow.add_conditional_edges(
        "arbiter",
        route_after_arbiter,
        {"consensus": END, "dispute": "dispute",
         "max_iterations": END, "error": "error"},
    )
    workflow.add_edge("dispute", "hunter")
    workflow.add_edge("dispute", "auditor")
    workflow.add_edge("error", END)

    return workflow.compile()


def run_audit(document: str) -> dict[str, Any]:
    """Run a single audit. Returns the final LangGraph state dict."""
    graph = build_graph()
    initial: AgentState = {
        "raw_document": document,
        "hunter_report": None,
        "auditor_report": None,
        "arbiter_decision": None,
        "audit_log": [],
        "consensus_met": False,
        "iterations": 0,
        "dispute_instructions": None,
        "last_error": None,
        "error_count": 0,
        "provenance_report": None,
        "consistency_anomalies": None,
    }
    return graph.invoke(initial, config={"recursion_limit": 50})

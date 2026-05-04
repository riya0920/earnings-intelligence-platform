"""
nodes.py
========

Agent nodes and conditional routers for the Adversarial Financial Auditor.

Adversarial Logic
-----------------
The three primary agents are designed to be *epistemically isolated*:

1. **Hunter (Extractor, Gemini 1.5 Pro)** — Optimistic. Sees only the raw
   document plus any prior dispute instructions. Reports every numeric
   field it can locate, with verbatim quotes.

2. **Forensic Auditor (Skeptic, Gemini 1.5 Pro)** — Independently re-reads
   the document. Crucially, it **never sees the Hunter's output**. It
   re-derives quantities from primitives (e.g., gross_profit must equal
   revenue minus COGS) and flags any internal inconsistency in the
   document itself.

3. **Arbiter (Judge, GPT-4o-mini)** — Receives both reports but never the
   raw document. It compares numeric fields, computes relative deltas, and
   either declares consensus or emits structured dispute instructions.
   Using a smaller cheaper model here is deliberate: adjudication is a
   shallow reasoning task once the inputs are structured.

Self-Healing
------------
Each LLM-calling node wraps its structured-output call in a retry loop. If
the model returns malformed JSON or fails Pydantic validation, the node
re-prompts with the validation error appended. After `MAX_ERROR_RETRIES`
failures, the node raises and the graph routes to `error_node`, which
records the failure in `audit_log` and ends the run with a non-consensus
verdict rather than crashing the process.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import ValidationError

from .state import (
    DELTA_THRESHOLD,
    MAX_ERROR_RETRIES,
    MAX_ITERATIONS,
    AgentState,
    ArbiterDecision,
    AuditorReport,
    HunterReport,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Model wiring. We keep this lazy so the module imports cleanly even when
# API keys are missing — useful for unit tests and CI.
# ----------------------------------------------------------------------------


def _get_hunter_llm() -> Runnable:
    """Hunter LLM — defaults to Gemini 2.5 Pro for large context + high recall.

    Configurable via HUNTER_MODEL environment variable:
      - "gemini-2.5-pro"    (default; replaces retired gemini-1.5-pro)
      - "gemini-2.5-flash"  (cheaper, slightly less capable)
      - "gpt-4o"            (different lab; uses OPENAI_API_KEY)
      - "gpt-4o-mini"       (cheapest)

    Note: Gemini 1.5 Pro was retired by Google in 2025 and now returns
    HTTP 404. The default has been updated to gemini-2.5-pro accordingly.
    Setting HUNTER_MODEL=gpt-4o lets you run the entire auditor on the
    OpenAI key alone — useful when Google API access is unavailable.
    """
    model = os.getenv("HUNTER_MODEL", "gemini-2.5-pro").strip().lower()
    if model.startswith("gpt-"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=0.0, max_retries=2)
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=model, temperature=0.0, max_retries=2,
    )


def _get_auditor_llm() -> Runnable:
    """Auditor LLM — defaults to Gemini 2.5 Pro for backward compatibility.

    Layer 1 (heterogeneous models) is enabled by setting AUDITOR_MODEL to
    a different lab's model:
      - "gemini-2.5-pro"  (default; same family as Hunter — NOT heterogeneous)
      - "gemini-2.5-flash" (cheaper Gemini variant)
      - "gpt-4o"          (different lab; uses OPENAI_API_KEY) — recommended
      - "gpt-4o-mini"     (cheaper; weaker reasoning, useful for ablation)

    The architectural argument for setting AUDITOR_MODEL=gpt-4o: when
    Hunter and Auditor are both Gemini, they share training-data-induced
    anchoring biases. A different lab's model is less likely to make the
    *same* mistake, so when they agree, the agreement is more informative.
    """
    model = os.getenv("AUDITOR_MODEL", "gemini-2.5-pro").strip().lower()
    if model.startswith("gpt-"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=0.0, max_retries=2)
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=model, temperature=0.0, max_retries=2,
    )


def _get_arbiter_llm() -> Runnable:
    """GPT-4o-mini for the Arbiter — cheap structured comparison."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        max_retries=2,
    )


# ----------------------------------------------------------------------------
# System prompts. Kept here (not in a separate file) to make the
# adversarial framing visible alongside the node implementations.
# ----------------------------------------------------------------------------


HUNTER_SYSTEM = """\
You are the HUNTER, a high-recall financial-data extractor.

Your job: pull every numeric financial metric you can find from the document
below, with verbatim provenance.

Mandatory rules:
1. For every value you report, include a `source_quote` that is a verbatim
   CONTIGUOUS substring of the document. NO ellipses (`...`). NO concatenating
   text from different sentences or table rows. The quote MUST be a single
   continuous span where the numeric value literally appears as written.
   If you cannot find such a contiguous quote, the value is not present in
   the document and you must return null.
2. For every value, set `is_actual` to `true` ONLY if the document presents
   the number as a reported actual result for the period in question.
   Forecasts, guidance, internal targets, "we expected", "we are guiding to",
   and forward-looking statements MUST be marked `is_actual = false` and
   should generally NOT be selected as the canonical value for that field.
3. Normalize all dollar values to millions of USD (e.g., "$1.2 billion" -> 1200.0).
   Normalize all percentages to plain percentage points (e.g., "38.7%" -> 38.7).
4. Include a 1-indexed `paragraph_number` for each quote. Paragraph numbers
   are explicit in the document — they appear as `[N]` prefixes (e.g. `[3]`)
   or as `Paragraph N:` prefixes. Use the integer N exactly as it appears.
5. NEVER compute, derive, or infer values not literally written in the
   document. If gross_profit is not stated as a line item in the document,
   you MUST return null for gross_profit even if you can compute it from
   revenue and cogs. Computation is the verifier's job, not yours. SEC
   income statements often go directly from "Cost of sales" to "Operating
   expenses" without a gross-profit line — in that case, gross_profit is null.

Primary fields (extract reported actuals only; forecasts get null):
  revenue, cogs, gross_profit, ebitda, net_income.

Secondary fields (extract ONLY if the document explicitly states them; do
NOT compute them yourself):
  gross_margin_pct        — stated gross margin, e.g. "gross margin of 38.7%"
  ebitda_margin_pct       — stated EBITDA margin
  net_margin_pct          — stated net margin
  yoy_revenue_growth_pct  — stated YoY revenue growth, e.g. "14% YoY growth"
  prior_period_revenue    — stated prior-quarter or prior-year revenue
                             (whichever the document references for comparison)

If a field is not present as a reported actual or is not explicitly stated,
return `null` for it. Do not infer or compute secondary fields — the
verification layer will recompute them from the primaries and compare.

Return ONLY valid JSON matching the provided schema. No prose, no markdown.
"""

AUDITOR_SYSTEM = """\
You are the FORENSIC AUDITOR, an adversarial skeptic.

You have NOT seen any other agent's extraction. You are reading the document
fresh. Your job is to:

1. Independently extract revenue, cogs, gross_profit, ebitda, net_income
   from the document, treating forecasts and guidance as INVALID for
   actual-period reporting (only include reported actuals).
2. Recompute derived quantities from primitives:
     - gross_profit_derived = revenue - cogs
     - ebitda_margin = ebitda / revenue
     - net_margin     = net_income / revenue
3. Compare your derived `gross_profit_derived` against any `gross_profit`
   stated in the document. If they disagree by more than 0.5%, raise a
   `flagged_anomaly`.
4. List `skeptical_critiques`: any place where the document mixes forecasts
   with actuals, omits a reconciliation, or is ambiguous about period.

Return ONLY valid JSON matching the provided schema. No prose, no markdown.
"""

ARBITER_SYSTEM = """\
You are the ARBITER. You do NOT read the source document. You compare two
independent reports — one from the Hunter and one from the Forensic Auditor —
and decide whether they have reached consensus.

For each metric (revenue, cogs, gross_profit, ebitda, net_income):
1. If both agents report a value, compute the relative delta:
       delta = abs(hunter - auditor) / max(abs(hunter), abs(auditor), 1e-9)
2. If `delta > 0.0001` (0.01%), the metric is DISPUTED.
3. If exactly one agent reports a value, that is also a DISPUTE
   (the other agent missed it or rejected it).
4. If neither agent reports a value, that metric is skipped.

`consensus_met` is true only if there are zero disputed metrics.

If `consensus_met` is false, write `dispute_instructions` that explicitly
direct both agents to:
  - Re-read the specific paragraphs cited by the Hunter for disputed fields.
  - State whether each disputed value is a reported actual or a forecast.
  - Justify their selection with a paragraph number.

Return ONLY valid JSON matching the provided schema. No prose, no markdown.
"""


# ----------------------------------------------------------------------------
# Self-healing structured output helper.
# ----------------------------------------------------------------------------


def _lenient_json_loads(text: str, *, raw_text: str = "") -> Any:
    """Parse JSON with a few common LLM-emission tolerances.

    Strict json.loads is the right default — but smaller models occasionally
    emit JSON with trailing commas, Python literal booleans, or smart
    quotes, which json.loads rejects with "Expecting ',' delimiter" or
    similar. Rather than failing the whole audit step over a stylistic
    glitch, we try strict first, then a series of cheap fixes.

    Each fix is applied only if strict parse failed, so well-formed JSON
    is unaffected. We never accept JS-style comments or unquoted keys —
    those usually indicate the model is hallucinating Python rather than
    JSON, and the self-heal retry is the right response there.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fix 1: trailing commas before } or ]. Common gpt-4o-mini glitch.
    fixed = re.sub(r",(\s*[}\]])", r"\1", text)
    # Fix 2: Python booleans/None.
    fixed = re.sub(r"\bTrue\b", "true", fixed)
    fixed = re.sub(r"\bFalse\b", "false", fixed)
    fixed = re.sub(r"\bNone\b", "null", fixed)
    # Fix 3: smart quotes.
    fixed = (fixed.replace("\u201c", '"').replace("\u201d", '"')
                  .replace("\u2018", "'").replace("\u2019", "'"))
    try:
        return json.loads(fixed)
    except json.JSONDecodeError as e:
        # Re-raise with the original text in the error so the self-heal
        # message carries useful context to the model.
        snippet = (raw_text or text)[:300]
        raise json.JSONDecodeError(
            f"{e.msg} (after lenient fixups). First 300 chars: {snippet}",
            e.doc, e.pos,
        ) from None


def _openai_structured_invoke(
    llm: Any,
    schema: type,
    system_prompt: str,
    user_prompt: str,
) -> Any:
    """Bypass langchain-openai's `with_structured_output` for OpenAI models.

    Why this exists
    ---------------
    langchain-openai's structured-output path tries to send the schema
    via OpenAI's `response_format` JSON-schema mode. That mode requires
    `additionalProperties: false` and `required: [every property]` at
    every level of the schema — constraints that Pydantic's default
    JSON-schema export doesn't satisfy. The result is a 400 BadRequest
    that no `method=` parameter consistently bypasses across versions.

    Instead, we drop to plain text generation, ask the model to emit JSON
    in the response, parse it with stdlib `json.loads`, and validate
    against the Pydantic schema ourselves. This is provider-agnostic and
    immune to langchain-openai version drift.

    Schema-echo defense
    -------------------
    Smaller models (gpt-4o-mini in particular) sometimes return the JSON
    Schema itself rather than an instance of it — i.e., they output
    `{"description": "...", "properties": {...}}` instead of
    `{"consensus_met": true, ...}`. We construct a minimal "instance
    skeleton" hint alongside the full schema to make the difference
    explicit. After parsing, if the result looks like a schema (has a
    top-level `properties` key but is missing required instance keys),
    we raise OutputParserException to trigger the self-heal retry with
    a more pointed correction message.
    """
    schema_dict = schema.model_json_schema()
    required_keys = schema_dict.get("required") or list(
        schema_dict.get("properties", {}).keys()
    )
    schema_hint = json.dumps(schema_dict, indent=2)

    # Build a minimal instance skeleton — just the required field names with
    # placeholder values — to disambiguate "what to return" from "the schema".
    skeleton: dict = {}
    for k in required_keys:
        prop = schema_dict.get("properties", {}).get(k, {})
        t = prop.get("type")
        if t == "string":
            skeleton[k] = "..."
        elif t == "integer":
            skeleton[k] = 0
        elif t == "number":
            skeleton[k] = 0.0
        elif t == "boolean":
            skeleton[k] = False
        elif t == "array":
            skeleton[k] = []
        elif t == "object":
            skeleton[k] = {}
        else:
            skeleton[k] = None
    skeleton_hint = json.dumps(skeleton, indent=2)

    enriched_system = (
        f"{system_prompt}\n\n"
        f"=== OUTPUT FORMAT ===\n"
        f"Return a JSON OBJECT (an instance) that conforms to this schema.\n"
        f"DO NOT return the schema definition itself — return an instance "
        f"with concrete values.\n\n"
        f"SCHEMA (do not return this):\n```json\n{schema_hint}\n```\n\n"
        f"INSTANCE SKELETON (return something shaped like this, with real "
        f"values filled in):\n```json\n{skeleton_hint}\n```\n\n"
        f"Output ONLY the instance JSON. No markdown fences, no prose, "
        f"no schema definition."
    )
    messages = [
        SystemMessage(content=enriched_system),
        HumanMessage(content=user_prompt),
    ]
    raw = llm.invoke(messages)
    text = raw.content if hasattr(raw, "content") else str(raw)

    # Strip optional markdown fences if the model wrapped the JSON.
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    # Find the outermost JSON object.
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise OutputParserException(f"No JSON object found in model output: {text[:300]}")
    candidate = s[start:end + 1]
    parsed = _lenient_json_loads(candidate, raw_text=text)

    # Schema-echo detection: did the model return the schema rather than
    # an instance? Schema objects have `properties` and/or `$defs` at the
    # top level and lack the required instance keys.
    if isinstance(parsed, dict):
        looks_like_schema = (
            ("properties" in parsed or "$defs" in parsed or
             parsed.get("type") == "object")
            and not any(k in parsed for k in required_keys)
        )
        if looks_like_schema:
            raise OutputParserException(
                "Model returned the JSON schema instead of an instance. "
                f"Required instance keys missing: {required_keys}. "
                "Return a JSON object with concrete values for each "
                "required key, NOT the schema definition."
            )

    return schema.model_validate(parsed)


def _invoke_with_self_heal(
    llm: Runnable,
    schema: type,
    system_prompt: str,
    user_prompt: str,
    *,
    node_name: str,
) -> Any:
    """Call `llm` with a Pydantic schema and retry on validation failure.

    The "self-healing" loop appends the validation error to the next prompt
    and asks the model to repair its output. After `MAX_ERROR_RETRIES`
    failures, the original exception is re-raised so the calling node can
    record an audit-log entry and the graph can route to `error_node`.

    Why this matters
    ----------------
    Even with `with_structured_output`, models occasionally emit JSON that
    fails business-level validation (e.g., a string where a float is
    required, an unknown enum, or a `field_name` that violates the
    `field_validator`). A blind retry typically produces the same garbage;
    showing the model its own error closes the loop.

    OpenAI dispatch
    ---------------
    OpenAI's `response_format` JSON-schema mode is fragile against the
    default Pydantic JSON-schema export (missing `required`-everywhere,
    missing `additionalProperties: false`). We bypass `with_structured_output`
    for OpenAI entirely (see `_openai_structured_invoke`) and use plain
    text + JSON parse + Pydantic validation. Other providers (Gemini)
    use the LangChain default path, which works.
    """
    is_openai = type(llm).__name__ == "ChatOpenAI"
    last_err: Optional[Exception] = None
    current_user = user_prompt

    if not is_openai:
        # Default LangChain structured-output path for Gemini & friends.
        structured = llm.with_structured_output(schema)

    for attempt in range(MAX_ERROR_RETRIES + 1):
        try:
            if is_openai:
                return _openai_structured_invoke(
                    llm, schema, system_prompt, current_user,
                )
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=current_user),
            ]
            return structured.invoke(messages)
        except (ValidationError, OutputParserException,
                json.JSONDecodeError) as e:
            last_err = e
            logger.warning(
                "[%s] structured-output failure on attempt %d/%d: %s",
                node_name,
                attempt + 1,
                MAX_ERROR_RETRIES + 1,
                e,
            )
            current_user = (
                f"{user_prompt}\n\n"
                f"---\nYour previous response failed schema validation with "
                f"this error:\n{e}\n\nRepair your output. Return ONLY valid "
                f"JSON matching the schema. No prose, no markdown fences."
            )

    assert last_err is not None
    raise last_err


# ----------------------------------------------------------------------------
# Agent nodes.
# ----------------------------------------------------------------------------


def hunter_node(state: AgentState) -> dict:
    """The Hunter — high-recall extraction with verbatim provenance.

    Adversarial role: optimistic. The Hunter casts a wide net; the Auditor
    is responsible for skepticism. Decoupling these roles prevents a single
    agent from anchoring on a wrong-but-plausible value (the standard LLM
    failure mode on adversarial transcripts).
    """
    iteration = state.get("iterations", 0)
    log_prefix = f"[hunter | iter={iteration}]"

    user_prompt_parts = [f"DOCUMENT:\n\n{state['raw_document']}"]
    if dispute := state.get("dispute_instructions"):
        user_prompt_parts.append(
            f"\n---\nDISPUTE INSTRUCTIONS FROM PRIOR ROUND:\n{dispute}\n"
            "Address each point above. Cite paragraph numbers."
        )
    user_prompt = "\n".join(user_prompt_parts)

    try:
        report: HunterReport = _invoke_with_self_heal(
            _get_hunter_llm(),
            HunterReport,
            HUNTER_SYSTEM,
            user_prompt,
            node_name="hunter",
        )
    except Exception as e:
        return {
            "last_error": f"hunter: {type(e).__name__}: {e}",
            "error_count": state.get("error_count", 0) + 1,
            "audit_log": [f"{log_prefix} extraction failed: {e}"],
        }

    return {
        "hunter_report": report.model_dump(),
        "audit_log": [
            f"{log_prefix} extracted "
            f"{sum(1 for v in report.model_dump().values() if isinstance(v, dict))} "
            f"actuals."
        ],
        "last_error": None,
    }


def auditor_node(state: AgentState) -> dict:
    """The Forensic Auditor — independent extraction + recomputation.

    Adversarial role: skeptic. Critically, the Auditor's prompt is
    constructed *without* any reference to the Hunter's output. The state
    dict does contain `hunter_report` by the time this node runs in
    sequential execution, but this node deliberately does not read that
    key. (For true parallelism, see `app.py`, where Hunter and Auditor are
    fanned out from START.)
    """
    iteration = state.get("iterations", 0)
    log_prefix = f"[auditor | iter={iteration}]"

    user_prompt_parts = [f"DOCUMENT:\n\n{state['raw_document']}"]
    if dispute := state.get("dispute_instructions"):
        user_prompt_parts.append(
            f"\n---\nDISPUTE INSTRUCTIONS FROM PRIOR ROUND:\n{dispute}\n"
            "Independently re-verify. Do not trust the Hunter's prior values."
        )
    user_prompt = "\n".join(user_prompt_parts)

    try:
        report: AuditorReport = _invoke_with_self_heal(
            _get_auditor_llm(),
            AuditorReport,
            AUDITOR_SYSTEM,
            user_prompt,
            node_name="auditor",
        )
    except Exception as e:
        return {
            "last_error": f"auditor: {type(e).__name__}: {e}",
            "error_count": state.get("error_count", 0) + 1,
            "audit_log": [f"{log_prefix} verification failed: {e}"],
        }

    return {
        "auditor_report": report.model_dump(),
        "audit_log": [
            f"{log_prefix} flagged {len(report.flagged_anomalies)} anomaly(ies), "
            f"{len(report.skeptical_critiques)} critique(s)."
        ],
        "last_error": None,
    }


def arbiter_node(state: AgentState) -> dict:
    """The Arbiter — structured comparison and consensus check.

    Adversarial role: judge. The Arbiter sees only the two reports, never
    the source document. This forces the upstream agents to be the source
    of truth about the document's contents and prevents the Arbiter from
    "rescuing" a sloppy extractor by re-reading the doc itself.
    """
    iteration = state.get("iterations", 0)
    log_prefix = f"[arbiter | iter={iteration}]"

    hunter = state.get("hunter_report") or {}
    auditor = state.get("auditor_report") or {}

    if not hunter and not auditor:
        return {
            "last_error": "arbiter: both upstream reports are empty",
            "error_count": state.get("error_count", 0) + 1,
            "audit_log": [f"{log_prefix} no inputs to adjudicate."],
        }

    user_prompt = (
        f"HUNTER REPORT:\n{json.dumps(hunter, indent=2, default=str)}\n\n"
        f"AUDITOR REPORT:\n{json.dumps(auditor, indent=2, default=str)}\n\n"
        f"Adjudicate per your system instructions. The delta threshold is "
        f"{DELTA_THRESHOLD} ({DELTA_THRESHOLD * 100:.4f}%)."
    )

    try:
        decision: ArbiterDecision = _invoke_with_self_heal(
            _get_arbiter_llm(),
            ArbiterDecision,
            ARBITER_SYSTEM,
            user_prompt,
            node_name="arbiter",
        )
    except Exception as e:
        return {
            "last_error": f"arbiter: {type(e).__name__}: {e}",
            "error_count": state.get("error_count", 0) + 1,
            "audit_log": [f"{log_prefix} adjudication failed: {e}"],
        }

    # Defense-in-depth: independently recompute the deltas from the
    # raw reports. The LLM may claim consensus on values it didn't
    # actually compare; verify it ourselves.
    deterministic = _deterministic_consensus(hunter, auditor)
    decision_dict = decision.model_dump()
    decision_dict["deterministic_check"] = deterministic

    # If the deterministic check disagrees with the Arbiter, trust the
    # determinism. The LLM is here to write `dispute_instructions`, not
    # to do floating-point arithmetic.
    if deterministic["consensus_met"] != decision.consensus_met:
        decision_dict["consensus_met"] = deterministic["consensus_met"]
        decision_dict["disputed_fields"] = deterministic["disputed_fields"]
        decision_dict["field_deltas"] = deterministic["field_deltas"]
        decision_dict["rationale"] = (
            f"OVERRIDE: deterministic check disagreed with LLM. "
            f"{decision.rationale}"
        )

    # Layer 3 — even if Hunter and Auditor numerically agree, check that
    # the agreed values are mutually consistent with stated ratios in the
    # document. A wrong-but-shared anchor (e.g., both grabbed the
    # forecast revenue) shows up here when the document also stated the
    # gross margin and the ratio doesn't reconcile.
    anomalies = _consistency_checks(hunter, auditor)
    decision_dict["consistency_anomalies"] = anomalies

    if anomalies and decision_dict["consensus_met"]:
        # Numerical agreement but the math doesn't add up. Force dispute.
        decision_dict["consensus_met"] = False
        consistency_dispute = _build_consistency_dispute(anomalies)
        existing = decision_dict.get("dispute_instructions") or ""
        decision_dict["dispute_instructions"] = (
            (existing + "\n\n" if existing else "") + consistency_dispute
        )
        decision_dict["rationale"] = (
            f"OVERRIDE: Hunter and Auditor agreed on values, but Layer 3 "
            f"consistency checks found {len(anomalies)} anomaly(ies). "
            f"{decision_dict['rationale']}"
        )

    return {
        "arbiter_decision": decision_dict,
        "consensus_met": decision_dict["consensus_met"],
        "dispute_instructions": decision_dict.get("dispute_instructions"),
        "consistency_anomalies": anomalies,
        "audit_log": [
            f"{log_prefix} consensus={decision_dict['consensus_met']} "
            f"disputed={decision_dict.get('disputed_fields') or []} "
            f"anomalies={len(anomalies)}"
        ],
        "last_error": None,
    }


def dispute_node(state: AgentState) -> dict:
    """Bookkeeping node entered when the Arbiter rejects the round.

    Increments the iteration counter and clears the prior reports so the
    next pass through Hunter/Auditor builds fresh state instead of
    accidentally inheriting stale extractions. Also clears the
    provenance and consistency state so the next round's verifier
    starts fresh.
    """
    iteration = state.get("iterations", 0) + 1
    return {
        "iterations": iteration,
        "hunter_report": None,
        "auditor_report": None,
        "provenance_report": None,
        "consistency_anomalies": None,
        "audit_log": [
            f"[dispute] entering iteration {iteration} of {MAX_ITERATIONS}; "
            f"instructions: {state.get('dispute_instructions') or '(none)'}"
        ],
    }


def error_node(state: AgentState) -> dict:
    """Terminal node for unrecoverable schema/parse failures.

    Records the failure cleanly. The graph then routes to END with
    `consensus_met = False`, which downstream callers should treat as
    "audit aborted" rather than "audit passed".
    """
    return {
        "consensus_met": False,
        "audit_log": [
            f"[error] aborting: {state.get('last_error') or 'unknown error'}"
        ],
    }


# ----------------------------------------------------------------------------
# Layer 2 — Provenance verifier node.
#
# Inserted between the parallel extractors and the Arbiter. The verifier
# runs deterministic Python (regex + string match) over the Hunter's
# report to catch hallucinated values and miscited paragraphs that
# multi-agent voting cannot detect (because both agents may share the
# same anchoring bias).
# ----------------------------------------------------------------------------


def provenance_verifier_node(state: AgentState) -> dict:
    """Verify every Hunter-extracted metric's provenance against the document.

    Failure routes back through the dispute path with a Hunter-specific
    instruction enumerating exactly which fields failed and why. The
    Auditor is unaffected — it does not produce per-metric provenance,
    so its report passes through unverified at this layer (Layer 3's
    consistency checks cover the Auditor).
    """
    from .provenance import verify_hunter_report, build_dispute_from_provenance

    iteration = state.get("iterations", 0)
    log_prefix = f"[provenance | iter={iteration}]"

    hunter = state.get("hunter_report") or {}
    if not isinstance(hunter, dict) or not hunter:
        # No Hunter report to verify; let downstream nodes handle.
        return {
            "provenance_report": None,
            "audit_log": [f"{log_prefix} no Hunter report to verify"],
        }

    document = state["raw_document"]
    verification = verify_hunter_report(hunter, document)

    if verification["all_passed"]:
        return {
            "provenance_report": verification,
            "audit_log": [
                f"{log_prefix} all {len(verification['verdicts'])} metric(s) verified"
            ],
        }

    dispute = build_dispute_from_provenance(verification)
    return {
        "provenance_report": verification,
        "dispute_instructions": dispute,
        "audit_log": [
            f"{log_prefix} {verification['failed_count']} provenance failure(s): "
            f"{verification['summary']}"
        ],
    }


# ----------------------------------------------------------------------------
# Layer 3 — Deterministic consistency checks.
#
# Even when Hunter and Auditor agree on the primaries (revenue, cogs,
# etc.), the document itself may contain stated ratios or growth rates
# that are inconsistent with those primaries. If the Hunter picked the
# wrong revenue (e.g., the forecast instead of the actual), and the
# document also says "gross margin of 38.7%", the recomputed
# gross_profit/revenue won't match 38.7% — anomaly flagged.
#
# Math is done in Python, not by the LLM. The LLM's role is extraction,
# not arithmetic.
# ----------------------------------------------------------------------------


def _consistency_checks(hunter: dict, auditor: dict) -> list[dict]:
    """Return a list of anomaly dicts. Empty list = all checks passed."""
    anomalies: list[dict] = []

    def _hv(field: str) -> Optional[float]:
        m = hunter.get(field) if isinstance(hunter, dict) else None
        if isinstance(m, dict):
            try:
                return float(m["value"])
            except (KeyError, TypeError, ValueError):
                return None
        return None

    revenue = _hv("revenue")
    cogs = _hv("cogs")
    gross_profit = _hv("gross_profit")
    ebitda = _hv("ebitda")
    net_income = _hv("net_income")
    gm_pct = _hv("gross_margin_pct")
    em_pct = _hv("ebitda_margin_pct")
    nm_pct = _hv("net_margin_pct")
    yoy_pct = _hv("yoy_revenue_growth_pct")
    prior_rev = _hv("prior_period_revenue")

    REL_TOL = 0.005   # 0.5% relative tolerance for ratio checks.
    PCT_TOL = 0.5     # 0.5 percentage points absolute for percentage checks.

    # Check 1: gross_profit consistency. Already in _deterministic_consensus
    # but we restate it here so anomalies are surfaced uniformly.
    if revenue is not None and cogs is not None and gross_profit is not None:
        recomputed = revenue - cogs
        denom = max(abs(gross_profit), 1e-9)
        if abs(recomputed - gross_profit) / denom > REL_TOL:
            anomalies.append({
                "check": "gross_profit_arithmetic",
                "expected": recomputed,
                "stated": gross_profit,
                "delta": gross_profit - recomputed,
                "severity": "high",
                "explanation": (
                    f"revenue ({revenue}) - cogs ({cogs}) = {recomputed:.2f}, "
                    f"but gross_profit was extracted as {gross_profit}. "
                    f"At least one of the three primaries is wrong."
                ),
            })

    # Check 2: gross_margin consistency. The killer Layer 3 check —
    # catches anchoring on wrong revenue when the document states the margin.
    if gm_pct is not None and gross_profit is not None and revenue not in (None, 0):
        recomputed_pct = (gross_profit / revenue) * 100.0
        if abs(recomputed_pct - gm_pct) > PCT_TOL:
            anomalies.append({
                "check": "gross_margin_consistency",
                "expected_pct": recomputed_pct,
                "stated_pct": gm_pct,
                "delta_pct": gm_pct - recomputed_pct,
                "severity": "high",
                "explanation": (
                    f"gross_profit ({gross_profit}) / revenue ({revenue}) = "
                    f"{recomputed_pct:.2f}%, but document states gross_margin "
                    f"of {gm_pct}%. Likely revenue or gross_profit is wrong."
                ),
            })

    # Check 3: ebitda margin consistency.
    if em_pct is not None and ebitda is not None and revenue not in (None, 0):
        recomputed_pct = (ebitda / revenue) * 100.0
        if abs(recomputed_pct - em_pct) > PCT_TOL:
            anomalies.append({
                "check": "ebitda_margin_consistency",
                "expected_pct": recomputed_pct,
                "stated_pct": em_pct,
                "delta_pct": em_pct - recomputed_pct,
                "severity": "high",
                "explanation": (
                    f"ebitda ({ebitda}) / revenue ({revenue}) = "
                    f"{recomputed_pct:.2f}%, but document states ebitda_margin "
                    f"of {em_pct}%."
                ),
            })

    # Check 4: net margin consistency.
    if nm_pct is not None and net_income is not None and revenue not in (None, 0):
        recomputed_pct = (net_income / revenue) * 100.0
        if abs(recomputed_pct - nm_pct) > PCT_TOL:
            anomalies.append({
                "check": "net_margin_consistency",
                "expected_pct": recomputed_pct,
                "stated_pct": nm_pct,
                "delta_pct": nm_pct - recomputed_pct,
                "severity": "high",
                "explanation": (
                    f"net_income ({net_income}) / revenue ({revenue}) = "
                    f"{recomputed_pct:.2f}%, but document states net_margin "
                    f"of {nm_pct}%."
                ),
            })

    # Check 5: YoY growth consistency.
    if yoy_pct is not None and prior_rev not in (None, 0) and revenue is not None:
        recomputed_pct = ((revenue - prior_rev) / prior_rev) * 100.0
        if abs(recomputed_pct - yoy_pct) > PCT_TOL:
            anomalies.append({
                "check": "yoy_growth_consistency",
                "expected_pct": recomputed_pct,
                "stated_pct": yoy_pct,
                "delta_pct": yoy_pct - recomputed_pct,
                "severity": "high",
                "explanation": (
                    f"(revenue {revenue} - prior {prior_rev}) / prior = "
                    f"{recomputed_pct:.2f}%, but document states YoY growth "
                    f"of {yoy_pct}%."
                ),
            })

    # Check 6: ordering sanity.
    # net_income should not exceed gross_profit (impossible without
    # large non-operating gains, in which case the Hunter should have
    # extracted them and we'd see ebitda > gross_profit too — that's
    # rare and we treat the violation as a soft anomaly).
    if net_income is not None and gross_profit is not None:
        if net_income > gross_profit * 1.005:
            anomalies.append({
                "check": "ordering_net_vs_gross",
                "stated_net_income": net_income,
                "stated_gross_profit": gross_profit,
                "severity": "medium",
                "explanation": (
                    f"net_income ({net_income}) exceeds gross_profit "
                    f"({gross_profit}). Implausible without large non-operating "
                    f"items; one of the values is likely wrong."
                ),
            })

    return anomalies


def _build_consistency_dispute(anomalies: list[dict]) -> str:
    """Render consistency anomalies as a Hunter+Auditor dispute instruction."""
    if not anomalies:
        return ""
    lines = [
        "CONSISTENCY FAILURE — extracted primary values are mathematically "
        "inconsistent with stated ratios or related quantities in the document. "
        "At least one extracted primary is wrong. Re-examine these specifically:",
    ]
    for a in anomalies:
        lines.append(f"  - [{a['check']}] {a['explanation']}")
    lines.append(
        "Re-read the document and identify which primary value (revenue, "
        "cogs, gross_profit, ebitda, or net_income) was incorrectly extracted. "
        "Pay close attention to whether the value you cited is a reported "
        "actual or a forecast/guidance/comparison-period figure."
    )
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Conditional routers.
# ----------------------------------------------------------------------------


def route_after_extraction(state: AgentState) -> str:
    """After parallel extraction, route to error or to provenance verifier."""
    if state.get("last_error"):
        return "error"
    # Both Hunter and Auditor must have produced output.
    if not state.get("hunter_report") or not state.get("auditor_report"):
        return "error"
    return "verifier"


def route_after_verifier(state: AgentState) -> str:
    """After provenance check, route to dispute or to arbiter."""
    if state.get("last_error"):
        return "error"
    pv = state.get("provenance_report") or {}
    if pv and not pv.get("all_passed", True):
        # Provenance failure short-circuits to dispute (or terminates if
        # we've exhausted iterations).
        if state.get("iterations", 0) >= MAX_ITERATIONS - 1:
            return "max_iterations"
        return "dispute"
    return "arbiter"


def route_after_arbiter(state: AgentState) -> str:
    """Route based on the Arbiter's decision."""
    if state.get("last_error"):
        return "error"
    if state.get("consensus_met"):
        return "consensus"
    if state.get("iterations", 0) >= MAX_ITERATIONS - 1:
        # We just finished iteration N; allowing dispute would push to N+1.
        # Cap here.
        return "max_iterations"
    return "dispute"


# ----------------------------------------------------------------------------
# Deterministic consensus check (used to guard the Arbiter's LLM output).
# ----------------------------------------------------------------------------


def _deterministic_consensus(hunter: dict, auditor: dict) -> dict:
    """Compute consensus arithmetically. The Arbiter LLM is sanity-checked
    against this. If the LLM claims consensus but the math disagrees, we
    override the LLM."""
    fields = ["revenue", "cogs", "gross_profit", "ebitda", "net_income"]
    deltas: dict[str, float] = {}
    disputed: list[str] = []

    auditor_extracted = auditor.get("independently_extracted") or {}

    for f in fields:
        h_metric = hunter.get(f) if isinstance(hunter, dict) else None
        h_val = h_metric.get("value") if isinstance(h_metric, dict) else None
        a_val = auditor_extracted.get(f)

        if h_val is None and a_val is None:
            continue
        if h_val is None or a_val is None:
            disputed.append(f)
            deltas[f] = float("inf")
            continue
        try:
            h_f = float(h_val)
            a_f = float(a_val)
        except (TypeError, ValueError):
            disputed.append(f)
            deltas[f] = float("inf")
            continue
        denom = max(abs(h_f), abs(a_f), 1e-9)
        d = abs(h_f - a_f) / denom
        deltas[f] = d
        if d > DELTA_THRESHOLD:
            disputed.append(f)

    return {
        "consensus_met": len(disputed) == 0,
        "field_deltas": deltas,
        "disputed_fields": disputed,
    }
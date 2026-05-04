"""
eval/report.py
==============

Aggregate the eval results JSONL into the comparison report.

Headline metrics (this is what the README and interview slide quote):

  Pass rate per pipeline (with Wilson 95% CI)
    "prose: 14/30 (46.7%) [95% CI 30.2%, 63.9%]"
    "verified: 24/30 (80.0%) [95% CI 62.7%, 90.5%]"

  Confident-wrong rate: questions where pipeline returned a number
  with confidence and was wrong. THE failure mode the system is
  designed to prevent.
    "prose confident-wrong: 11/30 (36.7%)"
    "verified confident-wrong: 2/30 (6.7%)"

  Per-bucket breakdown:
    Bucket A (clean): both should mostly pass.
    Bucket B (period disambiguation): expect verified > prose.
    Bucket C (adversarial): expect verified >> prose.

  McNemar's exact test on paired correctness — is the difference
  between prose and verified statistically significant given n=30?

Usage
-----
    python -m eval.report --in eval/results.jsonl
    python -m eval.report --in eval/results.jsonl --json eval/metrics.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% confidence interval for a proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    halfwidth = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    return (max(0.0, center - halfwidth), min(1.0, center + halfwidth))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value via binomial.

    b = system A correct, system B wrong (one direction)
    c = system A wrong,   system B correct (other direction)
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    cum = sum(math.comb(n, i) for i in range(0, k + 1))
    return min(1.0, cum * 2 / (2 ** n))


def load(path: str) -> list[dict]:
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _by_pipeline(rows: list[dict]) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        out[r["pipeline"]][r["qid"]] = r
    return dict(out)


def pass_rate(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    n = len(rows)
    n_pass = sum(1 for r in rows if r.get("is_pass"))
    lo, hi = wilson_ci(n_pass, n)
    return {"n": n, "passed": n_pass,
            "rate": n_pass / n if n else 0.0,
            "ci_low": lo, "ci_high": hi}


def confident_wrong_rate(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    n = len(rows)
    n_cw = sum(1 for r in rows
               if r.get("verdict") in ("confident_wrong", "confabulated"))
    lo, hi = wilson_ci(n_cw, n)
    return {"n": n, "confident_wrong": n_cw,
            "rate": n_cw / n if n else 0.0,
            "ci_low": lo, "ci_high": hi}


def verdict_breakdown(rows: Iterable[dict]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for r in rows:
        v = r.get("verdict") or "error"
        out[v] += 1
    return dict(out)


def latency_summary(rows: Iterable[dict]) -> dict:
    lats = [r["latency_s"] for r in rows if not r.get("error")]
    if not lats:
        return {"n": 0}
    s = sorted(lats)
    return {
        "n": len(lats),
        "mean_s": sum(s) / len(s),
        "median_s": s[len(s) // 2],
        "p95_s": s[min(int(0.95 * len(s)), len(s) - 1)],
    }


def per_bucket(rows: Iterable[dict]) -> dict[str, dict]:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r.get("bucket", "?")].append(r)
    return {bucket: pass_rate(rs) for bucket, rs in sorted(by.items())}


def paired_mcnemar(by_pipeline: dict[str, dict[str, dict]]) -> dict:
    """McNemar paired test: prose vs verified on same questions."""
    if "prose" not in by_pipeline or "verified" not in by_pipeline:
        return {"n_paired": 0, "note": "need both pipelines"}
    common = sorted(set(by_pipeline["prose"]) & set(by_pipeline["verified"]))
    prose_only = verified_only = 0
    both_pass = both_fail = 0
    for qid in common:
        p = by_pipeline["prose"][qid].get("is_pass") or False
        v = by_pipeline["verified"][qid].get("is_pass") or False
        if p and v:
            both_pass += 1
        elif not p and not v:
            both_fail += 1
        elif p and not v:
            prose_only += 1
        else:
            verified_only += 1
    return {
        "n_paired": len(common),
        "both_pass": both_pass, "both_fail": both_fail,
        "prose_only_pass": prose_only,
        "verified_only_pass": verified_only,
        "p_value": mcnemar_exact(prose_only, verified_only),
    }


def render_report(rows: list[dict]) -> str:
    by = _by_pipeline(rows)
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("VERIFIED RAG vs PROSE RAG — eval results")
    lines.append("=" * 78)

    # Headline pass rate.
    lines.append("\nOverall pass rate")
    lines.append("-" * 78)
    lines.append(f"{'Pipeline':<12} {'Pass':>10}  {'Rate':>8}  {'95% CI':<22}")
    for p in sorted(by):
        pr = pass_rate(by[p].values())
        lines.append(
            f"{p:<12} {pr['passed']:>3}/{pr['n']:<6}  "
            f"{pr['rate']:>7.1%}  "
            f"[{pr['ci_low']:.3f}, {pr['ci_high']:.3f}]"
        )

    # The headline failure-mode metric.
    lines.append("\nConfident-wrong rate "
                 "(returned a number with confidence; was wrong)")
    lines.append("-" * 78)
    lines.append(f"{'Pipeline':<12} {'Cnt':>10}  {'Rate':>8}  {'95% CI':<22}")
    for p in sorted(by):
        cw = confident_wrong_rate(by[p].values())
        lines.append(
            f"{p:<12} {cw['confident_wrong']:>3}/{cw['n']:<6}  "
            f"{cw['rate']:>7.1%}  "
            f"[{cw['ci_low']:.3f}, {cw['ci_high']:.3f}]"
        )

    # Per-bucket breakdown.
    lines.append("\nPer-bucket pass rate")
    lines.append("-" * 78)
    buckets = sorted({r["bucket"] for r in rows})
    header = f"{'Bucket':<24} " + "  ".join(f"{p:>14}" for p in sorted(by))
    lines.append(header)
    for bucket in buckets:
        cells = []
        for p in sorted(by):
            bucket_rows = [r for r in by[p].values() if r["bucket"] == bucket]
            pr = pass_rate(bucket_rows)
            cells.append(f"{pr['passed']:>2}/{pr['n']:<2} ({pr['rate']:>5.0%})")
        lines.append(f"{bucket:<24} " + "  ".join(f"{c:>14}" for c in cells))

    # Verdict breakdown.
    lines.append("\nVerdict distribution")
    lines.append("-" * 78)
    all_verdicts = sorted({v for p in by for v in verdict_breakdown(by[p].values())})
    header = f"{'Verdict':<22} " + "  ".join(f"{p:>10}" for p in sorted(by))
    lines.append(header)
    for v in all_verdicts:
        cells = [verdict_breakdown(by[p].values()).get(v, 0) for p in sorted(by)]
        lines.append(f"{v:<22} " + "  ".join(f"{c:>10}" for c in cells))

    # Paired McNemar.
    lines.append("\nPaired McNemar's exact test (prose vs verified)")
    lines.append("-" * 78)
    mc = paired_mcnemar(by)
    if "p_value" in mc:
        sig = ("***" if mc["p_value"] < 0.01
               else ("**" if mc["p_value"] < 0.05 else ""))
        lines.append(
            f"  paired questions:    {mc['n_paired']}\n"
            f"  both pass:           {mc['both_pass']}\n"
            f"  both fail:           {mc['both_fail']}\n"
            f"  prose_only pass:     {mc['prose_only_pass']}\n"
            f"  verified_only pass:  {mc['verified_only_pass']}\n"
            f"  p-value:             {mc['p_value']:.4f}  {sig}"
        )
    else:
        lines.append(f"  {mc.get('note', '?')}")

    # Latency.
    lines.append("\nLatency (seconds)")
    lines.append("-" * 78)
    lines.append(f"{'Pipeline':<12} {'n':>5}  {'mean':>8}  {'median':>8}  {'p95':>8}")
    for p in sorted(by):
        lat = latency_summary(by[p].values())
        if lat["n"]:
            lines.append(
                f"{p:<12} {lat['n']:>5}  {lat['mean_s']:>7.2f}s  "
                f"{lat['median_s']:>7.2f}s  {lat['p95_s']:>7.2f}s"
            )

    # Failures by qid.
    lines.append("\nQuestions both pipelines failed")
    lines.append("-" * 78)
    if "prose" in by and "verified" in by:
        failed_both = [qid for qid in sorted(set(by["prose"]) & set(by["verified"]))
                       if not by["prose"][qid].get("is_pass")
                       and not by["verified"][qid].get("is_pass")]
        if failed_both:
            for qid in failed_both:
                lines.append(f"  {qid}")
        else:
            lines.append("  (none — every question was caught by at least one pipeline)")

    lines.append("\n" + "=" * 78)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="eval/results.jsonl")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    rows = load(args.in_path)
    if not rows:
        print(f"No results found in {args.in_path}")
        return

    print(render_report(rows))

    if args.json_out:
        by = _by_pipeline(rows)
        payload = {
            "per_pipeline": {
                p: {
                    "pass_rate": pass_rate(by[p].values()),
                    "confident_wrong_rate": confident_wrong_rate(by[p].values()),
                    "verdict_breakdown": verdict_breakdown(by[p].values()),
                    "latency": latency_summary(by[p].values()),
                    "by_bucket": per_bucket(by[p].values()),
                }
                for p in sorted(by)
            },
            "paired_mcnemar": paired_mcnemar(by),
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        print(f"\nWrote machine-readable metrics to {args.json_out}")


if __name__ == "__main__":
    main()
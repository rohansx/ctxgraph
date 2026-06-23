#!/usr/bin/env python3
"""
OpenRouter model benchmark against ctxgraph gold fixtures.

Tests any OpenAI-compatible model on:
  - 50 tech episodes (benchmark_episodes.json)
  - 10 cross-domain episodes (cross_domain_episodes.json)

Reports three F1 variants for direct comparison:
  - strict: exact name + exact type/relation
  - fuzzy:  substring match on names (matches head_to_head.py)
  - pair:   entity-pair only, ignore relation type & direction (matches
            the generous scoring used for Graphiti in head_to_head.py)

Usage:
    OPENROUTER_API_KEY=sk-or-... \\
    python scripts/openrouter_bench.py \\
        --model google/gemma-4-26b-a4b-it \\
        --out /tmp/bench_gemma4_26b.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

FIXTURES_DIR = (
    Path(__file__).parent.parent / "crates/ctxgraph-extract/tests/fixtures"
)
TECH_FIXTURE = FIXTURES_DIR / "benchmark_episodes.json"
CD_FIXTURE = FIXTURES_DIR / "cross_domain_episodes.json"

ENTITY_TYPES = [
    "Person",
    "Component",
    "Service",
    "Language",
    "Database",
    "Infrastructure",
    "Decision",
    "Constraint",
    "Metric",
    "Pattern",
]
RELATION_TYPES = [
    "chose",
    "rejected",
    "replaced",
    "depends_on",
    "fixed",
    "introduced",
    "deprecated",
    "caused",
    "constrained_by",
]

SYSTEM_PROMPT = f"""Extract entities and relations from the user's text.
Reply with ONLY valid JSON — no prose, no markdown fences.

Entity types: {", ".join(ENTITY_TYPES)}
Relation types: {", ".join(RELATION_TYPES)}

Rules:
- Use SHORT canonical names ("Redis" not "Redis cache server", "Stripe" not "Stripe SDK v2")
- Teams/departments/roles are Person entities ("platform team", "treasury department")
- Constraints are compliance requirements, SLAs, certifications, budget caps
- Relation head/tail MUST be the EXACT name string from the entities list
- For migrations prefer "replaced" over "depends_on"

Schema: {{"entities":[{{"name":"...","entity_type":"..."}}], "relations":[{{"head":"...","relation":"...","tail":"..."}}]}}"""


# ── F1 scoring ──────────────────────────────────────────────────────


def fuzzy_match(a: str, b: str) -> bool:
    al, bl = a.lower().strip(), b.lower().strip()
    return al == bl or al in bl or bl in al


def f1(p: int, g: int, tp: int) -> float:
    if p == 0 and g == 0:
        return 1.0
    prec = tp / p if p else 0.0
    rec = tp / g if g else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def score_entities(
    predicted: list[dict[str, str]], expected: list[dict[str, Any]]
) -> dict[str, float]:
    """Return strict_f1 (name+type), fuzzy_f1 (substring name)."""
    pred_pairs = [(e.get("name", "").lower(), e.get("entity_type", "").lower()) for e in predicted]
    gold_pairs = [(e["name"].lower(), e["entity_type"].lower()) for e in expected]
    pred_names = [n for n, _ in pred_pairs]
    gold_names = [n for n, _ in gold_pairs]

    # strict: exact name + exact type
    matched = [False] * len(gold_pairs)
    tp_strict = 0
    for pn, pt in pred_pairs:
        for j, (gn, gt) in enumerate(gold_pairs):
            if not matched[j] and pn == gn and pt == gt:
                matched[j] = True
                tp_strict += 1
                break

    # fuzzy: substring name (any type)
    matched = [False] * len(gold_names)
    tp_fuzzy = 0
    for pn in pred_names:
        for j, gn in enumerate(gold_names):
            if not matched[j] and fuzzy_match(pn, gn):
                matched[j] = True
                tp_fuzzy += 1
                break

    return {
        "strict": f1(len(pred_pairs), len(gold_pairs), tp_strict),
        "fuzzy": f1(len(pred_names), len(gold_names), tp_fuzzy),
    }


def relation_pair_match(ph: str, pt: str, gh: str, gt: str) -> bool:
    h_ok = fuzzy_match(ph, gh)
    t_ok = fuzzy_match(pt, gt)
    h_rev = fuzzy_match(ph, gt)
    t_rev = fuzzy_match(pt, gh)
    return (h_ok and t_ok) or (h_rev and t_rev)


def score_relations(
    predicted: list[dict[str, str]], expected: list[dict[str, str]]
) -> dict[str, float]:
    """strict_f1 (head+rel+tail), pair_f1 (head+tail only, fuzzy, any direction)."""
    # strict: exact head+relation+tail
    p_strict = [
        (r.get("head", "").lower(), r.get("relation", "").lower(), r.get("tail", "").lower())
        for r in predicted
    ]
    g_strict = [
        (r["head"].lower(), r["relation"].lower(), r["tail"].lower()) for r in expected
    ]
    matched = [False] * len(g_strict)
    tp_strict = 0
    for pr in p_strict:
        for j, gr in enumerate(g_strict):
            if not matched[j] and pr == gr:
                matched[j] = True
                tp_strict += 1
                break

    # pair: fuzzy head+tail, any direction, any relation
    matched = [False] * len(expected)
    tp_pair = 0
    for pr in predicted:
        ph = pr.get("head", "").lower()
        pt = pr.get("tail", "").lower()
        for j, gr in enumerate(expected):
            if not matched[j] and relation_pair_match(ph, pt, gr["head"].lower(), gr["tail"].lower()):
                matched[j] = True
                tp_pair += 1
                break

    return {
        "strict": f1(len(predicted), len(expected), tp_strict),
        "pair": f1(len(predicted), len(expected), tp_pair),
    }


# ── OpenRouter call ─────────────────────────────────────────────────


def call_model(model: str, text: str, api_key: str, timeout: int = 120) -> tuple[dict, dict]:
    """Returns (parsed_json, usage_meta).

    Fairness fixes (bias audit): extraction is not a reasoning task, but several
    strong models (gpt-5-mini, MiniMax, GLM) are reasoning models — with reasoning
    ON they burned the token budget thinking and returned empty/non-JSON content,
    which the old parser scored as 0 (a harness artifact, not model quality). We
    now (1) disable reasoning, (2) request JSON output mode, (3) raise the token
    cap, and (4) fall back to reasoning_content and strip <think> blocks.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/rohansx/ctxgraph",
        "X-Title": "ctxgraph-bench",
    }

    def _post(extra: dict):
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "max_tokens": 2048,
        }
        payload.update(extra)
        return requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers, json=payload, timeout=timeout,
        )

    # Tiered fallback so every model gets a fair shot at emitting parseable JSON.
    # Some providers reject response_format, the `reasoning` param, or temperature=0
    # (e.g. GPT-5 reasoning models) — progressively drop the strictest params.
    attempts = [
        {"temperature": 0, "reasoning": {"enabled": False}, "response_format": {"type": "json_object"}},
        {"temperature": 0, "reasoning": {"enabled": False}},
        {"temperature": 0},
        {},  # maximally compatible: provider defaults
    ]
    resp = None
    for extra in attempts:
        resp = _post(extra)
        if resp.status_code < 400:
            break
    resp.raise_for_status()
    body = resp.json()

    msg = body["choices"][0]["message"]
    content = msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning") or ""
    # Strip reasoning <think>...</think> preamble if any leaked into content
    if "</think>" in content:
        content = content.split("</think>", 1)[1]
    # Strip code fences if present
    if "```" in content:
        if "```json" in content:
            content = content.split("```json", 1)[1].split("```", 1)[0]
        else:
            content = content.split("```", 1)[1].split("```", 1)[0]
    # Trim to JSON object
    if "{" in content and "}" in content:
        content = content[content.index("{"): content.rindex("}") + 1]
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        parsed = {"entities": [], "relations": [], "_parse_error": str(e), "_raw": content[:300]}
    return parsed, body.get("usage", {})


# ── Runner ──────────────────────────────────────────────────────────


def run_fixture(model: str, fixture: list[dict], label: str, api_key: str) -> dict:
    print(f"\n══ {model} on {label} ({len(fixture)} episodes) ══", flush=True)
    per_ep = []
    cost_total = 0.0
    time_total = 0.0
    err_count = 0
    for i, ep in enumerate(fixture):
        text = ep["text"]
        gold_ents = ep["expected_entities"]
        gold_rels = ep["expected_relations"]

        start = time.time()
        try:
            parsed, usage = call_model(model, text, api_key)
            elapsed = time.time() - start
            time_total += elapsed
            cost_total += float(usage.get("cost", 0) or 0)

            pred_ents = parsed.get("entities", []) or []
            pred_rels = parsed.get("relations", []) or []

            es = score_entities(pred_ents, gold_ents)
            rs = score_relations(pred_rels, gold_rels)

            per_ep.append({
                "idx": i,
                "domain": ep.get("domain", ep.get("source", "tech")),
                "ent_strict": es["strict"],
                "ent_fuzzy": es["fuzzy"],
                "rel_strict": rs["strict"],
                "rel_pair": rs["pair"],
                "elapsed_s": elapsed,
                "tokens": usage.get("total_tokens"),
                "cost": float(usage.get("cost", 0) or 0),
                "pred_ents": len(pred_ents),
                "pred_rels": len(pred_rels),
                "parse_error": parsed.get("_parse_error"),
            })
            domain = ep.get("domain", "tech")[:12]
            print(
                f"  ep{i:02d} [{domain:>12}] "
                f"ent_strict={es['strict']:.3f} ent_fuzzy={es['fuzzy']:.3f} "
                f"rel_strict={rs['strict']:.3f} rel_pair={rs['pair']:.3f} "
                f"t={elapsed:.1f}s ${float(usage.get('cost', 0) or 0)*1000:.3f}m",
                flush=True,
            )
        except Exception as e:
            err_count += 1
            elapsed = time.time() - start
            time_total += elapsed
            per_ep.append({"idx": i, "error": str(e)[:200], "elapsed_s": elapsed})
            print(f"  ep{i:02d}: ERROR {str(e)[:120]}", flush=True)

    n = len(fixture)
    ok = [e for e in per_ep if "error" not in e]
    if not ok:
        return {"model": model, "label": label, "n": n, "error": "all episodes failed"}

    def avg(k):
        return sum(e[k] for e in ok) / len(ok)

    summary = {
        "model": model,
        "label": label,
        "n_episodes": n,
        "n_ok": len(ok),
        "n_errors": err_count,
        "ent_f1_strict": avg("ent_strict"),
        "ent_f1_fuzzy": avg("ent_fuzzy"),
        "rel_f1_strict": avg("rel_strict"),
        "rel_f1_pair": avg("rel_pair"),
        "combined_f1_strict": (avg("ent_strict") + avg("rel_strict")) / 2,
        "combined_f1_fuzzy_pair": (avg("ent_fuzzy") + avg("rel_pair")) / 2,
        "avg_time_s": time_total / n,
        "total_time_s": time_total,
        "total_cost_usd": cost_total,
        "cost_per_1k_episodes_usd": cost_total / n * 1000 if n else 0.0,
        "per_episode": per_ep,
    }
    return summary


def print_summary(summary: dict) -> None:
    print(f"\n  ── SUMMARY: {summary['model']} / {summary['label']} ──")
    print(f"  episodes ok:        {summary['n_ok']}/{summary['n_episodes']} ({summary['n_errors']} errors)")
    print(f"  entity F1 (strict): {summary['ent_f1_strict']:.4f}")
    print(f"  entity F1 (fuzzy):  {summary['ent_f1_fuzzy']:.4f}")
    print(f"  relation F1 (strict head+rel+tail): {summary['rel_f1_strict']:.4f}")
    print(f"  relation F1 (pair, any direction):  {summary['rel_f1_pair']:.4f}")
    print(f"  combined F1 (strict / pair-fuzzy):  {summary['combined_f1_strict']:.4f}  /  {summary['combined_f1_fuzzy_pair']:.4f}")
    print(f"  avg time/episode:   {summary['avg_time_s']:.2f}s")
    print(f"  total cost:         ${summary['total_cost_usd']:.4f}")
    print(f"  $/1k episodes:      ${summary['cost_per_1k_episodes_usd']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="OpenRouter model id, e.g. google/gemma-4-26b-a4b-it")
    ap.add_argument("--out", required=True, help="JSON output path")
    ap.add_argument("--limit-tech", type=int, default=None, help="Cap tech episodes for smoke test")
    ap.add_argument("--limit-cd", type=int, default=None, help="Cap cross-domain episodes")
    ap.add_argument("--skip-tech", action="store_true")
    ap.add_argument("--skip-cd", action="store_true")
    ap.add_argument("--cd-fixture", default=None, help="Override cross-domain fixture path")
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    tech = json.loads(TECH_FIXTURE.read_text())
    cd_path = Path(args.cd_fixture) if args.cd_fixture else CD_FIXTURE
    cd = json.loads(cd_path.read_text())
    if args.limit_tech:
        tech = tech[: args.limit_tech]
    if args.limit_cd:
        cd = cd[: args.limit_cd]

    results = {"model": args.model, "tech": None, "cross_domain": None}

    if not args.skip_tech:
        results["tech"] = run_fixture(args.model, tech, "tech (50ep)", api_key)
        print_summary(results["tech"])
    if not args.skip_cd:
        results["cross_domain"] = run_fixture(args.model, cd, "cross-domain (10ep)", api_key)
        print_summary(results["cross_domain"])

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()

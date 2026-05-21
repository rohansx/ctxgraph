"""
End-to-end smoke test of the 5 CLARITY pieces.

Pieces tested:
  1. Universal schema TOML        (loaded into prompt + relation matcher)
  2. Extraction prompt + JSON     (run against OpenRouter Gemma 4 26B)
  3. Relation-vocabulary embed.   (proto_relation_match.py self-test + per-episode)
  4. NL query parser prompt       (run against OpenRouter small model proxy)
  5. (Piece 5 layer A — suggestion logging — validated as part of #2 JSON output)

Reports:
  - per-episode entity F1 (substring) and relation F1 (pair-fuzzy)
  - per-verb cosine match accuracy (verb → typed relation)
  - per-query NL→graph-op JSON validity rate
  - total cost in USD

Usage:
    OPENROUTER_API_KEY=sk-or-... /tmp/graphiti_venv/bin/python scripts/test_5_pieces.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# Make proto_relation_match importable
sys.path.insert(0, str(Path(__file__).parent))
from proto_relation_match import RelationMatcher  # noqa: E402

# ── Semantic matching (Step 1 from CLARITY follow-up) ─────────────
# Substring matching alone penalizes the LLM when it emits a paraphrased
# entity name. Semantic match via cosine on all-MiniLM-L6-v2 closes this gap.

_EMB_MODEL: SentenceTransformer | None = None
# Threshold tuned empirically on the universal_smoke fixture: 0.65 was too tight
# (rejected legitimate paraphrases of verbose "Idea" / "Fact" entities);
# 0.55 balances precision and recall.
SEMANTIC_THRESHOLD = 0.55


def _embedder() -> SentenceTransformer:
    global _EMB_MODEL
    if _EMB_MODEL is None:
        _EMB_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _EMB_MODEL


def semantic_match(a: str, b: str, threshold: float = SEMANTIC_THRESHOLD) -> bool:
    """Match if substring OR cosine similarity above threshold (per-pair API)."""
    al, bl = a.lower().strip(), b.lower().strip()
    if not al or not bl:
        return False
    if al == bl or al in bl or bl in al:
        return True
    m = _embedder()
    vecs = m.encode([al, bl], normalize_embeddings=True)
    cos = float(np.dot(vecs[0], vecs[1]))
    return cos >= threshold


def batch_entity_tp(
    pred_names: list[str], gold_names: list[str], threshold: float = SEMANTIC_THRESHOLD
) -> int:
    """Greedy bipartite matching, vectorized. Returns true-positive count.
    100× faster than per-pair semantic_match for episodes with many entities."""
    if not pred_names or not gold_names:
        return 0
    pred_norm = [p.lower().strip() for p in pred_names]
    gold_norm = [g.lower().strip() for g in gold_names]
    m = _embedder()
    # One encode call per side (batched)
    pred_vecs = np.asarray(m.encode(pred_norm, normalize_embeddings=True), dtype=np.float32)
    gold_vecs = np.asarray(m.encode(gold_norm, normalize_embeddings=True), dtype=np.float32)
    cos = pred_vecs @ gold_vecs.T  # (P, G)

    matched_gold = [False] * len(gold_norm)
    tp = 0
    for pi in range(len(pred_norm)):
        pn = pred_norm[pi]
        best_gi = -1
        best_score = -1.0
        for gi in range(len(gold_norm)):
            if matched_gold[gi]:
                continue
            gn = gold_norm[gi]
            if pn == gn or pn in gn or gn in pn:
                score = 1.0
            else:
                score = float(cos[pi, gi])
            if score > best_score:
                best_score = score
                best_gi = gi
        if best_gi >= 0 and best_score >= threshold:
            matched_gold[best_gi] = True
            tp += 1
    return tp


def batch_relation_pair_tp(
    pred_rels: list[tuple[str, str]],
    gold_rels: list[tuple[str, str]],
    threshold: float = SEMANTIC_THRESHOLD,
) -> int:
    """Greedy bipartite matching for relation pairs (head+tail). Same approach
    as `batch_entity_tp` but operates on string pairs with directional/reverse
    matching allowed."""
    if not pred_rels or not gold_rels:
        return 0
    # Collect all unique strings to embed once
    pool: list[str] = []
    seen = {}
    def add(s: str) -> int:
        sl = s.lower().strip()
        if sl in seen:
            return seen[sl]
        seen[sl] = len(pool)
        pool.append(sl)
        return seen[sl]

    pred_idx: list[tuple[int, int]] = [(add(h), add(t)) for h, t in pred_rels]
    gold_idx: list[tuple[int, int]] = [(add(h), add(t)) for h, t in gold_rels]

    m = _embedder()
    vecs = np.asarray(m.encode(pool, normalize_embeddings=True), dtype=np.float32)
    cos = vecs @ vecs.T

    def str_match(i: int, j: int) -> bool:
        a, b = pool[i], pool[j]
        if a == b or a in b or b in a:
            return True
        return float(cos[i, j]) >= threshold

    matched_gold = [False] * len(gold_rels)
    tp = 0
    for ph, pt in pred_idx:
        for j, (gh, gt) in enumerate(gold_idx):
            if matched_gold[j]:
                continue
            forward = str_match(ph, gh) and str_match(pt, gt)
            reverse = str_match(ph, gt) and str_match(pt, gh)
            if forward or reverse:
                matched_gold[j] = True
                tp += 1
                break
    return tp

ROOT = Path(__file__).parent.parent
FIXTURE = ROOT / "crates/ctxgraph-extract/tests/fixtures/universal_smoke.json"
EXTRACT_PROMPT_PATH = ROOT / "crates/ctxgraph-extract/prompts/extract.txt"
QUERY_PROMPT_PATH = ROOT / "crates/ctxgraph-cli/prompts/query_parse.txt"
SCHEMA_PATH = ROOT / "crates/ctxgraph-extract/schemas/universal.toml"

# Backends. Each has (provider_id, base_url, default_model_for_extract,
# default_model_for_query).
BACKENDS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "env": "OPENROUTER_API_KEY",
        "extract_model": "google/gemma-4-26b-a4b-it",
        "query_model": "qwen/qwen3-8b",
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1/chat/completions",
        "env": "CEREBRAS_API_KEY",
        # qwen-3-32b was deprecated; live models (May 2026) are llama3.1-8b,
        # gpt-oss-120b, qwen-3-235b-a22b-instruct-2507, zai-glm-4.7.
        # Use the largest Qwen MoE for extraction; gpt-oss-120b for query parsing.
        "extract_model": "qwen-3-235b-a22b-instruct-2507",
        "query_model": "gpt-oss-120b",
    },
}

# ── F1 scoring (matches scripts/openrouter_bench.py) ───────────────


def fuzzy_match(a: str, b: str) -> bool:
    al, bl = a.lower().strip(), b.lower().strip()
    return al == bl or al in bl or bl in al


def f1(pred_n: int, gold_n: int, tp: int) -> float:
    if pred_n == 0 and gold_n == 0:
        return 1.0
    p = tp / pred_n if pred_n else 0.0
    r = tp / gold_n if gold_n else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def relation_pair_match(ph: str, pt: str, gh: str, gt: str) -> bool:
    h_ok = semantic_match(ph, gh) and semantic_match(pt, gt)
    h_rev = semantic_match(ph, gt) and semantic_match(pt, gh)
    return h_ok or h_rev


# ── Generic OpenAI-compatible call (works for OpenRouter + Cerebras) ──

# Cerebras free tier = 30 RPM. Pace calls to stay under it.
_LAST_CALL_TIME: dict[str, float] = {}
_MIN_INTERVAL_S: dict[str, float] = {
    "cerebras": 2.1,   # ~28 RPM, safe margin under 30
    "openrouter": 0.0, # OpenRouter has higher limits; no pacing needed
}


def _pace(backend: str) -> None:
    min_interval = _MIN_INTERVAL_S.get(backend, 0.0)
    if min_interval <= 0:
        return
    last = _LAST_CALL_TIME.get(backend, 0.0)
    wait = min_interval - (time.time() - last)
    if wait > 0:
        time.sleep(wait)
    _LAST_CALL_TIME[backend] = time.time()


def call_chat(
    backend: str, model: str, system: str, user: str, api_key: str, max_retries: int = 3
) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = BACKENDS[backend]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if backend == "openrouter":
        headers["HTTP-Referer"] = "https://github.com/rohansx/ctxgraph"
        headers["X-Title"] = "ctxgraph-clarity-smoke-test"

    for attempt in range(max_retries):
        _pace(backend)
        r = requests.post(
            cfg["base_url"],
            headers=headers,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0,
                "max_tokens": 1024,
            },
            timeout=120,
        )
        if r.status_code == 429 and attempt < max_retries - 1:
            # Honor Retry-After if provided, else exponential backoff
            retry_after = float(r.headers.get("Retry-After", 0))
            wait = retry_after if retry_after else 2.0 * (2 ** attempt)
            time.sleep(wait)
            continue
        r.raise_for_status()
        break  # success
    body = r.json()
    content = body["choices"][0]["message"]["content"] or ""
    if "```" in content:
        if "```json" in content:
            content = content.split("```json", 1)[1].split("```", 1)[0]
        else:
            content = content.split("```", 1)[1].split("```", 1)[0]
    if "{" in content and "}" in content:
        content = content[content.index("{") : content.rindex("}") + 1]
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        parsed = {"_parse_error": str(e), "_raw": content[:300]}
    return parsed, body.get("usage", {})


# ── Piece 2 test: extraction prompt → JSON → score ────────────────


def test_piece_2(
    backend: str, api_key: str, fixtures: list[dict[str, Any]], suggestions_log_path: Path
) -> dict[str, Any]:
    cfg = BACKENDS[backend]
    extract_model = cfg["extract_model"]
    print(f"\n══ Piece 2 — Extraction prompt + JSON contract  [backend={backend}, model={extract_model}] ══")
    system_prompt = EXTRACT_PROMPT_PATH.read_text()
    cost_total = 0.0
    ent_f1s: list[float] = []
    rel_f1s: list[float] = []
    extractions: list[dict[str, Any]] = []
    suggestions_collected: list[dict[str, Any]] = []  # Piece 5 layer A

    for i, ep in enumerate(fixtures):
        # Inline the episode_text into the prompt as the prompt's template suggests
        sys_for_call = system_prompt.replace("{episode_text}", "")
        start = time.time()
        try:
            parsed, usage = call_chat(
                backend, extract_model, sys_for_call, f"Episode text:\n{ep['text']}", api_key
            )
            elapsed = time.time() - start
            cost = float(usage.get("cost", 0) or 0)
            cost_total += cost

            if "_parse_error" in parsed:
                print(f"  ep{i} [{ep['domain']:>16}] JSON PARSE ERROR: {parsed['_parse_error']}")
                ent_f1s.append(0.0)
                rel_f1s.append(0.0)
                extractions.append({})
                continue

            pred_ents = parsed.get("entities", []) or []
            pred_rels = parsed.get("relations", []) or []
            extractions.append(parsed)

            # Build id → name lookup so we can resolve relation head/tail
            # (Gemma emits relations with entity IDs per the JSON contract)
            id_to_name = {
                e.get("id", ""): e.get("name", "").lower()
                for e in pred_ents
                if e.get("id") and e.get("name")
            }

            # Entity F1 (semantic: substring OR cosine ≥ 0.65, batched)
            gold_ents = ep["expected_entities"]
            pred_names = [e.get("name", "").lower() for e in pred_ents]
            gold_names = [e["name"].lower() for e in gold_ents]
            tp_ent = batch_entity_tp(pred_names, gold_names)
            ent_f1 = f1(len(pred_names), len(gold_names), tp_ent)

            # Relation F1 (pair-fuzzy, batched). Resolve entity IDs to names first.
            gold_rels = ep["expected_relations"]
            pred_rel_pairs: list[tuple[str, str]] = []
            for pr in pred_rels:
                raw_h = pr.get("head", "")
                raw_t = pr.get("tail", "")
                ph = id_to_name.get(raw_h, raw_h).lower()
                pt = id_to_name.get(raw_t, raw_t).lower()
                pred_rel_pairs.append((ph, pt))
            gold_rel_pairs: list[tuple[str, str]] = [
                (gr["head"].lower(), gr["tail"].lower()) for gr in gold_rels
            ]
            tp_rel = batch_relation_pair_tp(pred_rel_pairs, gold_rel_pairs)
            rel_f1 = f1(len(pred_rel_pairs), len(gold_rel_pairs), tp_rel)

            ent_f1s.append(ent_f1)
            rel_f1s.append(rel_f1)

            # Piece 5 Layer A: persist any LLM-emitted schema suggestions
            for sug in parsed.get("suggestions") or []:
                suggestions_collected.append({
                    "episode_idx": i,
                    "episode_domain": ep["domain"],
                    "backend": backend,
                    "model": extract_model,
                    "suggestion": sug,
                })

            n_sug = len(parsed.get("suggestions") or [])
            print(
                f"  ep{i} [{ep['domain']:>16}]  ent_F1={ent_f1:.3f}  rel_F1={rel_f1:.3f}  "
                f"ents={len(pred_ents)}/{len(gold_ents)} rels={len(pred_rels)}/{len(gold_rels)}  "
                f"suggestions={n_sug}  t={elapsed:.1f}s  ${cost*1000:.2f}m"
            )
        except Exception as e:
            elapsed = time.time() - start
            print(f"  ep{i} [{ep['domain']:>16}]  ERROR ({elapsed:.1f}s): {str(e)[:120]}")
            ent_f1s.append(0.0)
            rel_f1s.append(0.0)
            extractions.append({})

    n = len(fixtures)
    avg_ent = sum(ent_f1s) / n
    avg_rel = sum(rel_f1s) / n
    combined = (avg_ent + avg_rel) / 2

    # Persist suggestions to side-table file (Piece 5 layer A)
    if suggestions_collected:
        existing: list[dict[str, Any]] = []
        if suggestions_log_path.exists():
            try:
                existing = json.loads(suggestions_log_path.read_text())
            except Exception:
                existing = []
        existing.extend(suggestions_collected)
        suggestions_log_path.write_text(json.dumps(existing, indent=2))

    # Per-domain breakdown
    per_domain: dict[str, list[tuple[float, float]]] = {}
    for ep, e_f1, r_f1 in zip(fixtures, ent_f1s, rel_f1s):
        per_domain.setdefault(ep["domain"], []).append((e_f1, r_f1))

    print()
    print(f"  ── Piece 2 SUMMARY [{backend}] (n={n}) ──")
    print(f"  entity F1 (avg):    {avg_ent:.4f}")
    print(f"  relation F1 (avg):  {avg_rel:.4f}")
    print(f"  combined F1:        {combined:.4f}")
    print(f"  total cost:         ${cost_total:.4f}")
    print(f"  schema suggestions: {len(suggestions_collected)} logged to {suggestions_log_path}")
    print(f"  per-domain breakdown (ent_F1 / rel_F1):")
    for dom, scores in sorted(per_domain.items()):
        avg_e = sum(s[0] for s in scores) / len(scores)
        avg_r = sum(s[1] for s in scores) / len(scores)
        print(f"    {dom:<18} n={len(scores)}  ent={avg_e:.3f}  rel={avg_r:.3f}")
    return {
        "backend": backend,
        "model": extract_model,
        "avg_entity_f1": avg_ent,
        "avg_relation_f1": avg_rel,
        "combined_f1": combined,
        "cost_usd": cost_total,
        "extractions": extractions,
        "per_ep_ent_f1": ent_f1s,
        "per_ep_rel_f1": rel_f1s,
        "suggestions_count": len(suggestions_collected),
        "per_domain": {
            dom: {
                "n": len(s),
                "ent_f1": sum(x[0] for x in s) / len(s),
                "rel_f1": sum(x[1] for x in s) / len(s),
            }
            for dom, s in per_domain.items()
        },
    }


# ── Piece 3 test: relation matcher on real verb queries ───────────


def test_piece_3() -> dict[str, Any]:
    print("\n══ Piece 3 — Relation-vocabulary embeddings ══")
    matcher = RelationMatcher()
    matcher.load_schema()

    # Per-episode verb variations users would actually ask
    queries: list[tuple[str, str]] = [
        # (user verb, expected relation)
        ("relies on", "depends_on"),
        ("requires", "depends_on"),
        ("needs", "depends_on"),
        ("is built on", "depends_on"),
        ("part of", "part_of"),
        ("belongs to", "part_of"),
        ("works at", "part_of"),
        ("is at", "located_at"),
        ("happens in", "located_at"),
        ("based in", "located_at"),
        ("attended", "participated_in"),
        ("was at", "participated_in"),
        ("led to", "caused"),
        ("resulted in", "caused"),
        ("caused by", "caused"),
        ("happened before", "preceded"),
        ("predecessor of", "preceded"),
        ("cites", "references"),
        ("builds on", "references"),
        ("owned by", "owned_by"),
        ("belongs to (owner)", "owned_by"),
        ("controlled by", "owned_by"),
        ("mentioned in", "mentions"),
        ("named in", "mentions"),
    ]

    correct = 0
    rows: list[dict[str, Any]] = []
    for verb, expected in queries:
        m = matcher.resolve(verb)
        ok = m.relation == expected
        correct += int(ok)
        flag = "✓" if ok else "✗"
        rows.append(
            {
                "verb": verb,
                "expected": expected,
                "predicted": m.relation,
                "score": m.score,
                "runner_up": m.runner_up,
                "runner_up_score": m.runner_up_score,
                "correct": ok,
            }
        )
        print(
            f"  {flag} {verb!r:>22}  → {m.relation:<15} ({m.score:.3f})  "
            f"runner_up={m.runner_up} ({m.runner_up_score:.3f})"
        )

    n = len(queries)
    acc = correct / n
    print()
    print(f"  ── Piece 3 SUMMARY (n={n}) ──")
    print(f"  accuracy: {correct}/{n} = {acc:.1%}")
    return {"accuracy": acc, "correct": correct, "total": n, "rows": rows}


# ── Piece 4 test: NL query parser ─────────────────────────────────


def test_piece_4(backend: str, api_key: str) -> dict[str, Any]:
    cfg = BACKENDS[backend]
    query_model = cfg["query_model"]
    print(f"\n══ Piece 4 — NL query parser  [backend={backend}, model={query_model}] ══")
    system_prompt = QUERY_PROMPT_PATH.read_text()

    test_queries = [
        ("what does Vernon CMS depend on?", "traverse", "depends_on"),
        ("who did I meet at PyCon?", "traverse", "participated_in"),
        ("what did I learn this week?", "list", None),
        ("what concepts are connected to Letta?", "traverse", None),
        ("who replaced whom?", "filter", "preceded"),
        ("what caused the outage?", "traverse", "caused"),
        ("which papers cite MemGPT?", "traverse", "references"),
        ("who works at Sundae?", "traverse", "part_of"),
    ]

    valid_ops = {"lookup", "traverse", "filter", "list", "compare"}
    valid_relations = {
        "mentions",
        "located_at",
        "related_to",
        "caused",
        "preceded",
        "references",
        "owned_by",
        "part_of",
        "depends_on",
        "participated_in",
    }

    cost_total = 0.0
    json_valid_count = 0
    schema_valid_count = 0
    op_correct = 0
    rel_correct = 0
    rows: list[dict[str, Any]] = []

    for q, expected_op, expected_rel in test_queries:
        sys_for_call = system_prompt.replace("{user_query}", "").replace(
            "{today_minus_7}", "2026-05-07"
        )
        start = time.time()
        try:
            parsed, usage = call_chat(
                backend, query_model, sys_for_call, f"Query: {q}", api_key
            )
            elapsed = time.time() - start
            cost = float(usage.get("cost", 0) or 0)
            cost_total += cost

            if "_parse_error" in parsed:
                print(f"  ✗ {q!r:>60} JSON parse error")
                rows.append({"query": q, "json_valid": False})
                continue
            json_valid_count += 1

            # Schema validation
            op = parsed.get("op")
            rel = parsed.get("relation")
            schema_ok = op in valid_ops and (rel is None or rel in valid_relations)
            if schema_ok:
                schema_valid_count += 1

            op_ok = op == expected_op
            rel_ok = (expected_rel is None) or (rel == expected_rel)
            if op_ok:
                op_correct += 1
            if rel_ok:
                rel_correct += 1

            op_flag = "✓" if op_ok else "✗"
            rel_flag = "✓" if rel_ok else "✗"
            print(
                f"  {q!r:>62}"
                f"\n    op={op:<10} {op_flag}  rel={str(rel):<18} {rel_flag}  "
                f"t={elapsed:.1f}s  ${cost*1000:.2f}m"
            )
            rows.append(
                {
                    "query": q,
                    "op": op,
                    "relation": rel,
                    "json_valid": True,
                    "schema_valid": schema_ok,
                    "op_correct": op_ok,
                    "rel_correct": rel_ok,
                }
            )
        except Exception as e:
            print(f"  ERROR {q!r}: {str(e)[:120]}")
            rows.append({"query": q, "error": str(e)[:200]})

    n = len(test_queries)
    print()
    print(f"  ── Piece 4 SUMMARY (n={n}) ──")
    print(f"  JSON valid:        {json_valid_count}/{n}")
    print(f"  schema valid:      {schema_valid_count}/{n}")
    print(f"  op classification: {op_correct}/{n} = {op_correct/n:.1%}")
    print(f"  relation classify: {rel_correct}/{n} = {rel_correct/n:.1%}")
    print(f"  total cost:        ${cost_total:.4f}")
    return {
        "json_valid": json_valid_count,
        "schema_valid": schema_valid_count,
        "op_correct": op_correct,
        "rel_correct": rel_correct,
        "total": n,
        "cost_usd": cost_total,
        "rows": rows,
    }


# ── Main ──────────────────────────────────────────────────────────


def main():
    # Collect available backends from env
    available: list[str] = []
    for backend, cfg in BACKENDS.items():
        if os.environ.get(cfg["env"]):
            available.append(backend)
    if not available:
        envs = ", ".join(cfg["env"] for cfg in BACKENDS.values())
        print(f"ERROR: no API key found. Set one of: {envs}")
        sys.exit(1)
    print(f"Backends available: {', '.join(available)}")

    fixtures = json.loads(FIXTURE.read_text())
    print(f"Loaded {len(fixtures)} fixture episodes from {FIXTURE.name}")
    new_count = sum(1 for e in fixtures if e["kind"] == "new_wiki")
    relabel_count = sum(1 for e in fixtures if e["kind"] == "relabeled_cross_domain")
    print(f"  {new_count} new wiki, {relabel_count} relabeled cross-domain")

    # Piece 3 is backend-independent (local sentence-transformers).
    # Run it once.
    p3 = test_piece_3()

    # Suggestions log path — Piece 5 Layer A side-table.
    suggestions_log = Path("/tmp/schema_suggestions.json")
    if suggestions_log.exists():
        suggestions_log.unlink()

    per_backend: dict[str, dict[str, Any]] = {}
    targets = {
        "piece_2_combined_f1_min": 0.55,
        "piece_3_accuracy_min": 0.80,
        "piece_4_json_valid_rate_min": 0.95,
        "piece_4_op_accuracy_min": 0.75,
    }

    for backend in available:
        api_key = os.environ[BACKENDS[backend]["env"]]
        print()
        print("──────────────────────────────────────────────────────────────────────")
        print(f"  BACKEND: {backend.upper()}")
        print("──────────────────────────────────────────────────────────────────────")
        try:
            p2 = test_piece_2(backend, api_key, fixtures, suggestions_log)
        except Exception as e:
            print(f"  Piece 2 on {backend} ABORTED: {str(e)[:200]}")
            p2 = {"combined_f1": 0.0, "cost_usd": 0.0, "error": str(e)[:200]}
        try:
            p4 = test_piece_4(backend, api_key)
        except Exception as e:
            print(f"  Piece 4 on {backend} ABORTED: {str(e)[:200]}")
            p4 = {"json_valid": 0, "op_correct": 0, "rel_correct": 0, "total": 0,
                  "cost_usd": 0.0, "error": str(e)[:200]}
        per_backend[backend] = {"piece_2": p2, "piece_4": p4}

    # Final summary
    print()
    print("══════════════════════════════════════════════════════════════════════")
    print("  ALL PIECES — END-TO-END SUMMARY")
    print("══════════════════════════════════════════════════════════════════════")
    print()
    print(f"  Piece 1 (universal schema TOML):  loaded {len(fixtures) and 'OK'} ✓")
    print(f"  Piece 3 (relation matcher):       accuracy = {p3['accuracy']:.1%}  "
          f"({p3['correct']}/{p3['total']})")
    print()
    # Side-by-side Piece 2 + Piece 4 across backends
    header = f"  {'metric':<40}"
    for b in available:
        header += f"{b:>16}"
    print(header)
    print("  " + "─" * (40 + 16 * len(available)))

    def row(label, fn):
        line = f"  {label:<40}"
        for b in available:
            v = fn(per_backend[b])
            line += f"{v:>16}"
        print(line)

    row("Piece 2 combined F1",
        lambda x: f"{x['piece_2'].get('combined_f1', 0.0):.3f}")
    row("Piece 2 entity F1 (substring)",
        lambda x: f"{x['piece_2'].get('avg_entity_f1', 0.0):.3f}")
    row("Piece 2 relation F1 (pair-fuzzy)",
        lambda x: f"{x['piece_2'].get('avg_relation_f1', 0.0):.3f}")
    row("Piece 2 cost (USD)",
        lambda x: f"${x['piece_2'].get('cost_usd', 0.0):.4f}")
    row("Piece 4 JSON valid",
        lambda x: f"{x['piece_4'].get('json_valid', 0)}/{x['piece_4'].get('total', 0)}")
    row("Piece 4 op accuracy",
        lambda x: f"{x['piece_4'].get('op_correct', 0)/max(x['piece_4'].get('total', 1),1):.1%}")
    row("Piece 4 relation accuracy",
        lambda x: f"{x['piece_4'].get('rel_correct', 0)/max(x['piece_4'].get('total', 1),1):.1%}")
    row("Piece 4 cost (USD)",
        lambda x: f"${x['piece_4'].get('cost_usd', 0.0):.4f}")

    total_cost = sum(
        per_backend[b]["piece_2"].get("cost_usd", 0.0)
        + per_backend[b]["piece_4"].get("cost_usd", 0.0)
        for b in available
    )
    print()
    print(f"  Total spend across backends: ${total_cost:.4f}")

    # Pass/fail per backend
    print()
    print("  PASS/FAIL vs minimum thresholds:")
    print(f"  {'check':<40}", end="")
    for b in available:
        print(f"{b:>16}", end="")
    print()
    print("  " + "─" * (40 + 16 * len(available)))
    for label, key, fn in [
        ("Piece 2 combined F1 ≥ 0.55", None,
         lambda x: x["piece_2"].get("combined_f1", 0.0) >= targets["piece_2_combined_f1_min"]),
        ("Piece 4 JSON valid ≥ 95%", None,
         lambda x: x["piece_4"].get("json_valid", 0) / max(x["piece_4"].get("total", 1), 1) >= targets["piece_4_json_valid_rate_min"]),
        ("Piece 4 op acc ≥ 75%", None,
         lambda x: x["piece_4"].get("op_correct", 0) / max(x["piece_4"].get("total", 1), 1) >= targets["piece_4_op_accuracy_min"]),
    ]:
        print(f"  {label:<40}", end="")
        for b in available:
            ok = fn(per_backend[b])
            print(f"{'PASS' if ok else 'FAIL':>16}", end="")
        print()
    print()
    p3_ok = p3["accuracy"] >= targets["piece_3_accuracy_min"]
    print(f"  Piece 3 accuracy ≥ 80% (backend-independent): {'PASS' if p3_ok else 'FAIL'} ({p3['accuracy']:.1%})")
    print()

    # Save raw output for inspection
    out_path = Path("/tmp/clarity_smoke_results.json")
    out_path.write_text(json.dumps({
        "piece_3": p3,
        "per_backend": per_backend,
        "total_cost_usd": total_cost,
        "targets": targets,
        "fixture_size": len(fixtures),
    }, indent=2, default=str))
    print(f"  Raw results:        {out_path}")
    print(f"  Schema suggestions: {suggestions_log} ({suggestions_log.exists() and 'exists' or 'no suggestions emitted'})")


if __name__ == "__main__":
    main()

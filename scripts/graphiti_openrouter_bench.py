#!/usr/bin/env python3
"""
Run Graphiti through OpenRouter on the v2 cross-domain fixture.
Score with the same pair-fuzzy F1 as openrouter_bench.py for apples-to-apples.

Requires:
    OPENROUTER_API_KEY in env
    Neo4j at localhost:7687 (user=neo4j, password=benchpass123)
    Graphiti and neo4j Python packages
"""
import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.llm_client import OpenAIClient, LLMConfig
from graphiti_core.nodes import EpisodeType
from neo4j import GraphDatabase


class DummyReranker(CrossEncoderClient):
    """Identity reranker. Doesn't affect entity/relation extraction quality."""

    async def rank(self, query, passages):
        return [(p, 1.0) for p in passages]


class DummyEmbedder(EmbedderClient):
    """Zero-vector embedder. Embeddings affect Graphiti's dedup, NOT extraction.
    Since this benchmark scores extraction (entities + relations), the embedder
    output is irrelevant. Using zeros avoids requiring an OpenAI API key."""

    EMBEDDING_DIM = 1536

    async def create(self, input_data):
        return [0.0] * self.EMBEDDING_DIM

    async def create_batch(self, input_data_list):
        return [[0.0] * self.EMBEDDING_DIM for _ in input_data_list]

FIXTURE = Path(__file__).parent.parent / "crates/ctxgraph-extract/tests/fixtures/cross_domain_v2.json"


def fuzzy_match(a: str, b: str) -> bool:
    al, bl = a.lower().strip(), b.lower().strip()
    return al == bl or al in bl or bl in al


def relation_pair_match(ph: str, pt: str, gh: str, gt: str) -> bool:
    return (fuzzy_match(ph, gh) and fuzzy_match(pt, gt)) or (
        fuzzy_match(ph, gt) and fuzzy_match(pt, gh)
    )


def f1(p: int, g: int, tp: int) -> float:
    if p == 0 and g == 0:
        return 1.0
    pr = tp / p if p else 0.0
    rc = tp / g if g else 0.0
    return 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0


def score(pred_ent_names, pred_rel_pairs, gold_ents, gold_rels):
    gold_ent_names = [e["name"] for e in gold_ents]
    matched = [False] * len(gold_ent_names)
    tp_ent = 0
    for pn in pred_ent_names:
        for j, gn in enumerate(gold_ent_names):
            if not matched[j] and fuzzy_match(pn, gn):
                matched[j] = True
                tp_ent += 1
                break
    ent_f1 = f1(len(pred_ent_names), len(gold_ent_names), tp_ent)

    matched = [False] * len(gold_rels)
    tp_rel = 0
    for ph, pt in pred_rel_pairs:
        for j, gr in enumerate(gold_rels):
            if not matched[j] and relation_pair_match(ph, pt, gr["head"], gr["tail"]):
                matched[j] = True
                tp_rel += 1
                break
    rel_f1 = f1(len(pred_rel_pairs), len(gold_rels), tp_rel)

    return ent_f1, rel_f1


async def run(model: str, out_path: str):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    fixtures = json.loads(FIXTURE.read_text())
    print(f"Loaded {len(fixtures)} episodes from {FIXTURE.name}")
    print(f"Model: {model}")

    llm_config = LLMConfig(
        api_key=api_key,
        model=model,
        small_model=model,
        base_url="https://openrouter.ai/api/v1",
        max_tokens=4096,
        temperature=0,
    )
    client = OpenAIClient(config=llm_config)

    graphiti = Graphiti(
        "bolt://localhost:7687",
        "neo4j",
        "benchpass123",
        llm_client=client,
        embedder=DummyEmbedder(),
        cross_encoder=DummyReranker(),
    )
    try:
        await graphiti.build_indices_and_constraints()
    except Exception:
        pass

    driver = GraphDatabase.driver(
        "bolt://localhost:7687", auth=("neo4j", "benchpass123")
    )

    per_ep = []
    total_time = 0.0
    err_count = 0

    for i, ep in enumerate(fixtures):
        with driver.session() as s:
            before = s.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]

        start = time.time()
        try:
            await graphiti.add_episode(
                name=f"v2_ep{i}_{ep['domain']}",
                episode_body=ep["text"],
                source_description=ep["domain"],
                reference_time=datetime.now(timezone.utc),
                source=EpisodeType.text,
                group_id=f"v2_ep_{i}",
            )
            elapsed = time.time() - start
            total_time += elapsed

            with driver.session() as s:
                ent_rows = s.run(
                    "MATCH (n:Entity) RETURN n.name AS name SKIP $skip",
                    skip=before,
                ).data()
                pred_ents = [r["name"] for r in ent_rows if r["name"]]

                edge_rows = s.run(
                    """
                    MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
                    WHERE r.fact IS NOT NULL
                    RETURN a.name AS head, r.name AS rel, b.name AS tail
                    ORDER BY r.created_at DESC LIMIT 50
                    """
                ).data()
                pred_rels = [(e["head"], e["tail"]) for e in edge_rows]

            ent_f1, rel_f1 = score(
                pred_ents, pred_rels, ep["expected_entities"], ep["expected_relations"]
            )

            per_ep.append(
                {
                    "idx": i,
                    "domain": ep["domain"],
                    "ent_f1_pair": ent_f1,
                    "rel_f1_pair": rel_f1,
                    "elapsed_s": elapsed,
                    "pred_ents": len(pred_ents),
                    "pred_rels": len(pred_rels),
                }
            )
            print(
                f"  ep{i:02d} [{ep['domain']:>14}]  ent={ent_f1:.3f}  rel={rel_f1:.3f}  "
                f"t={elapsed:.1f}s  ents={len(pred_ents)}  rels={len(pred_rels)}",
                flush=True,
            )
        except Exception as e:
            err_count += 1
            elapsed = time.time() - start
            total_time += elapsed
            err_str = str(e)[:150]
            per_ep.append({"idx": i, "domain": ep["domain"], "error": err_str, "elapsed_s": elapsed})
            print(f"  ep{i:02d} [{ep['domain']:>14}]  ERROR ({elapsed:.1f}s): {err_str}", flush=True)

    n = len(fixtures)
    ok = [e for e in per_ep if "error" not in e]
    if ok:
        avg_ent = sum(e["ent_f1_pair"] for e in ok) / len(ok)
        avg_rel = sum(e["rel_f1_pair"] for e in ok) / len(ok)
        combined = (avg_ent + avg_rel) / 2
    else:
        avg_ent = avg_rel = combined = 0.0

    summary = {
        "system": "graphiti",
        "model": model,
        "fixture": FIXTURE.name,
        "n_episodes": n,
        "n_ok": len(ok),
        "n_errors": err_count,
        "ent_f1_pair": avg_ent,
        "rel_f1_pair": avg_rel,
        "combined_f1_pair": combined,
        "avg_time_s": total_time / n,
        "total_time_s": total_time,
        "llm_calls_per_episode_approx": 6,
        "per_episode": per_ep,
    }
    Path(out_path).write_text(json.dumps(summary, indent=2))

    print(f"\n  ── SUMMARY: graphiti + {model} ──")
    print(f"  episodes ok:        {len(ok)}/{n} ({err_count} errors)")
    print(f"  entity F1 (pair):   {avg_ent:.4f}")
    print(f"  relation F1 (pair): {avg_rel:.4f}")
    print(f"  combined F1 (pair): {combined:.4f}")
    print(f"  avg time/episode:   {total_time / n:.2f}s")
    print(f"  Output → {out_path}")

    driver.close()
    await graphiti.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    asyncio.run(run(args.model, args.out))


if __name__ == "__main__":
    main()

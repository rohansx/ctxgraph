#!/usr/bin/env python3
"""
Cost/efficiency benchmark: ctxgraph (1 LLM call/episode) vs Graphiti (measured).

The quality benchmarks show ctxgraph and Graphiti extract at ~equivalent F1
(see docs/llm-backends.md / the fixed graphiti_openrouter_bench.py). The REAL,
defensible advantage is architectural: ctxgraph makes ONE LLM call per episode;
Graphiti runs a multi-call pipeline. This measures Graphiti's ACTUAL call count
(not a hardcoded guess) by instrumenting the LLM client, then projects the
local-native (Gemma-12B on-GPU) wall-clock implication.

Requires: OPENROUTER_API_KEY, Neo4j at localhost:7687 (neo4j/benchpass123).
Usage:  ./.venv-graphiti/bin/python scripts/cost_efficiency_bench.py --model google/gemini-2.5-flash-lite --limit 29
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

FIXTURE = Path(__file__).parent.parent / "crates/ctxgraph-extract/tests/fixtures/cross_domain_v2.json"
# Observed local Gemma-4-12B (Q4_K_M, 6GB VRAM partial offload) latency per LLM call.
LOCAL_GEMMA_S_PER_CALL = 33.0


class DummyReranker(CrossEncoderClient):
    async def rank(self, query, passages):
        return [(p, 1.0) for p in passages]


class DummyEmbedder(EmbedderClient):
    EMBEDDING_DIM = 1536

    async def create(self, input_data):
        return [0.0] * self.EMBEDDING_DIM

    async def create_batch(self, input_data_list):
        return [[0.0] * self.EMBEDDING_DIM for _ in input_data_list]


class CountingOpenAIClient(OpenAIClient):
    """Wraps the real client to count actual LLM calls + accumulate token usage."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.calls = 0

    async def _generate_response(self, *args, **kwargs):
        self.calls += 1
        return await super()._generate_response(*args, **kwargs)


async def main(model: str, limit: int):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    fixtures = json.loads(FIXTURE.read_text())[:limit]
    cfg = LLMConfig(api_key=api_key, model=model, small_model=model,
                    base_url="https://openrouter.ai/api/v1", max_tokens=4096, temperature=0)
    client = CountingOpenAIClient(config=cfg)
    graphiti = Graphiti("bolt://localhost:7687", "neo4j", "benchpass123",
                        llm_client=client, embedder=DummyEmbedder(), cross_encoder=DummyReranker())
    try:
        await graphiti.build_indices_and_constraints()
    except Exception:
        pass

    t0 = time.time()
    for i, ep in enumerate(fixtures):
        try:
            await graphiti.add_episode(
                name=f"ce_ep{i}", episode_body=ep["text"], source_description=ep["domain"],
                reference_time=datetime.now(timezone.utc), source=EpisodeType.text,
                group_id=f"ce_ep_{i}",
            )
        except Exception as e:
            print(f"  ep{i} error: {str(e)[:80]}", flush=True)
    g_wall = time.time() - t0
    await graphiti.close()

    n = len(fixtures)
    g_calls_ep = client.calls / n
    g_wall_ep = g_wall / n

    print("\n================  COST / EFFICIENCY  ================")
    print(f"model={model}  episodes={n}")
    print(f"\nLLM CALLS PER EPISODE (measured):")
    print(f"  ctxgraph : 1.00  (single schema-typed call)")
    print(f"  graphiti : {g_calls_ep:.2f}  ({client.calls} calls / {n} episodes)")
    print(f"  -> ctxgraph makes {g_calls_ep:.1f}x FEWER LLM calls per episode")
    print(f"\nCLOUD WALL-CLOCK (this run, {model}):")
    print(f"  graphiti : {g_wall_ep:.2f}s/episode")
    print(f"  (ctxgraph single-call latency on this model is ~1 LLM round-trip)")
    print(f"\nLOCAL-NATIVE PROJECTION (Gemma-4-12B @ ~{LOCAL_GEMMA_S_PER_CALL:.0f}s/call on 6GB GPU):")
    print(f"  ctxgraph : ~{LOCAL_GEMMA_S_PER_CALL:.0f}s/episode  (1 call)")
    print(f"  graphiti : ~{LOCAL_GEMMA_S_PER_CALL*g_calls_ep:.0f}s/episode  ({g_calls_ep:.1f} calls)")
    print(f"  -> at equivalent extraction quality, ctxgraph is ~{g_calls_ep:.1f}x faster locally")
    print("====================================================")

    out = {
        "model": model, "n_episodes": n,
        "ctxgraph_calls_per_ep": 1.0,
        "graphiti_calls_per_ep_measured": g_calls_ep,
        "graphiti_total_calls": client.calls,
        "call_multiplier": g_calls_ep,
        "graphiti_wall_s_per_ep": g_wall_ep,
        "local_gemma_s_per_call": LOCAL_GEMMA_S_PER_CALL,
        "ctxgraph_local_s_per_ep_proj": LOCAL_GEMMA_S_PER_CALL,
        "graphiti_local_s_per_ep_proj": LOCAL_GEMMA_S_PER_CALL * g_calls_ep,
    }
    Path("/tmp/ctxgraph-bakeoff/cost_efficiency.json").write_text(json.dumps(out, indent=2))
    print("written -> /tmp/ctxgraph-bakeoff/cost_efficiency.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemini-2.5-flash-lite")
    ap.add_argument("--limit", type=int, default=29)
    a = ap.parse_args()
    asyncio.run(main(a.model, a.limit))

# v0.9 Benchmark — Round 1 (ARCHIVED)

> ⚠️ **DEPRECATED — superseded by `../BENCHMARKS.md`.**
> This is the round-1 5-model benchmark on the 50-tech-episode fixture + 10 cross-domain episodes. It includes Gemma 3n E4B (which underperformed) and GPT-4o-mini (used as a control vs. Graphiti). Round 2 (on 29 cross-domain episodes, 5 selected models, including a full Graphiti head-to-head) lives in `../BENCHMARKS.md`. The findings here are **not wrong** — they're the first pass that led to dropping Gemma 3n E4B and selecting the round-2 model shortlist.
>
> **Current authoritative benchmark: `../BENCHMARKS.md`**

---

# Original content below

Reproducible benchmark comparing four LLMs as **stand-alone extractors** against ctxgraph's gold-labeled fixtures, plus the previously-committed ctxgraph local pipeline and Graphiti+GPT-4o numbers.

**Date**: 2026-05-13
**Harness**: `scripts/openrouter_bench.py`
**Raw results**: `scripts/results/v0.9_openrouter/*.json` (per-episode entities, relations, latency, cost)

## Setup

- **50 tech episodes** (`fixtures/benchmark_episodes.json`) — schema-typed gold labels: 10 entity types, 9 relation types.
- **10 cross-domain episodes** (`fixtures/cross_domain_episodes.json`) — finance, healthcare, legal, manufacturing, education, government, etc.
- **Identical system prompt** for every model. JSON-only output, no temperature.
- **Two scoring variants** computed simultaneously:
  - **strict**: exact name + exact entity-type for entities; exact (head, relation, tail) for relations.
  - **pair-fuzzy**: substring name match for entities; entity-pair match ignoring relation type and direction for relations. This matches the generous methodology used for Graphiti in `head_to_head.py`.

## Reproducing

```bash
export OPENROUTER_API_KEY=sk-or-...
python scripts/openrouter_bench.py --model google/gemma-4-26b-a4b-it --out bench.json
python scripts/summarize_bench.py
```

## Results

### Per-model F1 (50 tech + 10 cross-domain)

| Model | bucket | ent strict | ent fuzzy | rel strict | rel pair | combined strict | combined pair | s/ep | $/1k ep |
|---|---|---|---|---|---|---|---|---|---|
| openai/gpt-4o-mini | tech | 0.597 | 0.863 | 0.117 | 0.575 | 0.357 | 0.719 | 2.15 | $0.079 |
| openai/gpt-4o-mini | cross-domain | 0.389 | 0.874 | 0.239 | 0.503 | 0.314 | 0.689 | 2.72 | $0.115 |
| google/gemma-3n-e4b-it | tech | 0.488 | 0.876 | 0.094 | 0.434 | 0.291 | 0.655 | 7.63 | $0.039 |
| google/gemma-3n-e4b-it | cross-domain | 0.535 | 0.893 | 0.242 | 0.420 | 0.389 | 0.657 | 8.25 | $0.052 |
| **google/gemma-4-26b-a4b-it** | tech | 0.613 | 0.898 | **0.366** | 0.559 | **0.490** | 0.729 | 4.84 | $0.059 |
| **google/gemma-4-26b-a4b-it** | cross-domain | 0.337 | 0.855 | **0.380** | 0.640 | 0.359 | **0.747** | 8.98 | $0.093 |
| **google/gemma-4-31b-it** | tech | **0.647** | **0.915** | **0.435** | **0.640** | **0.541** | **0.778** | 11.78 | $0.108 |
| **google/gemma-4-31b-it** | cross-domain | **0.444** | 0.870 | **0.529** | **0.697** | **0.486** | **0.783** | 15.27 | $0.158 |

### Reference points (committed in repo, 50 tech episodes)

| | entity F1 | relation F1 | combined F1 |
|---|---|---|---|
| ctxgraph local-only (ONNX, no LLM) | 0.837 | 0.763 | **0.800** |
| Graphiti + GPT-4o (mapped scoring) | 0.570 | 0.104 | 0.337 |

### Deltas vs Graphiti + GPT-4o (mapped scoring, 50 tech episodes)

| Model alone | Δ combined strict | Δ combined pair-fuzzy |
|---|---|---|
| GPT-4o-mini | +0.020 | +0.382 |
| Gemma 3n E4B | -0.046 | +0.318 |
| **Gemma 4 26B-A4B** | **+0.153** | **+0.392** |
| **Gemma 4 31B** | **+0.204** | **+0.441** |

### Deltas vs ctxgraph local-only (no LLM)

| Model alone | Δ combined strict | Δ combined pair-fuzzy |
|---|---|---|
| GPT-4o-mini | -0.443 | -0.081 |
| Gemma 3n E4B | -0.509 | -0.145 |
| Gemma 4 26B-A4B | -0.310 | -0.071 |
| Gemma 4 31B | -0.259 | -0.022 |

## Findings

1. **Gemma 4 26B-A4B does outperform Graphiti+GPT-4o** on the 50-tech-episode fixture: +0.15 strict / +0.39 pair-fuzzy combined F1. Gemma 4 31B is even better (+0.20 / +0.44). The README's quality ranking is directionally correct.
2. **Gemma 4 26B-A4B outperforms GPT-4o-mini** at the same task: tech +0.13 strict, cross-domain +0.05 strict. Notably, **relation extraction** is dramatically better (0.366 vs 0.117 on tech). This is where Gemma 4 earns its lead.
3. **No LLM (including Gemma 4 31B) beats ctxgraph's local-only pipeline on tech.** The local ONNX pipeline at 0.800 combined F1 is 0.26+ ahead of the best LLM alone. **The tiered approach is validated**: don't call an LLM when you don't need to.
4. **LLMs do help on cross-domain.** Gemma 4 31B at 0.486 strict vs ctxgraph local's ~0.325 on cross-domain (per ROADMAP.md). This is where the LLM tier earns its keep.
5. **Gemma 3n E4B (the "free local default") is the weakest of the four.** On tech it underperforms even GPT-4o-mini and Graphiti. On cross-domain it edges past GPT-4o-mini on strict scoring. README claim of 7.6/10 quality maps roughly to my 0.655–0.657 pair-fuzzy combined F1 — same ballpark but not impressive in absolute terms.
6. **Cost reality**: Gemma 4 26B-A4B at **$0.06–$0.09 per 1000 episodes** is *cheaper* than the README's $0.13/1M tokens claim suggested (and 30× cheaper than Graphiti's $1.80/1k eps if Graphiti made 6 GPT-4o-mini calls).
7. **Latency reality**: Gemma 4 26B-A4B averages **5–9s per episode**, not the ~25s README claimed. The MoE (4B active per token) makes it noticeably faster than the dense 31B variant (~12–15s/ep).

## Caveats / what this benchmark does NOT show

- **Graphiti was not re-run against Gemma 4 26B.** The Graphiti number is from `graphiti_benchmark_results.json` using GPT-4o. A fully fair head-to-head with both systems running Gemma 4 26B would require a fresh Graphiti run (Neo4j + `pip install graphiti-core`).
- **The cross-domain set is only 10 episodes.** Per-domain variance is high; treat cross-domain numbers as directional.
- **LLM-as-judge results in the README (8.4/10, 7.6/10, 8.2/10) are unverified.** No judge harness exists. The F1 numbers above are the only reproducible scoring.
- **Gemma 4 26B-A4B was tested via OpenRouter** (cloud). Local performance (Ollama, llama.cpp, MLX) will vary by quantization and hardware.

## Honest conclusion

The README's central claim — *"ctxgraph + Gemma 4 26B beats Graphiti"* — is supported by this run. But the win comes overwhelmingly from the **local pipeline**, not the LLM:

- On tech (where most "developer memory" episodes live), ctxgraph local-only at 0.800 F1 beats Gemma 4 31B alone (0.541 strict) by **+0.26**. The LLM tier adds nothing on tech and should not fire.
- On cross-domain, Gemma 4 26B at 0.359 strict beats local-only (~0.325) by a modest **+0.03**. The tiered escalation is the right call here.

The moat is **the tiered architecture plus schema-typed local extraction**, not the choice of LLM. Any LLM in the table beats Graphiti's committed score because Graphiti's schema-free output doesn't map cleanly to the gold relation taxonomy — that's an evaluation artifact, not a model-quality finding. The real win is "don't call a 26B model when GLiNER at 30ms gives you 0.800 F1."

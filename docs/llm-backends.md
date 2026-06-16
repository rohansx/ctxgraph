# LLM backends for extraction

`ctxgraph`'s LLM extraction path (`LlmExtractor`, used by the v0.3 universal
pipeline) is a plain **OpenAI-compatible HTTP client**. It does not bind to any
particular runtime — it POSTs to `/v1/chat/completions` and parses the JSON
back. That means any of the following are drop-in backends, selected entirely by
environment variables.

## Configuration

`LlmExtractor::from_env()` resolves a backend in this order:

1. **Explicit (Tier 0)** — `CTXGRAPH_LLM_KEY` set → use `CTXGRAPH_LLM_URL` /
   `CTXGRAPH_LLM_MODEL` verbatim. Highest priority; this is how you force a
   specific local model.
2. **Ollama auto-detect (Tier 1)** — probes `localhost:11434`, picks a preferred
   small model. Free, private, zero config.
3. **Cloud (Tier 2)** — `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `OPENROUTER_API_KEY`.

| Variable | Purpose | Default |
|---|---|---|
| `CTXGRAPH_LLM_KEY` | API key (any non-empty string for local servers) | — |
| `CTXGRAPH_LLM_URL` | Full `/v1/chat/completions` URL | provider-specific |
| `CTXGRAPH_LLM_MODEL` | Model name sent in the request | `gpt-4o-mini` |
| `CTXGRAPH_LLM_TIMEOUT` | Per-request timeout, seconds | 60 (Ollama 120) |
| `CTXGRAPH_NO_LLM=1` | Disable the LLM path entirely | — |
| `CTXGRAPH_NO_OLLAMA=1` | Skip Ollama auto-detect | — |

> Large local models partly offloaded to CPU can take **minutes** per episode.
> Set `CTXGRAPH_LLM_TIMEOUT=600` so the client doesn't give up early.

## Backend A — Ollama (default, easiest)

```bash
ollama pull hf.co/yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF:Q4_K_M
env CTXGRAPH_LLM_KEY=ollama \
    CTXGRAPH_LLM_URL=http://localhost:11434/v1/chat/completions \
    CTXGRAPH_LLM_MODEL='hf.co/yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF:Q4_K_M' \
    CTXGRAPH_LLM_TIMEOUT=600 \
  cargo test -p ctxgraph-extract --test llm_cross_domain_test -- --ignored --nocapture
```

## Backend B — llama.cpp `llama-server` (more control)

Use this when you want direct control over GPU offload (`-ngl`), context size,
flash attention, or grammar-constrained output. `scripts/run-llama-server.sh`
serves a GGUF — by default **reusing the blob Ollama already downloaded**, so
there's no second copy on disk:

```bash
# terminal 1 — serve (resolves the GGUF from the ollama model tag)
scripts/run-llama-server.sh                       # default coder model, port 8080
# or: GGUF=/path/to/model.gguf NGL=24 PORT=8080 scripts/run-llama-server.sh

# terminal 2 — point ctxgraph at it
env CTXGRAPH_LLM_KEY=llamacpp \
    CTXGRAPH_LLM_URL=http://127.0.0.1:8080/v1/chat/completions \
    CTXGRAPH_LLM_MODEL=local \
    CTXGRAPH_LLM_TIMEOUT=600 \
  cargo test -p ctxgraph-extract --test llm_cross_domain_test -- --ignored --nocapture
```

The script's header documents how to build a CUDA `llama-server` into
`~/.local/bin` (no sudo needed if the CUDA toolkit is installed).

## Testing a model: the harness

`tests/llm_cross_domain_test.rs` exercises `LlmExtractor` directly (not the
GLiNER ONNX pipeline) so you can score any backend:

- `llm_smoke_test` — 3 off-fixture real-world snippets; asserts the model
  returns usable extraction JSON at all.
- `llm_cross_domain_hard_test` — per-domain entity + relation F1 across the 6
  `cross_domain_episodes.json` domains. Exploratory (no threshold asserted);
  never panics on bad model output — extraction errors are counted and reported.

## Reference: local-model comparison

Same harness, fixtures, and fuzzy-F1 scorer across all 10 cross-domain episodes
on a 6 GB-VRAM laptop (RTX 4050, partial CPU offload via Ollama):

| Model | Entity F1 (name) | Entity F1 (strict) | Relation F1 | Combined | Errors | Wall time |
|---|---|---|---|---|---|---|
| gemma-4-12B-coder (Q4_K_M) | 0.816 | 0.492 | 0.398 | 0.607 | 0/10 | ~20 min |
| qwen2.5:7b-instruct (Q4_K_M) | 0.850 | 0.383 | 0.263 | 0.557 | 0/10 | ~2.4 min |
| Gemma-4-12B-OBLITERATED (Q4_K_M) | 0.083 | 0.067 | 0.100 | 0.092 | 9/10 | ~105 min |

Takeaways:

- **Entity recall is the strong suit** of every working model; **relation F1 and
  strict (name+type) F1 are the weak spots** — partly real model error, partly
  the scorer requiring exact head/tail entity-name matches.
- **The coder fine-tune did not beat a stock 7B instruct** in a way that justifies
  8× the latency: qwen2.5:7b matches its entity recall and fits entirely in VRAM.
  The coder's code bias even leaks `snake_case` entity names that break relation
  matching (worst on the finance domain).
- **Abliterated models are unusable here** — `Gemma-4-12B-OBLITERATED` returned
  empty/unparseable output on 9/10 episodes. Abliteration strips the
  instruction-following that structured-JSON extraction relies on.

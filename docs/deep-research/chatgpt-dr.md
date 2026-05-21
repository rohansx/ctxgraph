# ctxgraph — Synthesis: The Final Brief Before HN

**Generated 2026-05-13. Synthesis of four research passes:** pass 2 (architectural baseline + 12-week plan), pass 3 (adversarial fact-check), External-A (mid-2026 model + provider sweep), External-B (hardware/caching/encoder architecture deep dive). Every claim cross-referenced; conflicts resolved with the most defensible source called out.

---

## 0. What changed between passes

Three concrete swaps the synthesis forces vs. pass 2:

1. **GLiNER2 replaces the GLiNER + GLiREL Tier-1 split.** Pass 2 mentioned GLiNER2 in passing; External-A and External-B both make it the headline local-tier swap. Pass 3 flagged GLiNER2 as needing partial verification — verified now: 205M params, single forward pass for NER + RE + hierarchical JSON, F1 0.590 on CrossNER vs GPT-4o 0.599, Apache-2.0 (`fastino-ai/GLiNER2`).
2. **NuExtract is the local extractive decoder, not just a Tier-2 model.** Its negative-sampling training (empty-string outputs for absent facts) eliminates an entire class of JSON-validation bugs. External-B's training-format diagram is the clearest source.
3. **Host-memory prompt caching gets first-class treatment, not a footnote.** External-B's measurement: 4.2s → 0.3s prefill TTFT on RTX 3090 with `--cram 256` and `keep_alive: -1` on a static ~8k-token prompt. This is the single biggest local-latency win and pass 2 mentioned it only generically.

The 12-week plan from pass 2 still stands. Two weeks of work shift order: weeks 1–2 now include GLiNER2 swap and host-memory caching, both of which are tractable before any fine-tune.

---

## 1. The headline benchmark — bulletproofed

The number you lead with on HN is unchanged: **same model, ctxgraph's single-call prompt vs Graphiti's 6-call pipeline = +0.227 combined F1, +5.8× relation F1**.

Pass 3's audit confirms this is your strongest defensible claim — it doesn't depend on model choice, schema bias, or LLM-as-judge methodology. The 0.687 (ctxgraph) and 0.460 (Graphiti) numbers are on your own 29-episode, 25-domain fixture; publish the fixture with the launch and invite community PRs.

What External-A adds: a sharper framing of *why* multi-call fails — "the model emits dense, free-form verbal constructions (~50 unaligned relation tags per episode such as `WAS_REWRITTEN_FROM` or `INTEGRATES_WITH`) ... [which] fail to resolve against target ontological categories." This is good launch-post material because it explains the mechanism, not just the metric.

What pass 3 corrected: drop two ancillary claims that won't survive `grep`:

- ❌ RL-Struct "38% less peak VRAM than PPO" → **40%** per arXiv 2512.00319 abstract.
- ❌ "Speculative decoding hurts at 4–14B" cited via three papers → only arXiv 2509.04474 actually supports this, and only for the specific Qwen3-8B + sub-1B-draft case (0.87× slowdown). EAGLE-3 still gets 2.91× on Qwen3-8B. Reframe as "naïve model-based spec sampling with a sub-1B draft is often net-negative on 8B targets; measure before shipping."
- ❌ "A-MEM SOTA across six base models on LoCoMo" → soften to "consistent improvement over baselines across six foundation models" (the paper's actual claim).
- ❌ Re-DocRED "34.7 avg triples per test doc" → **34.9** (official GitHub stats).
- ❌ ASER 2.0 "15 discourse relation types" → **14 PDTB discourse + 1 co-occurrence**.

---

## 2. The new architecture (synthesizing all four passes)

```
  Episode text in
        │
        ▼
  ┌──────────────────────────────────────────────────────┐
  │  Tier 1 — GLiNER2 ONNX                               │
  │  (NER + typed RE + hierarchical JSON, single pass)   │
  │  ~205M params, CPU-runnable, <20ms                   │
  │  Schema-aware: emits ctxgraph's 10/9 taxonomy        │
  └──────────────────────────────────────────────────────┘
        │
        ▼
  [Confidence gate]
   - entity density < 1.5/10 words
   - avg confidence < 0.4
   - <60% schema coverage
   - complexity markers (@, v2, ::, outage)
        │
        ▼
  ┌──────────────────────────────────────────────────────┐
  │  Tier 2 — Local Extractive Decoder (Ollama or        │
  │  mistral.rs embedded)                                │
  │  Default model selector (auto-detect free VRAM):     │
  │    < 4 GB → GLiNER2 only                             │
  │    4-8 GB → NuExtract 2.0-2B  (Apache 2.0)           │
  │    8-16 GB → NuExtract 2.0-4B or Qwen3-8B-LoRA       │
  │    16-24 GB → Hermes-4-14B-LoRA Q4_K_M               │
  │  Host-memory prompt cache (--cram 256, keep_alive=-1)│
  └──────────────────────────────────────────────────────┘
        │
        ▼
  ┌──────────────────────────────────────────────────────┐
  │  Tier 2.5 — Cerebras free tier (Qwen3-32B until      │
  │  Feb 2026 deprecation, then gpt-oss-120B)            │
  │  1M tokens/day, 30 RPM                               │
  └──────────────────────────────────────────────────────┘
        │
        ▼
  ┌──────────────────────────────────────────────────────┐
  │  Tier 3 — DeepInfra paid (Gemma-4-26B-A4B)           │
  │  $0.07/M in + $0.34/M out (verified by External-A)   │
  │  CloakPipe PII stripping pre-call                    │
  └──────────────────────────────────────────────────────┘
        │
        ▼
  ┌──────────────────────────────────────────────────────┐
  │  Tier 4 (offline, nightly) — Graph Judge             │
  │  arXiv 2411.17388 — binary keep/reject on triples    │
  │  ~1.5B model fine-tuned on (text, triple, gold) pairs│
  └──────────────────────────────────────────────────────┘
        │
        ▼
  Bi-temporal SQLite + FTS5 + sqlite-vec
```

Why this structure:
- **Tier 1 (GLiNER2)** kills the GLiNER + GLiREL split. One model, schema-aware, CPU-runnable, the encoder forward pass produces typed RE natively.
- **Tier 2 (extractive decoder)** uses negative-sampling-trained models so empty fields stay empty — JSON validation becomes trivial.
- **Tier 2.5 (Cerebras free)** is the new finding. Pass 1 surfaced it; pass 3 confirmed 1M tokens/day is genuine for development, with the Feb 2026 Qwen3-32B deprecation flagged.
- **Tier 3 (DeepInfra Gemma-4-26B-A4B)** replaces OpenRouter gpt-4o-mini as paid fallback. Pass 3 caught that DeepInfra Qwen3-32B is split-priced ($0.08 in / $0.28 out, $0.13 blended), but External-A independently confirms Gemma-4-26B-A4B at $0.07 in / $0.34 out — that's the model to default to, not the Qwen3.
- **Tier 4 (Graph Judge)** is the offline quality pass; ship it as a nightly cron, not online.

---

## 3. The fine-tuning recipe — fully consolidated

The dual-LoRA plan from pass 2 holds. External-A confirms it explicitly: *"Ship an optional `ctxgraph fine-tune` command (LoRA on Qwen3-14B or Gemma4-E4B using your own episodes + gold labels). One-click domain IE model in <16 GB. This is the biggest quality lever."*

### Decision

Train **two LoRAs** simultaneously on the same dataset:

| Model | Use case | License | RTX 4050 fit? |
|---|---|---|---|
| **Hermes-4-14B** (Qwen3-14B base) | Cloud + 16GB+ laptops | Apache 2.0 | Q4_K_M = ~9 GB, doesn't fit 6 GB native |
| **Qwen3-8B** | Truly local, RTX 4050 | Apache 2.0 | Q4_K_M ~5 GB, fits |

Pass 3's correction: Hermes 4 14B was trained on **~19B tokens** (the SFT corpus for the 14B model per arXiv 2508.18255), not the ~60B figure on the HF card (which is the full series across 14B/70B/405B). Use 19B when reasoning about distillation cost-equivalence.

### Synthetic data

Same plan as pass 2, refined:
- 12,000 short passages (200–600 tokens) from CC-News, Wikipedia abstracts, arXiv abstracts, READMEs, Stack Exchange, OpenReview, Reddit — domain-balanced across the 25 fixture domains.
- Two-stage labeling with Claude Opus 4.5 / GPT-5 using `distilabel` (pass 3's NEW2 found this is the right tool — declarative pipeline DSL, Apache 2.0, `TextGenerationToJSON` built-in).
- Consistency filter: keep pairs where Pass A ↔ Pass B agree on ≥60% of gold pairs (UniversalNER methodology, verified).
- 5% adversarial slice: 250 examples with deliberate contradictions in adjacent sentences, used to train `invalidates:` output (bi-temporal extraction prompt — see §5).
- Final 8,000 examples after filtering, ~9.6M tokens before epochs.

### Training config (Together AI, Axolotl-compatible YAML)

```yaml
base_model: NousResearch/Hermes-4-14B  # second pass: Qwen/Qwen3-8B
tokenizer_type: AutoTokenizer
trust_remote_code: true

datasets:
  - path: ./data/ctxgraph_synth_v1.jsonl
    type: completion
    field: text
sequence_len: 4096
sample_packing: true
val_set_size: 0.05

adapter: lora
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

optimizer: adamw_torch_fused
learning_rate: 1.0e-4
lr_scheduler: cosine
warmup_ratio: 0.03

micro_batch_size: 4
gradient_accumulation_steps: 4  # effective BS 16
num_epochs: 3
gradient_checkpointing: true
flash_attention: true
bf16: true

evals_per_epoch: 4
save_strategy: steps
save_steps: 200
save_total_limit: 3
seed: 1337
```

### Cost — verified numbers

Tokens: 8,000 × 1,200 × 3 epochs = 28.8M training + ~1.9M eval = ~30.7M total.

| Provider | Rate | Two-LoRA total | Wall-clock |
|---|---|---|---|
| **Together AI** (≤16B LoRA) | $0.48 / 1M tokens (pass 3 verified) | **~$30** | ~4–8h |
| **RunPod H100 PCIe** | $2.39/hr official, pass 3 corrected from $1.99 | **~$20–30** | ~6–10h + setup |
| **Modal H100 serverless** | ~$3.25/hr | **~$30–40** | ~5–6h |

Pass 3 caveat: Together's $0.48/M figure assumes you pay for eval tokens too. Add 5–10% buffer.

### Phase 2 — GRPO

After Phase 1 SFT converges, do a Phase 2 GRPO pass on the 5% adversarial slice. RL-Struct's reward function adapted:

```
r_total = 0.20·r_syntax + 0.20·r_schema + 0.30·r_pair_fuzzy_F1 + 0.30·r_entity_F1
```

Where (pass 3 verified): RL-Struct achieves 89.7% structural accuracy / 92.1% JSON validity / **40%** (not 38%) less peak VRAM than PPO. Skip the ">80% with 1000 samples" claim — pass 3 couldn't verify it.

---

## 4. Cloud routing — final numbers

Pass 3 caught several pricing errors. The verified routing as of May 2026:

| Tier | Provider | Model | Price | Notes |
|---|---|---|---|---|
| **2.5 Dev** | Cerebras free | Qwen3-32B | $0 up to 1M tok/day | Deprecating Qwen3-32B Feb 2026 → migrate to gpt-oss-120B |
| **2.5 Dev alt** | Groq free | Llama 3.3 70B | $0 up to 1K req/day | 30 RPM cap, 14.4K on 8B models |
| **3 Paid default** | DeepInfra | Gemma-4-26B-A4B | **$0.07 in / $0.34 out** | External-A verified; same model family as Tier 2 = clean cloud/local parity |
| **3 Paid alt cheap** | DeepInfra | Qwen3-32B | $0.08 in / $0.28 out | Pass 3 corrected from "flat $0.08" |
| **3 Paid high-quality** | OpenRouter | Hermes 4 70B | $0.13 in / $0.40 out | Still your IE quality leader |
| **3 Premium tooling** | Together AI | Llama 3.3 70B | $0.88/$0.88 | Best when you need fine-tuning on the same platform |

**Cost per 1k episodes (800 tokens, 600 in / 200 out):**

- Cerebras free: $0 up to ~1,250 eps/day
- DeepInfra Gemma-4-26B-A4B: 0.6 × $0.07 + 0.2 × $0.34 = **$0.11 / 1k eps**
- Old default (OpenRouter gpt-4o-mini): ~$0.24 / 1k eps

So the swap is **~2.2× cheaper**, not the 3× pass 2 quoted. Still a clean win.

---

## 5. The bi-temporal prompt moat

Pass 2 introduced this; External-B added critical detail about how to *implement* it on-disk.

**Single-call schema, bi-temporally aware**:

```json
{
  "events": [
    {"id": "e1", "type": "deploy", "time": "...", "location": "..."}
  ],
  "entities": [
    {"id": "x1", "name": "Vernon CMS", "type": "system"}
  ],
  "relations": [
    {"head": "x1", "relation": "depends_on", "tail": "x2"}
  ],
  "invalidates": ["edge_id_123", "edge_id_456"],
  "confidence": 0.87
}
```

The novel piece: **the LLM emits `invalidates:` directly**, not as a separate post-hoc pass. The current-facts context is bounded by retrieving top-K facts touching each entity mentioned in the episode (5–10 facts × N entities, well within 4k context).

External-B's contribution: enforce **prefix isolation** so prompt caching works. The schema/ontology/few-shot block stays static; the dynamic episode text and current-facts retrieval go at the *end* of the prompt sequence. Byte-for-byte prefix match is required for cache hits.

This is the part competitors literally cannot copy without retraining their models.

---

## 6. Host-memory prompt caching — production setup

External-B's measurement is the most rigorous I've seen. The setup that gets you the 93% TTFT reduction:

### llama.cpp / llama-server config

```bash
llama-server \
  --model ./models/qwen3-8b.q4_k_m.gguf \
  --ctx-size 32768 \
  --np 4 \
  --cram 256 \
  --flash-attn \
  --system-prompt-file ./ctxgraph_schema.txt \
  --debug-slot
```

Key params:
- `--cram 256` — 256 MB host-memory cache for pre-computed KV blocks
- `--system-prompt-file` — static schema/ontology block
- `--flash-attn` — required for stable prefix caching

### Ollama equivalent

```json
{
  "model": "qwen3:8b",
  "keep_alive": "-1",
  "options": { "num_ctx": 16384 },
  "messages": [...]
}
```

`keep_alive: -1` is the critical bit — without it, the model unloads on idle and the KV cache is purged.

### Measured result

8,192-token system instruction on RTX 3090:
- Stateless: 4.2s prefill per request
- With cache: 0.3s on subsequent requests
- Memory overhead: +1.9 GB host RAM

### Gotcha — Sliding Window Attention

Some Qwen variants use SWA, which conflicts with static KV caches. Force global attention with `--override-kv` to disable sliding-window restrictions. External-B's note.

For ctxgraph's ~800-token system prompt this is straightforward to enable by default whenever Ollama or llama.cpp is detected. Combined with the Tier 1 GLiNER2 swap and the Tier 2 NuExtract default, the realistic local-tier latency target drops from your current 17 s/ep to **3–5 s/ep**.

---

## 7. The competitive landscape — finalized

Pass 3 surfaced four 2026 competitors pass 2 missed entirely. Final positioning table:

| System | Position | Beats ctxgraph at | Loses to ctxgraph at |
|---|---|---|---|
| **Graphiti / Zep** | Production temporal KG, ~20K stars | Brand recognition, real-time updates | 6-call cost, free-form verbs, no local-only mode |
| **HippoRAG 2** (ICML'25) | Academic SOTA agent memory | Personalized PageRank for associative retrieval | Still LLM-per-episode for OpenIE, no temporal model |
| **Anthropic Memory tool** (Mar 2026) | Client-side `/memories` filesystem | Native Claude API integration | Not a KG, just a filesystem; locked to Claude |
| **Anthropic Managed Memory** (Apr 2026 beta) | Managed filesystem stores | Enterprise-managed, exportable | Locked to Claude, no schema-typed extraction |
| **Letta** (ex-MemGPT) | OSS three-tier memory | Active community, Apache 2.0, 74% LoCoMo | Python deps, no typed extraction |
| **Mem0** ($23.9M raised) | Personalized memory, 47.8K stars | Mature SDK, fast retrieval | Not a real KG (separate clusters), low typed F1 |
| **Mastra Observational Memory** (Feb 2026) | OSS LongMemEval leader | 84.23% gpt-4o, 94.87% gpt-5-mini | Multi-call, no schema-typed local tier |
| **Cognee** | Hierarchical graphs | HotPotQA correctness 0.93 | Python deps, GPU embedder |
| **EverMemOS** (Jan 2026) | Three-phase, 83% LongMemEval | High HaluMem score | Neo4j + GPU |
| **LinearRAG** (ICLR'26) | Relation-free graph build | Zero LLM tokens for indexing | Drops typed relations entirely — wrong tradeoff for "what depends on X" queries |
| **AutoSchemaKG** (HKUST) | Dynamic schema induction | 92% schema alignment at web scale | Not incremental, batch-only |

ctxgraph's unique intersection: **single-binary + SQLite-only + no-Python + no-Neo4j + no-GPU + single-call schema-typed + bi-temporal + truly local**. No one else hits all 8. That's the positioning sentence for HN.

---

## 8. Benchmarks worth running before launch

Pass 2 said run LongMemEval-S. Pass 3 added: LoCoMo + LongMemEval are saturated at 85–95% by the 2026 leaders; they no longer discriminate.

Final benchmark plan:

| Benchmark | Why run it | Defensibility |
|---|---|---|
| **Your 29-ep fixture** | The ground-truth headline result | Strong — publish the fixture |
| **Re-DocRED subset** | Standard document-level RE benchmark | Strong if you map your schema correctly |
| **LongMemEval-S** | What Zep cites; comparable to Graphiti | Medium — risk of comparison to saturated leaders |
| **BEAM-1M** | 1M-token scale where Python-based competitors fall over | Highest differentiation |
| **HaluMem** | Hallucination-in-memory | Underused; differentiates |
| **EverMemBench** (arXiv 2602.01313) | Multi-party, 1M+ token, 2,400 QA pairs | New, designed to break saturated systems |

External-A noted: *"Add DocRED-style scoring to your benchmark suite (pair-fuzzy already close). It survives HN scrutiny."* Confirmed.

Pass 3's hostile-reader prep: pre-empt the LoCoMo controversy in your eval methodology section. The Zep–Mem0 "Lies, Damn Lies & Statistics" debate is well-known; just state your methodology explicitly and link your scoring code.

---

## 9. The 10-move breakthrough plan — re-ranked after synthesis

| # | Move | Effort | Source consensus | Ship in |
|---|---|---|---|---|
| 1 | GLiNER2 swap — replace GLiNER+GLiREL in Tier 1 | ~1 week | Pass 2 ✓ External-A ✓ External-B ✓ | v0.3 |
| 2 | Host-memory prompt caching default + `--override-kv` for SWA models | 2 days | External-B explicit | v0.3 |
| 3 | NuExtract 2.0-4B as Tier 2 default; auto-detect VRAM tier | 2 days | All passes ✓ | v0.3 |
| 4 | Bi-temporal `invalidates:` in single-call extraction prompt | 1 week | Pass 2, sharpened by External-B prefix isolation | v0.3 |
| 5 | DeepInfra Gemma-4-26B-A4B as cloud Tier 3 default (replace OpenRouter gpt-4o-mini) | 1 day | All passes ✓ | v0.3 |
| 6 | Add `Event` node + 3 event relations to schema (13/12 total) | 3 days | Pass 2, AutoSchemaKG-supported | v0.3 |
| 7 | Train Hermes-4-14B + Qwen3-8B LoRAs on 8k synthetic dataset | 1 weekend, ~$30 | All passes ✓ | v0.4 |
| 8 | Embedded inference via mistral.rs (single-binary moat) | 2 weeks | Pass 2, External-A confirms | v0.4 |
| 9 | A-MEM-style append-only memory notes | 2 weeks | Pass 2; pass 3 confirmed MIT | v0.4 |
| 10 | Graph Judge nightly offline pass | 1 week | Pass 2, pass 3 deferred F1 verification | v0.5 |

The ordering shift vs pass 2: **moves 1 and 2 (GLiNER2 + prompt caching) jump to weeks 1–2**. Both are low-risk, high-impact, and don't require fine-tuning. Together they likely take the local-tier benchmark to a place where the LoRA in weeks 3–4 is icing rather than the only headline.

---

## 10. The 12-week plan — updated with synthesis

| Week | Deliverable | Source of confidence | Verifiable |
|---|---|---|---|
| **W1** | GLiNER2 wired into Tier 1; old GLiNER+GLiREL retired; re-run 29-ep fixture | All passes ✓ | Combined F1 delta vs pass-2 baseline |
| **W2** | Host-memory caching defaults + Ollama `keep_alive=-1` + `--override-kv` for SWA; NuExtract 2.0-4B as Tier 2 default; VRAM autodetect | External-B explicit | Per-ep latency before/after |
| **W3** | Bi-temporal `invalidates:` extraction prompt + Event node + 3 event relations | Pass 2 + External-A | 1,000-ep synthetic temporal-conflict eval |
| **W4** | DeepInfra Gemma-4-26B-A4B Tier 3; OpenRouter retired as default; updated README cost claims | Pass 3 verified pricing | Cost trace from real ingest |
| **W5** | 8k synthetic dataset generated via distilabel + Claude Opus 4.5 | Pass 2, pass 3 confirmed tool | jsonschema-validates 100%; manual spot-check 30 rows |
| **W6** | Two-LoRA train run on Together AI; HF model cards published | Pass 2 + External-A | Together cost logged; replication snippet works |
| **W7** | New evals: 29-ep fixture + Re-DocRED subset + LongMemEval-S + BEAM-1M (if time) + HaluMem | Pass 3 added BEAM/HaluMem | CSVs in repo |
| **W8** | **HN launch v0.3**: "ctxgraph 0.3 — single-binary Rust KG, +22 F1 over Graphiti same model, runs on 6GB GPU" | All passes ✓ | HN front page or not; survives `grep` |
| **W9** | mistral.rs embedded inference spike; `cargo install ctxgraph` includes the LLM | Pass 2 + External-A | Local extraction round-trip in pure Rust binary |
| **W10** | A-MEM `memory_notes` + append-only evolution; Graph Judge nightly offline pass | Pass 2 + pass 3 deferred items resolved | LongMemEval-S accuracy ≥ 75% |
| **W11** | MCP-as-resource endpoint; Claude Code integration demo; awesome-mcp-servers PR | Pass 3 NEW10 | Demo video, install in <60s |
| **W12** | **HN launch v0.4** + arXiv preprint: "ctxgraph — embedded LLM in Rust, A-MEM memory, GRPO-tuned extractor" | Pass 2 ✓ | Preprint up; reproduce-from-zero CI green |

---

## 11. Skeptically-test-before-claiming (pass 3 hostile-reader prep)

Read this list before writing the launch post:

1. **AutoSchemaKG "92% schema alignment"** — verified (pass 3). "90% content preservation vs 70% for entity-only" is a paraphrase you can't source; **don't cite it**.
2. **NuExtract 2.0 PRO "+9 F-Score over GPT-4.1"** — verified, but applies to the **closed PRO API** only. The open 8B tops out at 73 F-Score on the same benchmark.
3. **MIRIX 85.4% LoCoMo** — verified, but uses **gpt-4.1-mini**, not gpt-4o.
4. **Hindsight 91.4% LongMemEval** — verified, but requires a "larger backbone"; the open 20B backbone is 83.6%/85.67%.
5. **EmergenceMem 86%** — "EmergenceMem Internal" is **not publicly reproducible**. Public configs: 79–82.4%. Cite the public number.
6. **Zep LongMemEval 71.2%** — paper number with GPT-4o. The 72.27% is the blog. Independent reproduction (Gamgee 2026) reports 63.8%.
7. **Hermes 4 14B "60B tokens"** is wrong context — that's the full Hermes 4 series. The **14B SFT was 19B tokens**.
8. **DeepInfra Qwen3-32B "$0.08 flat"** is wrong — split-priced $0.08 in / $0.28 out, $0.13 blended. **DeepInfra Gemma-4-26B-A4B at $0.07 in / $0.34 out is the model to default to** instead.
9. **RunPod H100 PCIe "$1.99/hr"** is wrong — official price is **$2.39/hr** on-demand. Sub-$2 numbers exist only on Vast.ai and HPC-AI.
10. **mistral.rs "86 tok/s on A10"** has no third-party benchmark. Treat as vendor-reported.
11. **GraphRAG "LazyGraphRAG cut indexing cost to 0.1%"** mentioned in External-A — verify before citing in launch.
12. **PathHD hyperdimensional computing claims (40-60% latency reduction, 3-5× memory cut)** from External-B — speculative architecture, not productionized. **Don't put in v0.3; file in v0.5+ research roadmap.**

---

## 12. The launch post — three drafted titles

Ordered by defensibility:

1. **"ctxgraph: same LLM, 1 call vs 6 — 5.8× higher relation F1 (benchmark inside)"** — methodology-forward, gets engineers to click, every number is yours.
2. **"ctxgraph 0.3: knowledge graph for AI agents, single-binary Rust + SQLite, runs on a 6GB GPU"** — distribution-forward, leads with the unique intersection.
3. **"Show HN: ctxgraph — beating Graphiti's 6-call pipeline with one schema-typed prompt, fully local"** — competitor-forward, slightly more flame-bait, gets comments faster.

Avoid: "Graphiti killer," "the next generation of KG engines," anything with "blazingly fast."

The opening 90-second demo video matters more than the title. External-A's framing — "the model emits ~50 unaligned relation tags per episode" — is the single most compelling line for the demo voiceover.

---

## 13. Where each source disagreed

For transparency:

- **GLiNER2 status.** Pass 2 mentioned it cautiously; External-A made it the headline swap; pass 3 partial-verified. Synthesis: ship it in W1.
- **Speculative decoding.** Pass 2 said skip; External-A casually mentioned "2-3× throughput" without caveats; pass 3 audited and found two of pass 2's three citations don't support the original claim. Synthesis: skip for the local 4-14B path, measure before shipping anywhere else.
- **NuExtract role.** Pass 2 said Tier-2 model; External-B treats it as a class of "extractive decoders" with negative-sampling training as the key differentiator. Synthesis: NuExtract 2.0-4B as Tier 2 default; explain the negative-sampling mechanism in docs because that's *why* JSON validation gets easier.
- **Pricing.** External-A says DeepInfra Gemma-4-26B-A4B at $0.07 in / $0.34 out; pass 3 says DeepInfra Qwen3-32B is split-priced not flat; pass 2 had "$0.08 flat" everywhere. Synthesis: cite Gemma-4-26B-A4B as the default, Qwen3-32B as the cheap alt, both split-priced.
- **PathHD / hyperdimensional retrieval.** Only External-B mentioned this. Synthesis: noted but not v0.3 — speculative research vector.

---

## 14. Summary — what to do this week

1. Start the GLiNER2 swap. Pip-install `gliner2`, wrap the ONNX in a Rust crate (or PyO3 bridge), retire `glirel.rs` + most of `rel.rs`.
2. Wire host-memory prompt caching defaults into `llm_extract.rs` — `keep_alive: -1` for Ollama, `--cram 256 --system-prompt-file` for llama-server, `--override-kv` for SWA models.
3. Add NuExtract 2.0-4B to `OLLAMA_PREFERRED_MODELS` ahead of `gemma3n:e4b`.
4. Swap default cloud Tier 3 to DeepInfra `google/gemma-4-26b-a4b-it`.
5. Re-run the 29-ep fixture. **Target: combined F1 ≥ 0.745** (matching Hermes 4 70B teacher with a fully local stack).

If that number lands, you don't need the LoRA before the v0.3 launch — you have your headline already. The LoRA is the v0.4 follow-up.

---

*Final brief. All four passes synthesized. Use this as the working document through HN launch. Re-verify all pricing 24h before posting.*
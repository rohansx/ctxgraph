# ctxgraph — Roadmap

> **Status**: Authoritative roadmap as of 2026-05-14.
> **Master working doc**: `CLARITY.md` — this is the week-by-week schedule that operationalizes it.
> **Supersedes**: `archive/ROADMAP_v1.md` (pre-research phase plan, kept for history).
> **Synthesizes**: `CLARITY.md` (the 5-pieces framing), `deep-research/FINAL.md` § 9–10, measured benchmarks (`docs/research_brief.md`).

---

## TL;DR

Two landing zones, two launches:

- **v0.3 (~weeks 1–8) → HN launch #1**
  Headline: *"ctxgraph: typed knowledge graph for AI agents. One LLM call per write. Zero LLM calls for 90% of reads. Single Rust binary. Free at any reasonable scale via Cerebras."*
  Built on: GLiNER2 swap, host-memory prompt caching, **universal 9/10 schema**, **bi-temporal `invalidates:` single-call extraction**, **relation-vocabulary embeddings + NL query parser** (no-LLM read path), three-mode strategy (local-only / cloud-fallback / cloud-quality).

- **v0.4 (~weeks 9–12) → HN launch #2 + arXiv preprint**
  Headline: *"ctxgraph 0.4 — embedded LLM in Rust, A-MEM append-only memory, automatic schema-improvement loop."*
  Built on: dual LoRA fine-tune (Hermes-4-14B + Qwen3-8B), `mistral.rs` embedded inference, A-MEM memory notes, Piece 5 layers B + C (schema promotion job).

Total external spend through v0.4: **~$60** ($30 fine-tune × 2 + ~$0.50 in benchmark API calls + Together AI eval overage).

The five pieces that have to land for v0.3 are listed in `CLARITY.md` § 3. This doc slots them into a week-by-week schedule.

---

## What's already done (as of 2026-05-13)

| Component | Status | Reference |
|---|---|---|
| Tiered pipeline scaffold | Shipped v0.8.0 | `crates/ctxgraph-extract/src/pipeline.rs` |
| GLiNER + GLiREL local extraction | Shipped v0.8.0 | `ner.rs`, `glirel.rs`, `rel.rs` |
| Confidence gate | Shipped v0.8.0 | `pipeline.rs:242–246` |
| Ollama autodetect | Shipped v0.9.0 (unmerged) | commit `63eb8f8` |
| Auto-schema inference | Shipped v0.9.0 (unmerged) | commits `9dcb574`, `83f3487` |
| CloakPipe integration | Shipped v0.9.0 (unmerged) | feature flag `cloakpipe` |
| Bi-temporal SQLite + RRF search | Shipped | `ctxgraph-core/src/graph.rs` |
| MCP server (6 tools) | Shipped | `ctxgraph-mcp/` |
| Homebrew tap | Shipped | `.github/workflows/release.yml` |
| 50-tech-ep + 10-cross-domain F1 benchmark | Shipped | `tests/benchmark_test.rs`, `tests/cross_domain_test.rs` |
| 29-ep cross-domain-v2 benchmark | Shipped this session | `crates/ctxgraph-extract/tests/fixtures/cross_domain_v2.json`, `docs/research_brief.md` |
| Graphiti head-to-head with Gemma 4 26B + 31B | Shipped this session | `scripts/graphiti_openrouter_bench.py`, `scripts/results/v0.9_cross_domain_v2/` |

---

## The 5 pieces (the v0.3 unblock)

> Source: `CLARITY.md` § 3. Each piece is independently testable. Pieces 1–3 are buildable this weekend; pieces 4–5 are the next two weekends.

| # | Piece | Path | LOC est. | Effort | Ships in |
|---|---|---|---|---|---|
| 1 | **Universal schema TOML** (9 entity types, 10 relations) | `crates/ctxgraph-extract/schemas/universal.toml` | ~40 | 1 hr | v0.3 W3 |
| 2 | **Extraction prompt + JSON contract** (~500 tokens, schema baked in, `invalidates:` field) | `crates/ctxgraph-extract/prompts/extract.txt` | n/a | 1 day (iterate) | v0.3 W3 |
| 3 | **Relation-vocabulary embeddings** (synonym layer, cosine match user verb → typed relation, no LLM at query time) | `crates/ctxgraph-extract/src/relation_match.rs` | ~30 | 2 hr | v0.3 W3 |
| 4 | **NL query parser** (Qwen3-1.5B + few-shot, for the ~10% of queries the embedding match can't handle) | `crates/ctxgraph-cli/prompts/query_parse.txt` + 200 LOC dispatch | ~200 | 3 days | v0.3 W5 |
| 5 | **Automatic schema improvement** (3 layers: A = always-on suggestion logging, B = nightly promotion job, C = `ctxgraph schema edit`) | side-table + cron + CLI subcommand | ~400 | Layer A: W4 (2 days). Layers B+C: W10 (1 week) | v0.3 (A), v0.4 (B+C) |

**Pieces 1–3 together give you a demo.** Drop in the universal schema, feed 10 real episodes through the new prompt, query each with 2–3 verb variations ("depends on" / "relies on" / "needs"), confirm the embedding match resolves correctly. If that works on real data, you have a working v0.3 in concept; the rest is hardening + the read-path NL parser + the schema-evolution loop.

---

## What's also v0.3 (infrastructure pieces, not the 5 product pieces)

These are necessary plumbing the 5 pieces depend on. They were the original "10-move plan" from `deep-research/FINAL.md` § 9 minus the items that are now subsumed by Pieces 1–5:

| # | Move | Effort | Ships in |
|---|---|---|---|
| I-1 | **GLiNER2 swap** — replace GLiNER + GLiREL in Tier 1, retire `rel.rs` and `glirel.rs` (~2 500 LOC) | ~1 week | W1 |
| I-2 | **Host-memory prompt caching** — Ollama `keep_alive: -1`, llama.cpp `--cram 256 --system-prompt-file`, `--override-kv` for SWA models | 2 days | W2 |
| I-3 | **NuExtract 2.0-4B as Tier 2 default**; VRAM autodetect route 4–8 / 8–16 / 16–24 GB tiers | 2 days | W2 |
| I-4 | **Three-mode write-path strategy** (local-only / cloud-fallback / cloud-quality) + `~/.ctxgraph/config.toml` | 3 days | W6 |
| I-5 | **DeepInfra `google/gemma-4-26b-a4b-it`** as paid cloud default; **Cerebras free tier** as Mode B default | 1 day | W6 |
| I-6 | **`mistral.rs` embedded inference** (eliminates Ollama HTTP boundary, single-binary moat) | 2 weeks | v0.4 W9 |
| I-7 | **Hermes-4-14B + Qwen3-8B LoRA fine-tunes** on 8 k synthetic dataset | weekend, ~$30 on Together AI | v0.4 W5–6 |
| I-8 | **A-MEM append-only memory notes** (arXiv 2502.12110, MIT) | 2 weeks | v0.4 W10 |
| I-9 | **Graph Judge nightly offline pass** (arXiv 2411.17388) | 1 week | v0.5 |

**Ordering**: I-1, I-2, I-3 (weeks 1–2) are pure infrastructure with no product dependencies. Pieces 1–3 (week 3) are the smallest unit of "the new product." Piece 4 (week 5) extends the read path. Piece 5-A (week 4) and Piece 5-B/C (week 10) handle schema evolution. Everything else is plumbing or v0.4 follow-up.

---

## 12-week schedule

| Week | Deliverable | What lands |
|---|---|---|
| **W1** | **I-1**: GLiNER2 wired into Tier 1; old GLiNER + GLiREL retired; re-run 29-ep `cross_domain_v2` fixture | Combined F1 delta vs current 0.687 baseline |
| **W2** | **I-2 + I-3**: host-memory caching defaults; Ollama `keep_alive=-1`; NuExtract 2.0-4B as Tier 2; VRAM autodetect routing | Per-episode latency before/after; per-tier model matches VRAM probe |
| **W3** | **Pieces 1, 2, 3**: universal schema TOML + extraction prompt + relation-vocabulary embeddings | 10 real wiki episodes extract correctly; "depends on" / "relies on" / "needs" all resolve to `depends_on` |
| **W4** | **Piece 5 Layer A**: bi-temporal `invalidates:` in extraction prompt; suggestion logging to `schema_suggestions` table | 1 000-ep synthetic temporal-conflict eval shows invalidation precision/recall; suggestions appear in side-table |
| **W5** | **Piece 4**: NL query parser via Qwen3-1.5B + few-shot prompt; SQL op dispatch | 50-query test set: 90% resolve via simple path, 10% complex; all under 500ms |
| **W6** | **I-4 + I-5**: three-mode strategy + Cerebras + DeepInfra integration; mode-switching config UX | `ctxgraph init --mode cloud-fallback` works end-to-end with $0 cost on Cerebras free tier |
| **W7** | Benchmark expansion: 29-ep fixture + Re-DocRED subset + LongMemEval-S | CSVs committed under `scripts/results/v0.3_evals/` |
| **W8** | **HN launch v0.3** — *"ctxgraph: typed local KG, single binary, free at scale, +22 F1 vs Graphiti on same model"* | Every claim survives `grep`; demo video < 60s |
| **W9** | **I-6**: `mistral.rs` embedded inference spike; `cargo install ctxgraph` includes the LLM in-process | Local extraction round-trip in pure Rust binary, no Python |
| **W10** | **Piece 5 Layers B + C**: schema promotion job + `ctxgraph schema edit`; **I-8**: A-MEM memory notes | Promotion job fires nightly; LongMemEval-S accuracy ≥ 75% |
| **W11** | MCP polish; Claude Code integration demo; `awesome-mcp-servers` PR; sub-60-s install demo video | Install in < 60s; MCP resource shows up in Claude Code |
| **W12** | **HN launch v0.4** + arXiv preprint: *"ctxgraph 0.4 — embedded LLM in Rust, A-MEM memory, automatic schema improvement"* | Preprint URL; reproduce-from-zero CI workflow green |

---

## What we explicitly rejected

> Source: `CLARITY.md` § 7. Keep this list around so the same suggestions don't keep coming back during code review.

| Idea | Why rejected |
|---|---|
| Schema-less / freeform output | Same as Graphiti. We win on typed extraction. |
| Manual schema TOML in v0.3 | UX cliff. Most users never write a schema. Piece 5 Layer C handles power-user cases. |
| Pivot to brain-inspired DB now | Real long-term opportunity but 12–18 months. Phase 3 after ship. |
| Ship LoRA fine-tunes before universal schema launch | LoRA is icing. The universal schema + relation matching is the cake. |
| OpenRouter gpt-4o-mini as cloud default | DeepInfra Gemma-4-26B-A4B is cheaper *and* same family as local Tier 2. |
| Speculative decoding for inference acceleration | Two of three cited papers don't support the claim. Skip or measure. |
| LongMemEval / LoCoMo as headline benchmark | Saturated at 85–95%. Lead with the 29-ep fixture instead. |
| `ctxgraph-ingest` crate (git / shell / FS / browser connectors) | Headline doesn't depend on it. Defer to v0.5+. |
| Reflect API (`ctxgraph_reflect*` MCP tools) | Defer to v0.4 or v0.5. |
| Python SDK / web dashboard / daemon mode / TUI | All defer post-v0.4. |

---

## What to do this weekend (the 5 hours that unblock everything)

> Source: `CLARITY.md` § 10.

1. **Create `crates/ctxgraph-extract/schemas/universal.toml`** with Piece 1 (the 9 + 10).
2. **Create `crates/ctxgraph-extract/prompts/extract.txt`** with Piece 2.
3. **Test the extraction prompt against Cerebras free tier with 10 real episodes** from your personal wiki. Iterate the prompt 2-3 times based on what breaks.
4. **Implement Piece 3** (`relation_match.rs`, ~30 lines — startup-embed the 10 relation names, query-time cosine match).
5. **Run end-to-end**: add 10 episodes, query each with at least 2 verb variations ("depends on" vs "relies on" vs "needs"). Confirm the embedding match resolves correctly.

If pieces 1–3 work on real data, you have something to demo. Piece 4 (NL query parser) is week 5. Piece 5 (schema improvement) splits into week 4 (Layer A) + week 10 (Layers B + C).

---

## Benchmark plan before v0.3 launch

> Source: `deep-research/FINAL.md` § 8.

| Benchmark | Why | Defensibility |
|---|---|---|
| **Own 29-ep `cross_domain_v2` fixture** | The ground-truth headline result; same code, same data, two systems | Strong — publish the fixture, invite PRs |
| **Re-DocRED subset** | Standard document-level RE benchmark (96 relation types, 34.9 avg triples per test doc) | Strong if we map our schema to Re-DocRED's typology |
| **LongMemEval-S** | What Zep cites; comparable to Graphiti in literature | Medium — risk of comparison to saturated leaders |
| **BEAM-1M** | 1 M-token scale where Python-based competitors fall over | Highest differentiation |
| **HaluMem** | Hallucination-in-memory | Underused; differentiates |
| **EverMemBench** (arXiv 2602.01313) | Multi-party, 1 M+ token, 2 400 QA pairs, designed to break saturated systems | New, defensible |

**Hostile-reader prep**: pre-empt the LoCoMo controversy by stating the methodology explicitly. The Zep–Mem0 *"Lies, Damn Lies & Statistics"* debate is well-known; just link the scoring code. (`scripts/openrouter_bench.py` is already self-contained.)

---

## Fine-tuning recipe (v0.4 detail)

> Source: `deep-research/FINAL.md` § 3.

### What to train

Two LoRAs on the same dataset:

| Model | Use case | License | RTX 4050 fit? |
|---|---|---|---|
| **Hermes-4-14B** (Qwen3-14B base) | Cloud + 16 GB+ laptops | Apache 2.0 | Q4_K_M ≈ 9 GB, doesn't fit 6 GB native |
| **Qwen3-8B** | Truly local, RTX 4050 / 6 GB VRAM | Apache 2.0 | Q4_K_M ≈ 5 GB, fits |

### Synthetic dataset

12 000 short passages (200–600 tokens) from CC-News, Wikipedia abstracts, arXiv abstracts, READMEs, Stack Exchange, OpenReview, Reddit — domain-balanced across the 25 fixture domains. Two-stage labeling with Claude Opus 4.5 / GPT-5 via `distilabel`. Consistency filter: keep pairs where Pass A ↔ Pass B agree on ≥ 60 % of gold pairs. 5 % adversarial slice (250 examples with deliberate contradictions in adjacent sentences) used to train the `invalidates:` output. Final 8 000 examples after filtering, ~9.6 M tokens before epochs.

### Training config

```yaml
base_model: NousResearch/Hermes-4-14B   # second pass: Qwen/Qwen3-8B
adapter: lora
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
optimizer: adamw_torch_fused
learning_rate: 1.0e-4
lr_scheduler: cosine
warmup_ratio: 0.03
micro_batch_size: 4
gradient_accumulation_steps: 4
num_epochs: 3
gradient_checkpointing: true
flash_attention: true
bf16: true
sequence_len: 4096
sample_packing: true
```

### Cost — verified

Tokens: 8 000 × 1 200 × 3 epochs = 28.8 M training + ~1.9 M eval = ~30.7 M total.

| Provider | Rate | Two-LoRA total | Wall-clock |
|---|---|---|---|
| **Together AI** (≤ 16B LoRA) | $0.48 / 1 M tokens (pass-3 verified) | **~$30** | 4–8 h |
| RunPod H100 PCIe | $2.39/hr official (pass-3 corrected from $1.99) | ~$20–30 | 6–10 h + setup |
| Modal H100 serverless | ~$3.25/hr | ~$30–40 | 5–6 h |

Pass-3 caveat: Together's $0.48/M figure assumes you pay for eval tokens too. Add 5–10 % buffer.

### Phase 2 — GRPO

After Phase 1 SFT converges, do a Phase 2 GRPO pass on the 5 % adversarial slice. RL-Struct reward function adapted:

```
r_total = 0.20·r_syntax + 0.20·r_schema + 0.30·r_pair_fuzzy_F1 + 0.30·r_entity_F1
```

Where RL-Struct achieves 89.7 % structural accuracy / 92.1 % JSON validity / **40 %** (corrected from "38 %") less peak VRAM than PPO.

---

## Competitive landscape

> Source: `deep-research/FINAL.md` § 7. Four 2026 competitors that the original `ROADMAP.md` missed entirely are included here.

| System | Position | Beats ctxgraph at | Loses to ctxgraph at |
|---|---|---|---|
| **Graphiti / Zep** | Production temporal KG, ~20 K stars | Brand recognition, real-time updates | 6-call cost, free-form verbs, no local-only mode |
| **HippoRAG 2** (ICML '25) | Academic SOTA agent memory | Personalized PageRank for associative retrieval | Still LLM-per-episode for OpenIE, no temporal model |
| **Anthropic Memory tool** (Mar 2026) | Client-side `/memories` filesystem | Native Claude API integration | Not a KG; locked to Claude |
| **Anthropic Managed Memory** (Apr 2026 beta) | Managed filesystem stores | Enterprise-managed, exportable | Locked to Claude, no schema-typed extraction |
| **Letta** (ex-MemGPT) | OSS three-tier memory | Active community, Apache 2.0, 74 % LoCoMo | Python deps, no typed extraction |
| **Mem0** ($23.9 M raised) | Personalized memory, 47.8 K stars | Mature SDK, fast retrieval | Not a real KG (separate clusters), low typed F1 |
| **Mastra Observational Memory** (Feb 2026) | OSS LongMemEval leader | 84.23 % gpt-4o, 94.87 % gpt-5-mini | Multi-call, no schema-typed local tier |
| **Cognee** | Hierarchical graphs | HotPotQA correctness 0.93 | Python deps, GPU embedder |
| **EverMemOS** (Jan 2026) | Three-phase, 83 % LongMemEval | High HaluMem score | Neo4j + GPU |
| **LinearRAG** (ICLR '26) | Relation-free graph build | Zero LLM tokens for indexing | Drops typed relations — wrong tradeoff for "what depends on X" queries |
| **AutoSchemaKG** (HKUST) | Dynamic schema induction | 92 % schema alignment at web scale | Batch-only, not incremental |

**Unique intersection** that nobody else hits: single-binary + SQLite-only + no-Python + no-Neo4j + no-GPU-required + single-call schema-typed + bi-temporal + truly local. **That is the launch sentence.**

---

## Hostile-reader prep (claims to scrub or qualify)

> Source: `deep-research/FINAL.md` § 11.

1. **AutoSchemaKG "92 % schema alignment"** — verified. "90 % content preservation vs 70 % for entity-only" — **don't cite** (paraphrase you can't source).
2. **NuExtract 2.0 PRO "+9 F-Score over GPT-4.1"** — applies to the **closed PRO API only**. The open 8B tops out at 73 F-Score.
3. **MIRIX 85.4 % LoCoMo** — verified, but uses **gpt-4.1-mini**, not gpt-4o.
4. **Hindsight 91.4 % LongMemEval** — requires a larger backbone; the open 20B is 83.6 %.
5. **EmergenceMem 86 %** — *not publicly reproducible*. Public configs: 79–82.4 %. Cite the public number.
6. **Zep LongMemEval 71.2 %** — paper number with GPT-4o. Independent reproduction (Gamgee 2026) reports 63.8 %.
7. **Hermes 4 14B "60B tokens"** is wrong context — that's the full Hermes 4 series. **The 14B SFT was 19B tokens**.
8. **DeepInfra Qwen3-32B "$0.08 flat"** — split-priced $0.08 in / $0.28 out. Default to **Gemma-4-26B-A4B at $0.07 in / $0.34 out** instead.
9. **RunPod H100 PCIe "$1.99/hr"** — official price is **$2.39/hr** on-demand.
10. **`mistral.rs` "86 tok/s on A10"** — vendor-reported, no third-party benchmark yet.
11. **GraphRAG "LazyGraphRAG cut indexing cost to 0.1 %"** — verify before citing.
12. **PathHD hyperdimensional computing claims (40–60 % latency, 3–5× memory cut)** — speculative architecture, not productionized. **Defer to v0.5+ research roadmap**.

---

## Drafted launch titles

> Source: `deep-research/FINAL.md` § 12. Ordered by defensibility.

1. *"ctxgraph: same LLM, 1 call vs 6 — 5.8× higher relation F1 (benchmark inside)"* — methodology-forward
2. *"ctxgraph 0.3: knowledge graph for AI agents, single-binary Rust + SQLite, runs on a 6 GB GPU"* — distribution-forward
3. *"Show HN: ctxgraph — beating Graphiti's 6-call pipeline with one schema-typed prompt, fully local"* — competitor-forward

**Avoid**: "Graphiti killer," "the next generation of KG engines," anything with "blazingly fast."

---

## Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| GLiNER2 quality lower than current GLiNER + GLiREL on tech text | Low | High (regression) | Keep old path behind a `--legacy-tier1` flag; re-run 50-tech-ep benchmark in W1 |
| LoRA quality doesn't reach Hermes 4 70B parity | Medium | Medium (still ship without it) | v0.3 launch doesn't depend on the LoRA; LoRA is the v0.4 follow-up |
| DeepInfra changes Gemma 4 26B pricing before launch | Low | Low (model survives) | Verify 24 h before HN; have OpenRouter Gemma 4 26B as fallback |
| Graphiti ships a v2 with local-only mode | Low (no signal as of session) | High | Privacy + single-binary + bi-temporal-aware-prompt is still differentiated |
| HN post lands on a slow day | Medium | Medium | Schedule for Tuesday 09:00 PT; have demo video ready before posting |

---

## Launch pitch (final)

> Source: `CLARITY.md` § 11.

> **ctxgraph: typed knowledge graph for AI agents.** Single Rust binary. Single SQLite file. One LLM call per write (vs Graphiti's six). Reads run locally without an LLM in 90% of cases. Free at any reasonable scale via Cerebras. Plugs into Claude Code via MCP. On the same fixture with the same model, hits 5.8× higher relation extraction F1 than Graphiti.

Four sharp claims. Every one defensible by the four research passes + measured benchmarks. No hand-waving.

---

*End of roadmap. Re-verify pricing 24 h before any HN-facing claim. The week-1 GLiNER2 swap is the single most important infrastructure move; the week-3 universal-schema landing is the single most important product move.*

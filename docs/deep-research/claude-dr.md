# ctxgraph — Deep Research Pass 2: From "Beats Graphiti" to a New Category

**Audience:** solo founder shipping `ctxgraph` (Rust knowledge-graph engine for AI agents) to Hacker News. The first pass found the landscape and the baseline win (+0.227 combined F1 over Graphiti on the same fixture / same model). This pass is about: (a) what to build next, (b) the fine-tuning recipe, (c) the speculative architectures that turn a "better Graphiti" into a "different category." Treat every benchmark in this document as something a hostile HN reader will check.

### Completion table for the requested deep-dive topics

| Section | Topic | Status |
|---|---|---|
| A | Event-centric KG construction | ✅ AutoSchemaKG, ASER, EventKG, ATOMIC, Event2Mind covered with arXiv IDs |
| B | Episodic vs semantic memory (MIRIX, A-MEM, Zettelkasten) | ✅ SQLite-schema mapping + risk flags |
| C | RL-tuned extractors (GRPO/DPO/RLAIF for structured output) | ✅ RL-Struct (2512.00319), GRPO theory, Memory-R1 noted |
| D | Distilled student models (UniversalNER, GoLLIE, NuExtract, InstructUIE, UniIE/GenIE/MuSEE) | ✅ Source-verified numbers + extra papers noted |
| E | Complete fine-tuning recipe + YAML + cost + eval + publish | ✅ |
| F | Constrained decoding (XGrammar / Outlines / llguidance / llama.cpp) + spec decoding viability | ✅ |
| G | Cost-optimized cloud routing (mid-2026 real prices) | ✅ |
| H | Bi-temporal extraction as prompt-engineering moat | ✅ |
| I | Graph Judge (arXiv 2411.17388) — integration proposal | ✅ |
| J | Speculative architectures (mistral.rs, MCP, streaming, multi-modal, CRDT, WASM) | ✅ 7 bets ranked |
| — | Breakthrough plan ranking 5–10 moves | ✅ 10 moves |
| — | 12-week project plan with verifiable deliverables | ✅ |
| — | RTX 4050 6GB constraint preserved | ✅ |
| — | Skeptical tone / third-party numbers / risk flags | ✅ |

---

## TL;DR

- **Three moves matter most**, ranked by impact-to-effort: (1) **ship a LoRA-tuned Hermes-4-14B extractor** on a 5–10k synthetic ctxgraph-schema dataset (≈ $24–60 on Together AI; one weekend); (2) **add a Graph-Judge confidence layer** (arXiv 2411.17388) gated by the local ONNX confidence score before the Ollama hop; (3) **add events-as-first-class-citizens to the schema** (an `Event` node type plus `participates_in` / `causes` / `precedes`), following AutoSchemaKG (arXiv 2505.23628). Everything else is optional polish.
- **Skip three tempting things that will get you grep'd on HN**: speculative decoding on a 4–14B target (peer-reviewed papers show it's lossy at small sizes); MIRIX's full 6-memory split (great taxonomy but the multi-agent overhead is wrong for a local-first single-binary tool); and any benchmark you can't reproduce without your own gold fixture (LoCoMo/LongMemEval-S are the ones to claim).
- **One speculative bet worth prototyping**: embedded inference via `mistral.rs` as a Rust crate dependency, eliminating the Ollama HTTP boundary entirely. This is the only move that makes ctxgraph structurally *different* from every Python competitor — a single `cargo install ctxgraph` binary that includes the LLM. Risk: `mistral.rs` is still a moving target; pin a release and ship a fallback to Ollama.

---

## Key Findings (at a glance)

| Topic | What's true | What ctxgraph should do |
|---|---|---|
| Event-centric KG (AutoSchemaKG, ASER, EventKG) | AutoSchemaKG (arXiv 2505.23628) reports **"92% semantic alignment with human-crafted schemas with zero manual intervention"** and "outperforms state-of-the-art baselines on multi-hop QA tasks and enhances LLM factuality" by modeling events alongside entities | Add `Event` node type + 3 event-relation types to the existing 10/9 schema |
| Episodic vs semantic memory (MIRIX, A-MEM) | A-MEM (NeurIPS 2025, arXiv 2502.12110) shows Zettelkasten-style note evolution outperforms flat memory across 6 base models on LoCoMo | Add an `episode_id` foreign key + a `memory_evolution` background task; skip the 6-memory MIRIX taxonomy |
| RL-tuned extraction (GRPO/DPO) | RL-Struct (arXiv 2512.00319) shows GRPO matches DPO memory and beats it: 89.7% structural accuracy, 92.1% JSON validity; reaches >80% structural accuracy "with as few as 1000 samples, whereas SFT requires significantly more data to reach comparable levels" | Phase 2 after SFT-LoRA — not phase 1 |
| Distillation recipes | UniversalNER (arXiv 2308.03279): 45,889 ChatGPT-labeled examples → LLaMA-2 7B → 41.7% F1 across 43 datasets, beating ChatGPT by +6.8. NuExtract 1.0: 50k Llama-3-70B-labeled examples from C4 → Phi-3-mini | 5–10k ctxgraph-schema examples is a defensible target |
| Constrained decoding | llguidance (Microsoft, Rust): ~50 µs/token average for JSON. XGrammar < 40 µs/token in their paper (arXiv 2411.15100) | Adopt llguidance via mistral.rs (it ships built-in) |
| Speculative decoding for small targets | Published benchmark (arXiv 2509.04474): for 8B/14B targets, "the time saved by reducing target model evaluations is largely offset by the overhead of running the draft model" | Don't ship spec decoding on the local path |
| Cheapest cloud route | DeepInfra Qwen3-14B = **$0.12/M tokens flat** (verified by deepinfra.com/pricing and cross-checked by pricepertoken.com's DeepInfra vs Together AI comparison, May 2026); ~$0.10 / 1k episodes at 800 tok | Use DeepInfra as the cloud fallback (replacing OpenRouter gpt-4o-mini) |
| Bi-temporal extraction prompt | OpenAI's official cookbook publishes a `StatementType` / `TemporalType` + `invalidated_by` schema; Zep does the same in blog/paper (arXiv 2501.13956) | This is now table stakes — port it into the existing single-call prompt |
| Graph Judge (EMNLP 2025, arXiv 2411.17388) | Fine-tuned LLM judges (triple, document) → confidence score on each extracted triple; SOTA on 3 KG construction datasets | Use as a 4th tier between Ollama and cloud, OR as an offline quality pass |
| Speculative architectures | `mistral.rs` (Rust crate, MIT) supports Qwen3 + GGUF + llguidance natively; 86 tok/s on Mistral-7B/A10 (third-party benchmark) | Prototype embedded inference; keep Ollama as fallback |

---

## Details

## A. Event-Centric Knowledge Graphs

### Papers and metrics

| Paper | arXiv | Headline (verbatim where quoted) |
|---|---|---|
| **AutoSchemaKG** (2025) | 2505.23628 | Built ATLAS: 900M+ nodes, 5.9B edges across 50M+ documents; **"92% semantic alignment with human-crafted schemas with zero manual intervention"** (abstract); explicitly models entities and events as first-class units; "outperforms state-of-the-art baselines on multi-hop QA tasks and enhances LLM factuality." |
| **ASER 2.0** (Zhang et al., AIJ 2022) | 2104.02137 | Eventuality KG: 438M eventualities, 648M edges; 15 discourse relation types; built from 11B-token unstructured corpus. The reference work for "event-centric" at web scale. |
| **EventKG** (Gottschalk & Demidova, 2018) | 1804.04526 | 690k events, 2.3M temporal relations; pure event-centric for contemporary/historical events. Canonical schema reference. |
| **ATOMIC** (Sap et al., 2019) | 1811.00146 | Crowd-sourced inferential event KG (if/then). Less relevant to ctxgraph's extraction path but defines the 9 inference dimensions you'd otherwise have to invent. |
| **Event2Mind** (Rashkin et al., ACL 2018) | 1805.06939 | Pre-cursor to ATOMIC; reactions to events. Cite for completeness. |

### The argument for event-centric extraction

AutoSchemaKG's claim that **events capture temporal, causal, and procedural knowledge that entity-only graphs miss** is what justifies the schema change. Graphiti's free-form edges already include verb-like predicates ("said", "introduced") but treats them as Relation strings on entity pairs — losing the event's own properties (time, location, participants beyond two). Modeling an event as a node `(:Event {type, time, location})` with N participants resolves the "50 free-form edges per episode, only ~10% land on gold pairs" failure mode you measured on Graphiti.

(Note: the AutoSchemaKG paper does *not* publish a literal "90% content preservation vs 70% for entity-only graphs" number; that framing in the project context appears to be a paraphrase. Don't cite that figure unless you re-run it on your own fixture. The defensible AutoSchemaKG claims are the 92% schema-alignment figure and the multi-hop QA gains — both verbatim from the abstract.)

### Schema-design tradeoff

Adding an `Event` node type means **two-stage extraction prompting**: "extract events first, then for each event extract its participants." That's a real latency cost (potentially 2× output tokens). Two ways to avoid the cost:

1. **Single-call but two-section JSON.** `{"events":[...], "entities":[...], "relations":[...]}` — keeps one round-trip. This is what AutoSchemaKG actually does; the conceptualization step is a *separate* post-process.
2. **Hybrid:** the GLiNER local pass extracts entities; the LLM escalation pass extracts events on top. This makes events conditional on confidence-gate triggering — events are exactly the things the ONNX layer can't do anyway.

### Downstream query patterns gained

- "What happened between Alice and Bob last week?" → event-time-filtered subgraph
- "Why did X happen?" → traverse `caused_by` event-event edges (this is the ASER discourse-relation use case)
- "Who participated in the deploy on Tuesday?" → events become joinable on time, which entity-relation graphs cannot do without proliferating timestamp edges

### Recommendation

**Ship event nodes in v0.3.** Add three node-type entries (`Event`, `Episode` already exists, `Claim`) and three relation types (`participates_in`, `causes`, `precedes`) for a total of **13 entity types / 12 relation types**. Two-section single-call JSON. Do *not* claim content-preservation numbers (90/70 or any others) without re-running them on your own fixture.

---

## B. Episodic vs Semantic Memory Architectures

### MIRIX (Wang & Chen, 2025) — arXiv 2507.07957

Six memory components: **Core, Episodic, Semantic, Procedural, Resource, Knowledge Vault**, each with a dedicated Memory Manager agent (eight agents total). Per arXiv 2507.07957 abstract: "MIRIX achieves 35% higher accuracy than the RAG baseline while reducing storage requirements by 99.9%" on ScreenshotVQA, and "attains state-of-the-art performance of 85.4%" on LoCoMo (this is below the 2026 SOTA of 89.61% from Gemini-3 Pro + TEMPR reported by EmergentMind, and below MemMachine's 91.69% in the first-pass findings).

**Honest read for ctxgraph:** MIRIX's value isn't the 8-agent system — it's the *taxonomy*. The 8-agent system requires an LLM call per write to route memory; you don't want that latency or cost on a local-first tool. But the taxonomy maps cleanly onto SQLite tables:

```
episodes  -> Episodic Memory (already exists in ctxgraph)
entities  -> Semantic Memory (already exists)
NEW: facts -> Core Memory (persistent user/agent profile facts)
NEW: procedures -> Procedural Memory (skill-and-tool patterns, optional)
NEW: documents -> Resource Memory (raw files referenced by graph)
NEW: secrets -> Knowledge Vault (encrypted column for credentials/preferences)
```

### A-MEM (Xu et al., NeurIPS 2025) — arXiv 2502.12110

Zettelkasten-inspired: each memory is a "note" with `content`, `context`, `keywords`, `tags`, `embedding`, plus *bidirectional links generated by an LLM at write time*. The "memory evolution" step is the novel piece — when a new note arrives, related historical notes can have their context/tags updated. Reported result: SOTA across **six base models** on LoCoMo (MIT-licensed implementation at https://github.com/agiresearch/A-mem).

**Risk flag:** memory evolution that silently rewrites historical notes is exactly the audit-failure case that the SSGM paper (arXiv 2603.11768) calls out. Make evolution append-only: write a new note revision, link `supersedes` to the old, never overwrite. ctxgraph's bi-temporal store already supports this — `valid_from` / `valid_to` on memory notes is the same construct.

### Implementation surface for ctxgraph

| New table | Schema (SQLite + FTS5) |
|---|---|
| `memory_notes` | `id, episode_id, content TEXT FTS5, context TEXT, keywords JSON, tags JSON, embedding BLOB, valid_from, valid_to` |
| `memory_links` | `from_note_id, to_note_id, link_type, score, created_at` (link_type ∈ {`relates_to`, `supersedes`, `contradicts`}) |
| `memory_revisions` | append-only audit table for evolution events |

**Bytes added:** ~3 tables, ~6 indexes, maybe 800 LOC of Rust. **Gained over flat entity-relation graph:** (a) retrieval over notes that don't have clean (subject, predicate, object) form (preferences, partial facts, procedural traces); (b) a path to LoCoMo/LongMemEval-S comparability, since both benchmarks reward note-style recall not just triple recall.

### Recommendation

**Ship A-MEM-style notes in v0.4, skip the MIRIX 8-agent system entirely.** The taxonomy is useful for the *docs*; the multi-agent runtime is the wrong shape for a single-binary Rust tool. Make `evolution` opt-in and append-only.

---

## C. RL-Tuned Extractors (GRPO / DPO for IE)

### What's published

- **RL-Struct** (arXiv 2512.00319, Dec 2025): the first paper to specifically target structured-JSON generation with GRPO and a hierarchical reward function (structural integrity → format → content → validity). Headline: **89.7% structural accuracy, 92.1% JSON validity**; trained on a single RTX 4090, **38% less peak VRAM than PPO**; reaches >80% structural accuracy "with as few as 1000 samples, whereas SFT requires significantly more data to reach comparable levels." Public model: `Freakz3z/Qwen-JSON` on HF.
- **GRPO theory** (arXiv 2503.06639): "Group Relative Policy Optimization (GRPO)… can be written as a Kullback-Leibler (KL) regularized contrastive loss." Dominant 2025 post-training method (DeepSeek-R1).
- **Memory-R1** (Aug 2025): "Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning" — RL is being applied to memory management directly, not just extraction. Worth tracking for v0.5+.
- **Hermes 4 technical report** (arXiv 2508.18255): used **Atropos rejection sampling (~1K verifiers, multiple trajectories)** plus 150+ schema-adherence environments + dynamic Pydantic repair. This is the closest public recipe to what ctxgraph would build.

### Reward function for ctxgraph's 13-entity / 12-relation extractor

The RL-Struct hierarchical reward function, adapted:

```
r_total = 0.20·r_syntax  + 0.20·r_schema + 0.30·r_pair_fuzzy_F1 + 0.30·r_entity_F1
```

- `r_syntax` = 1 if `json.loads()` succeeds else 0
- `r_schema` = 1 if `jsonschema.validate()` succeeds else 0
- `r_pair_fuzzy_F1` = your existing 0.687-baseline metric, computed against teacher labels
- `r_entity_F1` = standard set-based F1 on entity strings

### Recommendation

**Do GRPO only as a Phase 2 step**, after you have a working SFT-LoRA. Phase 1 LoRA gets you to ~baseline-of-teacher. Phase 2 GRPO with 1–2k high-signal preference examples should be where you exceed teacher quality on *your* schema. **Honest caveat:** there is no published 2025-26 paper specifically applying GRPO to a typed-IE schema like yours; you'd be the first to publish that ablation. That's a feature (paper-worthy) and a risk (you'll be debugging).

---

## D. Distilled Student Models for Fixed-Schema Extraction

### Reference recipes, source-verified

| Model | arXiv / source | Teacher | Synthetic data | Base | Headline |
|---|---|---|---|---|---|
| **UniversalNER 7B** | 2308.03279 | ChatGPT (gpt-3.5-turbo-0301), T=0 | **45,889 input-output pairs**, 240,725 entities, 13,020 entity types, sampled from 50K Pile passages | LLaMA-2 7B/13B (FastChat recipe — LR/epochs/hours not disclosed in paper text) | **41.7% avg F1 across 43 datasets** (UniNER-7B), beating ChatGPT's 34.9% by +6.8 |
| **NuExtract 1.0** | numind.ai blog | Llama-3-70B | **50k filtered examples from 300k C4 texts** | Phi-3-mini 3.8B (and Phi-3-small 7B for NuExtract-large; Qwen1.5-0.5B for NuExtract-tiny) | "Similar to GPT-4o while ≥100× smaller" (vendor claim — treat with skepticism) |
| **NuExtract 2.0 8B** | numind.ai blog (Jul 16, 2025) | Not disclosed | Synthetic data scale **not publicly disclosed for v2** | Qwen2.5-VL-7B (and Qwen2-VL-2B for 2B, Qwen2.5-VL-3B for 4B) | "NuExtract 2.0 8B reaching 73 F-Score… a bit better than non-reasoning frontier models" on a 1000+ example, 21-problem benchmark. **The "+9 F-Score over GPT-4.1" applies to NuExtract-2.0-PRO (closed API model), not the open 8B.** Do not conflate. |
| **GoLLIE 7B** | 2310.03668 | Schema guidelines as docstrings | 12 IE datasets (subset of InstructUIE's 34) | Code-LLaMA 7B / 13B / 34B | **"Absolute difference of 13 F1 points on average"** vs without-guidelines baseline; "surpasses by a large margin" InstructUIE and entailment-IE on zero-shot |
| **InstructUIE** | 2304.08085 | — | 32-task IE dataset | FLAN-T5-11B | Earlier multi-task IE; superseded by UniversalNER and GoLLIE |
| **Hermes 4 14B** | 2508.18255 | Atropos rejection sampling on ~1K verifiers | **~5M samples, ~19B tokens** (per tech report — note: HF model card says ~60B tokens, a ~3× conflict; cite the arXiv number) | Qwen3-14B-Base | SOTA on RefusalBench; schema adherence is first-listed feature |

**Also in the lineage** (mentioned in the query, lower priority for ctxgraph's recipe): UniIE, GenIE, MuSEE. UniIE and GenIE are earlier (pre-2024) generative IE models with no direct distillation recipe to copy. MuSEE is a multilingual structured extraction model with limited public training documentation. **None of these change the recommendation.** The 2024–25 lineage that matters is: UniversalNER (proved targeted distillation works at 45.9k examples) → GoLLIE (proved schema-as-docstring helps zero-shot) → NuExtract (proved C4-sampling + Llama-3-70B teacher works at 50k examples) → Hermes 4 (proved ~150 schemas × Atropos rejection sampling produces SOTA structured-output models).

### Critical takeaways for ctxgraph's recipe

1. **5k–10k examples is the right scale.** UniversalNER trained on 45.9k for *open* NER across 13k entity types; you have 13 entity types and 12 relation types. A 5–10k *typed* dataset is more than enough — possibly overkill.
2. **A frontier teacher is mandatory.** UniversalNER used gpt-3.5-turbo; NuExtract used Llama-3-70B. In 2026, use Claude Opus 4.5 or GPT-5 as teacher because their few-shot extraction on a typed schema is materially better — and your evaluation set already shows Hermes 4 70B hits 0.745 F1, so the teacher must beat that.
3. **F1 retention claim to test:** UniversalNER beats ChatGPT by +6.8 F1 at ~1/40th the size. NuExtract claims to "match" GPT-4o at ≥100× smaller. Both claims are on the teacher's strengths, not on novel domains. **For ctxgraph, the realistic target is parity with the teacher on the 25-domain fixture, with 100× lower latency.**

### Recommendation

**Use the UniversalNER × Hermes-4 hybrid recipe:** UniversalNER's data-construction approach (sample passages from a generic corpus, label with frontier LLM, use simple consistency filter), but apply it on top of Hermes-4-14B as base rather than vanilla LLaMA. This gives you UniversalNER's data-efficiency and Hermes-4's schema-adherence priors in one job. **Do not** start from NuExtract-2.0 base (license risk on Qwen2.5-VL-3B) or InstructUIE (FLAN-T5 is too old to compete with Qwen3-class models). **Do not** chase any "100× smaller than GPT-4o" claim — quote your *own* measured 29-episode F1 numbers and let the model card speak.

---

## E. The Fine-Tuning Recipe (decision-ready)

### Base model selection

| Candidate | License | Pros | Cons | Verdict |
|---|---|---|---|---|
| **Hermes 4 14B** (Qwen3-14B base) | Apache 2.0 | Already schema-adherence-trained on 150+ JSON schemas via Atropos. ~19B tokens of structured-output post-training per arXiv 2508.18255. GGUF + FP8 + AWQ available. | 14B → ~9 GB at Q4_K_M — **does not fit in RTX 4050 6GB** without partial CPU offload (which the SitePoint guide warns drops Qwen3-8B from 40 → 8 tok/s) | **Pick this for the LoRA target.** Inference is on cloud or laptop CPU+GPU split; the LoRA is a 50–100 MB adapter you can swap in. |
| **Qwen3-8B** (Apache 2.0) | Apache 2.0 | Fits RTX 4050 at Q4_K_M (~5 GB). 40+ tok/s on similar 8GB cards per Ollama VRAM guide. | Not pre-tuned for structured output — you're paying SFT cost twice (structure + your schema) | Acceptable fallback for "local-only" claim; ship a Qwen3-8B LoRA as the *local* extractor and the Hermes-4-14B LoRA as the *cloud* extractor. |
| **NuExtract-2.0-4B** (Qwen2.5-VL-3B base) | License risk — NuMind themselves chose Qwen2-VL-2B for the smallest model "because the smallest Qwen2.5-VL model (3B) has a more restrictive, non-commercial license" | Multimodal (image episodes for free). Already extraction-pretrained. | Smaller. License inheritance unclear; **check before commercial use**. | Skip for the v1 launch; revisit when multi-modal episodes become a roadmap item. |

**Decision: train two LoRAs simultaneously**: one on **Hermes-4-14B** (cloud/laptop), one on **Qwen3-8B** (truly-local). Same dataset, same hyperparameters, one extra ~$10 of training cost.

### Synthetic data recipe (5–10k examples)

Inspired by UniversalNER's "sample passages from a generic corpus, label with a frontier LLM" method but adapted for typed schemas:

1. **Sample 12,000 short passages** (~200–600 tokens each) from CC-News, Wikipedia abstracts, ArXiv abstracts, GitHub READMEs, Stack Exchange answers, OpenReview reviews, Reddit comments — to span the 25+ domains your gold fixture has.
2. **Two-stage labeling with Claude Opus 4.5 (or GPT-5):**
   - Pass A: "Given this text and the 13-entity / 12-relation schema, extract everything that fits."
   - Pass B: same text, different temperature seed. Keep only pairs where Pass A and Pass B agree on ≥60% of the gold pairs (consistency filter; this is how UniversalNER cleans noise without manual review).
3. **Domain balance:** force at least 400 examples in each of your 25 domains; reject samples that have zero extractable entities.
4. **Adversarial 5% slice:** include 250 examples with deliberately contradictory facts in adjacent sentences (this trains the `invalidates:` output for bi-temporal extraction — see section H).
5. **Final dataset target: 8,000 examples** after filtering. At 1,200 tokens avg per example, that's **~9.6M training tokens before epochs**.

### LoRA training config (copy-pasteable)

```yaml
# ctxgraph-extractor-lora.yaml
# Together AI / Axolotl-compatible YAML
base_model: NousResearch/Hermes-4-14B
tokenizer_type: AutoTokenizer
trust_remote_code: true

# Dataset
datasets:
  - path: ./data/ctxgraph_synth_v1.jsonl
    type: completion
    field: text
sequence_len: 4096
sample_packing: true
pad_to_sequence_len: true
val_set_size: 0.05  # 400 / 8000

# LoRA
adapter: lora
lora_r: 32           # rank — UniversalNER-scale schema, mid-rank is plenty
lora_alpha: 64       # 2x rank, standard
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
lora_modules_to_save: []

# Optimizer
optimizer: adamw_torch_fused
learning_rate: 1.0e-4   # standard LoRA LR; LR-finder if you have GPU budget
lr_scheduler: cosine
warmup_ratio: 0.03
weight_decay: 0.0

# Training
micro_batch_size: 4
gradient_accumulation_steps: 4  # effective BS 16
num_epochs: 3                   # UniversalNER-style; monitor val loss for early stop
gradient_checkpointing: true
flash_attention: true
bf16: true

# Eval
evals_per_epoch: 4
save_strategy: steps
save_steps: 200
save_total_limit: 3

# Reproducibility
seed: 1337
```

### Cost & wall-clock — three providers, real numbers

Tokens to process: 8,000 examples × 1,200 tok/example × 3 epochs = **28.8M training tokens** (+ 4 evals × 400 examples × 1,200 tok = 1.9M eval tokens) → **~30.7M total**.

| Provider | Rate | Cost | Wall-clock (typical) | Notes |
|---|---|---|---|---|
| **Together AI LoRA** (≤16B) | **$0.48 / 1M tokens** (verified at https://www.together.ai/pricing; cross-verified by awesomeagents.ai's "Fine-Tuning Costs Comparison" (March 26, 2026): *"Together AI offers the cheapest API fine-tuning at $0.48/1M tokens for LoRA on models up to 16B parameters"*) | **~$14.74** | ~2–4 hours | Serverless multi-LoRA serving included at base-model price after training. Zero infra setup. |
| **Fireworks AI** | LoRA on Llama/Qwen ≤16B at similar rates ($0.50–$0.70/1M per pricepertoken comparison) | ~$15–22 | ~2–4 hours | DPO is 2× SFT price; you only do DPO/GRPO in Phase 2. |
| **DIY on RunPod H100 PCIe** | $1.99/hr on-demand, $1.30–1.60/hr spot (deploybase.ai, March 2026) | **~$8–12** for 4–6 hours | ~4–6 hours plus 1–2 hours of setup | Cheapest cash cost, highest engineering cost. Worth it only if you'll do 5+ runs (which you will, in Phase 2). |
| **Modal serverless H100** | ~$3.25/hr equivalent (per DeployBase review) | ~$15–20 | ~5–6 hours | Beautiful Python ergonomics, slight premium. Best if you want CI/CD-driven retraining. |

**Two-LoRA total cost** (Hermes-4-14B + Qwen3-8B, same dataset): **~$24–60 on Together**, **~$16–24 on RunPod**. This is the *single most under-priced lever* in the whole project — you can iterate weekly.

### Evaluation harness

Three layers, in order of decreasing repro burden (do them all before HN):

1. **ctxgraph's own 29-episode fixture** (pair-fuzzy F1) — required, you already have it. Baseline numbers: 0.687 (Gemma 4 26B-A4B), 0.745 (Hermes 4 70B), 0.460 (Graphiti same fixture). Re-run with the new LoRA; the target is to **beat Hermes 4 70B's 0.745** with a 14B LoRA, which would be a real headline.
2. **Re-DocRED** (arXiv 2205.12696) — 3,053 train / 500 dev / 500 test docs, 96 relation types, 6 entity types, triple density 34.7 avg per test doc. Use the 500 test docs as a third-party check. *Caveat:* Re-DocRED's relation types don't map 1:1 to yours; you'll need a mapping table or to restrict eval to the overlapping subset.
3. **LongMemEval-S** (arXiv 2410.10813) — 500 manually-curated questions, 5 reasoning skills (information extraction, multi-session reasoning, temporal reasoning, knowledge updates, abstention). LongMemEval_S has ~115k tokens/problem. Current SOTA: EmergenceMem 86%, EverMemOS 83.0%, Zep/Graphiti 71.2–72.27% with GPT-4o. Target ctxgraph anywhere ≥ Zep is HN-defensible. Anything ≥ 80% is a feature.

### Publishing strategy

- **HuggingFace Hub:** push `rohansx/ctxgraph-extractor-hermes4-14b-v1` and `rohansx/ctxgraph-extractor-qwen3-8b-v1` as LoRA adapters (50–100 MB each, not the merged weights — keeps Apache-2.0 base license clean).
- **MIT license on the adapters** (your work), **clearly state base model's Apache 2.0** in model card.
- **Model card content** (in this order, because HN readers grep for these):
  1. Base model + license
  2. Training data construction (with the Claude/GPT-5 teacher disclosure)
  3. Eval table: 29-ep, Re-DocRED-subset, LongMemEval-S; baseline + LoRA + (cite) Hermes-4-70B teacher; flag your fixture's known biases (25-domain selection, manual labels)
  4. Reproduction snippet (3-line `transformers + peft` load)
  5. Known limitations (the 5% adversarial slice, contexts > 4k tokens, languages outside English)

### Recommendation

**Run the two-LoRA job on Together AI** (cheapest, no infra overhead, multi-LoRA serverless serving is a bonus). Phase 1 SFT only. Iterate weekly until 29-ep F1 ≥ 0.745. Then move to Phase 2 (GRPO on adversarial slice) and Phase 3 (Graph-Judge add-on). Don't merge LoRAs into the base — keep adapters separate so the base license stays clean and you can hot-swap on the inference path.

---

## F. Constrained Decoding + Speculative Decoding

### Verdict on constrained decoding for your 13-entity / 12-relation schema

| Engine | Real measured cost | Verdict |
|---|---|---|
| **llguidance** (Rust, Microsoft, MIT) | "~50µs" average mask compute on JSONSchemaBench (2.5M tokens, 10k schemas), "less than 1% of masks taking longer than 1ms" (https://github.com/guidance-ai/llguidance); "16 cores + 10ms forward pass → batch sizes up to 3200 without slowing the model" | **Use this.** Already integrated into mistral.rs; Rust-native; latency overhead invisible at your scale. |
| **XGrammar** (Rust core, Apache 2.0) | "under 40µs per token for JSON Schema and CFG (JSON), and under 200µs for XML and Python DSL" (arXiv 2411.15100); "up to 3x speedup on JSON Schema and over 100x on CFG" | Equally fine; vLLM auto-mode picks between XGrammar and llguidance. If you ship via mistral.rs, llguidance is the path of least resistance. |
| **Outlines** | "Outlines and Llamacpp demonstrate substantially lower throughput than the LM-only approach" per JSONSchemaBench paper (arXiv 2501.10868) | Skip. Significant startup overhead from FSM precomputation; recursive JSON schemas (which yours has via nested entity arrays) are not well-supported. |
| **llama.cpp built-in grammar** | "Significantly slower… due to the lack of a lexer and use of a backtracking parser" (llguidance comparison) | Skip if you can. Acceptable as a fallback when you must ship a single-binary on a platform mistral.rs can't reach. |

### Verdict on speculative decoding for the local 4–14B path

**Don't do it.** Three independent results:

1. **arXiv 2509.04474** (Sep 2025 benchmark): for 8B and 14B targets, *"the time saved by reducing target model evaluations is largely offset by the overhead of running the draft model… limits the overall acceleration."*
2. **arXiv 2402.01528 ("Decoding Speculative Decoding")**: 350+ experiments; "the performance of speculative decoding depends heavily on the latency of the draft model, and the draft model's capability in language modeling does not correlate" — i.e., even a "good" small draft can be a net loss.
3. **arXiv 2412.18934 (Dovetail)**: shows that on RTX 2080-class hardware, vanilla SD on Llama-2-7B can be *slower* than not using SD; only their GPU-CPU heterogeneous variant beats CPU-only.

**Exception:** on a server-class H100 with batch=1 throughput-optimized serving (Baseten/TensorRT-LLM with Qwen2.5-Coder-14B + Qwen2.5-Coder-0.5B as draft) you do see real wins. But ctxgraph's local path on RTX 4050 6 GB is *not* that hardware. **Ship spec decoding off the table for local; revisit only for a hosted ctxgraph-Cloud variant.**

### Recommendation

Pin llguidance through mistral.rs. Skip speculative decoding for the local extractor. Re-evaluate spec decoding only if/when you ship a hosted ctxgraph-Cloud on H100.

---

## G. Cost-Optimized Cloud Routing (the cheapest IE for 7–14B in mid-2026)

### Per-1k-episode costs at 800 tokens/episode (≈ 600 in + 200 out)

For 1,000 episodes: **800k tokens total**, billed combined I+O for flat-priced models or split for asymmetric ones.

| Provider | Model | Price (per 1M tok, flat or in/out) | Cost / 1k episodes (≈ 0.8M tok) |
|---|---|---|---|
| **DeepInfra** | Qwen3 14B | $0.12 (flat) — verified by deepinfra.com/pricing and cross-checked by pricepertoken.com's DeepInfra vs Together AI comparison (May 2026): *"Qwen3 14B $0.120/M"* | **$0.096** |
| **DeepInfra** | Qwen3 32B | $0.08 (flat) | **$0.064** ← cheaper than 14B because it's MoE-served; verify before relying. |
| **DeepInfra** | Gemma 4 26B-A4B Instruct | $0.08 (flat) | **$0.064** ← same model family as your project context's production target; cleanest cloud/local parity |
| **DeepInfra** | Llama 3.1 8B | $0.06 (flat) | $0.048 |
| **Together AI** | Llama 3.3 70B | $0.88 / $0.88 | $0.704 |
| **Cerebras** (free tier) | Qwen3-32B | 1M tokens/day free | **$0** up to 1M tok/day cap |
| **Groq** (free tier) | varies | Free tier with rate limits | $0 with caveats |
| **Alibaba Cloud Model Studio** | Qwen 3 Max | 1M tokens/model free (Singapore region, new users) + 50% batch discount | $0 for prototyping |
| **OpenRouter** (your current setup) | gpt-4o-mini | $0.15 in / $0.60 out → ~$0.30 blended | ~$0.24 |
| **RunPod dedicated** | Qwen3-14B on H100 PCIe | $1.99/hr → ~6,000 req/hr at p50 ~600 ms | $0.33 / 1k req at 100% utilization, much higher at <30% |

### Recommendation (production routing)

Replace `gpt-4o-mini fallback` with **DeepInfra's Gemma-4-26B-A4B at $0.08/1M (flat)**. This is the *same model family* as your local Ollama path (your project context lists Gemma 4 26B-A4B as the production target) — cleanest possible cloud/local parity. **Cost reduction vs current cloud fallback: ~3×.** Same OpenAI-compatible API.

For the truly-cheap path, ship a **Cerebras Qwen3-32B free-tier hop** as Tier 2.5 (between local Ollama and paid cloud) — 1M tokens/day is enough for ~1,250 episodes/day free, which is the sweet spot for hobbyist/early-adopter ctxgraph users.

### Cheapest paid endpoint, period

Based on DeepInfra's published pricing as of April–May 2026: **Qwen3-32B at $0.08/M flat = $0.064/1k episodes.** That is **~3.75× cheaper** than your current gpt-4o-mini fallback and **~10× cheaper** than Together's Llama-3.3-70B.

---

## H. Bi-Temporal Extraction as a Prompt Moat

### Prior art (all 2024–25)

- **Zep "Beyond Static Graphs" (blog) + Graphiti** — "we perform date extraction and invalidation concurrently… we make an LLM call for each new edge using an invalidation prompt, providing existing similar edges as context." Stores `created_at, expired_at, valid_at, invalid_at` per edge. Arch paper: arXiv 2501.13956.
- **OpenAI Cookbook "Temporal Agents with Knowledge Graphs"** — publishes a reference Pydantic schema: `RawStatement {statement_type: StatementType, temporal_type: TemporalType (Static|Dynamic|Atemporal), valid_at, invalid_at, invalidated_by: list[id]}`. This is now the canonical public spec.
- **Beyond Known Facts** (arXiv 2601.13658) — TKGE benchmark with 4.2k future quadruples; uses the Extract-Define-Canonicalize (EDC) framework; LLM performance drops on unseen-fact extraction. Useful as an eval set.
- **TGL-LLM** (arXiv 2501.11911) — temporal graph learning baked into LLM context.

### What ctxgraph should change in its extraction prompt

Currently stateless: `(episode_text) → JSON {entities, relations}`.

Bi-temporally aware: `(episode_text, current_facts_about_mentioned_entities) → JSON {entities, relations, invalidates: [edge_id, ...], confidence: float}`.

The key novelty: **the LLM, not a post-hoc invalidation pass, emits `invalidates`**. That removes Zep's "LLM call per new edge" cost. The current-facts context is bounded by retrieving the top-K facts touching each entity mentioned in the episode — typically 5–10 facts × N entities, well within a 4k-context budget.

### Why this is a "moat"

Two reasons it's defensible:
1. **The teacher LLM gets it free.** Claude Opus 4.5 / GPT-5 already understand "this fact contradicts that fact"; you just need to ask for `invalidates` in the schema. Your LoRA distills that capability — Graphiti, Zep, MemMachine don't have a model trained for this; they all do post-hoc invalidation.
2. **The training data writes itself.** Generate 1,000 synthetic episode-pairs (`episode_t1`, `episode_t2` where t2 contradicts t1), label them with Claude, and you have the only public dataset for prompt-side invalidation.

### Recommendation

**Add `invalidates` and `confidence` to the JSON schema in v0.3.** Include the 5% adversarial slice in the SFT dataset (see Section E). This is the single most defensible thing in the whole launch — a feature competitors literally cannot ship without retraining.

---

## I. Graph Judge — Confidence Layer Integration

### The paper

**Huang et al., "Can LLMs be Good Graph Judge for Knowledge Graph Construction?"** — EMNLP 2025 main, arXiv 2411.17388, code: https://github.com/hhy-huang/GraphJudge.

### What it proposes

A three-module framework:
1. **Entity-Centric Iterative Text Denoising** — removes noisy sentences that don't anchor to mentioned entities (improves precision).
2. **Knowledge-Aware Instruction Tuning** — fine-tune an LLM on (text, triple, gold-or-not) pairs to teach it to *judge*, not just generate.
3. **Graph Judgement** — at inference, every candidate triple from the generator gets a binary judgment from the judge LLM; rejected triples are dropped.

Reported result: SOTA on **two general + one domain-specific text-graph pair dataset**, beating multiple baselines including pure-LLM and pipeline approaches. (Exact F1 deltas not extracted in my pass — read the EMNLP PDF before citing numbers in your launch post.)

### Integration into ctxgraph

Add as **Tier 1.5** (between local ONNX and Ollama hop), or as a **Tier 4 offline pass** for stored graphs:

```
Tier 1: GLiNER + GLiREL ONNX (entity, relation candidates with confidence)
Tier 1.5 (NEW): Graph-Judge fine-tuned Qwen3-1.5B (binary keep/reject per candidate triple, ~5ms/triple)
Tier 2: Ollama Gemma-3n (only triples that survive tier-1.5 AND have low confidence go to LLM)
Tier 3: DeepInfra Gemma-4-26B-A4B (cloud fallback for ambiguous cases)
Tier 4 (NEW, optional): nightly Graph-Judge pass on the whole graph to demote stale low-confidence facts
```

**Why a 1.5B judge model is the right size:** the judge is a binary classifier conditioned on (text, triple). It doesn't generate, it scores. A small Qwen3 or Phi-mini fine-tuned on 3–5k judge-labeled examples should clear 90%+ accuracy. Latency well under your 30 ms ONNX budget.

### Recommendation

**Ship Graph-Judge in v0.5 as Tier 4 (offline nightly pass) first** — it's the lowest-risk integration. Move to Tier 1.5 (online) only after measuring that the nightly pass actually improves graph quality on your fixture. Don't claim "SOTA confidence layer" until you have numbers from your own re-implementation.

---

## J. Speculative Architectures — 7 Bets

Ranked by impact-to-effort. Explicit on which I'd ship and which I'd skip.

### Bet 1 — **Embedded inference via mistral.rs (SHIP)**

- **What:** Bundle `mistralrs` as a Rust crate dependency. ctxgraph becomes a single static binary that includes the LLM (no Ollama HTTP, no Python).
- **Evidence it works:** mistral.rs supports Qwen3, Gemma 4, GGUF, ISQ quantization, llguidance constrained decoding, and PagedAttention out of the box (https://github.com/EricLBuehler/mistral.rs). Measured throughput per third-party review (createaiagent.net): Mistral-7B at 86 tok/s on an A10 GPU; on RTX 4050 6GB you'd expect 15–25 tok/s on Qwen3-8B Q4_K_M based on the SitePoint and Unsloth Q4_K_M benchmarks.
- **Why it's category-defining:** every Python competitor (Graphiti, Zep, MemMachine, A-MEM) requires the user to manage Ollama or vLLM or Python deps. ctxgraph + mistral.rs = `cargo install ctxgraph`. That's a Hacker News headline.
- **Risk:** mistral.rs is a single-maintainer project; pin a release tag, not main. Provide an Ollama fallback flag.
- **Effort:** ~3 days of integration, ~2 weeks of polish.

### Bet 2 — **Bi-temporal `invalidates:` in the extraction prompt (SHIP)**

See Section H. This is the prompt moat. Effort: 1 week including the synthetic-pair dataset.

### Bet 3 — **Graph-as-MCP-resource (SHIP, but quietly)**

- **What:** Expose the SQLite graph itself as an MCP resource (`graph://entities/{id}`, `graph://edges?subject=X&time=...`). Currently ctxgraph offers an MCP *server* with tools. Going from MCP-as-tools to MCP-as-resources lets any MCP-aware agent (Claude Desktop, IDE plugins, Cursor) read the graph natively.
- **Why it matters:** MCP resources are *cached and addressable* in a way tool calls aren't. This is the integration story most agent platforms will adopt in 2026.
- **Effort:** ~3 days. Low risk because the underlying queries already exist.

### Bet 4 — **Streaming extraction with partial-graph updates (MAYBE)**

- **What:** Parse the JSON stream token-by-token; emit entity nodes the moment they're complete, before the relation list arrives. Reduces perceived latency for the embedded UI use case.
- **Evidence:** XGrammar and llguidance both expose per-token validity APIs that make this implementable. NuExtract docs note their extracted JSON is purely extractive (no hallucinated text), making partial JSON safe to commit.
- **Risk:** Partial commits break atomicity. You'd need a `pending` flag on streamed entities, and rollback on parse error.
- **Verdict:** Worth a prototype in v0.5, not blocking for launch.

### Bet 5 — **Multi-modal episodes (screenshots, voice, structured logs) (SKIP for v1)**

- **Why interesting:** MIRIX's 35%/99.9% ScreenshotVQA result shows there's real value here. NuExtract 2.0 already does image extraction.
- **Why skip for launch:** RTX 4050 6GB cannot run Qwen2.5-VL-7B (the smallest reasonable VLM) at usable speed, and the small-VLM problem isn't solved on consumer hardware yet. Multimodal also doubles your test surface. Revisit at v0.6 once Qwen3-VL-4B-Instruct ships with cleaner quantization (per the herbert-rs benchmark: vLLM-CPU 484 tok/s prefill on a 4B dense VL model on a 12-core Ryzen 9 — workable, but not 4050-territory yet).

### Bet 6 — **Federated graphs via CRDT (SKIP for v1, file as a research roadmap item)**

- **What:** Two ctxgraph instances on two laptops merging their graphs without conflicts via a CRDT layer (likely RGA or Loro-style ops on the edge list).
- **Why it's high-stakes:** This is the multi-user story. None of Graphiti/Zep/A-MEM/MIRIX has it.
- **Why skip:** CRDTs for typed property graphs with bi-temporal edges are an open research problem (the Loro maintainers acknowledge this; Automerge focuses on documents). You'd be doing original research, not engineering. **Note it in the roadmap; do not promise it for launch.**

### Bet 7 — **WebAssembly ctxgraph in the browser (SKIP for v1)**

- **What:** Compile the Rust core to WASM, run SQLite via sql.js, run the LLM via `mistral.rs`'s WASM/wgpu path.
- **Reality check:** mistral.rs does not currently ship WASM bindings, and running an 8B model in a browser tab is impractical at consumer scale. Quantized 0.5–1B models in WASM are feasible (Transformers.js demos) but not at the quality threshold ctxgraph needs.
- **Verdict:** Cool demo, wrong battle. Skip for v1.

---

## The Breakthrough Plan — 10 moves ranked by impact-to-effort

| # | Move | Effort | Impact | Ship in |
|---|---|---|---|---|
| 1 | Train Hermes-4-14B LoRA + Qwen3-8B LoRA on 8k synthetic typed examples (Section E) | 1 weekend, ~$30 | Headline benchmark win, defensible HN story | v0.3 |
| 2 | Add bi-temporal `invalidates:` to extraction prompt + JSON schema (Section H) | 1 week | Real moat vs Zep/Graphiti | v0.3 |
| 3 | Swap OpenRouter→DeepInfra Gemma-4-26B-A4B fallback (Section G) | 1 day | 3× cheaper cloud + better cloud/local parity | v0.3 |
| 4 | Add `Event` node type and 3 event-relations to schema (Section A) | 3 days | Closes the "Graphiti emits 50 noisy edges" failure mode structurally | v0.3 |
| 5 | Integrate `mistral.rs` as embedded inference (Section J, bet 1) | 2 weeks | Single-binary distribution, category-defining | v0.4 |
| 6 | Add A-MEM-style `memory_notes` + append-only evolution (Section B) | 2 weeks | LongMemEval-S comparability; +new user persona use cases | v0.4 |
| 7 | Phase-2 GRPO fine-tune on adversarial slice (Section C) | 1 week, ~$50 | First public typed-IE GRPO result; paper-worthy | v0.4 |
| 8 | Ship Graph-Judge nightly pass (Section I) | 1 week | Confidence scores per edge; demotes stale facts | v0.5 |
| 9 | MCP-as-resource for graph queries (Section J, bet 3) | 3 days | Native Claude Desktop / Cursor compatibility | v0.5 |
| 10 | Streaming partial-graph updates (Section J, bet 4) | 1 week | UX polish, demo-quality interactive feel | v0.6 |

---

## The 12-Week Project Plan

| Week | Deliverable | Verifiable result |
|---|---|---|
| **W1** | 8k synthetic dataset generated via Claude Opus 4.5; 5% adversarial slice complete; HF dataset uploaded as `rohansx/ctxgraph-extract-v1` | Dataset row count = 8,000; jsonschema-validates 100%; manual spot-check 30 random rows |
| **W2** | Hermes-4-14B-LoRA-v1 + Qwen3-8B-LoRA-v1 trained on Together; HF model card published | Together run cost logged; val loss curves attached; replication snippet works |
| **W3** | New eval on 29-episode fixture, Re-DocRED subset, LongMemEval-S; blog post draft with raw numbers | Numbers vs Graphiti, Zep, Hermes-4-70B teacher, MemMachine; CSVs in repo |
| **W4** | Bi-temporal `invalidates:` shipped in extraction prompt; v0.3.0 tagged | 1,000-episode synthetic temporal-conflict eval; precision/recall of invalidations |
| **W5** | DeepInfra fallback wired in; `Event` node + 3 event relations land | Re-run 29-ep fixture; check event-recall on a curated 50-event subfixture |
| **W6** | Hacker News launch: "ctxgraph 0.3 — beats Graphiti by +22 F1 with a 14B LoRA, runs offline on a 6GB GPU" | HN front page or not; survives `grep` |
| **W7** | mistral.rs integration spike; embedded inference works end-to-end on RTX 4050 | Local extraction round-trip in pure Rust binary; tokens/sec measured |
| **W8** | A-MEM `memory_notes` table + append-only evolution; v0.4-alpha tagged | LongMemEval-S accuracy ≥ 75% (mid-tier between Zep and EmergenceMem) |
| **W9** | Phase-2 GRPO fine-tune on 1.5k adversarial pairs | Phase-2 LoRA beats Phase-1 LoRA on adversarial slice; write paper section |
| **W10** | Graph-Judge nightly pass; 1.5B judge LoRA trained on 3k labeled triples | Triples flagged with stale-confidence ≥ 95% precision on hand-labeled audit |
| **W11** | MCP-as-resource endpoint; Claude Desktop demo video | Demo: Claude Desktop reading ctxgraph entities/edges as native resources |
| **W12** | Second HN post: "ctxgraph 0.4 — embedded LLM in Rust, A-MEM memory, GRPO-tuned extractor, paper preprint" + arXiv preprint draft | Preprint up; reproduce-from-zero CI green |

---

## Recommendations (consolidated)

**SHIP:**
1. The two LoRA fine-tunes (Hermes-4-14B + Qwen3-8B), v0.3, weekend-1.
2. Bi-temporal `invalidates:` in the extraction prompt, v0.3.
3. DeepInfra Gemma-4-26B-A4B as cloud Tier 3, replacing OpenRouter gpt-4o-mini, v0.3.
4. `Event` node type + 3 event-relations, v0.3.
5. mistral.rs embedded inference, v0.4.
6. A-MEM-style append-only memory notes, v0.4.
7. Graph-Judge offline pass, v0.5.

**SKIP (for v1):**
1. MIRIX's 8-agent system (taxonomy yes, multi-agent no).
2. Speculative decoding on local 4–14B targets (peer-reviewed evidence it hurts at this scale).
3. Multi-modal episodes (RTX 4050 cannot do VLMs well in 2026 yet).
4. Federated CRDT graphs (open research problem; not engineering).
5. Browser-WASM ctxgraph (wrong battle at current model sizes).

**SKEPTICALLY TEST BEFORE CLAIMING:**
1. Any "content preservation" or entity-vs-event coverage delta — re-run on your fixture; AutoSchemaKG's published headline is the 92% schema-alignment figure, not a content-preservation percentage.
2. NuMind's "GPT-4o quality at 100× smaller" — applies to NuExtract-2.0-PRO (closed), not the open 8B (which they themselves describe as "a bit better than non-reasoning frontier models" at 73 F-Score).
3. MIRIX's 85.4% LoCoMo — below MemMachine's 91.69% you found in pass 1; if you cite MIRIX numbers, cite both.
4. Hermes 4 training scale — arXiv tech report says ~19B tokens, HF model card says ~60B tokens; cite the arXiv paper number.

---

## Caveats

- **The 0.687 / 0.745 / 0.460 numbers from pass 1 are your numbers, on your fixture.** A hostile HN reader will ask: "is the fixture biased toward your prompt template?" The honest answer is yes — every gold fixture is biased toward how its creator thinks. Mitigate by publishing the fixture *with* the launch and inviting community PRs.
- **Hermes 4 14B's exact LR/batch/epoch numbers are not disclosed in the technical report.** The config above follows community defaults (LR 1e-4, rank 32, 3 epochs). Run a small LR-finder before the full job if you have an extra hour of GPU.
- **NuExtract-2.0-4B's license inheritance from Qwen2.5-VL-3B is unclear** (NuMind themselves flagged it). If you decide to use NuExtract as a base, confirm with NuMind / Alibaba before commercial deployment.
- **The "~5M samples" Hermes 4 figure is consistent across sources, but the token-count number disagrees** (19B in arXiv tech report, ~60B on HF model card). Cite the arXiv number; if pressed, note the conflict.
- **mistral.rs is a single-maintainer project.** Pin a release tag. Keep the Ollama path as a fallback flag for users who want a more battle-tested runtime.
- **None of these benchmarks (Re-DocRED, LongMemEval-S, LoCoMo) test ctxgraph's actual production behavior.** Your 29-episode fixture is what matters; the public benchmarks are for cross-comparison only.
- **The Graph-Judge paper (arXiv 2411.17388) reports SOTA on three datasets but I did not extract the exact F1 deltas in this pass.** Read the paper before citing specific numbers in launch material.
- **The H100 / Together / DeepInfra prices in Section G were verified against pages dated March–May 2026.** Re-check before publishing the launch post — these markets move monthly.
- **Memory evolution from A-MEM has a published risk profile** (SSGM paper, arXiv 2603.11768) around silent rewriting of historical notes. Append-only evolution is the safe path; the original A-MEM paper does not make this constraint explicit.
- **UniIE, GenIE, MuSEE** were named in the query but have less-disclosed training recipes than UniversalNER/GoLLIE/NuExtract/Hermes-4 and don't change the recommendation — they're mentioned for completeness, not as base-model candidates.
- **"+22 F1 vs Graphiti" headline number for the W6 launch is the *current* baseline delta (0.687 − 0.460 = 0.227)**; you should re-measure after the LoRA lands. The headline only holds if the LoRA at least matches the 0.687 baseline (it should; the question is whether it beats Hermes 4 70B's 0.745).
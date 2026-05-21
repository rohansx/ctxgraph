# ctxgraph Pass-3 Adversarial Fact-Check

**Pass-3 verdict: pass 2 is mostly defensible, but ~10 claims will not survive a hostile `grep` on HN.** The big landmines are (a) RL-Struct VRAM number (38% → actually 40%), (b) the speculative-decoding citation chain (two of three papers are miscited), (c) several benchmark numbers cite the wrong backbone model, (d) pricing claims that round vendor numbers in your favor, and (e) four 2026 competitors (Anthropic Managed Memory, Letta, Mastra OM, EverMemOS) that pass 2 missed entirely. Below: precisely what's wrong, what's unverifiable, what verifies, what was deferred, and the new angles you need before HN day.

---

## 1. Claims that are WRONG

### B10. RL-Struct VRAM reduction — pass 2 says "38% less peak VRAM than PPO", paper says **40%**
arXiv 2512.00319 abstract: *"Leveraging Gradient Regularized Policy Optimization (GRPO), we enable the model to internalize these constraints without the need for a separate critic network, reducing peak VRAM usage by 40% compared to PPO."* The 89.7% structural / 92.1% JSON validity numbers verify. The ">80% with 1000 samples" claim is **not in the abstract or fetched sections** — it needs a direct page-pin from a v2 PDF table before you ship. **Action**: change to "40% less peak VRAM than PPO" and drop the 1000-sample sentence unless you can quote the table.

### R1. Speculative decoding citation chain — TWO OF THREE PAPERS ARE MISCITED
This is your single biggest HN risk because the citations are trivially checkable.

- **arXiv 2509.04474** (Sun et al., "Scaling Up, Speeding Up", Aug 2025): *partially* supports the claim. Model-based SpS with a 0.6B Qwen3 draft gives **0.87× speedup on Qwen3-8B** (a slowdown). But EAGLE-3 still achieves **2.91× on QW3-8B** and 2.23× on QW3-14B. The honest claim is: "naïve model-based speculative sampling with a sub-1B draft can be net-negative on 8B targets" — NOT "speculative decoding hurts at 4-14B."
- **arXiv 2402.01528** ("Decoding Speculative Decoding", Yan et al., 2024): **does not study 4-14B at all.** Target models are LLaMA-65B and OPT-66B. Cite for the "draft latency dominates throughput" principle only.
- **arXiv 2412.18934** (Dovetail, Dec 2024): **does not argue spec decoding hurts at 4-14B.** It proposes CPU/GPU heterogeneous draft+target placement for consumer hardware. Orthogonal.

**Action**: collapse this section to a single sentence: "For ctxgraph's 4-14B target band, naïve model-based speculative sampling with a small draft is often net-negative (arXiv 2509.04474 §4, Qwen3-8B at 0.87×); EAGLE-3 style methods still pay off, so the right default is *no draft model unless you measure a win*."

### R5. ASER 2.0 — "15 discourse relation types" is slightly wrong
ASER 2.0 uses **14 PDTB discourse relations + 1 Co-Occurrence relation = 15 relation types in 5 categories** (Temporal, Contingency, Comparison, Expansion, Co-Occurrence). Co-Occurrence is not a PDTB discourse relation. 438M eventualities, 648M edges, 11B-token corpus all verify. **Action**: rewrite as "15 relation types (14 PDTB discourse + 1 co-occurrence)."

### B19. Re-DocRED avg triples per test doc — 34.9, not 34.7
Official Re-DocRED GitHub README stats: Avg # Triples is **28.1 train / 34.6 dev / 34.9 test**. 34.7 is the pooled dev+test average from the paper text. 3,053/500/500 docs, 96 relation types, 6 entity types, MIT license all verify. **Action**: change to "34.9 avg triples per test doc."

### B12. A-MEM "SOTA across six base models on LoCoMo" — misleading paraphrase
The paper actually says *"Empirical experiments on six foundation models show superior improvement against existing SOTA baselines"* — A-MEM beats baselines across six models, it does not claim SOTA *across* them. The later MemoryOS paper (arXiv 2506.06326) explicitly shows A-MEM is no longer SOTA on LoCoMo as of mid-2025. **Action**: soften to "A-MEM reports consistent improvement over baselines across six foundation models on LoCoMo (Xu et al., 2025, arXiv 2502.12110)."

### B16. EmergenceMem 86% on LongMemEval — verifies as a number, but mislabel risk
The 86.00% is **EmergenceMem Internal**, which Emergence AI explicitly say is **not publicly reproducible** (emergence.ai/blog/sota-on-longmemeval-with-rag). Their two open configs are **EmergenceMem Simple 82.40%** and **EmergenceMem Simple Fast 79.00%**. Citing "86%" bare will be challenged. **Action**: cite as "EmergenceMem Internal 86% (closed, non-reproducible); public configs 79–82.4%."

### B18. Zep/Graphiti LongMemEval — pass 2 conflates two numbers
- **71.2%** is Zep with **GPT-4o** in the original Jan 2025 paper (Rasmussen et al., arXiv 2501.13956). Verified.
- **72.27%** is the April 2025 follow-up blog (blog.getzep.com/gpt-4-1-and-o4-mini-is-openai-overselling-long-context) — Zep used **GPT-4o** for graph build and retrieval against an o4-mini full-context baseline. Same architecture, slight prompting tweaks.

Cite separately and name the model. Also: an independent reproduction (cited in Gamgee 2026, arxiv:2512.13564) reports Zep at **63.8%**, calling 71.2% self-reported.

### Pricing — three pass-2 numbers are wrong
- **P2.** DeepInfra Qwen3-32B is **$0.08 input / $0.28 output, $0.13 blended (3:1)** — not "$0.08 flat" (Artificial Analysis, deepinfra.com/Qwen/Qwen3-32B, April 2026). Pass 2's "$0.08/M flat" understates output cost by 3.5×.
- **P6.** RunPod H100 PCIe is **$2.39/hr on-demand** per the official runpod.io/gpu-models/h100-pcie page (not $1.99). Sub-$2 numbers exist only on Vast.ai ($1.49–$1.87) and HPC-AI ($1.99); cite those sources separately if you want the low end.
- **B9.** Hermes 4 14B tokens — *both* numbers are correct but pass 2 conflates them. The **~19B** figure (arXiv 2508.18255: *"approximately 5 million samples totaling 19 billion tokens"*) is the **14B model's SFT corpus**. The **~60B** figure on the HF card is the **full Hermes 4 series corpus across 14B/70B/405B** (5M samples × ~5× reasoning-heavy token density). Use 19B as the comparison for ctxgraph distillation cost.

---

## 2. Claims that are UNVERIFIABLE — don't cite without primary source

| # | Claim | Why unverifiable |
|---|---|---|
| **B6** | NuExtract 2.0 8B "73 F-Score" | NuMind blog shows the chart but doesn't publish the benchmark set, sample sizes, or full competitor list. Label as "NuMind-reported." |
| **B10b** | RL-Struct ">80% with 1000 samples" | Not in abstract; pin a table page or drop. |
| **B13** | MemMachine 91.69% LoCoMo / 93.0% LongMemEval | Self-published on memmachine.ai; arXiv 2604.04853 is a self-preprint. A Sept 2025 MemMachine blog cited only **0.8487 LoCoMo** — so 0.9169 is a recent v0.2 jump. Cite as "MemMachine v0.2-reported, preprint, not peer-reviewed." |
| **C3** | mistral.rs "86 tok/s on A10 for Mistral-7B" | No third-party benchmark found. Treat as vendor-reported or drop. |
| **P3, P4** | DeepInfra Gemma 4 26B-A4B "$0.08/M flat" and Llama 3.1 8B "$0.06/M flat" | Could not confirm these exact SKUs/prices in DeepInfra pricing pages or Artificial Analysis catalog (artificialanalysis.ai/providers/deepinfra cites Llama 3.1 8B at $0.02–0.03 blended, not "$0.06 flat"). **Re-verify on the day of launch.** |
| **P10** | "Together $0.48/M covers training" | Together docs explicitly say *"Total tokens processed = (n_epochs × n_tokens_per_training_dataset) + (n_evals × n_tokens_per_validation_dataset)"*. The pass-2 math (8,000 × 1,200 × 3 = 28.8M × $0.48 = $13.82) is correct **only if you do no evals**. Add the asterisk. |

---

## 3. Claims that VERIFY — move along

- **B1** AutoSchemaKG 92% semantic alignment — verified (arXiv 2505.23628 v3 abstract; pass 1's 95% was the v1 ResearchGate draft, 92% is current).
- **B2** 900M+ nodes, 5.9B edges, 50M docs — verified verbatim.
- **B3** UniversalNER 7B 41.7 avg F1 vs ChatGPT 34.9 — verified (paper §5; "+6.8" sits inside the paper's "7–9 absolute F1 points" claim).
- **B4** 45,889 / 240,725 / 13,020 / Pile — verified.
- **B5** NuExtract 1.0 50k from 300k C4 — verified verbatim.
- **B7** NuExtract 2.0 PRO **+9 F-Score over GPT-4.1** — verified for the **closed PRO API** only ("20/21 problems used as Claude could not process images past a certain size"). The open 2B–8B models top out at 73.
- **B8** GoLLIE 13 F1 absolute vs no-guidelines baseline — verified (Table 3, arXiv 2310.03668).
- **B11** MIRIX 35% / 99.9% / 85.4% — all verified. Critical caveat: **85.4% on LoCoMo uses gpt-4.1-mini, not gpt-4o**; the 410%/93.3% stats are vs Gemini long-context on ScreenshotVQA, not LoCoMo.
- **B14** Hindsight 91.4% / 89.61% — verified (arXiv 2512.12818). 91.4% requires a "larger backbone"; the open 20B backbone gets 83.6% / 85.67%. Cite the backbone.
- **B17** EverMemOS 83.0% LongMemEval — verified (arXiv 2601.02163; +5.2pp over MemOS).
- **R2** llguidance ~50µs JSONSchemaBench — verified verbatim in README: *"the average mask computation in JSON Schema Bench (2.5M tokens, 10k schemas) is under 50μs, with less than 1% of masks taking longer than 1ms"*.
- **R3** XGrammar <40µs — verified verbatim (arXiv 2411.15100 §4.1: *"under 40 µs per token for JSON Schema and CFG (JSON)"*; Figure 9 shows 36 µs for both).
- **R6** EventKG 690k events / 2.3M temporal relations — verified verbatim ("over 690 thousand … over 2.3 million temporal relations").
- **L1** Hermes 4 14B Apache 2.0 base — verified (Qwen3-14B is Apache 2.0; Hermes 4 14B inherits).
- **L4** mistral.rs MIT — verified.
- **L5** A-MEM MIT — verified (*"This project is licensed under the MIT License"*, github.com/agiresearch/A-mem).
- **P1** Together $0.48/M LoRA up to 16B — verified (CloudZero, ToolHalla March 2026, PricePerToken).
- **P5** Together Llama 3.3 70B $0.88/$0.88 — verified (aipricing.guru, April 2026).
- **P8** Cerebras free 1M tokens/day — verified, **but Qwen3-32B is deprecating Feb 16, 2026** per the Cerebras docs (one inference-docs page lists May 27, 2026 for Qwen3-235B). The tier survives; that specific model does not. Plan migration to gpt-oss-120B or Llama 3.1 8B.
- **P9** Groq free tier — verified at **30 RPM, 1K req/day on 70B models, 14.4K on 8B**. Useful for dev, tight for production.

---

## 4. Items DEFERRED — not investigated this pass (call out so you know to verify)

These were in pass 2 but I did not get a tool round to them and the search budget exhausted. **Treat as unverified** and run quick checks before launch:

| # | Claim | What to verify |
|---|---|---|
| L2 | Qwen2.5-VL-3B "non-commercial license per NuMind" | Check huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct for the actual license file. Qwen's policy historically: ≤72B research, >72B commercial — confirm the 3B variant. |
| L3 | NuExtract-2.0-2B "commercially OK" because Qwen2-VL-2B base | Check the Qwen2-VL-2B HF card — Qwen2-VL-2B is generally Apache 2.0 but verify. |
| L6 | AutoSchemaKG GitHub license | Check github.com/HKUST-KnowComp/AutoSchemaKG — paper is CC BY 4.0 per ResearchGate, but code license may differ. |
| L7 | Graphiti license | Independent sources cite Apache 2.0 and the project has 24.8K stars per Gamgee 2026. **Likely verifies but I did not directly confirm.** |
| R4 | Graph Judge (arXiv 2411.17388) "SOTA on two general + one domain-specific text-graph pair dataset" | Did not re-investigate this pass. |
| P7 | Modal H100 serverless "~$3.25/hr equivalent" | Did not check modal.com pricing directly. |
| C1, C2, C4 | Python-only deps, mistral.rs capabilities, Cerebras/Groq production-suitability | Plausible based on what I gathered, but not individually fact-checked. |

---

## 5. NEW angles pass 2 missed

### NEW1. Latest 2026 agent memory benchmarks beyond LoCoMo/LongMemEval

LoCoMo and LongMemEval_S are saturated; top scores cluster at 85–95%, so they no longer discriminate. The actual 2026 battleground:

- **BEAM-1M** (1M-token scale). Hindsight reports 64.1% at 10M tokens; True Memory Pro 76.6% at 1M (arXiv 2605.04897). **Where Python-based Graphiti/Mem0 fall over and where ctxgraph can win.**
- **EverMemBench** (arXiv 2602.01313, late 2025) — first multi-party, multi-group, 1M+ token benchmark with 2,400 QA pairs; designed to break LoCoMo-saturated systems. **Lead with this for differentiation.**
- **HaluMem** — hallucination-in-memory; EverMind reports 93.04%. Underused, differentiating.
- **MemBench (ACL 2025)** — 8,500 items; MemPalace hybrid top-5 R@5 = 80.3%.

**Recommendation**: don't lead with LoCoMo. Lead with **BEAM-1M + HaluMem + EverMemBench**.

### NEW2. Synthetic data generation tooling for typed-schema training data

The realistic 2026 stack for ctxgraph's teacher→student distillation:

- **distilabel** (Argilla / HF, Apache 2.0) — declarative pipeline DSL with built-in `TextGenerationToJSON`, OpenAI/vLLM backends. Best fit for "use Claude / GPT-5 / DeepSeek V4 to label C4 + domain corpus, filter, ship."
- **NVIDIA NeMo Curator** — heavy, trillion-token GPU pipeline. Overkill for ctxgraph.
- **NuMind's NuExtract recipe** — the C4→template→filter pipeline IS the GoLLIE/NuExtract methodology. Reuse, don't reinvent.
- **Ai2 open-instruct + lighteval** — SFT loop + eval, both Apache 2.0.

**Recommendation**: distilabel + a teacher (Qwen3-235B or DeepSeek V4 Flash on DeepInfra) + NuExtract-style filtering.

### NEW3. Proprietary alternatives — MASSIVE pass-2 gap

Pass 2 missed what shipped between Dec 2025 and May 2026:

- **Anthropic Memory tool (API)** — platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool. Client-side filesystem-style tool exposing a `/memories` directory; Claude makes tool calls, app executes locally. ZDR-eligible (tool type `memory_20250818`). **Directly competitive with ctxgraph for API customers — your highest-priority positioning target.**
- **Claude.ai chat memory** — rolled to all users (free tier included) **March 2, 2026**. Prospective-only.
- **Anthropic Managed Agents Memory** — public beta **April 23, 2026**. Managed-filesystem memory stores, exportable via API/Console. Rakuten case study (Yusuke Kaji, GM AI for Business, on claude.com/blog/claude-managed-agents-memory, April 23, 2026): *"Our agents distill lessons from every session, delivering 97% fewer first-pass errors at 27% lower cost and 34% lower latency."* Other early adopters: Netflix, Wisedocs, Ando.
- **Anthropic "Dreaming"** — May 7, 2026 research preview; scheduled background memory consolidation.
- **OpenAI ChatGPT Memory** (May 5, 2026): layered scopes (Saved Memories + Reference Chat History + Memory Sources panel + Project Memory + Gmail connector). No public mechanism paper.
- **Letta (ex-MemGPT, github.com/letta-ai/letta)** — **active, Apache 2.0**, three-tier Core/Recall/Archival memory. Cited at **LoCoMo 74.0% (Filesystem)** and **LongMemEval ~83%**. Pricing: Pro $20/mo, Max $200/mo; self-host free. **Letta Code shipped December 16, 2025** and is the "#1 model-agnostic OSS harness on TerminalBench, achieving 42.5% overall score, ranking 4th overall and 2nd among agents using Claude 4 Sonnet" (letta.com/blog/terminal-bench, Dec 16, 2025). **Your closest open-source competitor.**
- **Mem0 (core: Apache 2.0, 47.8K+ stars)** — Per mem0.ai/pricing as summarized by Vectorize (vectorize.io/articles/mem0-alternatives) and DEV Community 2026: *"Free tier gives you 10K memories and 1K retrieval calls/month. After that, it's $19/mo for 50K memories, then $249/mo for Pro. The $19 to $249 jump is a 13× price increase."* Pro adds graph memory + analytics; Enterprise is custom. Independent benchmarks put Mem0 at **49.0% on LongMemEval** vs Mem0's own higher self-reports.
- **Mastra Observational Memory (OM)** — open-source, released **February 9, 2026** per Tyler Barnes (@tylbar) tweet of Feb 9, 2026: *"Announcing a new SOTA memory system, Observational Memory (OM)… gpt-4o 84.2%, gpt-5-mini 94.9%"*. **84.23% LongMemEval with gpt-4o is the highest openly-reproducible score**; 94.87% with gpt-5-mini is the record. Fully open end-to-end at mastra.ai/research/observational-memory.
- **EverMemOS / EverMind** — Engram-inspired three-phase; **83.0% LongMemEval, 94.5% LoCoMo, 93.04% HaluMem**. Uses Neo4j + GPU-served embedder.

**Implication**: pass 2's framing ("Python competitors need Python deps") understates the threat. The real threats are **Anthropic Managed Memory** for Claude-using teams and **Letta** for self-hosters. ctxgraph must differentiate as: single Rust binary, SQLite-only, no Python, no Neo4j, no GPU. That story holds — none of the above can claim all five.

### NEW4. Rust AI infra ecosystem story — real, not marketing

Production projects to lean on:
- **Qdrant** — Rust vector DB, **31K stars** (GitHub org repos page, updated May 12, 2026: "Rust 31,242 Apache-2.0"; Ventureburn March 2026 cited 29K at its $50M Series B). Apache 2.0.
- **LanceDB** — Rust columnar vector DB (Arrow-backed). Apache 2.0.
- **candle** (HF) — Rust ML framework. Apache 2.0.
- **burn** — Rust DL framework. Apache 2.0.
- **mistral.rs** — Rust inference. MIT.
- **tantivy** — Rust full-text search. MIT.
- **SurrealDB** — multi-model, Rust. AGPL/commercial.
- **Polars** — Rust dataframes. MIT.

**Play**: position ctxgraph as part of "the Rust-native AI stack." Pair docs: ctxgraph + LanceDB + candle + mistral.rs = **zero Python end-to-end**. None of the Python competitors can match this.

### NEW5. Best Rust-AI-infra HN posts to model in 2025–26

What worked:
- **sqlite-vec Show HN (Alex Garcia)** — single C file, embedded, no daemon. Lead with the install one-liner.
- **LanceDB Show HN** — benchmark table vs FAISS/Pinecone *in the post*, not the README.
- **candle Show HN** — local Whisper demo GIF.
- **Qdrant launch** — "Rust + production scale + customer logos."
- **MIRIX Show HN (Jul 2025)** — screenshot demo video; visceral, not benchmark-led.
- **mistral.rs Show HN** — supported features as a checklist, not prose.

What fails: leading with benchmark tables. Lead with a **90-second video** of ctxgraph ingesting Claude Code logs and answering queries, single binary, zero setup.

### NEW6. SQLite + vector + graph competition

- **sqlite-vec** (Alex Garcia, MIT, single C file) — the 2026 de facto SQLite vector extension. **Embed or interop; don't reimplement HNSW in Rust.**
- **libsql / Turso** — SQLite fork + HTTP + vectors. AGPL.
- **sqlite-vss** — older, FAISS-based, deprecated.
- **USearch** — header-only C++ vector index with SQLite bindings.
- **No production "SQLite + knowledge graph" project I could find.** ctxgraph's gap.

**Recommendation**: wrap sqlite-vec for vectors and explain the choice in your README's architecture section.

### NEW7. Distribution and packaging — the "single binary" claim

To deliver "single binary":
- **cargo binstall** — free if you publish to crates.io with binaries in CI.
- **homebrew tap** — one-day setup.
- **AUR** — community-maintained once you have stars.
- **debian package** via `cargo-deb`.
- **Docker image** — multi-arch, statically linked.
- **MCP server entry** in `awesome-mcp-servers` and `modelcontextprotocol/servers` — highest leverage.

Path to top-3 in awesome-mcp-servers: PR with working demo, one-line install, real benchmark in README.

### NEW8. Bi-temporal modeling — citable 2025–26 work

Beyond Zep/Graphiti and the OpenAI cookbook:
- **Graphiti paper** (arXiv 2501.13956, Jan 2025).
- **TiMem** (Li et al., Jan 2026) — Temporal Memory Tree, **76.88% LongMemEval, 27% memory footprint reduction**.
- **EverMemOS** (Hu et al., Jan 2026, arXiv 2601.02163) — Engram-inspired three-phase.
- **TKGE benchmark suite**: TKBC, ICEWS14/05-15, Wikidata12k, YAGO11k.
- **Hindsight** (arXiv 2512.12818) — temporal/entity-aware memory layer.

**Recommendation**: cite Graphiti + TiMem + Hindsight in your README's "related work."

### NEW9. GLiNER2 status

Pass-2 sources I gathered don't deeply cover GLiNER2. As of late 2025, **GLiNER and GLiNER-multi remain SOTA-competitive for zero-shot NER under 1GB**. The field has bifurcated: **GLiNER for pure NER, NuExtract for full structured extraction**. Still viable for ctxgraph's NER component; **partial-verify** the current GLiNER2 release before citing as "current best."

### NEW10. Agent integration patterns — where ctxgraph slots in

- **Claude Code** — Anthropic Memory tool via `/memories` filesystem. ctxgraph becomes the backend via an MCP server reading/writing `/memories`-shaped paths. **Highest-leverage integration.**
- **Cursor** — `.cursor/rules` + proprietary chat memory. Less hookable.
- **Cline** — Memory Bank pattern (markdown files in repo). Natural fit.
- **Windsurf** — Cascade memory. Less open.
- **OpenCode / Aider** — ad-hoc.

**README's first code block**: `cargo install ctxgraph` → `ctxgraph serve --mcp` → 5-line `.claude/mcp.json` → Claude Code persists to ctxgraph.

---

## 6. Implications for pass 2's 12-week plan

| Change | Why |
|---|---|
| **Collapse the speculative-decoding paragraph to one sentence** | Two of three citations don't support the claim. Use only arXiv 2509.04474 with the Qwen3-8B 0.87× figure. |
| **Stop leading with LoCoMo / LongMemEval as headline benchmark** | Saturated at 85–95% by Mastra OM, OMEGA, MIRIX, EverMemOS. Lead with **BEAM-1M + HaluMem + EverMemBench**. |
| **Add Anthropic Managed Memory + Letta as primary competitors in README** | Bigger threats than Graphiti as of May 2026. Position ctxgraph as the **only** single-binary + SQLite-only + no-Python + no-Neo4j + no-GPU option. |
| **Don't reimplement vector indexing — embed sqlite-vec** | Or explain in the README why you didn't. HN will ask. |
| **Reframe Hermes 4 distillation budget around 19B, not 60B tokens** | The 19B is the 14B SFT corpus (arXiv 2508.18255 abstract); 60B is the cross-series total. Use 19B for cost calcs. |
| **Fix the pricing table** | DeepInfra Qwen3-32B is $0.08 in / $0.28 out, $0.13 blended (not $0.08 flat). RunPod H100 PCIe is $2.39/hr official (not $1.99). |
| **Add a "benchmark methodology" callout** | Acknowledge LongMemEval/LoCoMo numbers are largely vendor-self-reported (Zep–Mem0 "Lies, Damn Lies & Statistics" controversy is well-known). Pre-empt the HN comment by stating your evaluation methodology explicitly. |
| **Plan Cerebras model migration** | Qwen3-32B on Cerebras deprecates Feb 16, 2026. Migrate to gpt-oss-120B or Llama 3.1 8B on the same free tier. |
| **First README code block: Claude Code + MCP integration** | Highest-leverage adoption path. Don't bury it. |
| **Three small spec fixes hostile readers will catch** | RL-Struct **40%** VRAM not 38%; A-MEM "improvement over baselines" not "SOTA across six models"; Re-DocRED **34.9** triples not 34.7. |
| **Add "deferred for verification" list to the repo before launch** | L2, L3, L6, L7 licenses and R4 Graph Judge are unconfirmed — run them before HN. |

---

## Coverage table

| Item | Status |
|---|---|
| B1–B5, B7, B8, B11, B14, B17 | Verified |
| B6 | Unverifiable (vendor-only chart) |
| B9 | Reconciled (19B = 14B SFT corpus; 60B = full series) |
| B10 | **Corrected** (38% → 40% VRAM); >80%/1000-sample claim unverifiable |
| B12 | **Softened** ("improvement over baselines," not "SOTA across six models") |
| B13 | Flagged as MemMachine-self-reported preprint |
| B15 | User's own fixture number, ground truth |
| B16 | **Caveated** (Internal closed vs Simple/Simple Fast public) |
| B18 | **Reconciled** (71.2% Zep paper; 72.27% Zep blog; 63.8% independent) |
| B19 | **Corrected** (34.7 → 34.9) |
| P1, P5, P8, P9 | Verified |
| P2 | **Corrected** ($0.08 flat → $0.08 in / $0.28 out) |
| P3, P4, P7 | **Deferred** — re-verify on day of launch |
| P6 | **Corrected** ($1.99 → $2.39/hr official) |
| P10 | Math correct only without eval overhead; add caveat |
| R1 | **Corrected** (2 of 3 citations miscited) |
| R2, R3, R6 | Verified verbatim |
| R4 | **Deferred** |
| R5 | **Corrected** (15 discourse → 14 PDTB + 1 co-occurrence) |
| L1, L4, L5 | Verified |
| L2, L3, L6, L7 | **Deferred** — L7 (Graphiti) likely Apache 2.0 but unconfirmed |
| C1, C2, C4 | Plausible, not individually fact-checked |
| C3 | **Unverifiable** — no third-party benchmark for "86 tok/s on A10" |
| NEW1–NEW10 | Covered with named sources |
# ctxgraph — Documentation Index

This folder is structured so you find the **right doc in one hop**, regardless of why you opened it.

---

## Canonical docs — start here

| If you want… | Read this |
|---|---|
| **The master working doc** — product, decisions, the 5 pieces to build, launch pitch | **`CLARITY.md`** ← start here |
| The architecture (as-built + v0.3 target, the universal schema, the read path, the 3 modes) | `ARCHITECTURE.md` |
| The roadmap (5 pieces + infrastructure + 12-week schedule + this-weekend todo) | `ROADMAP.md` |
| The benchmark results (the headline F1 numbers, hostile-reader audit) | `BENCHMARKS.md` |
| The raw session findings + the prompt for deep research | `research_brief.md` |
| The synthesis of four deep-research passes that informs everything | `deep-research/FINAL.md` |

Each canonical doc is **internally consistent and authoritative** as of 2026-05-14. If two docs disagree, the canonical one wins; if a canonical doc contradicts itself, file an issue.

**`CLARITY.md` is the master.** Everything else elaborates a section of it:
- `CLARITY.md` § 3 (the 5 pieces) → `ROADMAP.md` § "The 5 pieces" + 12-week schedule
- `CLARITY.md` § 4 (model strategy / 3 modes) → `ARCHITECTURE.md` § 8
- `CLARITY.md` § 5 (read path) → `ARCHITECTURE.md` § 7
- `CLARITY.md` § 6 (architecture diagram) → `ARCHITECTURE.md` overall
- `CLARITY.md` § 7 (rejected ideas) → `ROADMAP.md` § "What we explicitly rejected"
- `CLARITY.md` § 11 (launch pitch) → `ROADMAP.md` § "Launch pitch"
- `BENCHMARKS.md` is the measured-evidence backing for `CLARITY.md`'s headline claims.

---

## Folder layout

```
docs/
├── README.md                ← you are here (navigation index)
├── CLARITY.md               ← master working doc: product + 5 pieces + decisions + launch
├── ARCHITECTURE.md          ← authoritative architecture (as-built §1-4 + target v0.3 §5-14)
├── ROADMAP.md               ← authoritative roadmap (5 pieces + 12-week schedule)
├── BENCHMARKS.md            ← authoritative benchmark results + hostile-reader audit
├── research_brief.md        ← session findings + prompt for deep-research models
│
├── adr/                     ← Architecture Decision Records (historical, still authoritative for the decisions they document)
│   ├── 001-sqlite-over-neo4j.md
│   ├── 002-tiered-extraction.md
│   ├── 003-bitemporal-model.md
│   ├── 004-onnx-runtime-for-ml.md
│   ├── 005-rrf-search-fusion.md
│   └── 006-unified-gliner2-model.md
│
├── blog/                    ← drafted launch posts and dev.to articles
│   ├── sqlite-as-graph-database.md
│   ├── sqlite-as-graph-database-devto.md
│   └── we-replaced-neo4j-with-45-sql-statements.md
│
├── deep-research/           ← raw output from four deep-research passes (source material)
│   ├── FINAL.md             ← synthesis of all four passes (most important)
│   ├── claude-dr.md         ← pass 2 (architectural baseline)
│   ├── claude-dr-2.md       ← pass 3 (adversarial fact-check)
│   ├── chatgpt-dr.md        ← External-A (model + provider sweep)
│   ├── gemini-dr.md         ← External-B (hardware/caching/encoder)
│   └── grok.md              ← External-C (independent run on the deep-research prompt)
│
└── archive/                 ← superseded docs, preserved for history
    ├── ARCHITECTURE_v1.md   ← pre-research aspirational architecture
    ├── ROADMAP_v1.md        ← original 4-phase plan
    ├── benchmark_v0.6.md    ← original tech-only F1 vs Graphiti
    └── benchmark_v0.9_round1.md  ← round-1 5-model benchmark (Gemma 3n E4B + GPT-4o-mini)
```

Each archived doc has a deprecation banner at the top pointing to its replacement.

---

## Reading order for a new contributor

If you're picking this project up fresh, read in this order:

1. **`CLARITY.md`** end-to-end (~15 min) — the product, the 5 pieces, the model strategy, the read path, the rejected ideas, the launch pitch. Everything else is reference for this.
2. **`BENCHMARKS.md`** § "TL;DR" + the apples-to-apples table — the single most important measurement that backs the launch claim.
3. **`ARCHITECTURE.md`** § 1–4 (as-built) — what's actually in `crates/` today.
4. **`ARCHITECTURE.md`** § 5–14 (target v0.3) — what's changing in the next 8 weeks, organized by the 5 pieces + infrastructure plumbing.
5. **`ROADMAP.md`** § "The 5 pieces" + § "12-week schedule" + § "This weekend" — what to ship and when.

For deeper context:
- **`deep-research/FINAL.md`** — the synthesis of four research passes that informs the v0.3 target architecture.
- **`research_brief.md`** — the raw session findings + the deep-research prompt that produced FINAL.md.
- **`adr/`** — past architectural decisions (SQLite over Neo4j, ONNX over PyTorch, etc.) and the reasoning behind them.

---

## Where to file what

| You want to… | Put it in… |
|---|---|
| Propose a new architectural decision (not a feature) | New `adr/00N-*.md` file |
| Update measured benchmark numbers | Edit `BENCHMARKS.md`, drop raw JSON in `scripts/results/` |
| Add a finding from a new research pass | New `deep-research/*.md` file, then update `deep-research/FINAL.md` if it changes the synthesis |
| Update the implementation plan | Edit `ROADMAP.md` directly |
| Update what's in the code or what should be in the code | Edit `ARCHITECTURE.md` directly (use the §-A as-built / §-T target distinction) |
| Write launch/marketing content | New `blog/*.md` file |
| Preserve a doc that's being replaced | Move to `archive/` with a deprecation banner |

---

## What's NOT in this folder

| Looking for… | Look here |
|---|---|
| Source code | `../crates/` |
| Reproducible benchmark harnesses | `../scripts/` |
| Raw per-episode benchmark JSON outputs | `../scripts/results/` |
| Hand-labeled fixtures | `../crates/ctxgraph-extract/tests/fixtures/` |
| Top-level project README | `../README.md` (public-facing, currently being updated to match `BENCHMARKS.md`) |
| Contribution guide | `../CONTRIBUTING.md` |

---

*Index last updated 2026-05-13. If you add a new top-level doc to `docs/`, add a row to the "Canonical docs" table above.*

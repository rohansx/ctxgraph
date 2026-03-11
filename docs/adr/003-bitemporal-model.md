# ADR-003: Bi-Temporal Data Model

**Status**: Accepted
**Date**: 2026-03-11

## Context

Facts change over time. "Alice works at Google" may become false when she joins Meta. ctxgraph needs to handle evolving facts without losing history.

Options:

1. **Overwrite** — Update edges in place. Simple but loses history.
2. **Append-only** — Never modify, just add new records. Preserves history but makes "current state" queries expensive.
3. **Bi-temporal** — Two time dimensions: valid time (real-world truth) and transaction time (when recorded). Industry standard for temporal databases.

## Decision

Use **bi-temporal timestamps** on all edges:

- `valid_from` / `valid_until` — When was this fact true in the real world?
- `recorded_at` — When was this fact recorded in ctxgraph?

Facts are never deleted. They are invalidated by setting `valid_until`.

## Rationale

- **Context graphs are fundamentally about "why did we do X?"** — answering this requires historical context, not just current state.
- **Time-travel queries are a core feature.** "Who worked at Google in 2024?" requires knowing what was true at that point.
- **Audit trail comes free.** `recorded_at` tracks when information entered the system, supporting compliance and debugging.
- **Contradiction detection (Tier 3) depends on it.** When a new fact contradicts an old one, the old edge needs `valid_until` set, not deletion.

## Consequences

- **Positive**: Full history preservation. No data loss on updates.
- **Positive**: Time-travel queries are natural.
- **Positive**: Audit trail for when facts were recorded.
- **Negative**: "Current state" queries need `WHERE valid_until IS NULL` filter.
- **Negative**: More storage than overwrite model (but minimal — text edges are tiny).
- **Mitigation**: Default query mode is "current only". Time-travel is opt-in via `search_at()` or `--as-of` flag.

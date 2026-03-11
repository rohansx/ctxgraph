# ADR-001: SQLite over Neo4j for Graph Storage

**Status**: Accepted
**Date**: 2026-03-11

## Context

ctxgraph needs a graph storage backend. The two main options are:

1. **Neo4j** (or FalkorDB) — purpose-built graph database with Cypher query language.
2. **SQLite** — embedded relational database with recursive CTEs for graph traversal.

## Decision

Use **SQLite with recursive CTEs** as the sole storage backend.

## Rationale

| Factor | SQLite | Neo4j |
|---|---|---|
| Deployment | Embedded, zero config | Requires Docker or server process |
| Setup time | 0 seconds | 15-30 minutes |
| Dependencies | Single file (bundled via rusqlite) | JVM + Docker or native install |
| Full-text search | FTS5 built-in | Requires Lucene plugin |
| Graph traversal | Recursive CTEs | Native Cypher (faster at scale) |
| Scale ceiling | ~100K-1M nodes | Millions+ |
| Embeddability | Yes (in-process) | No (client-server) |
| Privacy | Nothing leaves process | Separate server process |

ctxgraph targets solo developers, small teams, and privacy-constrained environments — exactly the use cases where Neo4j's operational overhead is unjustified. Recursive CTEs provide sufficient graph traversal for graphs under 1M nodes, which covers the vast majority of target use cases.

## Consequences

- **Positive**: Zero infrastructure, instant startup, single-file database, embeddable in other Rust apps.
- **Positive**: FTS5 comes free — no separate search infrastructure.
- **Negative**: Graph traversal performance degrades at very large scale (>1M nodes). Not a concern for target use cases.
- **Negative**: No native graph query language. Recursive CTEs are more verbose than Cypher.
- **Mitigation**: If users outgrow SQLite, they can export to JSON and import into Neo4j. This is a "graduate out" path, not a limitation.

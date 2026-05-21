"""
Piece 3 — relation-vocabulary embeddings (Python prototype).

At startup: embed each of the 10 relation names + descriptions, cache.
At query time: embed the user's verb, cosine match, pick top-1.

The Rust target lives at crates/ctxgraph-extract/src/relation_match.rs.
This prototype validates the embedding approach with the same model
(all-MiniLM-L6-v2, 384-dim) before we commit to the Rust translation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import tomllib
from sentence_transformers import SentenceTransformer

DEFAULT_SCHEMA = (
    Path(__file__).parent.parent
    / "crates/ctxgraph-extract/schemas/universal.toml"
)


@dataclass(frozen=True)
class RelationMatch:
    relation: str
    score: float
    runner_up: str
    runner_up_score: float


class RelationMatcher:
    """Cosine-match a user's verb to one of the 10 typed relations."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.relation_names: list[str] = []
        self.relation_vectors: np.ndarray | None = None  # shape: (N, 384)

    def load_schema(self, path: Path | None = None) -> None:
        path = path or DEFAULT_SCHEMA
        with open(path, "rb") as f:
            data = tomllib.load(f)
        relations = data.get("relations", {})
        if not relations:
            raise ValueError(f"no [relations] table in {path}")
        self.relation_names = list(relations.keys())
        texts = [f"{name}: {desc}" for name, desc in relations.items()]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        self.relation_vectors = np.asarray(embeddings, dtype=np.float32)

    def resolve(self, user_verb: str) -> RelationMatch:
        if self.relation_vectors is None:
            raise RuntimeError("call load_schema() first")
        q = self.model.encode([user_verb], normalize_embeddings=True)[0]
        scores = self.relation_vectors @ q  # both normalized → cosine
        ranked = np.argsort(-scores)
        return RelationMatch(
            relation=self.relation_names[ranked[0]],
            score=float(scores[ranked[0]]),
            runner_up=self.relation_names[ranked[1]],
            runner_up_score=float(scores[ranked[1]]),
        )

    def resolve_many(self, verbs: Iterable[str]) -> list[RelationMatch]:
        return [self.resolve(v) for v in verbs]


def _selftest() -> None:
    """Quick correctness check — exits 0 on pass, 1 on fail."""
    matcher = RelationMatcher()
    matcher.load_schema()

    test_cases: list[tuple[str, str]] = [
        # (user verb, expected relation)
        ("depends on", "depends_on"),
        ("relies on", "depends_on"),
        ("needs", "depends_on"),
        ("requires", "depends_on"),
        ("is part of", "part_of"),
        ("belongs to", "part_of"),
        ("is owned by", "owned_by"),
        ("controlled by", "owned_by"),
        ("located at", "located_at"),
        ("happens in", "located_at"),
        ("caused", "caused"),
        ("led to", "caused"),
        ("resulted in", "caused"),
        ("happened before", "preceded"),
        ("cites", "references"),
        ("builds on", "references"),
        ("links to", "references"),
        ("participated in", "participated_in"),
        ("attended", "participated_in"),
        ("mentioned", "mentions"),
    ]

    correct = 0
    failures: list[str] = []
    for verb, expected in test_cases:
        m = matcher.resolve(verb)
        ok = m.relation == expected
        correct += int(ok)
        flag = "✓" if ok else "✗"
        print(
            f"  {flag} {verb!r:>22}  → {m.relation:<15} ({m.score:.3f})  "
            f"runner_up={m.runner_up} ({m.runner_up_score:.3f})"
        )
        if not ok:
            failures.append(f"{verb!r} → {m.relation}, expected {expected}")

    print()
    print(f"  accuracy: {correct}/{len(test_cases)} = {correct/len(test_cases):.1%}")
    if failures:
        print("  failures:")
        for f in failures:
            print(f"    {f}")

    # Threshold: ≥80% accuracy or we shouldn't ship this approach
    target_pct = 0.80
    if correct / len(test_cases) < target_pct:
        print(f"\n  FAILED — below {target_pct:.0%} threshold")
        raise SystemExit(1)
    print(f"  PASSED — ≥{target_pct:.0%} threshold")


if __name__ == "__main__":
    _selftest()

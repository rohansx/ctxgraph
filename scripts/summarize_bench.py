#!/usr/bin/env python3
"""Print a comparison table across all benchmark JSON results."""

import json
from pathlib import Path

RESULTS = [
    ("/tmp/bench_gpt4o_mini.json", "openai/gpt-4o-mini"),
    ("/tmp/bench_gemma3n_e4b.json", "google/gemma-3n-e4b-it"),
    ("/tmp/bench_gemma4_26b.json", "google/gemma-4-26b-a4b-it"),
    ("/tmp/bench_gemma4_31b.json", "google/gemma-4-31b-it"),
]

# Committed reference numbers
CTXGRAPH_LOCAL_TECH_COMBINED = 0.800  # from benchmark_comparison.json
CTXGRAPH_LOCAL_TECH_ENT = 0.837
CTXGRAPH_LOCAL_TECH_REL = 0.763
GRAPHITI_TECH_ENT = 0.570
GRAPHITI_TECH_REL_MAPPED = 0.104
GRAPHITI_TECH_COMBINED_MAPPED = 0.337


def fmt(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) else str(v)


def main():
    rows = []
    for path, name in RESULTS:
        if not Path(path).exists():
            print(f"missing: {path}")
            continue
        r = json.loads(Path(path).read_text())
        for bucket_key in ("tech", "cross_domain"):
            b = r.get(bucket_key)
            if not b:
                continue
            rows.append({
                "model": name,
                "bucket": bucket_key,
                "n": b["n_ok"],
                "ent_strict": b["ent_f1_strict"],
                "ent_fuzzy": b["ent_f1_fuzzy"],
                "rel_strict": b["rel_f1_strict"],
                "rel_pair": b["rel_f1_pair"],
                "combined_strict": b["combined_f1_strict"],
                "combined_pair": b["combined_f1_fuzzy_pair"],
                "avg_time": b["avg_time_s"],
                "cost_1k": b["cost_per_1k_episodes_usd"],
            })

    print("\n=== ENTITIES + RELATIONS, F1 (strict = exact name+type+rel) ===")
    print(
        f"{'model':<32} {'bucket':<14} {'n':>3} {'ent_str':>8} {'ent_fz':>8} "
        f"{'rel_str':>8} {'rel_pr':>8} {'cmb_str':>8} {'cmb_pr':>8} {'s/ep':>6} {'$/1kep':>8}"
    )
    print("-" * 130)
    for r in rows:
        print(
            f"{r['model']:<32} {r['bucket']:<14} {r['n']:>3} "
            f"{r['ent_strict']:>8.3f} {r['ent_fuzzy']:>8.3f} "
            f"{r['rel_strict']:>8.3f} {r['rel_pair']:>8.3f} "
            f"{r['combined_strict']:>8.3f} {r['combined_pair']:>8.3f} "
            f"{r['avg_time']:>6.2f} {r['cost_1k']:>8.3f}"
        )

    print("\n=== REFERENCE POINTS (committed in repo, 50 tech episodes) ===")
    print(f"  ctxgraph local-only         entity F1={CTXGRAPH_LOCAL_TECH_ENT:.3f}  relation F1={CTXGRAPH_LOCAL_TECH_REL:.3f}  combined={CTXGRAPH_LOCAL_TECH_COMBINED:.3f}")
    print(f"  Graphiti + GPT-4o (mapped)  entity F1={GRAPHITI_TECH_ENT:.3f}  relation F1={GRAPHITI_TECH_REL_MAPPED:.3f}  combined={GRAPHITI_TECH_COMBINED_MAPPED:.3f}")

    print("\n=== DELTAS vs Graphiti+GPT-4o (50 tech episodes, mapped scoring) ===")
    tech_rows = [r for r in rows if r["bucket"] == "tech"]
    for r in tech_rows:
        delta_strict = r["combined_strict"] - GRAPHITI_TECH_COMBINED_MAPPED
        delta_pair = r["combined_pair"] - GRAPHITI_TECH_COMBINED_MAPPED
        print(
            f"  {r['model']:<32} Δstrict={delta_strict:+.3f}   Δpair-fuzzy={delta_pair:+.3f}"
        )

    print("\n=== DELTAS vs ctxgraph local-only (50 tech episodes) ===")
    for r in tech_rows:
        delta_strict = r["combined_strict"] - CTXGRAPH_LOCAL_TECH_COMBINED
        delta_pair = r["combined_pair"] - CTXGRAPH_LOCAL_TECH_COMBINED
        print(
            f"  {r['model']:<32} Δstrict={delta_strict:+.3f}   Δpair-fuzzy={delta_pair:+.3f}"
        )


if __name__ == "__main__":
    main()

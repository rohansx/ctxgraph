#!/usr/bin/env python3
"""Compare all v2 benchmark results (LLM-alone + Graphiti) on the 29-ep cross-domain fixture.
All scores are pair-fuzzy F1 (most fair to free-form output like Graphiti's)."""

import json
from pathlib import Path

LLM_RESULTS = [
    ("/tmp/v2_gemma3_27b.json", "Gemma 3 27B (dense)"),
    ("/tmp/v2_gemma4_26b.json", "Gemma 4 26B-A4B (MoE)"),
    ("/tmp/v2_gemma4_31b.json", "Gemma 4 31B (dense)"),
    ("/tmp/v2_hermes4_70b.json", "Hermes 4 70B (IE-tuned)"),
    ("/tmp/v2_qwen3_30b.json", "Qwen 3 30B A3B (MoE)"),
]

GRAPHITI_RESULTS = [
    ("/tmp/v2_graphiti_gemma4_26b.json", "Graphiti + Gemma 4 26B"),
    ("/tmp/v2_graphiti_gemma4_31b.json", "Graphiti + Gemma 4 31B"),
]


def fmt(x, w=8, p=3):
    if x is None:
        return "n/a".rjust(w)
    return f"{x:>{w}.{p}f}"


def load(path):
    if not Path(path).exists():
        return None
    return json.loads(Path(path).read_text())


def main():
    print()
    print("=" * 110)
    print("CROSS-DOMAIN HEAD-TO-HEAD (29 episodes, 25 domains)")
    print("=" * 110)
    print()
    print(
        f"{'system':<40} {'n_ok':>6} {'ent_F1':>10} {'rel_F1':>10} "
        f"{'combined':>10} {'s/ep':>8} {'$/1k_ep':>10}"
    )
    print("-" * 110)

    # LLM-alone results (pair-fuzzy bucket from openrouter_bench.py)
    for path, name in LLM_RESULTS:
        r = load(path)
        if not r:
            print(f"{name:<40}  MISSING ({path})")
            continue
        b = r.get("cross_domain") or {}
        if not b:
            continue
        ent = b.get("ent_f1_fuzzy")  # pair-fuzzy for entities = substring match
        rel = b.get("rel_f1_pair")
        comb = b.get("combined_f1_fuzzy_pair")
        t = b.get("avg_time_s")
        cost = b.get("cost_per_1k_episodes_usd")
        n_ok = b.get("n_ok")
        print(
            f"{name+' (alone)':<40} {str(n_ok)+'/29':>6} {fmt(ent,10):>10} "
            f"{fmt(rel,10):>10} {fmt(comb,10):>10} {fmt(t,8,2):>8} ${fmt(cost,9,3)}"
        )

    print()
    # Graphiti results
    for path, name in GRAPHITI_RESULTS:
        r = load(path)
        if not r:
            print(f"{name:<40}  MISSING ({path})  (likely still running or failed)")
            continue
        ent = r.get("ent_f1_pair")
        rel = r.get("rel_f1_pair")
        comb = r.get("combined_f1_pair")
        t = r.get("avg_time_s")
        n_ok = r.get("n_ok")
        print(
            f"{name:<40} {str(n_ok)+'/29':>6} {fmt(ent,10):>10} "
            f"{fmt(rel,10):>10} {fmt(comb,10):>10} {fmt(t,8,2):>8} {'~6x cost':>10}"
        )

    print()
    print("=" * 110)
    print("APPLES-TO-APPLES (same LLM, different system)")
    print("=" * 110)

    # Direct ctxgraph vs Graphiti comparison with same model
    ct_26b = load("/tmp/v2_gemma4_26b.json")
    gr_26b = load("/tmp/v2_graphiti_gemma4_26b.json")
    if ct_26b and gr_26b:
        b = ct_26b["cross_domain"]
        ct_ent = b["ent_f1_fuzzy"]
        ct_rel = b["rel_f1_pair"]
        ct_comb = b["combined_f1_fuzzy_pair"]
        gr_ent = gr_26b["ent_f1_pair"]
        gr_rel = gr_26b["rel_f1_pair"]
        gr_comb = gr_26b["combined_f1_pair"]
        print(f"\n  Both systems using google/gemma-4-26b-a4b-it:")
        print(f"  {'metric':<28} {'ctxgraph (alone)':>22} {'Graphiti':>14} {'Δ (ctxgraph - graphiti)':>26}")
        print(f"  {'entity F1 (pair-fuzzy)':<28} {ct_ent:>22.3f} {gr_ent:>14.3f} {ct_ent - gr_ent:>+26.3f}")
        print(f"  {'relation F1 (pair-fuzzy)':<28} {ct_rel:>22.3f} {gr_rel:>14.3f} {ct_rel - gr_rel:>+26.3f}")
        print(f"  {'combined F1 (pair-fuzzy)':<28} {ct_comb:>22.3f} {gr_comb:>14.3f} {ct_comb - gr_comb:>+26.3f}")

    ct_31b = load("/tmp/v2_gemma4_31b.json")
    gr_31b = load("/tmp/v2_graphiti_gemma4_31b.json")
    if ct_31b and gr_31b:
        b = ct_31b["cross_domain"]
        ct_ent = b["ent_f1_fuzzy"]
        ct_rel = b["rel_f1_pair"]
        ct_comb = b["combined_f1_fuzzy_pair"]
        gr_ent = gr_31b["ent_f1_pair"]
        gr_rel = gr_31b["rel_f1_pair"]
        gr_comb = gr_31b["combined_f1_pair"]
        print(f"\n  Both systems using google/gemma-4-31b-it:")
        print(f"  {'metric':<28} {'ctxgraph (alone)':>22} {'Graphiti':>14} {'Δ (ctxgraph - graphiti)':>26}")
        print(f"  {'entity F1 (pair-fuzzy)':<28} {ct_ent:>22.3f} {gr_ent:>14.3f} {ct_ent - gr_ent:>+26.3f}")
        print(f"  {'relation F1 (pair-fuzzy)':<28} {ct_rel:>22.3f} {gr_rel:>14.3f} {ct_rel - gr_rel:>+26.3f}")
        print(f"  {'combined F1 (pair-fuzzy)':<28} {ct_comb:>22.3f} {gr_comb:>14.3f} {ct_comb - gr_comb:>+26.3f}")


if __name__ == "__main__":
    main()

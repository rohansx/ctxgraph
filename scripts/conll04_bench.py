#!/usr/bin/env python3
"""
Third-party benchmark: CoNLL04 relation extraction (schema NEITHER tool authored).

Addresses the bias-audit's "open-book gold" finding — the cross_domain fixtures
were labeled in ctxgraph's own schema. CoNLL04 is a standard, externally-defined
RE dataset (entities Person/Org/Location/Other; directional relations Work_For,
Live_In, Located_In, OrgBased_In, Kill). We score with a STRICT DIRECTIONAL +
TYPED relation metric (head→tail order and relation type must match) — the metric
that actually matters for a typed knowledge graph, not the lenient pair-fuzzy.

Adapt data first (writes /tmp/ctxgraph-bakeoff/conll04_episodes.json), then:
  OPENROUTER_API_KEY=... ./.venv-bench/bin/python scripts/conll04_bench.py \
      --model deepseek/deepseek-v3.2 --limit 100 [--verify] --out OUT.json
"""
import argparse, json, os, sys, time
from pathlib import Path
import requests

EPISODES = Path("/tmp/ctxgraph-bakeoff/conll04_episodes.json")
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"  # overridable via --base-url (e.g. ollama)
ENTITY_TYPES = ["Person", "Organization", "Location", "Other"]
RELATION_TYPES = ["Work_For", "Live_In", "Located_In", "OrgBased_In", "Kill"]

SYSTEM_PROMPT = f"""Extract entities and relations from the sentence.
Reply with ONLY valid JSON — no prose, no fences.

Entity types: {", ".join(ENTITY_TYPES)}
Relation types (DIRECTION matters — head is the subject):
- Work_For:    head=Person,        tail=Organization   ("X works for Y")
- Live_In:     head=Person,        tail=Location       ("X lives in Y")
- Located_In:  head=Location,      tail=Location       ("X is located in Y")
- OrgBased_In: head=Organization,  tail=Location       ("org X is based in Y")
- Kill:        head=Person,        tail=Person         ("X killed Y")

Rules:
- Use the entity's exact surface name from the sentence.
- head/tail MUST be names in your entities list, in the correct direction.
- Only emit relations explicitly stated. Prefer fewer correct over many guesses.

Schema: {{"entities":[{{"name":"...","entity_type":"..."}}], "relations":[{{"head":"...","relation":"...","tail":"..."}}]}}"""

VERIFY_PROMPT = """Audit this CoNLL04 extraction. Return CORRECTED JSON (same schema):
fix reversed relation direction (head=subject), delete unsupported/ungrounded
relations, do not add entities. Reply ONLY with corrected JSON."""


def ensure_episodes():
    """Fetch + adapt CoNLL04 test split from HF datasets-server if not cached.
    Reproducible: no committed third-party data, regenerated on demand."""
    if EPISODES.exists():
        return
    import urllib.request
    EPISODES.parent.mkdir(parents=True, exist_ok=True)
    tmap = {"Peop": "Person", "Org": "Organization", "Loc": "Location", "Other": "Other"}
    rows = []
    for off in (0, 100):
        u = ("https://datasets-server.huggingface.co/rows?dataset=DFKI-SLT/conll04"
             f"&config=default&split=test&offset={off}&length=100")
        rows += json.load(urllib.request.urlopen(u, timeout=60)).get("rows", [])
    eps = []
    for rr in rows:
        r = rr["row"]; toks = r["tokens"]; ents = r["entities"]
        sp = lambda e: " ".join(toks[e["start"]:e["end"]])
        rels = [{"head": sp(ents[x["head"]]), "relation": x["type"], "tail": sp(ents[x["tail"]])}
                for x in r.get("relations", [])]
        if rels:
            eps.append({"domain": "conll04", "text": " ".join(toks),
                        "expected_entities": [{"name": sp(e), "entity_type": tmap.get(e["type"], e["type"])} for e in ents],
                        "expected_relations": rels})
    EPISODES.write_text(json.dumps(eps, indent=1))
    print(f"fetched + adapted {len(eps)} CoNLL04 episodes -> {EPISODES}", file=sys.stderr)


def fuzzy(a, b):
    a, b = a.lower().strip(), b.lower().strip()
    return a == b or a in b or b in a


def chat(model, system, user, key, timeout=120):
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "HTTP-Referer": "https://github.com/rohansx/ctxgraph", "X-Title": "ctxgraph-conll04"}
    def post(extra):
        p = {"model": model, "messages": [{"role": "system", "content": system},
             {"role": "user", "content": user}], "max_tokens": 1536}
        p.update(extra)
        return requests.post(BASE_URL, headers=headers, json=p, timeout=timeout)
    for extra in ({"temperature": 0, "reasoning": {"enabled": False}, "response_format": {"type": "json_object"}},
                  {"temperature": 0, "reasoning": {"enabled": False}}, {"temperature": 0}, {}):
        r = post(extra)
        if r.status_code < 400:
            break
    r.raise_for_status()
    b = r.json(); m = b["choices"][0]["message"]
    c = m.get("content") or m.get("reasoning_content") or m.get("reasoning") or ""
    if "</think>" in c: c = c.split("</think>", 1)[1]
    if "```json" in c: c = c.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in c: c = c.split("```", 1)[1].split("```", 1)[0]
    if "{" in c and "}" in c: c = c[c.index("{"):c.rindex("}") + 1]
    try:
        return json.loads(c), b.get("usage", {})
    except json.JSONDecodeError:
        return {"entities": [], "relations": [], "_err": True}, b.get("usage", {})


def f1(p, g, tp):
    if p == 0 and g == 0: return 1.0
    pr, rc = (tp / p if p else 0.0), (tp / g if g else 0.0)
    return 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0


def score(pred, gold):
    pe = pred.get("entities", []) or []; pr = pred.get("relations", []) or []
    # entity F1 (name, fuzzy)
    gn = [e["name"] for e in gold["expected_entities"]]; used = [False] * len(gn); tp = 0
    for e in pe:
        nm = (e.get("name") or "")
        for j, g in enumerate(gn):
            if not used[j] and fuzzy(nm, g): used[j] = True; tp += 1; break
    ent = f1(len(pe), len(gn), tp)
    # relation F1: STRICT directional + typed (head->tail order, type exact)
    gr = gold["expected_relations"]; used = [False] * len(gr); tps = 0
    for p in pr:
        ph, pt, prl = (p.get("head") or ""), (p.get("tail") or ""), (p.get("relation") or "").lower()
        for j, g in enumerate(gr):
            if not used[j] and prl == g["relation"].lower() and fuzzy(ph, g["head"]) and fuzzy(pt, g["tail"]):
                used[j] = True; tps += 1; break
    rel_dir = f1(len(pr), len(gr), tps)
    # relation F1: undirected + untyped (lenient, for context)
    used = [False] * len(gr); tpu = 0
    for p in pr:
        ph, pt = (p.get("head") or ""), (p.get("tail") or "")
        for j, g in enumerate(gr):
            if not used[j] and ((fuzzy(ph, g["head"]) and fuzzy(pt, g["tail"])) or (fuzzy(ph, g["tail"]) and fuzzy(pt, g["head"]))):
                used[j] = True; tpu += 1; break
    rel_pair = f1(len(pr), len(gr), tpu)
    return ent, rel_dir, rel_pair


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--base-url", default=BASE_URL, help="OpenAI-compatible endpoint (e.g. http://localhost:11434/v1/chat/completions for local ollama)")
    a = ap.parse_args()
    globals()["BASE_URL"] = a.base_url
    key = os.environ.get("OPENROUTER_API_KEY", "ollama")  # local servers ignore the key
    ensure_episodes()
    eps = json.loads(EPISODES.read_text())[:a.limit]
    print(f"== {a.model} on CoNLL04 ({len(eps)} eps){' +verify' if a.verify else ''} ==", flush=True)
    E = R = RP = cost = 0.0; errs = 0; per = []
    for i, ep in enumerate(eps):
        try:
            t = time.time()
            parsed, usage = chat(a.model, SYSTEM_PROMPT, ep["text"], key)
            if a.verify and not parsed.get("_err") and (parsed.get("entities") or parsed.get("relations")):
                payload = json.dumps({"entities": parsed.get("entities", []), "relations": parsed.get("relations", [])})
                vp, vu = chat(a.model, VERIFY_PROMPT, f"SENTENCE:\n{ep['text']}\n\nEXTRACTION:\n{payload}", key)
                if not vp.get("_err"): parsed = vp
                cost += float((vu or {}).get("cost", 0) or 0)
            cost += float((usage or {}).get("cost", 0) or 0)
            e, rd, rp = score(parsed, ep)
            E += e; R += rd; RP += rp
            if parsed.get("_err"): errs += 1
            per.append({"i": i, "ent": e, "rel_dir": rd, "rel_pair": rp, "dt": time.time() - t})
        except Exception as ex:
            errs += 1; per.append({"i": i, "error": str(ex)[:120]})
    n = len(eps); ok = n  # scored all (errors counted as low scores already)
    res = {"model": a.model, "dataset": "conll04", "n": n, "verify": a.verify,
           "entity_f1": E / n, "relation_f1_directional_typed": R / n,
           "relation_f1_pair_lenient": RP / n, "parse_errors": errs,
           "cost_per_1k": (cost / n) * 1000}
    Path(a.out).write_text(json.dumps({**res, "per_episode": per}, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

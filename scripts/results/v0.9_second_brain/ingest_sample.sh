#!/usr/bin/env bash
# Ingest a curated slice of /home/rsx/Desktop/second-brain into ctxgraph via the
# universal pipeline. Uses Cerebras free tier ($0).

set -euo pipefail

REPO=/home/rsx/Desktop/projx/ctxgraph
BRAIN=/home/rsx/Desktop/second-brain
BIN="$REPO/target/debug/ctxgraph"

export CTXGRAPH_UNIVERSAL_SCHEMA="$REPO/crates/ctxgraph-extract/schemas/universal.toml"
export CTXGRAPH_UNIVERSAL_PROMPT="$REPO/crates/ctxgraph-extract/prompts/extract.txt"

cd "$BRAIN"

# Curated 12 files: 5 projects + 4 concepts + 3 entities. Avoid the noisy
# Excalidraw / Untitled junk.
FILES=(
    "wiki/projects/ctxgraph.md"
    "wiki/projects/cloakpipe.md"
    "wiki/projects/reflect.md"
    "wiki/projects/workz.md"
    "wiki/projects/leadecho.md"
    "wiki/concepts/agent-brain.md"
    "wiki/concepts/mcp-servers.md"
    "wiki/concepts/local-llm-inference.md"
    "wiki/concepts/credential-isolation.md"
    "wiki/entities/gemma-4.md"
    "wiki/entities/candle.md"
    "wiki/entities/gbrain.md"
)

i=0
for f in "${FILES[@]}"; do
    i=$((i+1))
    if [ ! -f "$f" ]; then
        echo "  ep$i: SKIP (missing) $f"; continue
    fi
    # Strip YAML frontmatter (the --- ... --- block at the top)
    body=$(awk 'BEGIN{fm=0} /^---$/{fm++; next} fm<2{next} {print}' "$f")
    # Trim leading/trailing blank lines
    body=$(echo "$body" | sed -e '/./,$!d' -e ':a' -e '$!{N;ba' -e '}' -e 's/[[:space:]]*$//')
    # Cap to 1500 chars so we don't blow out context on first try
    body=${body:0:1500}
    name=$(basename "$f" .md)
    echo "─── ep$i: $name (${#body} chars) ───"
    "$BIN" log --universal "$body" --source "$f" --tags "$(basename $(dirname $f))" || echo "  log failed"
    # Cerebras rate limit: 30 RPM, ~2s between calls
    sleep 2
done

echo
echo "═══ Final stats ═══"
"$BIN" stats
echo
echo "═══ Entities (top 25) ═══"
"$BIN" entities list --limit 25 2>&1 | head -30

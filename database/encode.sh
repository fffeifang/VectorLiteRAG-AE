#!/usr/bin/env bash
set -euo pipefail

###############################################
# Resolve script directory and project root
###############################################
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( realpath "${SCRIPT_DIR}/.." )"

cd "$PROJECT_ROOT"

###############################################
# Positional arguments
#  $1 = dataset name (orcas1k, orcas2k, wikiall)
#  $2 = target (queries|base|empty)
#  $3 = shard number (optional)
###############################################
SRC="${1:-none}"
TARGET="${2:-}"
SHARD="${3:-}"

###############################################
# Snapshot detection
###############################################
MODEL_DIR="$PROJECT_ROOT/database/embedding_model/models--dunzhang--stella_en_1.5B_v5/snapshots"
PYTHON_SCRIPT="$PROJECT_ROOT"/database/embedding.py

if ! SNAPSHOT=$(ls -d "$MODEL_DIR"/* 2>/dev/null | head -n 1); then
    echo "[!] No snapshot found, initializing model..."
    python "$PYTHON_SCRIPT" --init
    SNAPSHOT=$(ls -d "$MODEL_DIR"/* | head -n 1)
fi

echo "[+] Using snapshot: $SNAPSHOT"

###############################################
# Helpers
###############################################
copy_module_json() {
    local module_file="$1"
    cp "$module_file" "$SNAPSHOT/modules.json"
}

encode_queries() {
    local outdir="$1"
    python "$PYTHON_SCRIPT"  --path query_database --output "$outdir"
}

encode_base() {
    local outdir="$1"
    if [ -n "$SHARD" ]; then
        python "$PYTHON_SCRIPT"  --path text_database --output "$outdir" --shard "$SHARD"
    else
        python "$PYTHON_SCRIPT"  --path text_database --output "$outdir"
    fi
}

###############################################
# wikiall (no embedding, only split)
###############################################
wikiall() {
    python "$PROJECT_ROOT"/database/split_queries.py --dataset wikiall
}

###############################################
# orcas1k
###############################################
orcas1k() {
    mkdir -p orcas1k
    copy_module_json "$PROJECT_ROOT/database/embedding_model/modules_1024.json"

    case "$TARGET" in
        "")
            encode_queries orcas1k/queries
            encode_base    orcas1k/base
            python split_queries.py --dataset orcas1k
            ;;
        queries)
            encode_queries orcas1k/queries
            python split_queries.py --dataset orcas1k
            ;;
        base)
            encode_base orcas1k/base
            ;;
        *)
            echo "Invalid target. Use '', queries, or base."
            exit 1
            ;;
    esac
}

###############################################
# orcas2k
###############################################
orcas2k() {
    mkdir -p orcas2k
    copy_module_json "$PROJECT_ROOT/database/embedding_model/modules_2048.json"

    case "$TARGET" in
        "")
            encode_queries orcas2k/queries
            encode_base    orcas2k/base
            python split_queries.py --dataset orcas2k
            ;;
        queries)
            encode_queries orcas2k/queries
            python split_queries.py --dataset orcas2k
            ;;
        base)
            encode_base orcas2k/base
            ;;
        *)
            echo "Invalid target. Use '', queries, or base."
            exit 1
            ;;
    esac
}

###############################################
# Dispatcher
###############################################
case "$SRC" in
    orcas1k)
        orcas1k
        ;;
    orcas2k)
        orcas2k
        ;;
    wikiall)
        wikiall
        ;;
    test)
        wikiall
        ;;
    *)
        echo "Usage: $0 {orcas1k|orcas2k|wikiall} [queries|base] [shard]"
        exit 1
        ;;
esac

echo "[✓] Done."
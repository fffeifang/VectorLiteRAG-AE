#!/usr/bin/env bash
set -euo pipefail

#######################################
# Resolve project root
#######################################
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PRJ_ROOT="$( realpath "${SCRIPT_DIR}/.." )"
cd "$PRJ_ROOT"

#######################################
# Preset parsing
#######################################
PRESET="${1:-}"

if [[ -z "$PRESET" ]]; then
    echo "Usage: $0 {test|wikiall|orcas1k|orcas2k}"
    exit 1
fi

OUTPUT_FS="build_fs"

case "$PRESET" in
    test)
        DATASET="wikiall"
        NLIST=1
        ;;
    wikiall)
        DATASET="wikiall"
        NLIST=256      # 256K
        ;;
    orcas1k)
        DATASET="orcas1k"
        NLIST=1024       # 1M
        ;;
    orcas2k)
        DATASET="orcas2k"
        NLIST=1024       # 1M
        ;;
    *)
        echo "Unknown preset: $PRESET"
        echo "Valid presets: test | wikiall | orcas1k | orcas2k"
        exit 1
        ;;
esac

#######################################
# Run trainer
#######################################
echo "[+] Index build preset: ${PRESET}"
echo "    dataset : ${DATASET}"
echo "    nlist   : $((NLIST * 1024)) "
echo "    output  : ${OUTPUT_FS}"

python -m index.trainer \
    --dataset "${DATASET}" \
    -o "${OUTPUT_FS}" \
    -g \
    -n "${NLIST}"

echo "[+] Index build completed for preset: ${PRESET}"

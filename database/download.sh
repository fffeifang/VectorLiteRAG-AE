#!/usr/bin/env bash
set -euo pipefail

############################################
# Resolve project root and dataset path
############################################
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( realpath "${SCRIPT_DIR}/.." )"
DATABASE_DIR="${PROJECT_ROOT}/database"
DATASET_DIR="${DATABASE_DIR}/dataset"

mkdir -p "$DATASET_DIR/wikidump"

num_threads=128
wiki_dump_url="https://dumps.wikimedia.org/enwiki/20250901/enwiki-20250901-pages-articles.xml.bz2"
orcas_url="https://msmarco.z22.web.core.windows.net/msmarcoranking/orcas-doctrain-queries.tsv.gz"

############################################
# Wikipedia dump
############################################
download_wikidump() {
    echo "[+] Downloading Wikipedia dump"
    mkdir -p "${DATASET_DIR}/wikidump"

    wget -O "${DATASET_DIR}/wikidump/wikidump.xml.bz2" "$wiki_dump_url"
    bzip2 -d "${DATASET_DIR}/wikidump/wikidump.xml.bbz2"

    wikiextractor --json \
        --processes "$num_threads" \
        --bytes 32G \
        --sections \
        --output "${DATASET_DIR}/wikidump" \
        "${DATASET_DIR}/wikidump/wikidump.xml"

    echo "[OK] Wikipedia dump downloaded & extracted"
}

############################################
# ORCAS dataset
############################################
download_orcas() {
    echo "[+] Downloading ORCAS"
    mkdir -p "$DATASET_DIR"

    wget -P "$DATASET_DIR" "$orcas_url"
    gunzip "${DATASET_DIR}/orcas-doctrain-queries.tsv.gz"

    echo "[OK] ORCAS downloaded & extracted"
}

############################################
# RAPIDS wiki_all (full)
############################################
download_wikiall() {
    echo "[+] Downloading wiki_all"
    mkdir -p "${DATABASE_DIR}/wikiall"

    curl -s -i https://data.rapids.ai/raft/datasets/wiki_all/wiki_all.tar.{00..9} \
        | tar -xf - -C "${DATABASE_DIR}/wikiall"

    mv "${DATABASE_DIR}/wikiall"/base*.fbin "${DATABASE_DIR}/wikiall/base.fbin" 2>/dev/null || true

    echo "[OK] wiki_all dataset downloaded & extracted"
}

############################################
# TEST dataset (wiki_all_1M)
############################################
download_test() {
    TEST_URL="https://data.rapids.ai/raft/datasets/wiki_all_1M/wiki_all_1M.tar"
    TARGET_DIR="${DATABASE_DIR}/wikiall"

    echo "[+] Downloading TEST dataset (wiki_all_1M)"
    mkdir -p "$TARGET_DIR"

    # Follow redirects (-L) and save tar
    curl -L "$TEST_URL" -o "${TARGET_DIR}/wiki_all_1M.tar"

    echo "[+] Extracting wiki_all_1M.tar"
    tar -xf "${TARGET_DIR}/wiki_all_1M.tar" -C "$TARGET_DIR"
    rm "${TARGET_DIR}/wiki_all_1M.tar"

    echo "[OK] Test dataset downloaded & extracted into ${TARGET_DIR}"
}

############################################
# Download ALL datasets
############################################
download_all() {
    download_wikidump
    download_orcas
    download_wikiall
}

############################################
# Download preprocessed wikiall
############################################
download_preprocessed_wikiall() {
    echo "[+] Downloading Preprocessed Wikiall Index and query vectors"
    mkdir -p "${DATABASE_DIR}/wikiall"

    huggingface-cli download --repo-type dataset nobellant215/wikiall ivfpq.index queries.fbin --local-dir ${DATABASE_DIR}/wikiall/
    python "$PROJECT_ROOT"/database/split_queries.py --dataset wikiall
}

############################################
# Download preprocessed orcas 1k
############################################
download_preprocessed_orcas_1k() {
    echo "[+] Downloading Preprocessed ORCAS 1k Index and query vectors"
    mkdir -p "${DATABASE_DIR}/orcas1k"

    huggingface-cli download --repo-type dataset nobellant215/orcas1k ivfpq.index queries.fbin --local-dir ${DATABASE_DIR}/orcas1k/
    python "$PROJECT_ROOT"/database/split_queries.py --dataset orcas1k
}

############################################
# Download preprocessed orcas 2k
############################################
download_preprocessed_orcas_2k() {
    echo "[+] Downloading Preprocessed ORCAS 2k Index and query vectors"
    mkdir -p "${DATABASE_DIR}/orcas2k"

    huggingface-cli download --repo-type dataset nobellant215/orcas2k ivfpq.index queries.fbin --local-dir ${DATABASE_DIR}/orcas2k/
    python "$PROJECT_ROOT"/database/split_queries.py --dataset orcas2k
}

############################################
# Download ALL preprocessed datasets
############################################
download_all_preprocessed() {
    download_preprocessed_wikiall
    download_preprocessed_orcas_1k
    download_preprocessed_orcas_2k
}

############################################
# CLI dispatcher
############################################
ACTION="${1:-all}"

case "$ACTION" in
    wikidump)
        download_wikidump
        ;;
    orcas)
        download_orcas
        ;;
    wikiall)
        download_wikiall
        ;;
    test)
        download_test
        ;;
    all)
        echo "[*] Downloading ALL datasets"
        download_all
        ;;
    wikiall_pp)
        download_preprocessed_wikiall
        ;;
    orcas1k_pp)
        download_preprocessed_orcas_1k
        ;;
    orcas2k_pp)
        download_preprocessed_orcas_2k
        ;;
    all_pp)
        echo "[*] Downloading ALL preprocessed datasets"
        download_all_preprocessed
        ;;
    *)
        echo "Usage: $0 {wikidump|orcas|wikiall|test|all|wikiall_pp|orcas1k_pp|orcas2k_pp|all_pp}"
        exit 1
        ;;
esac
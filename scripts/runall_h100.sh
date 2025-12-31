#!/usr/bin/env bash
set -euo pipefail

#######################################
# Resolve project root directory
#######################################
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
VLITE_ROOT="$( realpath "${SCRIPT_DIR}/.." )"

export VLITE_ROOT

#######################################
# Command helpers
#######################################
run() {
    # 모든 python 실행은 이 wrapper를 사용
    python3 "${VLITE_ROOT}/main.py" "$@"
}

#######################################
# main : figure 10 ~ 12
#######################################
main() {
    index=$1
    echo "[+] Running H100 main experiment"

    run --model qwen32b \
        --index all \
        --is_profiling \
        --sweep \
        --tag main
    run --model qwen32b \
        --index all \
        --search_mode all \
        --sweep \
        --tag main

    run --model llama70b \
        --index all \
        --is_profiling \
        --sweep \
        --tag main
    run --model llama70b \
        --index all \
        --search_mode all \
        --sweep \
        --tag main
}

#######################################
# input/output length sweep
#######################################
inout_length() {
    inlen=$1
    outlen=$2
    echo "[+] I/O sweep: input=${inlen}, output=${outlen}"

    run --model llama70b \
        --index orcas2k \
        --sweep \
        --search_mode all \
        --input_len "$inlen" \
        --output_len "$outlen"
}

#######################################
# vary SLO
#######################################
vary_slo() {
    slo=$1
    echo "[+] SLO sweep: slo=${slo}"

    run --is_profiling \
        --search_slo "$slo" \
        --model qwen32b \
        --index orcas1k

    run --search_slo "$slo" \
        --model qwen32b \
        --index orcas1k \
        --sweep \
        --search_mode all
}

#######################################
# vary number of GPUs
#######################################
num_gpu() {
    g=$1
    echo "[+] GPU number sweep ngpu=${g}"

    run --is_profiling \
        --num_gpus "$g" \
        --model qwen32b \
        --index orcas2k

    run --num_gpus "$g" \
        --model qwen32b \
        --index orcas2k
}

#######################################
# Dispatch by CLI argument
#######################################
OPTION="${1:-none}"

case "$OPTION" in
    main)
        main
        ;;

    inout)
        inout_length 1024 128
        inout_length 1024 512
        inout_length 512 256
        inout_length 2048 256
        ;;

    slo)
        vary_slo 100
        vary_slo 150
        vary_slo 200
        vary_slo 250
        ;;

    ngpu)
        num_gpu 4
        num_gpu 6
        ;;

    *)
        echo "Unknown option: $OPTION"
        echo "Usage: $0 {main|inout|slo|ngpu|dispatcher}"
        exit 1
        ;;
esac

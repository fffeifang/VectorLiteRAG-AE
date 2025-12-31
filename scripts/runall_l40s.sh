#!/usr/bin/env bash
set -euo pipefail

#######################################
# Resolve project root directory
#######################################
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
VLITE_ROOT="$( realpath "${SCRIPT_DIR}/.." )"

export VLITE_ROOT

#######################################
# python wrapper
#######################################
run() {
    python3 "${VLITE_ROOT}/main.py" "$@"
}

#######################################
# Test run: wikiall + L40S CPU baseline
#######################################
test() {
    echo "[+] Running L40S test mode (wikiall, cpu search)"
    run --model llama8b \
        --index wikiall \
        --search_mode cpu \
        --sweep \
        --running_time 20 \
        --arrival_rate 16 \
        --tag test

    ORIG_OUT_DIR=${VLITE_ROOT}/results/wikiall/llama8b/8gpus/cpu
    TEST_OUT_DIR=${VLITE_ROOT}/results/test
    mkdir ${TEST_OUT_DIR}
    cp ${ORIG_OUT_DIR}/raw/*  ${TEST_OUT_DIR}/
    cp ${ORIG_OUT_DIR}/summary/*  ${TEST_OUT_DIR}/

    echo "[+] Cleanup test dummy index"
    rm ${VLITE_ROOT}/database/wikiall/*
}

#######################################
# Main experiment for L40S
#######################################
main() {
    echo "[+] Running L40S main experiment"

    run --model llama8b \
        --index all \
        --is_profiling \
        --sweep \
        --tag main
    run --model llama8b \
        --index all \
        --search_mode all \
        --sweep \
        --tag main

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
# Dispatcher ablation
#######################################
dispatcher() {
    rate=$1
    echo "[+] Dispatcher ablation, rate=$rate"

    run --model llama8b \
        --index orcas2k \
        --search_mode vlite \
        --arrival_rate "$rate" \
        --disable_dispatcher \
        --tag dispatcher

    run --model llama8b \
        --index orcas2k \
        --search_mode vlite \
        --arrival_rate "$rate" \
        --tag dispatcher
}

#######################################
# Input/Output length sweep
#######################################
inout_length() {
    inlen=$1
    outlen=$2

    echo "[+] I/O sweep: input=${inlen}, output=${outlen}"

    run --model llama8b \
        --index orcas2k \
        --sweep \
        --search_mode all \
        --input_len "$inlen" \
        --output_len "$outlen"

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
# Dispatch CLI
#######################################
OPTION="${1:-none}"

case "$OPTION" in
    test)
        test
        ;;

    main)
        main
        ;;

    inout)
        inout_length 1024 128
        inout_length 1024 512
        inout_length 512 256
        inout_length 2048 256
        ;;

    dispatcher)
        dispatcher 24
        dispatcher 32
        dispatcher 41
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
        echo "Usage: $0 {test|main|inout|dispatcher|slo|ngpus}"
        exit 1
        ;;
esac
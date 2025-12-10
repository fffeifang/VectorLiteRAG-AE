# Resolve project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PRJ_ROOT="$( realpath "${SCRIPT_DIR}/.." )"

export PYTHONPATH="${PRJ_ROOT}:${PYTHONPATH:-}"

PLOT_PY="${PRJ_ROOT}/analysis/plot.py"

if [ ! -f "$PLOT_PY" ]; then
    echo "[ERROR] plot.py not found at $PLOT_PY"
    exit 1
fi

usage() {
    echo "Usage:"
    echo "  $0 all              # Plot all figures"
    echo "  $0 main             # Plot main figures (10,11,12)"
    echo "  $0 10 [11 12 ...]   # Plot specific figures"
    exit 1
}

if [ "$#" -eq 0 ]; then
    usage
fi

ARGS=()

for arg in "$@"; do
    case "$arg" in
        all)
            ARGS+=(--all)
            ;;
        main)
            ARGS+=(--main)
            ;;
        10|11|12|14|15|16|17)
            ARGS+=(--f${arg})
            ;;
        *)
            echo "[ERROR] Unknown argument: $arg"
            usage
            ;;
    esac
done

echo "[VLITE] Running plots: ${ARGS[*]}"
python3 "$PLOT_PY" "${ARGS[@]}"
echo "[VLITE] Plotting completed."

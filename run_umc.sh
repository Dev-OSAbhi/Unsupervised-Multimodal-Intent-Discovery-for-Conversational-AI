#!/usr/bin/env bash
# =============================================================================
# run_umc.sh  —  Train & evaluate UMC on all three benchmark datasets
# =============================================================================
# Usage:
#   bash run_umc.sh                        # run all datasets
#   bash run_umc.sh MIntRec                # run a single dataset
#   SKIP_PRETRAIN=1 bash run_umc.sh        # skip pre-training (load saved weights)
# =============================================================================

set -e  # exit immediately on error

# ── Configurable paths ────────────────────────────────────────────────────────
BERT_PATH="${BERT_PATH:-bert-base-uncased}"
DATA_PATH="${DATA_PATH:-Datasets}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs}"
SEED="${SEED:-0}"

# ── Dataset selection ─────────────────────────────────────────────────────────
if [ -n "$1" ]; then
    DATASETS=("$1")
else
    DATASETS=("MIntRec" "MELD-DA" "IEMOCAP-DA")
fi

# ── Helper: print a section banner ───────────────────────────────────────────
banner() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
}

# ── Main loop ─────────────────────────────────────────────────────────────────
for DATASET in "${DATASETS[@]}"; do
    banner "Dataset: $DATASET"

    PRETRAIN_WEIGHTS="${OUTPUT_PATH}/${DATASET}/pretrain.pt"

    if [ "${SKIP_PRETRAIN:-0}" = "1" ] && [ -f "$PRETRAIN_WEIGHTS" ]; then
        # ── Skip pre-training, load existing weights ──────────────────────────
        echo "[INFO] Skipping pre-training. Loading weights from: $PRETRAIN_WEIGHTS"
        python run.py \
            --dataset        "$DATASET" \
            --data_path      "$DATA_PATH" \
            --bert_path      "$BERT_PATH" \
            --output_path    "${OUTPUT_PATH}/${DATASET}" \
            --seed           "$SEED" \
            --pretrain_path  "$PRETRAIN_WEIGHTS" \
            --train \
            --save_model
    else
        # ── Full pipeline: pre-train then train ───────────────────────────────
        echo "[INFO] Running full pipeline (pre-train + train) for $DATASET"
        python run.py \
            --dataset      "$DATASET" \
            --data_path    "$DATA_PATH" \
            --bert_path    "$BERT_PATH" \
            --output_path  "${OUTPUT_PATH}/${DATASET}" \
            --seed         "$SEED" \
            --pretrain \
            --train \
            --save_model
    fi

    echo "[INFO] Finished: $DATASET  →  results saved to ${OUTPUT_PATH}/${DATASET}/"
done

banner "All runs complete"
echo "Results are stored under: ${OUTPUT_PATH}/"

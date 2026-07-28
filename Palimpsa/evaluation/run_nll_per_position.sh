#!/bin/bash
# Usage: ./evaluation/run_nll_per_position.sh [GPU_ID] [MODEL_NAME] [STEP] [MAX_LENGTH]
# Example: ./evaluation/run_nll_per_position.sh 0 palimpsa-170M 3000 4096

GPU_ID=${1:-0}
MODEL_NAME=${2:-"palimpsa-170M"}
STEP=${3:-3000}
MAX_LENGTH=${4:-4096}

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR="$(pwd)"
EXP_DIR="${ROOT_DIR}/../exp/${MODEL_NAME}"
HF_MODEL_PATH="${EXP_DIR}/hf_model_step_${STEP}"

# ─── WandB ────────────────────────────────────────────────────────────────────
export WANDB_HOST=wandb.fz-juelich.de
export WANDB_API_KEY="local-3617f3396eb6ef1ff72c930182864107c0faed03"
WANDB_ENTITY="hybrid_nns"
WANDB_PROJECT="eval_ppl_per_token"

# ─── Settings ─────────────────────────────────────────────────────────────────
DATASET="wikitext"
SPLIT="train"
BATCH_SIZE=4
DTYPE="bfloat16"

RUN_NAME="${MODEL_NAME}-step${STEP}"
OUTPUT_PATH="${EXP_DIR}"

# ─── Safety check ─────────────────────────────────────────────────────────────
if [ ! -d "$HF_MODEL_PATH" ]; then
    echo "❌ Error: HF Model not found at ${HF_MODEL_PATH}"
    echo "Please convert the checkpoint first:"
    echo "   python tools/convert_dcp_to_hf.py --exp ${EXP_DIR} --step ${STEP}"
    exit 1
fi

# ─── Run ──────────────────────────────────────────────────────────────────────
echo "📊 NLL-per-position eval on GPU ${GPU_ID}"
echo "   Model : ${MODEL_NAME} (step ${STEP})"
echo "   Length: ${MAX_LENGTH} tokens"
echo "   WandB : ${WANDB_PROJECT} / ${RUN_NAME}"

export CUDA_VISIBLE_DEVICES=$GPU_ID
mkdir -p logs/

python evaluation/eval_nll_per_position.py \
    --model_path    "$HF_MODEL_PATH" \
    --run_name      "$RUN_NAME" \
    --max_length    "$MAX_LENGTH" \
    --output_path   "$OUTPUT_PATH" \
    --dataset       "$DATASET" \
    --dataset_split "$SPLIT" \
    --batch_size    "$BATCH_SIZE" \
    --dtype         "$DTYPE" \
    --device        "cuda" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_entity  "$WANDB_ENTITY" \
    --max_chunks 10000 \
    | tee "logs/nll_per_pos_${MODEL_NAME}_step${STEP}.log"

echo "✅ Done."
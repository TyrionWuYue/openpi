#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   CONFIG=pi05_xxx_lora REPO_ID=sii_team9/task ASSET_ID=agilex_xxx scripts/train_agilex_lora.sh
#   NUM_TRAIN_STEPS=3000 scripts/train_agilex_lora.sh
#   SAVE_INTERVAL=1000 scripts/train_agilex_lora.sh
#   OVERWRITE=1 scripts/train_agilex_lora.sh   # 明确删除同名旧实验后重训
#
# 训练前先运行：
#   scripts/prepare_agilex_data.sh --overwrite

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-}"
EXP_NAME="${EXP_NAME:-overfit_10}"
ASSET_ID="${ASSET_ID:-}"
REPO_ID="${REPO_ID:-}"
LEROBOT_ROOT="${LEROBOT_ROOT:-agilex_data}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/inspire/hdd/project/embodied-intelligent-robot-system/czxs25120101/openpi_cache}"
BASE_PARAMS_PATH="${BASE_PARAMS_PATH:-$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base/params}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
KEEP_PERIOD="${KEEP_PERIOD:-$SAVE_INTERVAL}"
OVERWRITE="${OVERWRITE:-0}"

export OPENPI_DATA_HOME OPENPI_OFFLINE="${OPENPI_OFFLINE:-1}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

[[ "$LEROBOT_ROOT" = /* ]] && DATASET_DIR="$LEROBOT_ROOT/$REPO_ID" || DATASET_DIR="$REPO_ROOT/$LEROBOT_ROOT/$REPO_ID"
NORM_STATS="$REPO_ROOT/assets/$CONFIG/$ASSET_ID/norm_stats.json"

require_env() {
  local name="$1"
  [[ -n "${!name:-}" ]] || { echo "[missing] set $name" >&2; exit 1; }
}

require_env CONFIG
require_env REPO_ID
require_env ASSET_ID

echo "[AgileX train]"
echo "  config=$CONFIG exp=$EXP_NAME asset=$ASSET_ID"
echo "  dataset=$DATASET_DIR"
echo "  base=$BASE_PARAMS_PATH"
echo "  save_interval=$SAVE_INTERVAL keep_period=$KEEP_PERIOD overwrite=$OVERWRITE"

[[ -d "$DATASET_DIR" ]] || { echo "[missing] dataset: $DATASET_DIR" >&2; exit 1; }
[[ -f "$NORM_STATS" ]] || { echo "[missing] norm stats: $NORM_STATS" >&2; exit 1; }
[[ -d "$BASE_PARAMS_PATH" ]] || { echo "[missing] pi0.5 base params: $BASE_PARAMS_PATH" >&2; exit 1; }
[[ -f "$TOKENIZER_PATH" ]] || { echo "[missing] tokenizer: $TOKENIZER_PATH" >&2; exit 1; }

echo "Step 1: Starting LoRA training..."
TRAIN_ARGS=()
[[ -n "${NUM_TRAIN_STEPS:-}" ]] && TRAIN_ARGS+=(--num-train-steps "$NUM_TRAIN_STEPS")
TRAIN_ARGS+=(--save-interval "$SAVE_INTERVAL")
TRAIN_ARGS+=(--keep-period "$KEEP_PERIOD")
[[ "$OVERWRITE" == "1" ]] && TRAIN_ARGS+=(--overwrite)

uv run scripts/train.py "$CONFIG" \
  --exp-name "$EXP_NAME" \
  --data.assets.asset-id "$ASSET_ID" \
  --weight-loader.params-path "$BASE_PARAMS_PATH" \
  "${TRAIN_ARGS[@]}"

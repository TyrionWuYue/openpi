#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   RAW_DIR=/path/to/hdf5 REPO_ID=sii_team9/task TASK="task prompt" CONFIG=pi05_xxx_lora ASSET_ID=agilex_xxx scripts/prepare_agilex_data.sh --overwrite
#   MODE=convert scripts/prepare_agilex_data.sh    # 只转 HDF5 -> LeRobot
#   MODE=norm scripts/prepare_agilex_data.sh       # 只计算 norm stats
#
# 说明：默认 ACTION_SOURCE=next_qpos。STATIONARY_ACTION_DIMS=auto 会在原始数据转换时清理静止关节。
# 右臂是否参与训练/推理由 config 里的 masked_action_dims/action_loss_mask 控制。

MODE="${MODE:-all}"
OVERWRITE="${OVERWRITE:-0}"
for arg in "$@"; do
  case "$arg" in
    all|check|convert|norm) MODE="$arg" ;;
    --overwrite) OVERWRITE=1 ;;
    -h|--help)
      sed -n '1,12p' "$0"
      exit 0
      ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RAW_DIR="${RAW_DIR:-}"
LEROBOT_ROOT="${LEROBOT_ROOT:-agilex_data}"
REPO_ID="${REPO_ID:-}"
TASK="${TASK:-}"
FPS="${FPS:-30}"
ACTION_SOURCE="${ACTION_SOURCE:-next_qpos}"
STATIONARY_ACTION_DIMS="${STATIONARY_ACTION_DIMS:-auto}"
STATIONARY_DELTA_THRESHOLD="${STATIONARY_DELTA_THRESHOLD:-1e-4}"
CONFIG="${CONFIG:-}"
ASSET_ID="${ASSET_ID:-}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/inspire/hdd/project/embodied-intelligent-robot-system/czxs25120101/openpi_cache}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model}"

export OPENPI_DATA_HOME OPENPI_OFFLINE="${OPENPI_OFFLINE:-1}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

[[ "$LEROBOT_ROOT" = /* ]] && LEROBOT_ROOT_PATH="$LEROBOT_ROOT" || LEROBOT_ROOT_PATH="$REPO_ROOT/$LEROBOT_ROOT"
DATASET_DIR="$LEROBOT_ROOT_PATH/$REPO_ID"
NORM_STATS="$REPO_ROOT/assets/$CONFIG/$ASSET_ID/norm_stats.json"

require_env() {
  local name="$1"
  [[ -n "${!name:-}" ]] || { echo "[missing] set $name" >&2; exit 1; }
}

require_env RAW_DIR
require_env REPO_ID
require_env TASK
require_env CONFIG
require_env ASSET_ID

echo "[AgileX data]"
echo "  raw=$RAW_DIR"
echo "  dataset=$DATASET_DIR"
echo "  task=$TASK"
echo "  fps=$FPS action_source=$ACTION_SOURCE stationary_dims=$STATIONARY_ACTION_DIMS"
echo "  config=$CONFIG asset=$ASSET_ID overwrite=$OVERWRITE"

[[ -d "$RAW_DIR" ]] || { echo "[missing] RAW_DIR: $RAW_DIR" >&2; exit 1; }
[[ -f "$TOKENIZER_PATH" ]] || { echo "[missing] tokenizer: $TOKENIZER_PATH" >&2; exit 1; }

if [[ "$MODE" == "check" ]]; then
  [[ -d "$DATASET_DIR" ]] && echo "[ok] dataset" || echo "[pending] dataset"
  [[ -f "$NORM_STATS" ]] && echo "[ok] norm stats" || echo "[pending] norm stats"
  exit 0
fi

if [[ "$MODE" == "all" || "$MODE" == "convert" ]]; then
  if [[ -d "$DATASET_DIR" && "$OVERWRITE" != "1" ]]; then
    echo "[skip] dataset exists; use --overwrite to rebuild"
  else
    echo "Step 1: Converting AgileX HDF5 to LeRobot..."
    CONVERT_ARGS=()
    [[ "$OVERWRITE" == "1" ]] && CONVERT_ARGS+=(--overwrite)
    uv run examples/aloha_real/convert_agilex_hdf5_to_lerobot.py \
      --raw-dir "$RAW_DIR" \
      --repo-id "$REPO_ID" \
      --task "$TASK" \
      --fps "$FPS" \
      --action-source "$ACTION_SOURCE" \
      --stationary-action-dims "$STATIONARY_ACTION_DIMS" \
      --stationary-delta-threshold "$STATIONARY_DELTA_THRESHOLD" \
      --local-dir "$LEROBOT_ROOT_PATH" \
      "${CONVERT_ARGS[@]}"
  fi
fi

if [[ "$MODE" == "all" || "$MODE" == "norm" ]]; then
  [[ -d "$DATASET_DIR" ]] || { echo "[missing] dataset: $DATASET_DIR" >&2; exit 1; }
  if [[ -f "$NORM_STATS" && "$OVERWRITE" != "1" ]]; then
    echo "[skip] norm stats exist; use --overwrite to recompute"
  else
    echo "Step 2: Computing AgileX norm stats..."
    uv run scripts/compute_norm_stats.py \
      --config-name "$CONFIG" \
      --asset-id "$ASSET_ID"
  fi
fi

echo "[done] AgileX data is ready."

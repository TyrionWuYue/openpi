#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/train_agilex_cube_in_bowl_lora.sh [check|train]

Default mode is "train". Run scripts/prepare_agilex_cube_in_bowl_data.sh first.

Environment overrides:
  LEROBOT_ROOT         Local LeRobot root directory. Dataset is stored under $LEROBOT_ROOT/$REPO_ID.
  REPO_ID              LeRobot repo id.
  CONFIG               OpenPI train config.
  ASSET_ID             Norm-stats asset id.
  EXP_NAME             Training experiment name.
  OPENPI_DATA_HOME     Local OpenPI artifact cache.
  BASE_PARAMS_PATH     Local pi0.5 base params directory.
  TOKENIZER_PATH       Local PaliGemma tokenizer model file.
  NUM_TRAIN_STEPS      Optional train step override.
  WANDB_MODE           Defaults to disabled. Set to offline or online if needed.
EOF
}

MODE="${1:-train}"
if [[ "$MODE" == "-h" || "$MODE" == "--help" ]]; then
  usage
  exit 0
fi

case "$MODE" in
  check|train) ;;
  *)
    usage >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

LEROBOT_ROOT="${LEROBOT_ROOT:-agilex_data}"
REPO_ID="${REPO_ID:-sii_team9/cube_in_bowl}"
CONFIG="${CONFIG:-pi05_agilex_cube_in_bowl_lora}"
ASSET_ID="${ASSET_ID:-agilex_cube_in_bowl}"
EXP_NAME="${EXP_NAME:-overfit_10}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/inspire/hdd/project/embodied-intelligent-robot-system/czxs25120101/openpi_cache}"
BASE_PARAMS_PATH="${BASE_PARAMS_PATH:-$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base/params}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-}"

export OPENPI_DATA_HOME
export OPENPI_OFFLINE="${OPENPI_OFFLINE:-1}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

if [[ "$LEROBOT_ROOT" = /* ]]; then
  LEROBOT_ROOT_PATH="$LEROBOT_ROOT"
else
  LEROBOT_ROOT_PATH="$REPO_ROOT/$LEROBOT_ROOT"
fi

DATASET_DIR="$LEROBOT_ROOT_PATH/$REPO_ID"
NORM_STATS_PATH="$REPO_ROOT/assets/$CONFIG/$ASSET_ID/norm_stats.json"

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "$path" ]]; then
    echo "[missing] $label: $path" >&2
    return 1
  fi
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "[missing] $label: $path" >&2
    return 1
  fi
}

check_dataset() {
  require_dir "$DATASET_DIR" "converted LeRobot dataset"
}

check_tokenizer() {
  require_file "$TOKENIZER_PATH" "PaliGemma tokenizer"
}

check_base_params() {
  require_dir "$BASE_PARAMS_PATH" "pi0.5 base params"
}

check_norm_stats() {
  require_file "$NORM_STATS_PATH" "AgileX norm stats"
}

print_config() {
  cat <<EOF
[config]
  mode=$MODE
  lerobot_root=$LEROBOT_ROOT_PATH
  repo_id=$REPO_ID
  dataset_dir=$DATASET_DIR
  config=$CONFIG
  asset_id=$ASSET_ID
  exp_name=$EXP_NAME
  openpi_data_home=$OPENPI_DATA_HOME
  tokenizer=$TOKENIZER_PATH
  base_params=$BASE_PARAMS_PATH
  norm_stats=$NORM_STATS_PATH
  offline=$OPENPI_OFFLINE
  wandb_mode=$WANDB_MODE
EOF
}

run_check() {
  print_config
  check_dataset
  check_norm_stats
  check_tokenizer
  check_base_params
  echo "[ok] training artifact check passed"
}

run_train() {
  check_dataset
  check_norm_stats
  check_tokenizer
  check_base_params

  train_cmd=(
    uv run scripts/train.py
    "$CONFIG"
    --exp-name "$EXP_NAME"
    --overwrite
    --data.assets.asset-id "$ASSET_ID"
    --weight-loader.params-path "$BASE_PARAMS_PATH"
  )
  if [[ -n "$NUM_TRAIN_STEPS" ]]; then
    train_cmd+=(--num-train-steps "$NUM_TRAIN_STEPS")
  fi

  XLA_PYTHON_CLIENT_MEM_FRACTION="$XLA_PYTHON_CLIENT_MEM_FRACTION" "${train_cmd[@]}"
}

print_config
case "$MODE" in
  check)
    run_check
    ;;
  train)
    run_train
    ;;
esac

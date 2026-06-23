#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/prepare_agilex_cube_in_bowl_data.sh [all|check|convert|norm] [--overwrite|--no-overwrite]

Default mode is "all". Default overwrite is off.

Environment overrides:
  RAW_DIR              Raw HDF5 directory.
  LEROBOT_ROOT         Local LeRobot root directory. Dataset is stored under $LEROBOT_ROOT/$REPO_ID.
  REPO_ID              LeRobot repo id.
  TASK                 Training prompt/task string.
  CONFIG               OpenPI train config.
  ASSET_ID             Norm-stats asset id.
  OPENPI_DATA_HOME     Local OpenPI artifact cache.
  TOKENIZER_PATH       Local PaliGemma tokenizer model file.
  OVERWRITE            Set to 1 to rebuild converted data and norm stats.
EOF
}

MODE="all"
if [[ $# -gt 0 && "$1" != --* ]]; then
  MODE="$1"
  shift
fi

if [[ "$MODE" == "-h" || "$MODE" == "--help" ]]; then
  usage
  exit 0
fi

case "$MODE" in
  all|check|convert|norm) ;;
  *)
    usage >&2
    exit 2
    ;;
esac

OVERWRITE="${OVERWRITE:-0}"
for arg in "$@"; do
  case "$arg" in
    --overwrite)
      OVERWRITE=1
      ;;
    --no-overwrite)
      OVERWRITE=0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
done

is_truthy() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RAW_DIR="${RAW_DIR:-/inspire/qb-ilm2/project/embodied-intelligent-robot-system/public/Cube_in_Bowl}"
LEROBOT_ROOT="${LEROBOT_ROOT:-agilex_data}"
REPO_ID="${REPO_ID:-sii_team9/cube_in_bowl}"
TASK="${TASK:-put the cube in the bowl}"
CONFIG="${CONFIG:-pi05_agilex_cube_in_bowl_lora}"
ASSET_ID="${ASSET_ID:-agilex_cube_in_bowl}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/inspire/hdd/project/embodied-intelligent-robot-system/czxs25120101/openpi_cache}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model}"

export OPENPI_DATA_HOME
export OPENPI_OFFLINE="${OPENPI_OFFLINE:-1}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

if [[ "$LEROBOT_ROOT" = /* ]]; then
  LEROBOT_ROOT_PATH="$LEROBOT_ROOT"
else
  LEROBOT_ROOT_PATH="$REPO_ROOT/$LEROBOT_ROOT"
fi

DATASET_DIR="$LEROBOT_ROOT_PATH/$REPO_ID"
NORM_STATS_PATH="$REPO_ROOT/assets/$CONFIG/$ASSET_ID/norm_stats.json"
OVERWRITE_ENABLED=0
CONVERT_ARGS=()
if is_truthy "$OVERWRITE"; then
  OVERWRITE_ENABLED=1
  CONVERT_ARGS+=(--overwrite)
fi

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

check_raw() {
  require_dir "$RAW_DIR" "raw HDF5 directory"
}

check_dataset() {
  require_dir "$DATASET_DIR" "converted LeRobot dataset"
}

check_tokenizer() {
  require_file "$TOKENIZER_PATH" "PaliGemma tokenizer"
}

check_norm_stats() {
  require_file "$NORM_STATS_PATH" "AgileX norm stats"
}

print_config() {
  cat <<EOF
[config]
  mode=$MODE
  raw_dir=$RAW_DIR
  lerobot_root=$LEROBOT_ROOT_PATH
  repo_id=$REPO_ID
  dataset_dir=$DATASET_DIR
  task=$TASK
  config=$CONFIG
  asset_id=$ASSET_ID
  openpi_data_home=$OPENPI_DATA_HOME
  tokenizer=$TOKENIZER_PATH
  norm_stats=$NORM_STATS_PATH
  offline=$OPENPI_OFFLINE
  overwrite=$OVERWRITE_ENABLED
EOF
}

run_check() {
  print_config
  check_raw
  check_tokenizer
  if [[ -d "$DATASET_DIR" ]]; then
    echo "[ok] converted LeRobot dataset: $DATASET_DIR"
  else
    echo "[pending] converted LeRobot dataset will be created at: $DATASET_DIR"
  fi
  if [[ -f "$NORM_STATS_PATH" ]]; then
    echo "[ok] AgileX norm stats: $NORM_STATS_PATH"
  else
    echo "[pending] AgileX norm stats will be created at: $NORM_STATS_PATH"
  fi
  echo "[ok] data preparation check passed"
}

run_convert() {
  check_raw
  if [[ -d "$DATASET_DIR" && "$OVERWRITE_ENABLED" == 0 ]]; then
    echo "[skip] converted LeRobot dataset exists: $DATASET_DIR"
    echo "       pass --overwrite or set OVERWRITE=1 to rebuild it"
    return
  fi

  uv run examples/aloha_real/convert_agilex_hdf5_to_lerobot.py \
    --raw-dir "$RAW_DIR" \
    --repo-id "$REPO_ID" \
    --task "$TASK" \
    --local-dir "$LEROBOT_ROOT_PATH" \
    "${CONVERT_ARGS[@]}"
  check_dataset
}

run_norm() {
  check_dataset
  check_tokenizer
  if [[ -f "$NORM_STATS_PATH" && "$OVERWRITE_ENABLED" == 0 ]]; then
    echo "[skip] AgileX norm stats exist: $NORM_STATS_PATH"
    echo "       pass --overwrite or set OVERWRITE=1 to recompute them"
    return
  fi

  uv run scripts/compute_norm_stats.py \
    --config-name "$CONFIG" \
    --asset-id "$ASSET_ID"
  check_norm_stats
}

print_config
case "$MODE" in
  check)
    run_check
    ;;
  convert)
    run_convert
    ;;
  norm)
    run_norm
    ;;
  all)
    run_convert
    run_norm
    ;;
esac

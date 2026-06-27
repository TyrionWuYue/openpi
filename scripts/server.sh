#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   scripts/server.sh                 # 默认启动最新 AgileX ALOHA checkpoint
#   scripts/server.sh 5000            # 启动最新实验里的第 5000 step
#   scripts/server.sh EXP_NAME 5000   # 启动指定实验的第 5000 step
#   POLICY_DIR=/path/to/ckpt scripts/server.sh
#
# 常用覆盖：
#   PORT=8001 scripts/server.sh
#   WARMUP_STEPS=0 scripts/server.sh
#   POLICY_CONFIG=pi05_xxx_lora ASSET_ID=agilex_xxx DEFAULT_PROMPT="task" scripts/server.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PORT="${PORT:-8000}"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-put the cube in the bowl}"
POLICY_CONFIG="${POLICY_CONFIG:-pi05_agilex_aloha}"
ASSET_ID="${ASSET_ID:-agilex_cube_in_bowl}"
EXP_NAME="${EXP_NAME:-latest}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-checkpoints}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/inspire/hdd/project/embodied-intelligent-robot-system/czxs25120101/openpi_cache}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
JAX_CACHE_DIR="${JAX_CACHE_DIR:-$HOME/.cache/jax}"

export OPENPI_DATA_HOME OPENPI_OFFLINE="${OPENPI_OFFLINE:-1}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

usage() {
  cat <<'EOF'
用法：
  scripts/server.sh                 # 默认启动最新 AgileX ALOHA checkpoint
  scripts/server.sh 5000            # 启动最新实验里的第 5000 step
  scripts/server.sh EXP_NAME 5000   # 启动指定实验的第 5000 step
  POLICY_DIR=/path/to/ckpt scripts/server.sh

常用覆盖：
  PORT=8001 scripts/server.sh
  WARMUP_STEPS=0 scripts/server.sh
  POLICY_CONFIG=pi05_xxx_lora ASSET_ID=agilex_xxx DEFAULT_PROMPT="task" scripts/server.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ge 1 ]]; then
  if [[ "$1" =~ ^[0-9]+$ || "$1" == "latest" ]]; then
    CHECKPOINT_STEP="$1"
  elif [[ "$1" == */* ]]; then
    POLICY_DIR="$1"
  else
    EXP_NAME="$1"
  fi
fi

if [[ $# -ge 2 ]]; then
  CHECKPOINT_STEP="$2"
fi

has_checkpoint_params() {
  [[ -d "$1/params" ]]
}

has_checkpoint_assets() {
  [[ -f "$1/assets/$ASSET_ID/norm_stats.json" ]]
}

latest_step_in_run() {
  local run_dir="$1"
  local best=""
  local candidate step

  for candidate in "$run_dir"/*; do
    [[ -d "$candidate" ]] || continue
    step="$(basename "$candidate")"
    [[ "$step" =~ ^[0-9]+$ ]] || continue
    has_checkpoint_params "$candidate" || continue
    [[ -z "$best" || "$step" -gt "$(basename "$best")" ]] && best="$candidate"
  done

  [[ -n "$best" ]] && printf '%s\n' "$best"
}

latest_checkpoint() {
  local config_dir="$CHECKPOINT_BASE_DIR/$POLICY_CONFIG"
  local run_dir candidate best=""

  if [[ "${EXP_NAME:-latest}" != "latest" ]]; then
    run_dir="$config_dir/$EXP_NAME"
    if [[ "${CHECKPOINT_STEP:-latest}" == "latest" ]]; then
      latest_step_in_run "$run_dir"
    else
      candidate="$run_dir/$CHECKPOINT_STEP"
      has_checkpoint_params "$candidate" && printf '%s\n' "$candidate"
    fi
    return
  fi

  for run_dir in "$config_dir"/*; do
    [[ -d "$run_dir" ]] || continue
    if [[ "${CHECKPOINT_STEP:-latest}" == "latest" ]]; then
      candidate="$(latest_step_in_run "$run_dir" || true)"
    else
      candidate="$run_dir/$CHECKPOINT_STEP"
      has_checkpoint_params "$candidate" || candidate=""
    fi
    [[ -n "$candidate" ]] || continue
    [[ -z "$best" || "$candidate" -nt "$best" ]] && best="$candidate"
  done

  [[ -n "$best" ]] && printf '%s\n' "$best"
}

if [[ -z "${POLICY_DIR:-}" ]]; then
  POLICY_DIR="$(latest_checkpoint || true)"
fi

if [[ -z "$POLICY_DIR" ]]; then
  echo "[missing] checkpoint not found under $CHECKPOINT_BASE_DIR/$POLICY_CONFIG/$EXP_NAME/${CHECKPOINT_STEP:-latest}" >&2
  echo "Hint: check the experiment name and step, or pass POLICY_DIR=/path/to/checkpoint." >&2
  exit 1
fi
[[ -d "$POLICY_DIR/params" ]] || { echo "[missing] checkpoint params: $POLICY_DIR/params" >&2; exit 1; }
has_checkpoint_assets "$POLICY_DIR" || {
  echo "[missing] AgileX norm stats: $POLICY_DIR/assets/$ASSET_ID/norm_stats.json" >&2
  exit 1
}

echo "[AgileX server]"
echo "  port=$PORT prompt=$DEFAULT_PROMPT"
echo "  config=$POLICY_CONFIG asset=$ASSET_ID exp=$EXP_NAME step=${CHECKPOINT_STEP:-latest}"
echo "  checkpoint=$POLICY_DIR"
echo "  warmup_steps=$WARMUP_STEPS jax_cache_dir=$JAX_CACHE_DIR"

echo "Step 1: Starting policy server..."
uv run scripts/serve_policy.py \
  --port "$PORT" \
  --default-prompt "$DEFAULT_PROMPT" \
  --asset-id "$ASSET_ID" \
  --warmup-steps "$WARMUP_STEPS" \
  --jax-cache-dir "$JAX_CACHE_DIR" \
  policy:checkpoint \
  --policy.config="$POLICY_CONFIG" \
  --policy.dir="$POLICY_DIR"

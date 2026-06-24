#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   POLICY_CONFIG=pi05_xxx_lora ASSET_ID=agilex_xxx DEFAULT_PROMPT="task prompt" scripts/server.sh
#   CHECKPOINT_STEP=5000 scripts/server.sh
#   POLICY_DIR=/path/to/checkpoint scripts/server.sh
#
# 说明：这是通用 AgileX policy server 启动脚本。policy config、prompt、asset_id 都由环境变量传入。

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PORT="${PORT:-8000}"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-}"
POLICY_CONFIG="${POLICY_CONFIG:-}"
ASSET_ID="${ASSET_ID:-}"
EXP_NAME="${EXP_NAME:-overfit_10}"
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

require_env() {
  local name="$1"
  [[ -n "${!name:-}" ]] || { echo "[missing] set $name" >&2; exit 1; }
}

require_env POLICY_CONFIG
require_env ASSET_ID
require_env DEFAULT_PROMPT

if [[ -z "${POLICY_DIR:-}" ]]; then
  RUN_DIR="$CHECKPOINT_BASE_DIR/$POLICY_CONFIG/$EXP_NAME"
  if [[ -z "${CHECKPOINT_STEP:-}" || "${CHECKPOINT_STEP:-}" == "latest" ]]; then
    POLICY_DIR=""
    for candidate in "$RUN_DIR"/*; do
      [[ -d "$candidate" ]] || continue
      step="$(basename "$candidate")"
      [[ "$step" =~ ^[0-9]+$ ]] || continue
      [[ -z "$POLICY_DIR" || "$step" -gt "$(basename "$POLICY_DIR")" ]] && POLICY_DIR="$candidate"
    done
  else
    POLICY_DIR="$RUN_DIR/$CHECKPOINT_STEP"
  fi
fi

[[ -n "$POLICY_DIR" && -d "$POLICY_DIR/params" ]] || { echo "[missing] checkpoint params: $POLICY_DIR/params" >&2; exit 1; }
[[ -f "$POLICY_DIR/assets/$ASSET_ID/norm_stats.json" ]] || {
  echo "[missing] AgileX norm stats: $POLICY_DIR/assets/$ASSET_ID/norm_stats.json" >&2
  exit 1
}

echo "[AgileX server]"
echo "  port=$PORT prompt=$DEFAULT_PROMPT"
echo "  config=$POLICY_CONFIG asset=$ASSET_ID"
echo "  checkpoint=$POLICY_DIR"

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

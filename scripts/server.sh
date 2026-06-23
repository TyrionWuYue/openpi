#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/server.sh

Starts the policy server for real AgileX inference with the fine-tuned
pi0.5 LoRA checkpoint and AgileX norm stats.

Environment overrides:
  PORT                 Server port. Default: 8000.
  DEFAULT_PROMPT       Prompt used when the robot observation omits one.
  POLICY_CONFIG        OpenPI policy config.
  ASSET_ID             Norm-stats asset id. Default: agilex_cube_in_bowl.
  EXP_NAME             Training experiment name. Default: overfit_10.
  CHECKPOINT_STEP      Checkpoint step to serve. Default: latest numeric step.
  POLICY_DIR           Explicit checkpoint directory. Overrides EXP_NAME/CHECKPOINT_STEP.
  CHECKPOINT_BASE_DIR  Base checkpoint directory. Default: checkpoints.
  OPENPI_DATA_HOME     Local OpenPI artifact cache.
  WARMUP_STEPS         Dummy inference calls before opening server. Default: 2.
  JAX_CACHE_DIR        Persistent JAX compilation cache. Default: $HOME/.cache/jax.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PORT="${PORT:-8000}"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-put the cube in the bowl}"
POLICY_CONFIG="${POLICY_CONFIG:-pi05_agilex_cube_in_bowl_lora}"
ASSET_ID="${ASSET_ID:-agilex_cube_in_bowl}"
EXP_NAME="${EXP_NAME:-overfit_10}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-checkpoints}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/inspire/hdd/project/embodied-intelligent-robot-system/czxs25120101/openpi_cache}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
JAX_CACHE_DIR="${JAX_CACHE_DIR:-$HOME/.cache/jax}"

export OPENPI_DATA_HOME
export OPENPI_OFFLINE="${OPENPI_OFFLINE:-1}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

latest_checkpoint_step() {
  local run_dir="$1"
  local latest=""
  local step=""
  local candidate=""

  for candidate in "$run_dir"/*; do
    [[ -d "$candidate" ]] || continue
    step="$(basename "$candidate")"
    [[ "$step" =~ ^[0-9]+$ ]] || continue
    if [[ -z "$latest" || "$step" -gt "$latest" ]]; then
      latest="$step"
    fi
  done

  if [[ -z "$latest" ]]; then
    echo "[missing] no numeric checkpoint steps found under: $run_dir" >&2
    return 1
  fi
  printf '%s\n' "$latest"
}

if [[ -z "${POLICY_DIR:-}" ]]; then
  RUN_DIR="$CHECKPOINT_BASE_DIR/$POLICY_CONFIG/$EXP_NAME"
  if [[ -z "${CHECKPOINT_STEP:-}" || "${CHECKPOINT_STEP:-}" == "latest" ]]; then
    CHECKPOINT_STEP="$(latest_checkpoint_step "$RUN_DIR")"
  fi
  POLICY_DIR="$RUN_DIR/$CHECKPOINT_STEP"
fi

if [[ "$POLICY_DIR" != gs://* ]]; then
  if [[ ! -d "$POLICY_DIR" ]]; then
    echo "[missing] checkpoint directory: $POLICY_DIR" >&2
    exit 1
  fi
  if [[ ! -d "$POLICY_DIR/params" ]]; then
    echo "[missing] checkpoint params directory: $POLICY_DIR/params" >&2
    exit 1
  fi
  if [[ ! -f "$POLICY_DIR/assets/$ASSET_ID/norm_stats.json" ]]; then
    echo "[missing] AgileX norm stats: $POLICY_DIR/assets/$ASSET_ID/norm_stats.json" >&2
    echo "Refusing to serve because this checkpoint would not use the real AgileX normalization." >&2
    exit 1
  fi
fi

cat <<EOF
[server]
  port=$PORT
  prompt=$DEFAULT_PROMPT
  policy_config=$POLICY_CONFIG
  policy_dir=$POLICY_DIR
  asset_id=$ASSET_ID
  openpi_data_home=$OPENPI_DATA_HOME
  warmup_steps=$WARMUP_STEPS
  jax_cache_dir=$JAX_CACHE_DIR
EOF

uv run scripts/serve_policy.py \
  --port "$PORT" \
  --default-prompt "$DEFAULT_PROMPT" \
  --asset-id "$ASSET_ID" \
  --warmup-steps "$WARMUP_STEPS" \
  --jax-cache-dir "$JAX_CACHE_DIR" \
  policy:checkpoint \
  --policy.config="$POLICY_CONFIG" \
  --policy.dir="$POLICY_DIR"

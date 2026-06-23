#!/usr/bin/env bash
set -Eeuo pipefail

PORT="${PORT:-8000}"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-fold the towel}"
POLICY_CONFIG="${POLICY_CONFIG:-pi0_aloha_towel}"
POLICY_DIR="${POLICY_DIR:-gs://openpi-assets/checkpoints/pi0_aloha_towel}"

uv run scripts/serve_policy.py \
  --port "$PORT" \
  --default_prompt "$DEFAULT_PROMPT" \
  --warmup-steps "${WARMUP_STEPS:-2}" \
  --jax-cache-dir "${JAX_CACHE_DIR:-$HOME/.cache/jax}" \
  policy:checkpoint \
  --policy.config="$POLICY_CONFIG" \
  --policy.dir="$POLICY_DIR"

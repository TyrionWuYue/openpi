#!/usr/bin/env bash
set -Eeuo pipefail

uv run scripts/serve_policy.py \
  --port 8000 \
  --default_prompt "Transfer Cube" \
  policy:checkpoint \
  --policy.config=pi05_aloha \
  --policy.dir=/inspire/hdd/project/embodied-intelligent-robot-system/czxs25120101/openpi_cache/openpi-assets/checkpoints/pi05_base

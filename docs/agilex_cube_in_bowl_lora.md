# AgileX Cube-in-Bowl LoRA Fine-Tuning

This pipeline fine-tunes pi0.5 with LoRA on AgileX ALOHA-style HDF5 episodes.
It keeps OpenPI's standard flow:

1. Convert HDF5 episodes to LeRobot.
2. Compute norm stats from the converted AgileX dataset.
3. LoRA fine-tune from pi0.5 base weights.
4. Serve the fine-tuned checkpoint.

## 1. Convert HDF5 to LeRobot

```bash
uv run examples/aloha_real/convert_agilex_hdf5_to_lerobot.py \
  --raw-dir /inspire/qb-ilm2/project/embodied-intelligent-robot-system/public/Cube_in_Bowl \
  --repo-id sii_team9/cube_in_bowl \
  --task "put the cube in the bowl" \
  --overwrite
```

The converter expects files named `episode_1.hdf5`, `episode_2.hdf5`, and so on.
It writes a local LeRobot dataset under the default LeRobot cache for
`sii_team9/cube_in_bowl`.

## 2. Compute AgileX Norm Stats

Use your own asset id, not `trossen`.

```bash
uv run scripts/compute_norm_stats.py \
  --config-name pi05_agilex_cube_in_bowl_lora \
  --asset-id agilex_cube_in_bowl
```

This writes:

```text
assets/pi05_agilex_cube_in_bowl_lora/agilex_cube_in_bowl/norm_stats.json
```

If you choose a different asset id, use the same value for norm stats, training,
and serving.

## 3. LoRA Fine-Tune

The config defaults to the local pi0.5 base weights at:

```text
/inspire/hdd/project/embodied-intelligent-robot-system/czxs25120101/openpi_cache/openpi-assets/checkpoints/pi05_base/params
```

Run:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi05_agilex_cube_in_bowl_lora \
  --exp-name overfit_10 \
  --overwrite \
  --data.assets.asset-id agilex_cube_in_bowl
```

The default AgileX LoRA config trains for `15_000` steps and saves every `500`
steps. This is intended as a roughly six-hour overfit run on a single RTX 4090,
assuming the converted 10-demo dataset is already local and the pi0.5 base
checkpoint is cached.

For a quick overfit check on 10 demos, you can lower steps without changing the
pipeline:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi05_agilex_cube_in_bowl_lora \
  --exp-name overfit_10 \
  --overwrite \
  --num-train-steps 3000 \
  --save-interval 500 \
  --data.assets.asset-id agilex_cube_in_bowl
```

If the local base checkpoint is unavailable, override the weight loader params
path with the official GCS checkpoint:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi05_agilex_cube_in_bowl_lora \
  --exp-name overfit_10 \
  --overwrite \
  --weight-loader.params-path gs://openpi-assets/checkpoints/pi05_base/params \
  --data.assets.asset-id agilex_cube_in_bowl
```

## 4. Serve the Fine-Tuned Checkpoint

Replace `3000` with the checkpoint step you want to serve.

```bash
uv run scripts/serve_policy.py \
  --port 8000 \
  --default-prompt "put the cube in the bowl" \
  --asset-id agilex_cube_in_bowl \
  policy:checkpoint \
  --policy.config pi05_agilex_cube_in_bowl_lora \
  --policy.dir checkpoints/pi05_agilex_cube_in_bowl_lora/overfit_10/3000
```

The robot side should use the same prompt and connect to this server as usual.

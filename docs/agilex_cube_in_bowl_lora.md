# AgileX Cube-in-Bowl LoRA Fine-Tuning

This pipeline fine-tunes pi0.5 with LoRA on AgileX ALOHA-style HDF5 episodes.
It keeps OpenPI's standard flow:

1. Convert HDF5 episodes to LeRobot.
2. Compute norm stats from the converted AgileX dataset.
3. LoRA fine-tune from pi0.5 base weights.
4. Serve the fine-tuned checkpoint.

For a fully local run, launch commands from the OpenPI repo root. The converted
LeRobot dataset is kept under `agilex_data/`. OpenPI remote artifacts should be
pre-populated under a local `OPENPI_DATA_HOME`; on the current server we use:

```bash
export OPENPI_DATA_HOME=/inspire/hdd/project/embodied-intelligent-robot-system/czxs25120101/openpi_cache
export OPENPI_OFFLINE=1
export HF_HOME="$PWD/.hf_cache"
export HF_DATASETS_CACHE="$PWD/.hf_cache/datasets"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
```

Required local artifacts for `pi05_agilex_cube_in_bowl_lora`:

```text
agilex_data/sii_team9/cube_in_bowl/
assets/pi05_agilex_cube_in_bowl_lora/agilex_cube_in_bowl/norm_stats.json
/inspire/hdd/project/embodied-intelligent-robot-system/czxs25120101/openpi_cache/openpi-assets/checkpoints/pi05_base/params/
/inspire/hdd/project/embodied-intelligent-robot-system/czxs25120101/openpi_cache/big_vision/paligemma_tokenizer.model
```

The tokenizer path above is the local cache for:

```text
gs://big_vision/paligemma_tokenizer.model
```

With `OPENPI_OFFLINE=1`, OpenPI will fail fast if a required remote artifact is
not already present at its local cache path instead of attempting a network
download.

Quick local artifact check:

```bash
test -d agilex_data/sii_team9/cube_in_bowl
test -f assets/pi05_agilex_cube_in_bowl_lora/agilex_cube_in_bowl/norm_stats.json
test -d /inspire/hdd/project/embodied-intelligent-robot-system/czxs25120101/openpi_cache/openpi-assets/checkpoints/pi05_base/params
test -f /inspire/hdd/project/embodied-intelligent-robot-system/czxs25120101/openpi_cache/big_vision/paligemma_tokenizer.model
```

## Two-Step Pipeline

From the OpenPI repo root:

```bash
scripts/prepare_agilex_cube_in_bowl_data.sh
scripts/train_agilex_cube_in_bowl_lora.sh
```

The first script prepares data:

1. HDF5 to LeRobot conversion.
2. AgileX norm stats computation.

The second script runs pi0.5 LoRA training only.

Data preparation does not overwrite by default. To rebuild the converted dataset
and recompute norm stats:

```bash
scripts/prepare_agilex_cube_in_bowl_data.sh --overwrite
```

You can also run one data stage at a time:

```bash
scripts/prepare_agilex_cube_in_bowl_data.sh check
scripts/prepare_agilex_cube_in_bowl_data.sh convert --overwrite
scripts/prepare_agilex_cube_in_bowl_data.sh norm --overwrite
```

Training checks and overrides:

```bash
scripts/train_agilex_cube_in_bowl_lora.sh check
scripts/train_agilex_cube_in_bowl_lora.sh train
```

Useful overrides:

```bash
EXP_NAME=overfit_10 NUM_TRAIN_STEPS=6250 scripts/train_agilex_cube_in_bowl_lora.sh train
WANDB_MODE=offline scripts/train_agilex_cube_in_bowl_lora.sh train
```

## 1. Convert HDF5 to LeRobot

```bash
uv run examples/aloha_real/convert_agilex_hdf5_to_lerobot.py \
  --raw-dir /inspire/qb-ilm2/project/embodied-intelligent-robot-system/public/Cube_in_Bowl \
  --repo-id sii_team9/cube_in_bowl \
  --task "put the cube in the bowl" \
  --local-dir agilex_data \
  --overwrite
```

The converter expects files named `episode_1.hdf5`, `episode_2.hdf5`, and so on.
It writes a local LeRobot dataset to:

```text
agilex_data/sii_team9/cube_in_bowl
```

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

The default AgileX LoRA config trains for `6_250` steps and saves every `500`
steps. Based on the observed `~4.5s/step` RTX 4090 run, this targets roughly
eight hours assuming the converted 10-demo dataset is already local and the
pi0.5 base checkpoint is cached.

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

The default server script serves the latest numeric checkpoint under
`checkpoints/pi05_agilex_cube_in_bowl_lora/overfit_10/` and refuses to start
unless the checkpoint contains the AgileX norm stats at
`assets/agilex_cube_in_bowl/norm_stats.json`.

```bash
scripts/server.sh
```

To serve a specific checkpoint step:

```bash
CHECKPOINT_STEP=6000 scripts/server.sh
```

To serve an explicit checkpoint directory:

```bash
POLICY_DIR=checkpoints/pi05_agilex_cube_in_bowl_lora/overfit_10/6249 scripts/server.sh
```

The robot side should use the same prompt and connect to this server as usual.

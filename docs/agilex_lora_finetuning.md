# AgileX LoRA 微调速查

当前 AgileX 真机三路 RGB 相机约 30Hz，所以数据转换和机器人推理默认按 30fps/30Hz。

## 1. 准备数据

```bash
RAW_DIR=/path/to/hdf5_episodes \
REPO_ID=sii_team9/task_name \
TASK="task prompt" \
CONFIG=pi05_your_agilex_task_lora \
ASSET_ID=agilex_task_name \
scripts/prepare_agilex_data.sh --overwrite
```

这一步会生成：

```text
agilex_data/<REPO_ID>/
assets/<CONFIG>/<ASSET_ID>/norm_stats.json
```

## 2. 训练

```bash
CONFIG=pi05_your_agilex_task_lora \
REPO_ID=sii_team9/task_name \
ASSET_ID=agilex_task_name \
scripts/train_agilex_lora.sh
```

常用覆盖：

```bash
NUM_TRAIN_STEPS=3000 scripts/train_agilex_lora.sh
SAVE_INTERVAL=1000 scripts/train_agilex_lora.sh
EXP_NAME=my_run WANDB_MODE=offline scripts/train_agilex_lora.sh
```

## 3. 起推理 Server

```bash
POLICY_CONFIG=pi05_your_agilex_task_lora \
ASSET_ID=agilex_task_name \
DEFAULT_PROMPT="task prompt" \
scripts/server.sh
```

指定 checkpoint：

```bash
CHECKPOINT_STEP=5000 scripts/server.sh
POLICY_DIR=/path/to/checkpoint scripts/server.sh
```

server 会检查 checkpoint 内是否带有对应 `ASSET_ID` 的 AgileX norm stats，避免误用其他机器或其他任务的 normalization。

## 4. 机器人端

机器人端仍然运行：

```bash
./run_aloha_inference.sh
```

默认 `CONTROL_HZ=30`、`ACTION_HORIZON=25`。`ACTION_HORIZON` 要和训练 config 保持一致。

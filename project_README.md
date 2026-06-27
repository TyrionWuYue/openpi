# AgileX OpenPI LoRA 实验说明

本文档记录本项目中 AgileX ALOHA 机器人上的 pi0.5 LoRA 微调流程，包含环境配置、数据准备、训练/推理运行步骤和复现实验要点。默认实验配置是 `pi05_agilex_aloha`，任务提示词是 `put the cube in the bowl`。

## 1. 项目目标

本项目基于 Physical Intelligence 的 `openpi`，使用 pi0.5 base checkpoint 在 AgileX ALOHA 双臂平台上做 LoRA 微调。当前配置只训练和执行左臂相关动作：

- 状态/动作接口仍使用 ALOHA 的 14 维格式：`[left_arm_6, left_gripper, right_arm_6, right_gripper]`。
- 模型内部 `action_dim=32`，原始 14 维动作会 pad 到 32 维。
- `action_loss_mask=(1.0,) * 7 + (0.0,) * 25`：训练 loss 只计算前 7 维，即左臂 6 个关节和左夹爪；后 25 维不参与 loss。
- `masked_action_dims=tuple(range(7, 14))`：右臂 7 维在训练输入中置零，推理输出时拷回当前 state，从而保持右臂不被策略控制。

这两个 mask 作用不同：`action_loss_mask` 控制“哪些维度算 loss”，`masked_action_dims` 控制“哪些维度不学习、不执行”。

## 2. 环境配置

推荐环境：

- Ubuntu 22.04。
- Python 3.11。
- NVIDIA GPU，LoRA 微调建议至少 24GB 显存；全量微调需要更大显存。
- CUDA 12 兼容驱动。
- `uv` 用于依赖管理。

首次克隆或补齐子模块：

```bash
git submodule update --init --recursive
```

安装 Python 依赖：

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

如果使用 RLDS 相关功能，再额外安装：

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync --group rlds
```

训练和推理脚本默认以离线方式运行，避免运行时访问 HuggingFace、Transformers 或 GCS。需要提前准备好：

```text
$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base/params
$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model
```

本项目脚本默认：

```bash
export OPENPI_DATA_HOME=/inspire/hdd/project/embodied-intelligent-robot-system/czxs25120101/openpi_cache
```

如果换机器，建议显式设置为自己的缓存目录：

```bash
export OPENPI_DATA_HOME=/path/to/openpi_cache
export BASE_PARAMS_PATH=$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base/params
export TOKENIZER_PATH=$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model
```

脚本会自动设置以下离线变量：

```bash
OPENPI_OFFLINE=1
HF_HUB_OFFLINE=1
HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1
WANDB_MODE=disabled
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
```

如需使用 W&B，运行训练前覆盖：

```bash
export WANDB_MODE=online
```

## 3. 数据准备

### 3.1 原始 HDF5 格式

原始数据目录需要包含若干 episode 文件：

```text
raw_dir/
  episode_0.hdf5
  episode_1.hdf5
  ...
```

每个 HDF5 文件必须包含：

```text
/action
/observations/qpos
/observations/images/cam_high
/observations/images/cam_left_wrist
/observations/images/cam_right_wrist
```

可选字段：

```text
/observations/qvel
/observations/effort
```

要求：

- `/action`、`/observations/qpos` 和所有相机帧数一致。
- state/action 维度为 14，顺序为左臂 6 关节、左夹爪、右臂 6 关节、右夹爪。
- 三路相机默认是 `cam_high`、`cam_left_wrist`、`cam_right_wrist`。
- 图像可以是未压缩 RGB 数组，也可以是 HDF5 内压缩图像；转换脚本会解码后写入 LeRobot 数据集。

### 3.2 HDF5 转 LeRobot 并计算 norm stats

一条命令完成转换和归一化统计：

```bash
RAW_DIR=/path/to/hdf5_episodes \
REPO_ID=sii_team9/cube_in_bowl \
TASK="put the cube in the bowl" \
CONFIG=pi05_agilex_aloha \
ASSET_ID=agilex_cube_in_bowl \
scripts/prepare_agilex_data.sh --overwrite
```

输出位置：

```text
agilex_data/sii_team9/cube_in_bowl/
assets/pi05_agilex_aloha/agilex_cube_in_bowl/norm_stats.json
```

也可以分步执行：

```bash
# 只转换数据
MODE=convert RAW_DIR=/path/to/hdf5_episodes \
REPO_ID=sii_team9/cube_in_bowl \
TASK="put the cube in the bowl" \
CONFIG=pi05_agilex_aloha \
ASSET_ID=agilex_cube_in_bowl \
scripts/prepare_agilex_data.sh --overwrite

# 只重新计算 norm stats
MODE=norm RAW_DIR=/path/to/hdf5_episodes \
REPO_ID=sii_team9/cube_in_bowl \
TASK="put the cube in the bowl" \
CONFIG=pi05_agilex_aloha \
ASSET_ID=agilex_cube_in_bowl \
scripts/prepare_agilex_data.sh --overwrite
```

常用数据参数：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `LEROBOT_ROOT` | `agilex_data` | LeRobot 数据集根目录。可以是相对路径或绝对路径。 |
| `FPS` | `30` | 数据集帧率，当前真机推理也默认 30Hz。 |
| `ACTION_SOURCE` | `next_qpos` | 训练 action 来源。`next_qpos` 表示用下一帧 qpos 作为当前帧动作目标。 |
| `STATIONARY_ACTION_DIMS` | `auto` | 转换时自动找静止维度，并把这些维度的 action 设成当前 state。 |
| `STATIONARY_DELTA_THRESHOLD` | `1e-4` | `auto` 判断静止维度时使用的 qpos delta 阈值。 |

注意：右臂是否参与训练/推理不是由数据转换决定，而是由训练 config 里的 `masked_action_dims` 和 `action_loss_mask` 决定。

## 4. 训练配置

当前核心配置位于 `src/openpi/training/config.py` 的 `pi05_agilex_aloha`。

| 项 | 当前值 |
| --- | --- |
| base model | pi0.5 base |
| model variants | `gemma_2b_lora` + `gemma_300m_lora` |
| `action_horizon` | `50` |
| `action_dim` | `32` |
| `batch_size` | `32` |
| `num_train_steps` | `2_000` |
| `learning rate` | `5e-5` |
| LR schedule | cosine schedule，但 `decay_lr=5e-5`，实际保持 5e-5 |
| warmup | `1_000` steps |
| optimizer | AdamW |
| gradient clipping | global norm `1.0` |
| EMA | `None` |
| 保存间隔 | `500` steps |
| 保留周期 | `500` steps |
| 训练随机种子 | `42` |

LoRA 冻结逻辑来自 `Pi0Config.get_freeze_filter()`：使用 LoRA variant 时，基础 Gemma 权重冻结，只训练 LoRA 参数和未被冻结的相关参数。

### 4.1 启动训练

```bash
CONFIG=pi05_agilex_aloha \
REPO_ID=sii_team9/cube_in_bowl \
ASSET_ID=agilex_cube_in_bowl \
DEFAULT_PROMPT="put the cube in the bowl" \
EXP_NAME=overfit_10 \
scripts/train_agilex_lora.sh
```

训练前脚本会检查：

- `agilex_data/<REPO_ID>` 是否存在。
- `assets/<CONFIG>/<ASSET_ID>/norm_stats.json` 是否存在。
- `BASE_PARAMS_PATH` 是否存在。
- `TOKENIZER_PATH` 是否存在。

checkpoint 默认保存到：

```text
checkpoints/pi05_agilex_aloha/<EXP_NAME>/<step>/
```

每个 checkpoint 会包含：

```text
params/
assets/<ASSET_ID>/norm_stats.json
train_state/
```

推理 server 会检查 checkpoint 内的 `assets/<ASSET_ID>/norm_stats.json`，避免误用其他任务或其他机器人数据的 normalization。

### 4.2 常用训练覆盖

```bash
# 改训练步数
NUM_TRAIN_STEPS=3000 scripts/train_agilex_lora.sh

# 改保存频率
SAVE_INTERVAL=1000 KEEP_PERIOD=1000 scripts/train_agilex_lora.sh

# 覆盖旧实验目录并重训
OVERWRITE=1 scripts/train_agilex_lora.sh

# 使用指定 base checkpoint
BASE_PARAMS_PATH=/path/to/pi05_base/params scripts/train_agilex_lora.sh
```

如果要改 learning rate，推荐在 `src/openpi/training/config.py` 中固定记录，保证复现实验时 config 自包含。例如：

```python
lr_schedule=_optimizer.CosineDecaySchedule(
    warmup_steps=1_000,
    peak_lr=5e-5,
    decay_steps=1_000_000,
    decay_lr=5e-5,
)
```

也可以直接用 `scripts/train.py` 做一次性覆盖：

```bash
uv run scripts/train.py pi05_agilex_aloha \
  --exp-name lr_1e-5 \
  --data.assets.asset-id agilex_cube_in_bowl \
  --data.repo-id sii_team9/cube_in_bowl \
  --data.default-prompt "put the cube in the bowl" \
  --weight-loader.params-path "$BASE_PARAMS_PATH" \
  --lr-schedule.peak-lr 1e-5 \
  --lr-schedule.decay-lr 1e-5 \
  --lr-schedule.warmup-steps 1000 \
  --num-train-steps 3000 \
  --save-interval 500 \
  --keep-period 500
```

## 5. 启动推理 Server

默认启动最新 checkpoint：

```bash
POLICY_CONFIG=pi05_agilex_aloha \
ASSET_ID=agilex_cube_in_bowl \
DEFAULT_PROMPT="put the cube in the bowl" \
scripts/server.sh
```

指定实验名或 step：

```bash
# 指定 latest 实验中的 500 step
POLICY_CONFIG=pi05_agilex_aloha \
ASSET_ID=agilex_cube_in_bowl \
scripts/server.sh 500

# 指定某个实验的 1000 step
POLICY_CONFIG=pi05_agilex_aloha \
ASSET_ID=agilex_cube_in_bowl \
scripts/server.sh overfit_10 1000

# 直接指定 checkpoint 目录
POLICY_DIR=/path/to/checkpoints/pi05_agilex_aloha/overfit_10/1000 \
ASSET_ID=agilex_cube_in_bowl \
scripts/server.sh
```

常用 server 参数：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PORT` | `8000` | policy websocket server 端口。 |
| `DEFAULT_PROMPT` | `put the cube in the bowl` | 请求中没有 prompt 时使用的默认提示词。 |
| `WARMUP_STEPS` | `2` | 开服务前做几次 dummy inference，提前触发 JAX/XLA 编译。 |
| `JAX_CACHE_DIR` | `$HOME/.cache/jax` | JAX 编译缓存目录。 |
| `CHECKPOINT_BASE_DIR` | `checkpoints` | checkpoint 根目录。 |

## 6. 机器人端运行

机器人端在部署目录运行推理脚本；本仓库对应文件是 `scripts/run_aloha_inference.sh`，部署到机器人工作目录后通常按下面方式执行：

```bash
./run_aloha_inference.sh
```

如果 policy server 是远程代理地址：

```bash
POLICY_URL="https://.../proxy/8000/" ./run_aloha_inference.sh
```

只做连通性和 ROS topic 预检查：

```bash
PREFLIGHT_ONLY=1 ./run_aloha_inference.sh
```

当前机器人端默认：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `ACTION_HORIZON` | `50` | 每次 policy 返回并执行的 action chunk 长度，应和训练 config 的 `action_horizon` 一致。 |
| `CONTROL_HZ` | `30` | 开环执行频率。 |
| `MAX_EPISODE_STEPS` | `1000` | 单次 episode 最大步数。 |
| `INITIAL_RESET` | `1` | episode 前是否 reset。 |
| `FINAL_RESET` | `1` | episode 后是否 reset。 |
| `OPENPI_POLICY_ACTION_MODE` | `clamp` | 动作安全模式，可选 `clamp`、`hold`、`raw`。 |
| `OPENPI_MAX_POLICY_ARM_DELTA` | `0.08` | `clamp` 模式下单步关节最大变化。 |
| `OPENPI_MAX_POLICY_GRIPPER_DELTA` | `0.20` | `clamp` 模式下单步夹爪最大变化。 |
| `OPENPI_AGILEX_GRIPPER_MIN` | `0.0` | AgileX/Piper 夹爪最小值。 |
| `OPENPI_AGILEX_GRIPPER_MAX` | `0.10` | AgileX/Piper 夹爪最大值。 |

如果示教数据没有自动 reset，可以把机器人手动放到类似示教初始姿态后运行：

```bash
INITIAL_RESET=0 FINAL_RESET=0 ./run_aloha_inference.sh
```

## 7. 复现实验步骤

建议每次实验固定以下信息：

```bash
export CONFIG=pi05_agilex_aloha
export REPO_ID=sii_team9/cube_in_bowl
export ASSET_ID=agilex_cube_in_bowl
export TASK="put the cube in the bowl"
export DEFAULT_PROMPT="put the cube in the bowl"
export EXP_NAME=overfit_10
export RAW_DIR=/path/to/hdf5_episodes
export OPENPI_DATA_HOME=/path/to/openpi_cache
```

完整复现：

```bash
# 1. 安装依赖
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# 2. 转换数据并计算 norm stats
RAW_DIR=$RAW_DIR \
REPO_ID=$REPO_ID \
TASK="$TASK" \
CONFIG=$CONFIG \
ASSET_ID=$ASSET_ID \
scripts/prepare_agilex_data.sh --overwrite

# 3. 训练
CONFIG=$CONFIG \
REPO_ID=$REPO_ID \
ASSET_ID=$ASSET_ID \
DEFAULT_PROMPT="$DEFAULT_PROMPT" \
EXP_NAME=$EXP_NAME \
OVERWRITE=1 \
scripts/train_agilex_lora.sh

# 4. 启动 server
POLICY_CONFIG=$CONFIG \
ASSET_ID=$ASSET_ID \
DEFAULT_PROMPT="$DEFAULT_PROMPT" \
EXP_NAME=$EXP_NAME \
scripts/server.sh
```

为了让结果可复现，记录：

- git commit hash：`git rev-parse HEAD`。
- `git diff`，尤其是 `src/openpi/training/config.py`、`scripts/*.sh`、数据转换脚本。
- 原始 HDF5 数据路径、episode 数量、是否删过坏 episode。
- `REPO_ID`、`ASSET_ID`、`TASK`、`DEFAULT_PROMPT`。
- `ACTION_SOURCE`、`STATIONARY_ACTION_DIMS`、`STATIONARY_DELTA_THRESHOLD`。
- `num_train_steps`、`batch_size`、learning rate、warmup、save interval。
- 使用的 base checkpoint 路径和 tokenizer 路径。
- 推理时的 checkpoint step、`ACTION_HORIZON`、`CONTROL_HZ` 和动作安全参数。

## 8. 常见问题

### 找不到 norm stats

报错类似：

```text
[missing] norm stats: assets/<CONFIG>/<ASSET_ID>/norm_stats.json
```

先运行：

```bash
MODE=norm RAW_DIR=/path/to/hdf5_episodes \
REPO_ID=sii_team9/cube_in_bowl \
TASK="put the cube in the bowl" \
CONFIG=pi05_agilex_aloha \
ASSET_ID=agilex_cube_in_bowl \
scripts/prepare_agilex_data.sh --overwrite
```

确认 `CONFIG`、`ASSET_ID` 和训练命令完全一致。

### checkpoint 已存在

训练默认不会覆盖已有目录。需要继续训练用 `--resume` 或直接使用脚本覆盖：

```bash
OVERWRITE=1 scripts/train_agilex_lora.sh
```

注意：`OVERWRITE=1` 会删除同名实验目录后重训。

### server 找不到 checkpoint

确认目录结构为：

```text
checkpoints/<POLICY_CONFIG>/<EXP_NAME>/<STEP>/params
```

可以直接指定：

```bash
POLICY_DIR=/absolute/path/to/checkpoint scripts/server.sh
```

### 首次推理很慢

第一次 JAX/XLA 编译会慢。server 默认 `WARMUP_STEPS=2`，会在监听机器人请求前先 warmup。调试启动时可以临时关闭：

```bash
WARMUP_STEPS=0 scripts/server.sh
```

正式跑机器人不建议关闭 warmup。

### 右臂发生非预期动作

检查三处是否一致：

- config 中 `masked_action_dims=tuple(range(7, 14))`。
- config 中 `action_loss_mask=(1.0,) * 7 + (0.0,) * 25`。
- 机器人端 `OPENPI_POLICY_ACTION_MODE=clamp`，不要在未经确认时使用 `raw`。

## 9. 关键文件索引

| 文件 | 作用 |
| --- | --- |
| `src/openpi/training/config.py` | 训练 config、learning rate、mask、数据路径、LoRA 配置。 |
| `src/openpi/models/pi0.py` | `action_loss_mask` 在 flow matching loss 中生效的位置。 |
| `src/openpi/transforms.py` | `ZeroActionDims` 和 `CopyStateToActionDims` 的实现。 |
| `src/openpi/policies/aloha_policy.py` | ALOHA/AgileX 输入输出转换、夹爪范围和图像 mask。 |
| `examples/aloha_real/convert_agilex_hdf5_to_lerobot.py` | HDF5 到 LeRobot 的数据转换。 |
| `scripts/prepare_agilex_data.sh` | 数据转换和 norm stats 一键脚本。 |
| `scripts/train_agilex_lora.sh` | LoRA 训练入口。 |
| `scripts/server.sh` | policy server 启动入口。 |
| `scripts/run_aloha_inference.sh` | 机器人端推理入口。 |

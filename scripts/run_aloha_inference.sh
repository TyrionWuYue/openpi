#!/usr/bin/env bash
set -Eeuo pipefail

# OpenPI ALOHA reset/inference launcher.
# Hardware initialization belongs in ./env.sh.
#
# If your demonstrations were collected without an automatic reset, place the robot
# in a demonstration-like start pose and run:
#   INITIAL_RESET=0 FINAL_RESET=0 ./run_aloha_inference.sh

TEAM9_DIR="${TEAM9_DIR:-/home/agilex/team9}"
PIPER_WS="${PIPER_WS:-/home/agilex/cobot_magic/Piper_ros_private-ros-noetic}"
CONDA_SH="${CONDA_SH:-/home/agilex/miniconda3/etc/profile.d/conda.sh}"

DEFAULT_POLICY_URL="https://nat-notebook-inspire.sii.edu.cn/ws-6040202d-b785-4b37-98b0-c68d65dd52ce/project-aa6664a6-c21b-426f-b8b9-f3d84953ccff/user-acc4c516-9dde-4134-b1f4-bcc86baad8f6/vscode/3bcf2884-8d7e-4feb-8145-964349be5580/dec52103-8ead-4f7a-ad20-9d664add10cb/proxy/8000/"
POLICY_URL="${POLICY_URL:-${1:-$DEFAULT_POLICY_URL}}"

FRONT_TOPIC="${FRONT_TOPIC:-/camera_f/color/image_raw}"
LEFT_TOPIC="${LEFT_TOPIC:-/camera_l/color/image_raw}"
RIGHT_TOPIC="${RIGHT_TOPIC:-/camera_r/color/image_raw}"

ACTION_HORIZON="${ACTION_HORIZON:-25}"
CONTROL_HZ="${CONTROL_HZ:-30}"
NUM_EPISODES="${NUM_EPISODES:-1}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-1000}"
INITIAL_RESET="${INITIAL_RESET:-1}"
FINAL_RESET="${FINAL_RESET:-1}"
INFERENCE_TIMEOUT_SEC="${INFERENCE_TIMEOUT_SEC:-300}"
OPENPI_POLICY_ACTION_MODE="${OPENPI_POLICY_ACTION_MODE:-clamp}"
OPENPI_MAX_POLICY_ARM_DELTA="${OPENPI_MAX_POLICY_ARM_DELTA:-0.08}"
OPENPI_MAX_POLICY_GRIPPER_DELTA="${OPENPI_MAX_POLICY_GRIPPER_DELTA:-0.20}"
OPENPI_AGILEX_GRIPPER_MIN="${OPENPI_AGILEX_GRIPPER_MIN:-0.0}"
OPENPI_AGILEX_GRIPPER_MAX="${OPENPI_AGILEX_GRIPPER_MAX:-0.10}"
OPENPI_POLICY_ACTION_LOG_STEPS="${OPENPI_POLICY_ACTION_LOG_STEPS:-8}"
OPENPI_POLICY_PUBLISH_TICKS="${OPENPI_POLICY_PUBLISH_TICKS:-1}"
OPENPI_POLICY_PUBLISH_RATE="${OPENPI_POLICY_PUBLISH_RATE:-30}"
OPENPI_RUNTIME_TIMING_LOG_STEPS="${OPENPI_RUNTIME_TIMING_LOG_STEPS:-8}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
RESET_ONLY="${RESET_ONLY:-0}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-/tmp/openpi_aloha_${RUN_ID}}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/client.log}"

usage() {
  cat <<EOF
Usage:
  ./run_aloha_inference.sh
  POLICY_URL='https://.../proxy/8000/' ./run_aloha_inference.sh
  PREFLIGHT_ONLY=1 ./run_aloha_inference.sh
  RESET_ONLY=1 ./run_aloha_inference.sh

Run ./env.sh first after boot, camera re-plug, or arm power-cycle.

Main env:
  ACTION_HORIZON=${ACTION_HORIZON}
  CONTROL_HZ=${CONTROL_HZ}
  MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS}
  INITIAL_RESET=${INITIAL_RESET}
  FINAL_RESET=${FINAL_RESET}
  INFERENCE_TIMEOUT_SEC=${INFERENCE_TIMEOUT_SEC}
  OPENPI_POLICY_ACTION_MODE=${OPENPI_POLICY_ACTION_MODE}
  OPENPI_MAX_POLICY_ARM_DELTA=${OPENPI_MAX_POLICY_ARM_DELTA}
  OPENPI_MAX_POLICY_GRIPPER_DELTA=${OPENPI_MAX_POLICY_GRIPPER_DELTA}
  OPENPI_AGILEX_GRIPPER_MIN=${OPENPI_AGILEX_GRIPPER_MIN}
  OPENPI_AGILEX_GRIPPER_MAX=${OPENPI_AGILEX_GRIPPER_MAX}
  OPENPI_POLICY_PUBLISH_TICKS=${OPENPI_POLICY_PUBLISH_TICKS}
  OPENPI_POLICY_PUBLISH_RATE=${OPENPI_POLICY_PUBLISH_RATE}
  OPENPI_RUNTIME_TIMING_LOG_STEPS=${OPENPI_RUNTIME_TIMING_LOG_STEPS}
  FRONT_TOPIC=${FRONT_TOPIC}
  LEFT_TOPIC=${LEFT_TOPIC}
  RIGHT_TOPIC=${RIGHT_TOPIC}
EOF
}

source_ros() {
  set +u
  source /opt/ros/noetic/setup.bash
  [[ -f "${PIPER_WS}/devel/setup.bash" ]] && source "${PIPER_WS}/devel/setup.bash"
  set -u
}

conda_aloha() {
  set +u
  source "$CONDA_SH"
  conda activate aloha
  set -u
}

normalize_policy_url() {
  case "$POLICY_URL" in
    */proxy) POLICY_URL="${POLICY_URL}/8000/" ;;
    */proxy/) POLICY_URL="${POLICY_URL}8000/" ;;
    */proxy/8000) POLICY_URL="${POLICY_URL}/" ;;
  esac
}

need_msg() {
  local topic="$1"
  echo "[check] ${topic}"
  if ! timeout 15 bash -lc "source /opt/ros/noetic/setup.bash; rostopic echo -n1 '$topic' >/dev/null"; then
    echo "[error] no message from ${topic}" >&2
    echo "[hint] run ./env.sh first, then retry ./run_aloha_inference.sh" >&2
    return 1
  fi
}

need_subscriber() {
  local topic="$1"
  echo "[check-sub] ${topic}"
  if ! timeout 10 bash -lc "source /opt/ros/noetic/setup.bash; rostopic info '$topic' 2>/dev/null | awk '/Subscribers:/{seen=1; next} seen && /\\*/{found=1} END{exit !found}'"; then
    echo "[error] no subscriber on ${topic}" >&2
    echo "[hint] run ./env.sh first, then retry ./run_aloha_inference.sh" >&2
    return 1
  fi
}

preflight_policy_server() {
  echo "[check] policy server"
  cd "$TEAM9_DIR"
  source_ros
  conda_aloha
  POLICY_URL="$POLICY_URL" python - <<'PY' 2>&1 | tee "${LOG_DIR}/policy_preflight.log"
import os
import socket
from openpi_client import websocket_client_policy

try:
    client = websocket_client_policy.WebsocketClientPolicy(host=os.environ["POLICY_URL"], port=8000)
except (socket.timeout, TimeoutError) as exc:
    raise RuntimeError(
        "Policy server websocket TLS handshake timed out. "
        "The GPU server/proxy URL is not reachable from the robot right now; "
        "check that the server is running and that DEFAULT_POLICY_URL/POLICY_URL is current."
    ) from exc
metadata = client.get_server_metadata()
print("metadata", metadata, flush=True)
if not isinstance(metadata, dict):
    raise RuntimeError(f"Policy server metadata should be a dict, got {type(metadata).__name__}: {metadata!r}")
client._ws.close()
PY
}

validate_config() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
  fi
  normalize_policy_url
}

preflight_env() {
  source_ros
  need_msg /puppet/joint_left
  need_msg /puppet/joint_right
  need_msg "$FRONT_TOPIC"
  need_msg "$LEFT_TOPIC"
  need_msg "$RIGHT_TOPIC"
  need_subscriber /master/joint_left
  need_subscriber /master/joint_right
  need_subscriber /enable_flag
}

run_reset_only() {
  echo "[reset] log=${RUN_LOG}"
  cd "$TEAM9_DIR"
  conda_aloha
  python - <<'PY' 2>&1 | tee "$RUN_LOG"
from examples.aloha_real import real_env

env = real_env.make_real_env(init_node=True, reset_position=None)
env.reset()
print("[openpi] Reset-only completed", flush=True)
PY
}

run_inference() {
  echo "[run] log=${RUN_LOG}"
  echo "[run] ACTION_HORIZON=${ACTION_HORIZON} CONTROL_HZ=${CONTROL_HZ} MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS} INITIAL_RESET=${INITIAL_RESET} FINAL_RESET=${FINAL_RESET} OPENPI_POLICY_ACTION_MODE=${OPENPI_POLICY_ACTION_MODE}"
  cd "$TEAM9_DIR"
  conda_aloha
  export OPENPI_INFERENCE_TIMEOUT_SEC="$INFERENCE_TIMEOUT_SEC"
  export OPENPI_POLICY_ACTION_MODE
  export OPENPI_MAX_POLICY_ARM_DELTA
  export OPENPI_MAX_POLICY_GRIPPER_DELTA
  export OPENPI_AGILEX_GRIPPER_MIN
  export OPENPI_AGILEX_GRIPPER_MAX
  export OPENPI_POLICY_ACTION_LOG_STEPS
  export OPENPI_POLICY_PUBLISH_TICKS
  export OPENPI_POLICY_PUBLISH_RATE
  export OPENPI_RUNTIME_TIMING_LOG_STEPS
  reset_args=()
  if [[ "$INITIAL_RESET" == "1" ]]; then
    reset_args+=(--args.initial-reset)
  else
    reset_args+=(--args.no-initial-reset)
  fi
  if [[ "$FINAL_RESET" == "1" ]]; then
    reset_args+=(--args.final-reset)
  else
    reset_args+=(--args.no-final-reset)
  fi
  python -m examples.aloha_real.main \
    --args.host "$POLICY_URL" \
    --args.port 8000 \
    --args.action-horizon "$ACTION_HORIZON" \
    --args.control-hz "$CONTROL_HZ" \
    --args.num-episodes "$NUM_EPISODES" \
    --args.max-episode-steps "$MAX_EPISODE_STEPS" \
    "${reset_args[@]}" \
    2>&1 | tee "$RUN_LOG"
}

main() {
  validate_config "$@"
  mkdir -p "$LOG_DIR"
  if [[ "$RESET_ONLY" != "1" ]]; then
    preflight_policy_server || {
      echo "[hint] policy preflight failed; inspect ${LOG_DIR}/policy_preflight.log" >&2
      exit 1
    }
  fi
  preflight_env
  if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
    echo "[preflight] ok"
    exit 0
  fi
  if [[ "$RESET_ONLY" == "1" ]]; then
    run_reset_only
  else
    run_inference
  fi
}

main "$@"

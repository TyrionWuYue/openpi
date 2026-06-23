#!/usr/bin/env bash
set -Eeuo pipefail

# ROS1 launcher for the official OpenPI ALOHA real-runtime on AgileX.
# The policy loop is examples.aloha_real.main; only examples/aloha_real/real_env.py
# and robot_utils.py contain AgileX-specific hardware adaptation.

TEAM9_DIR="${TEAM9_DIR:-/home/agilex/team9}"
PIPER_WS="${PIPER_WS:-/home/agilex/cobot_magic/Piper_ros_private-ros-noetic}"
FRONT_CAMERA_DIR="${FRONT_CAMERA_DIR:-/home/agilex/cobot_magic/collect_data}"
CONDA_SH="${CONDA_SH:-/home/agilex/miniconda3/etc/profile.d/conda.sh}"

DEFAULT_POLICY_URL="https://nat-notebook-inspire.sii.edu.cn/ws-6040202d-b785-4b37-98b0-c68d65dd52ce/project-aa6664a6-c21b-426f-b8b9-f3d84953ccff/user-acc4c516-9dde-4134-b1f4-bcc86baad8f6/vscode/3bcf2884-8d7e-4feb-8145-964349be5580/3d51b084-c174-4d41-afed-4a30e650473d/proxy/8000/"
POLICY_URL="${POLICY_URL:-${1:-$DEFAULT_POLICY_URL}}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${RUN_LOG:-/tmp/openpi_aloha_real_${RUN_ID}.log}"

FRONT_TOPIC="${FRONT_TOPIC:-/camera_f/color/image_raw}"
LEFT_TOPIC="${LEFT_TOPIC:-/camera_l/color/image_raw}"
RIGHT_TOPIC="${RIGHT_TOPIC:-/camera_r/color/image_raw}"

PIPER_MODE="${PIPER_MODE:-1}"
PIPER_AUTO_ENABLE="${PIPER_AUTO_ENABLE:-true}"
RESTART_PIPER="${RESTART_PIPER:-1}"
DISABLE_ON_EXIT="${DISABLE_ON_EXIT:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
RESET_ONLY="${RESET_ONLY:-0}"

ACTION_HORIZON="${ACTION_HORIZON:-25}"
NUM_EPISODES="${NUM_EPISODES:-1}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-1000}"
if [[ "$RESET_ONLY" == "1" ]]; then
  NUM_EPISODES=0
fi

ROSCORE_LOG="${ROSCORE_LOG:-/tmp/openpi_roscore_${RUN_ID}.log}"
PIPER_LOG="${PIPER_LOG:-/tmp/openpi_piper_${RUN_ID}.log}"
FRONT_CAMERA_LOG="${FRONT_CAMERA_LOG:-/tmp/openpi_front_camera_${RUN_ID}.log}"

usage() {
  cat <<EOF
Usage:
  POLICY_URL='https://.../proxy/8000/' ./run_aloha_inference.sh
  PREFLIGHT_ONLY=1 POLICY_URL='https://.../proxy/8000/' ./run_aloha_inference.sh
  RESET_ONLY=1 POLICY_URL='https://.../proxy/8000/' ./run_aloha_inference.sh

Main env:
  ACTION_HORIZON=${ACTION_HORIZON}
  MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS}
  FRONT_TOPIC=${FRONT_TOPIC}
  LEFT_TOPIC=${LEFT_TOPIC}
  RIGHT_TOPIC=${RIGHT_TOPIC}

Camera topics must already be published before running this script.
EOF
}

source_ros() {
  set +u
  source /opt/ros/noetic/setup.bash
  [[ -f "${PIPER_WS}/devel/setup.bash" ]] && source "${PIPER_WS}/devel/setup.bash"
  set -u
}

conda_env() {
  set +u
  source "$CONDA_SH"
  conda activate "$1"
  set -u
}

normalize_policy_url() {
  case "$POLICY_URL" in
    */proxy) POLICY_URL="${POLICY_URL}/8000/" ;;
    */proxy/) POLICY_URL="${POLICY_URL}8000/" ;;
    */proxy/8000) POLICY_URL="${POLICY_URL}/" ;;
  esac
}

topic_exists() {
  source_ros
  rostopic list 2>/dev/null | grep -Fxq "$1"
}

topic_alive() {
  timeout 5 bash -lc "source /opt/ros/noetic/setup.bash; rostopic echo -n1 '$1' >/dev/null" >/dev/null 2>&1
}

wait_topic() {
  echo "[wait] $1"
  timeout 45 bash -lc "source /opt/ros/noetic/setup.bash; rostopic echo -n1 '$1' >/dev/null"
}

wait_subscriber() {
  local topic="$1"
  echo "[wait-sub] ${topic}"
  for _ in $(seq 1 30); do
    if timeout 5 bash -lc "source /opt/ros/noetic/setup.bash; rostopic info '$topic' 2>/dev/null | awk '/Subscribers:/{seen=1; next} seen && /\\*/{found=1} END{exit !found}'"; then
      return 0
    fi
    sleep 1
  done
  rostopic info "$topic" || true
  return 1
}

kill_nodes_matching() {
  source_ros
  rosnode list 2>/dev/null | grep -E "$1" | xargs -r rosnode kill >/dev/null 2>&1 || true
}

start_roscore_if_needed() {
  source_ros
  if rosnode list >/dev/null 2>&1; then
    return
  fi
  echo "[start] roscore -> ${ROSCORE_LOG}"
  bash -lc "source /opt/ros/noetic/setup.bash; exec roscore" >"$ROSCORE_LOG" 2>&1 &
  for _ in $(seq 1 30); do
    rosnode list >/dev/null 2>&1 && return
    sleep 1
  done
  echo "[error] roscore did not become ready" >&2
  exit 1
}

start_piper_if_needed() {
  if [[ "$RESTART_PIPER" != "1" ]] && topic_exists /puppet/joint_left && topic_exists /puppet/joint_right; then
    return
  fi
  echo "[start] piper mode=${PIPER_MODE} auto_enable=${PIPER_AUTO_ENABLE} -> ${PIPER_LOG}"
  kill_nodes_matching "piper_.*agilex"
  bash -lc "
    source /opt/ros/noetic/setup.bash
    source '${CONDA_SH}'
    conda activate aloha
    cd '${PIPER_WS}'
    bash ./can_config.sh
    source devel/setup.bash
    exec roslaunch piper start_ms_piper.launch mode:='${PIPER_MODE}' auto_enable:='${PIPER_AUTO_ENABLE}'
  " >"$PIPER_LOG" 2>&1 &
}

start_front_camera_if_needed() {
  if topic_alive "$FRONT_TOPIC"; then
    return
  fi
  echo "[start] front camera -> ${FRONT_CAMERA_LOG}"
  kill_nodes_matching "front_camera_node"
  bash -lc "
    source /opt/ros/noetic/setup.bash
    source '${CONDA_SH}'
    conda activate rm_aloha
    cd '${FRONT_CAMERA_DIR}'
    exec python -u front_cam_node.py
  " >"$FRONT_CAMERA_LOG" 2>&1 &
}

cleanup_robot() {
  if [[ "$DISABLE_ON_EXIT" == "1" ]]; then
    source_ros
    rostopic pub -1 /enable_flag std_msgs/Bool "data: false" >/dev/null 2>&1 || true
  fi
}

preflight_policy_server() {
  echo "[preflight] policy server"
  cd "$TEAM9_DIR"
  source_ros
  conda_env aloha
  POLICY_URL="$POLICY_URL" python - <<'PY'
import os
from openpi_client import websocket_client_policy

client = websocket_client_policy.WebsocketClientPolicy(host=os.environ["POLICY_URL"], port=8000)
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
  if [[ -z "$POLICY_URL" ]]; then
    usage
    exit 2
  fi
  if [[ "$LEFT_TOPIC" == "$RIGHT_TOPIC" ]]; then
    echo "[error] left/right wrist topics are identical: $LEFT_TOPIC" >&2
    exit 2
  fi
  if [[ "$FRONT_TOPIC" == "$LEFT_TOPIC" || "$FRONT_TOPIC" == "$RIGHT_TOPIC" ]]; then
    echo "[error] front camera topic must differ from wrist topics: $FRONT_TOPIC" >&2
    exit 2
  fi
  normalize_policy_url
}

main() {
  validate_config "$@"
  trap cleanup_robot EXIT
  trap 'trap - EXIT; cleanup_robot; exit 130' INT TERM

  start_roscore_if_needed
  start_piper_if_needed
  start_front_camera_if_needed

  wait_topic /puppet/joint_left
  wait_topic /puppet/joint_right
  wait_topic "$FRONT_TOPIC"
  wait_topic "$LEFT_TOPIC"
  wait_topic "$RIGHT_TOPIC"

  if [[ "$PIPER_MODE" == "1" ]]; then
    wait_subscriber /master/joint_left
    wait_subscriber /master/joint_right
    wait_subscriber /enable_flag
  fi

  preflight_policy_server
  if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
    echo "[preflight] ok"
    exit 0
  fi

  rostopic pub -1 /enable_flag std_msgs/Bool "data: true" >/dev/null

  echo "[run] log=${RUN_LOG}"
  cd "$TEAM9_DIR"
  python -m examples.aloha_real.main \
    --host "$POLICY_URL" \
    --port 8000 \
    --action-horizon "$ACTION_HORIZON" \
    --num-episodes "$NUM_EPISODES" \
    --max-episode-steps "$MAX_EPISODE_STEPS" \
    2>&1 | tee "$RUN_LOG"
}

main "$@"

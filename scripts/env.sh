#!/usr/bin/env bash
set -Eeuo pipefail

# Hardware/ROS environment setup for AgileX ALOHA.
# Run this after boot, camera re-plug, arm power-cycle, or any stale ROS state.

PIPER_WS="${PIPER_WS:-/home/agilex/cobot_magic/Piper_ros_private-ros-noetic}"
CAMERA_DIR="${CAMERA_DIR:-/home/agilex/cobot_magic/collect_data}"
CONDA_SH="${CONDA_SH:-/home/agilex/miniconda3/etc/profile.d/conda.sh}"

LEFT_DEV="${LEFT_DEV:-/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._Dabai_DC1_CC15C4300CX-video-index0}"
RIGHT_DEV="${RIGHT_DEV:-/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._Dabai_DC1_CC15C430057-video-index0}"

FRONT_TOPIC="${FRONT_TOPIC:-/camera_f/color/image_raw}"
LEFT_TOPIC="${LEFT_TOPIC:-/camera_l/color/image_raw}"
RIGHT_TOPIC="${RIGHT_TOPIC:-/camera_r/color/image_raw}"

RESTART_WRIST_CAMERAS="${RESTART_WRIST_CAMERAS:-1}"
RESTART_PIPER="${RESTART_PIPER:-1}"
PIPER_AUTO_ENABLE="${PIPER_AUTO_ENABLE:-true}"
PIPER_STARTUP_WAIT_SEC="${PIPER_STARTUP_WAIT_SEC:-6}"
DEVICE_WAIT_SEC="${DEVICE_WAIT_SEC:-12}"
WRIST_CAMERA_START_WAIT_SEC="${WRIST_CAMERA_START_WAIT_SEC:-8}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-/tmp/openpi_env_${RUN_ID}}"

usage() {
  cat <<EOF
Usage:
  ./env.sh
  RESTART_WRIST_CAMERAS=0 ./env.sh
  RESTART_PIPER=0 ./env.sh

Main env:
  LEFT_DEV=${LEFT_DEV}
  RIGHT_DEV=${RIGHT_DEV}
  FRONT_TOPIC=${FRONT_TOPIC}
  LEFT_TOPIC=${LEFT_TOPIC}
  RIGHT_TOPIC=${RIGHT_TOPIC}
  RESTART_WRIST_CAMERAS=${RESTART_WRIST_CAMERAS}
  RESTART_PIPER=${RESTART_PIPER}
  PIPER_AUTO_ENABLE=${PIPER_AUTO_ENABLE}
  PIPER_STARTUP_WAIT_SEC=${PIPER_STARTUP_WAIT_SEC}
  DEVICE_WAIT_SEC=${DEVICE_WAIT_SEC}
  WRIST_CAMERA_START_WAIT_SEC=${WRIST_CAMERA_START_WAIT_SEC}
EOF
}

source_ros() {
  set +u
  source /opt/ros/noetic/setup.bash
  [[ -f "${PIPER_WS}/devel/setup.bash" ]] && source "${PIPER_WS}/devel/setup.bash"
  set -u
}

ensure_root() {
  if [[ "${EUID}" -eq 0 ]]; then
    return
  fi
  local script_path
  script_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/$(basename -- "${BASH_SOURCE[0]}")"
  echo "[sudo] restarting as root for CAN, cameras, and Piper access"
  exec sudo -E "$script_path" "$@"
}

start_roscore() {
  source_ros
  if timeout 2 rostopic list >/dev/null 2>&1; then
    echo "[ok] roscore"
    return
  fi
  echo "[start] roscore"
  nohup bash -lc "source /opt/ros/noetic/setup.bash; exec roscore" >"${LOG_DIR}/roscore.log" 2>&1 &
  for _ in $(seq 1 20); do
    timeout 2 rostopic list >/dev/null 2>&1 && return
    sleep 1
  done
  echo "[error] roscore did not become reachable" >&2
  return 1
}

configure_can() {
  echo "[config] CAN"
  bash -lc "cd '${PIPER_WS}' && bash ./can_config.sh" >"${LOG_DIR}/can_config.log" 2>&1
}

wait_device() {
  local label="$1"
  local dev="$2"
  local waited=0
  while [[ "$waited" -lt "$DEVICE_WAIT_SEC" ]]; do
    if [[ -e "$dev" ]]; then
      echo "[ok] ${label} ${dev}"
      return
    fi
    sleep 1
    waited=$((waited + 1))
  done
  echo "[error] ${label} device is missing: ${dev}" >&2
  echo "[hint] check USB cable/hub/power; current /dev/v4l/by-id:" >&2
  ls -l /dev/v4l/by-id >&2 || true
  return 1
}

topic_has_msg() {
  local topic="$1"
  timeout 5 bash -lc "source /opt/ros/noetic/setup.bash; rostopic echo -n1 '$topic' >/dev/null" >/dev/null 2>&1
}

start_camera_if_needed() {
  local topic="$1"
  local log="$2"
  shift 2
  if topic_has_msg "$topic"; then
    echo "[ok] ${topic}"
    return
  fi
  echo "[start] ${topic}"
  nohup "$@" >"$log" 2>&1 &
}

start_camera() {
  local topic="$1"
  local log="$2"
  shift 2
  echo "[start] ${topic}"
  nohup "$@" >"$log" 2>&1 &
}

wait_topic_or_show_log() {
  local label="$1"
  local topic="$2"
  local log="$3"
  local waited=0
  while [[ "$waited" -lt "$WRIST_CAMERA_START_WAIT_SEC" ]]; do
    if topic_has_msg "$topic"; then
      echo "[ok] ${label} ${topic}"
      return
    fi
    sleep 1
    waited=$((waited + 1))
  done
  echo "[error] ${label} did not publish ${topic}" >&2
  echo "[hint] tail of ${log}:" >&2
  tail -n 120 "$log" >&2 || true
  echo "[hint] current /dev/v4l/by-id:" >&2
  ls -l /dev/v4l/by-id >&2 || true
  return 1
}

stop_wrist_camera_nodes() {
  source_ros
  for _ in $(seq 1 5); do
    rosnode kill /camera_l/camera /camera_r/camera >/dev/null 2>&1 || true
    yes | rosnode cleanup >/dev/null 2>&1 || true
    if ! rosnode list 2>/dev/null | grep -Eq '^/camera_[lr]/camera$'; then
      return
    fi
    sleep 1
  done
  echo "[error] stale wrist camera ROS nodes are still registered:" >&2
  rosnode list 2>/dev/null | grep -E '^/camera_[lr]/camera$' >&2 || true
  return 1
}

restart_wrist_cameras() {
  if [[ "$RESTART_WRIST_CAMERAS" != "1" ]]; then
    return
  fi
  echo "[restart] wrist cameras"
  stop_wrist_camera_nodes
  pkill -f '[a]stra_camera' >/dev/null 2>&1 || true
  pkill -f '[o]pencv_camera_pub.py' >/dev/null 2>&1 || true
  sleep 2
}

start_cameras() {
  restart_wrist_cameras

  wait_device "left wrist camera" "$LEFT_DEV"
  wait_device "right wrist camera" "$RIGHT_DEV"

  v4l2-ctl --device="$LEFT_DEV" --set-fmt-video=width=640,height=480,pixelformat=MJPG --set-parm=30 >/dev/null 2>&1 || true
  v4l2-ctl --device="$RIGHT_DEV" --set-fmt-video=width=640,height=480,pixelformat=MJPG --set-parm=30 >/dev/null 2>&1 || true

  start_camera "$LEFT_TOPIC" "${LOG_DIR}/camera_left.log" \
    bash -lc "source /opt/ros/noetic/setup.bash; cd '${CAMERA_DIR}'; exec /usr/bin/python3 opencv_camera_pub.py --device '${LEFT_DEV}' --topic '${LEFT_TOPIC}' --frame_id camera_l_color_link --fps 30"
  start_camera "$RIGHT_TOPIC" "${LOG_DIR}/camera_right.log" \
    bash -lc "source /opt/ros/noetic/setup.bash; cd '${CAMERA_DIR}'; exec /usr/bin/python3 opencv_camera_pub.py --device '${RIGHT_DEV}' --topic '${RIGHT_TOPIC}' --frame_id camera_r_color_link --fps 30"
  wait_topic_or_show_log "left wrist camera" "$LEFT_TOPIC" "${LOG_DIR}/camera_left.log"
  wait_topic_or_show_log "right wrist camera" "$RIGHT_TOPIC" "${LOG_DIR}/camera_right.log"

  start_camera_if_needed "$FRONT_TOPIC" "${LOG_DIR}/camera_front.log" \
    bash -lc "source /opt/ros/noetic/setup.bash; source '${CONDA_SH}'; conda activate rm_aloha; cd '${CAMERA_DIR}'; exec python -u front_cam_node.py"
}

restart_piper_replay() {
  if [[ "$RESTART_PIPER" != "1" ]]; then
    return
  fi
  echo "[restart] Piper replay mode"
  source_ros
  rosnode list 2>/dev/null | awk '/^\/piper_/ {print}' | xargs -r rosnode kill >/dev/null 2>&1 || true
  pkill -f '[r]oslaunch piper start_ms_piper.launch' >/dev/null 2>&1 || true
  pkill -f '[p]iper_start_ms_node.py' >/dev/null 2>&1 || true
  sleep 2
}

start_piper_replay() {
  restart_piper_replay
  if [[ "$RESTART_PIPER" != "1" ]] && topic_has_msg /puppet/joint_left && topic_has_msg /puppet/joint_right; then
    echo "[ok] Piper replay mode"
    return
  fi
  echo "[start] Piper replay mode auto_enable=${PIPER_AUTO_ENABLE}"
  nohup bash -lc "
    source /opt/ros/noetic/setup.bash
    source '${CONDA_SH}'
    conda activate aloha
    source '${PIPER_WS}/devel/setup.bash'
    cd '${PIPER_WS}'
    exec roslaunch piper start_ms_piper.launch mode:=1 auto_enable:='${PIPER_AUTO_ENABLE}'
  " >"${LOG_DIR}/piper_replay.log" 2>&1 &
  sleep "${PIPER_STARTUP_WAIT_SEC}"
}

need_msg() {
  local topic="$1"
  echo "[check] ${topic}"
  if ! timeout 15 bash -lc "source /opt/ros/noetic/setup.bash; rostopic echo -n1 '$topic' >/dev/null"; then
    echo "[error] no message from ${topic}" >&2
    return 1
  fi
}

need_subscriber() {
  local topic="$1"
  echo "[check-sub] ${topic}"
  timeout 10 bash -lc "source /opt/ros/noetic/setup.bash; rostopic info '$topic' 2>/dev/null | awk '/Subscribers:/{seen=1; next} seen && /\\*/{found=1} END{exit !found}'"
}

preflight_env() {
  need_msg "$LEFT_TOPIC"
  need_msg "$RIGHT_TOPIC"
  need_msg "$FRONT_TOPIC"
  need_msg /puppet/joint_left
  need_msg /puppet/joint_right
  need_subscriber /master/joint_left
  need_subscriber /master/joint_right
  need_subscriber /enable_flag
}

main() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
  fi
  ensure_root "$@"
  mkdir -p "$LOG_DIR"
  start_roscore
  configure_can
  start_cameras
  start_piper_replay
  preflight_env || {
    echo "[hint] inspect logs in ${LOG_DIR}" >&2
    tail -n 120 "${LOG_DIR}/camera_left.log" >&2 || true
    tail -n 120 "${LOG_DIR}/camera_right.log" >&2 || true
    tail -n 120 "${LOG_DIR}/camera_front.log" >&2 || true
    tail -n 120 "${LOG_DIR}/piper_replay.log" >&2 || true
    exit 1
  }
  echo "[env] ready; logs=${LOG_DIR}"
}

main "$@"

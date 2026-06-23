# Ignore lint errors because this file is mostly copied from ACT (https://github.com/tonyzhaozh/act).
# ruff: noqa
import collections
import logging
import time
from typing import Optional, List

import dm_env
import numpy as np

from examples.aloha_real import constants
from examples.aloha_real import robot_utils


_LOGGER = logging.getLogger(__name__)

DEFAULT_RESET_POSITION = [0, -0.96, 1.16, 0, -0.3, 0]
DEFAULT_RESET_GRIPPER_POSITION = 0.2

AGILEX_RESET_POSITION_LEFT = [
    -0.00133514404296875,
    0.00209808349609375,
    0.01583099365234375,
    -0.032616615295410156,
    -0.00286102294921875,
    0.00095367431640625,
    3.557830810546875,
]
AGILEX_RESET_POSITION_RIGHT = [
    -0.00133514404296875,
    0.00438690185546875,
    0.034523963928222656,
    -0.053597450256347656,
    -0.00476837158203125,
    -0.00209808349609375,
    3.557830810546875,
]


def _build_reset_targets(reset_position, *, use_server_reset_pose=False):
    if reset_position is None or not use_server_reset_pose:
        return list(AGILEX_RESET_POSITION_LEFT), list(AGILEX_RESET_POSITION_RIGHT)

    values = [float(x) for x in reset_position]
    if len(values) == 14:
        return values[:7], values[7:]
    if len(values) == 7:
        return list(values), list(values)
    if len(values) == 6:
        target = values + [DEFAULT_RESET_GRIPPER_POSITION]
        return list(target), list(target)
    raise ValueError(f"reset_position must have 6, 7, or 14 values, got {len(values)}")


def _round_values(values):
    if values is None:
        return None
    return [round(float(x), 4) for x in values]


def _agilex_gripper_to_policy(value):
    return float(value)


class RealEnv:
    """
    AgileX ROS1-backed ALOHA real environment.

    Action/state order follows the official ALOHA contract:
    [left_arm_qpos(6), left_gripper(1), right_arm_qpos(6), right_gripper(1)].
    Arm actions are absolute joint targets. Gripper values stay in the AgileX/Piper opening range.
    """

    def __init__(self, init_node, *, reset_position: Optional[List[float]] = None, setup_robots: bool = True):
        self.args = robot_utils.get_arguments()
        self._reset_position_left, self._reset_position_right = _build_reset_targets(
            reset_position,
            use_server_reset_pose=self.args.use_server_reset_pose,
        )
        self._policy_step = 0
        self._ts = None
        _LOGGER.info("Creating RosOperator")
        self.ros_operator = robot_utils.RosOperator(self.args, init_node=init_node)
        _LOGGER.info("RosOperator ready")
        print("[openpi] RosOperator ready", flush=True)

        if setup_robots:
            self.setup_robots()

    def setup_robots(self):
        return 0

    def build_image_dict(self, img_front: np.ndarray, img_left: np.ndarray, img_right: np.ndarray) -> dict:
        return {
            "cam_high": img_front,
            "cam_high_depth": None,
            "cam_left_wrist": img_left,
            "cam_left_wrist_depth": None,
            "cam_right_wrist": img_right,
            "cam_right_wrist_depth": None,
        }

    def get_observation(self):
        img_front, img_left, img_right, puppet_arm_left, puppet_arm_right = robot_utils.get_ros_observation(
            self.args, self.ros_operator
        )

        qpos = np.concatenate((np.asarray(puppet_arm_left.position), np.asarray(puppet_arm_right.position)), axis=0)
        qvel = np.concatenate((np.asarray(puppet_arm_left.velocity), np.asarray(puppet_arm_right.velocity)), axis=0)
        effort = np.concatenate((np.asarray(puppet_arm_left.effort), np.asarray(puppet_arm_right.effort)), axis=0)

        raw_left_gripper = float(qpos[6])
        raw_right_gripper = float(qpos[13])
        qpos[6] = _agilex_gripper_to_policy(raw_left_gripper)
        qpos[13] = _agilex_gripper_to_policy(raw_right_gripper)

        if self._policy_step < self.args.policy_action_log_steps:
            print(
                "[openpi] Observation gripper "
                f"raw_left={raw_left_gripper:.5f} policy_left={qpos[6]:.4f} "
                f"raw_right={raw_right_gripper:.5f} policy_right={qpos[13]:.4f}",
                flush=True,
            )

        obs = collections.OrderedDict()
        obs["qpos"] = qpos
        obs["qvel"] = qvel
        obs["effort"] = effort
        obs["images"] = self.build_image_dict(img_front, img_left, img_right)
        return obs

    def get_reward(self):
        return 0

    def reset(self, *, fake: bool = False):
        _LOGGER.info("RealEnv reset started fake=%s", fake)
        if not fake:
            _LOGGER.info("Publishing reset targets")
            print("[openpi] Reset: publishing target pose", flush=True)
            self.ros_operator.puppet_arm_publish_continuous(
                self._reset_position_left,
                self._reset_position_right,
                wait_for_feedback=self.args.require_reset_feedback,
                warn_if_feedback_far=not self.args.require_reset_feedback,
            )
            _LOGGER.info("Reset target publish completed")
            print("[openpi] Reset: target pose published", flush=True)
        _LOGGER.info("Waiting for reset observation")
        print("[openpi] Reset: waiting for synchronized observation", flush=True)
        self._policy_step = 0
        self._ts = dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST, reward=self.get_reward(), discount=None, observation=self.get_observation()
        )
        return self._ts

    def _safe_policy_targets(self, left_target, right_target):
        mode = self.args.policy_action_mode
        if mode not in {"clamp", "hold", "raw"}:
            raise RuntimeError(f"OPENPI_POLICY_ACTION_MODE must be clamp, hold, or raw; got {mode!r}")

        raw_left = left_target.copy()
        raw_right = right_target.copy()
        current_ts = getattr(self, "_ts", None)
        current = None if current_ts is None else np.asarray(current_ts.observation["qpos"], dtype=float)
        changed = False

        if current is not None and current.shape[0] >= 14:
            current_left = current[:7].copy()
            current_right = current[7:14].copy()
            current_left[6] = np.clip(current_left[6], self.args.agilex_gripper_min, self.args.agilex_gripper_max)
            current_right[6] = np.clip(current_right[6], self.args.agilex_gripper_min, self.args.agilex_gripper_max)
        else:
            current_left = None
            current_right = None

        if mode == "hold":
            if current_left is None or current_right is None:
                raise RuntimeError("Cannot hold policy action without a current qpos observation")
            left_target = current_left
            right_target = current_right
            changed = True
        elif mode == "clamp":
            left_target[6] = np.clip(left_target[6], self.args.agilex_gripper_min, self.args.agilex_gripper_max)
            right_target[6] = np.clip(right_target[6], self.args.agilex_gripper_min, self.args.agilex_gripper_max)
            if current_left is not None and current_right is not None:
                arm_delta = self.args.max_policy_arm_delta
                gripper_delta = self.args.max_policy_gripper_delta
                safe_left = left_target.copy()
                safe_right = right_target.copy()
                safe_left[:6] = current_left[:6] + np.clip(left_target[:6] - current_left[:6], -arm_delta, arm_delta)
                safe_right[:6] = current_right[:6] + np.clip(
                    right_target[:6] - current_right[:6], -arm_delta, arm_delta
                )
                safe_left[6] = current_left[6] + np.clip(left_target[6] - current_left[6], -gripper_delta, gripper_delta)
                safe_right[6] = current_right[6] + np.clip(
                    right_target[6] - current_right[6], -gripper_delta, gripper_delta
                )
                safe_left[6] = np.clip(safe_left[6], self.args.agilex_gripper_min, self.args.agilex_gripper_max)
                safe_right[6] = np.clip(safe_right[6], self.args.agilex_gripper_min, self.args.agilex_gripper_max)
                changed = bool(
                    not np.allclose(safe_left, left_target, atol=1e-6)
                    or not np.allclose(safe_right, right_target, atol=1e-6)
                )
                left_target = safe_left
                right_target = safe_right

        if self._policy_step < self.args.policy_action_log_steps or changed:
            print(
                "[openpi] Policy action "
                f"step={self._policy_step} mode={mode} changed={changed} "
                f"current_left={_round_values(current_left)} raw_left={_round_values(raw_left)} "
                f"cmd_left={_round_values(left_target)} current_right={_round_values(current_right)} "
                f"raw_right={_round_values(raw_right)} cmd_right={_round_values(right_target)}",
                flush=True,
            )

        return left_target, right_target

    def step(self, action):
        state_len = len(action) // 2
        left_target = np.asarray(action[:state_len], dtype=float).copy()
        right_target = np.asarray(action[state_len:], dtype=float).copy()
        left_target, right_target = self._safe_policy_targets(left_target, right_target)
        self._policy_step += 1

        _LOGGER.debug("Publishing policy action")
        self.ros_operator.puppet_arm_publish_policy(left_target.tolist(), right_target.tolist())

        time.sleep(constants.DT)
        self._ts = dm_env.TimeStep(
            step_type=dm_env.StepType.MID, reward=self.get_reward(), discount=None, observation=self.get_observation()
        )
        return self._ts


def get_action(master_bot_left, master_bot_right):
    action = np.zeros(14)
    action[:6] = master_bot_left.dxl.joint_states.position[:6]
    action[7 : 7 + 6] = master_bot_right.dxl.joint_states.position[:6]
    action[6] = constants.MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot_left.dxl.joint_states.position[6])
    action[13] = constants.MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot_right.dxl.joint_states.position[6])
    return action


def make_real_env(init_node, *, reset_position: Optional[List[float]] = None, setup_robots: bool = True) -> RealEnv:
    return RealEnv(init_node, reset_position=reset_position, setup_robots=setup_robots)

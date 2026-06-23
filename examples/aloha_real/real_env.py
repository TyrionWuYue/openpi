# Ignore lint errors because this file is mostly copied from ACT (https://github.com/tonyzhaozh/act).
# ruff: noqa
import collections
import time
from typing import Optional, List

import dm_env
import numpy as np

from examples.aloha_real import constants
from examples.aloha_real import robot_utils


DEFAULT_RESET_POSITION = [0, -0.96, 1.16, 0, -0.3, 0]

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


class RealEnv:
    """
    AgileX ROS1-backed ALOHA real environment.

    Action/state order follows the official ALOHA contract:
    [left_arm_qpos(6), left_gripper_norm(1), right_arm_qpos(6), right_gripper_norm(1)].
    Arm actions are absolute joint targets. Gripper actions are normalized ALOHA values.
    """

    def __init__(self, init_node, *, reset_position: Optional[List[float]] = None, setup_robots: bool = True):
        self._reset_position = reset_position[:6] if reset_position else DEFAULT_RESET_POSITION
        self._reset_position_left = AGILEX_RESET_POSITION_LEFT
        self._reset_position_right = AGILEX_RESET_POSITION_RIGHT

        self.args = robot_utils.get_arguments()
        self.ros_operator = robot_utils.RosOperator(self.args, init_node=init_node)

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

        qpos[6] = constants.PUPPET_GRIPPER_POSITION_NORMALIZE_FN(qpos[6])
        qpos[13] = constants.PUPPET_GRIPPER_POSITION_NORMALIZE_FN(qpos[13])
        qvel[6] = constants.PUPPET_GRIPPER_POSITION_NORMALIZE_FN(qvel[6])
        qvel[13] = constants.PUPPET_GRIPPER_POSITION_NORMALIZE_FN(qvel[13])

        obs = collections.OrderedDict()
        obs["qpos"] = qpos
        obs["qvel"] = qvel
        obs["effort"] = effort
        obs["images"] = self.build_image_dict(img_front, img_left, img_right)
        return obs

    def get_reward(self):
        return 0

    def reset(self, *, fake: bool = False):
        if not fake:
            self.ros_operator.puppet_arm_publish_continuous(self._reset_position_left, self._reset_position_right)
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST, reward=self.get_reward(), discount=None, observation=self.get_observation()
        )

    def step(self, action):
        state_len = len(action) // 2
        left_target = np.asarray(action[:state_len], dtype=float).copy()
        right_target = np.asarray(action[state_len:], dtype=float).copy()

        left_target[-1] = constants.PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(left_target[-1])
        right_target[-1] = constants.PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(right_target[-1])

        self.ros_operator.puppet_arm_publish_continuous(left_target.tolist(), right_target.tolist())

        time.sleep(constants.DT)
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID, reward=self.get_reward(), discount=None, observation=self.get_observation()
        )


def get_action(master_bot_left, master_bot_right):
    action = np.zeros(14)
    action[:6] = master_bot_left.dxl.joint_states.position[:6]
    action[7 : 7 + 6] = master_bot_right.dxl.joint_states.position[:6]
    action[6] = constants.MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot_left.dxl.joint_states.position[6])
    action[13] = constants.MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot_right.dxl.joint_states.position[6])
    return action


def make_real_env(init_node, *, reset_position: Optional[List[float]] = None, setup_robots: bool = True) -> RealEnv:
    return RealEnv(init_node, reset_position=reset_position, setup_robots=setup_robots)

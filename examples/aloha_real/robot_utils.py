# Ignore lint errors because this file is mostly copied from ACT (https://github.com/tonyzhaozh/act).
# ruff: noqa
from collections import deque
from types import SimpleNamespace

from cv_bridge import CvBridge
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header


def get_ros_observation(args, ros_operator):
    rate = rospy.Rate(args.publish_rate)
    logged_wait = False

    while not rospy.is_shutdown():
        result = ros_operator.get_frame()
        if result:
            return result[0], result[1], result[2], result[6], result[7]
        if not logged_wait:
            rospy.loginfo("Waiting for synchronized AgileX camera/joint observation")
            logged_wait = True
        rate.sleep()

    raise RuntimeError("ROS shutdown while waiting for observation")


class RosOperator:
    def __init__(self, args, *, init_node=True):
        self.args = args
        self.bridge = CvBridge()

        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.img_front_deque = deque()
        self.img_left_depth_deque = deque()
        self.img_right_depth_deque = deque()
        self.img_front_depth_deque = deque()
        self.puppet_arm_left_deque = deque()
        self.puppet_arm_right_deque = deque()
        self.robot_base_deque = deque()

        self.init_ros(init_node=init_node)

    def puppet_arm_publish(self, left, right):
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = rospy.Time.now()
        joint_state_msg.name = ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        joint_state_msg.position = left
        self.puppet_arm_left_publisher.publish(joint_state_msg)
        joint_state_msg.position = right
        self.puppet_arm_right_publisher.publish(joint_state_msg)

    def robot_base_publish(self, vel):
        vel_msg = Twist()
        vel_msg.linear.x = vel[0]
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = vel[1]
        self.robot_base_publisher.publish(vel_msg)

    def puppet_arm_publish_continuous(self, left, right):
        rate = rospy.Rate(self.args.publish_rate)
        left_arm = None
        right_arm = None
        while not rospy.is_shutdown():
            if self.puppet_arm_left_deque:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if self.puppet_arm_right_deque:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is not None and right_arm is not None:
                break
            rate.sleep()

        if left_arm is None or right_arm is None:
            raise RuntimeError("No AgileX joint feedback available for continuous publish")

        left = list(left)
        right = list(right)
        left_symbol = [1 if left[i] - left_arm[i] > 0 else -1 for i in range(len(left))]
        right_symbol = [1 if right[i] - right_arm[i] > 0 else -1 for i in range(len(right))]

        running = True
        while running and not rospy.is_shutdown():
            left_diff = [abs(left[i] - left_arm[i]) for i in range(len(left))]
            right_diff = [abs(right[i] - right_arm[i]) for i in range(len(right))]
            running = False

            for i in range(len(left)):
                if left_diff[i] < self.args.arm_steps_length[i]:
                    left_arm[i] = left[i]
                else:
                    left_arm[i] += left_symbol[i] * self.args.arm_steps_length[i]
                    running = True
            for i in range(len(right)):
                if right_diff[i] < self.args.arm_steps_length[i]:
                    right_arm[i] = right[i]
                else:
                    right_arm[i] += right_symbol[i] * self.args.arm_steps_length[i]
                    running = True

            self.puppet_arm_publish(left_arm, right_arm)
            rate.sleep()

    def get_frame(self):
        if len(self.img_left_deque) == 0 or len(self.img_right_deque) == 0 or len(self.img_front_deque) == 0:
            return False
        if self.args.use_depth_image and (
            len(self.img_left_depth_deque) == 0
            or len(self.img_right_depth_deque) == 0
            or len(self.img_front_depth_deque) == 0
        ):
            return False

        if self.args.use_depth_image:
            frame_time = min(
                [
                    self.img_left_deque[-1].header.stamp.to_sec(),
                    self.img_right_deque[-1].header.stamp.to_sec(),
                    self.img_front_deque[-1].header.stamp.to_sec(),
                    self.img_left_depth_deque[-1].header.stamp.to_sec(),
                    self.img_right_depth_deque[-1].header.stamp.to_sec(),
                    self.img_front_depth_deque[-1].header.stamp.to_sec(),
                ]
            )
        else:
            frame_time = min(
                [
                    self.img_left_deque[-1].header.stamp.to_sec(),
                    self.img_right_deque[-1].header.stamp.to_sec(),
                    self.img_front_deque[-1].header.stamp.to_sec(),
                ]
            )

        if self.img_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if self.img_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if self.img_front_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_left_deque) == 0 or self.puppet_arm_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_right_deque) == 0 or self.puppet_arm_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if self.args.use_robot_base and (
            len(self.robot_base_deque) == 0 or self.robot_base_deque[-1].header.stamp.to_sec() < frame_time
        ):
            return False

        img_left = self._pop_image(self.img_left_deque, frame_time)
        img_right = self._pop_image(self.img_right_deque, frame_time)
        img_front = self._pop_image(self.img_front_deque, frame_time)
        puppet_arm_left = self._pop_msg(self.puppet_arm_left_deque, frame_time)
        puppet_arm_right = self._pop_msg(self.puppet_arm_right_deque, frame_time)

        img_left_depth = None
        img_right_depth = None
        img_front_depth = None
        if self.args.use_depth_image:
            img_left_depth = self._pop_image(self.img_left_depth_deque, frame_time)
            img_right_depth = self._pop_image(self.img_right_depth_deque, frame_time)
            img_front_depth = self._pop_image(self.img_front_depth_deque, frame_time)

        robot_base = None
        if self.args.use_robot_base:
            robot_base = self._pop_msg(self.robot_base_deque, frame_time)

        return (
            img_front,
            img_left,
            img_right,
            img_front_depth,
            img_left_depth,
            img_right_depth,
            puppet_arm_left,
            puppet_arm_right,
            robot_base,
        )

    def _pop_msg(self, queue, frame_time):
        while len(queue) > 1 and queue[0].header.stamp.to_sec() < frame_time:
            queue.popleft()
        return queue.popleft()

    def _pop_image(self, queue, frame_time):
        return self.bridge.imgmsg_to_cv2(self._pop_msg(queue, frame_time), "passthrough")

    def _bounded_append(self, queue, msg):
        if len(queue) >= self.args.queue_size:
            queue.popleft()
        queue.append(msg)

    def img_left_callback(self, msg):
        self._bounded_append(self.img_left_deque, msg)

    def img_right_callback(self, msg):
        self._bounded_append(self.img_right_deque, msg)

    def img_front_callback(self, msg):
        self._bounded_append(self.img_front_deque, msg)

    def img_left_depth_callback(self, msg):
        self._bounded_append(self.img_left_depth_deque, msg)

    def img_right_depth_callback(self, msg):
        self._bounded_append(self.img_right_depth_deque, msg)

    def img_front_depth_callback(self, msg):
        self._bounded_append(self.img_front_depth_deque, msg)

    def puppet_arm_left_callback(self, msg):
        self._bounded_append(self.puppet_arm_left_deque, msg)

    def puppet_arm_right_callback(self, msg):
        self._bounded_append(self.puppet_arm_right_deque, msg)

    def robot_base_callback(self, msg):
        self._bounded_append(self.robot_base_deque, msg)

    def init_ros(self, *, init_node=True):
        if init_node:
            rospy.init_node("openpi_aloha_agilex_ros_operator", anonymous=True)
        rospy.Subscriber(self.args.img_left_topic, Image, self.img_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_right_topic, Image, self.img_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_front_topic, Image, self.img_front_callback, queue_size=1000, tcp_nodelay=True)
        if self.args.use_depth_image:
            rospy.Subscriber(
                self.args.img_left_depth_topic, Image, self.img_left_depth_callback, queue_size=1000, tcp_nodelay=True
            )
            rospy.Subscriber(
                self.args.img_right_depth_topic, Image, self.img_right_depth_callback, queue_size=1000, tcp_nodelay=True
            )
            rospy.Subscriber(
                self.args.img_front_depth_topic, Image, self.img_front_depth_callback, queue_size=1000, tcp_nodelay=True
            )
        rospy.Subscriber(
            self.args.puppet_arm_left_topic, JointState, self.puppet_arm_left_callback, queue_size=1000, tcp_nodelay=True
        )
        rospy.Subscriber(
            self.args.puppet_arm_right_topic, JointState, self.puppet_arm_right_callback, queue_size=1000, tcp_nodelay=True
        )
        rospy.Subscriber(self.args.robot_base_topic, Odometry, self.robot_base_callback, queue_size=1000, tcp_nodelay=True)
        self.puppet_arm_left_publisher = rospy.Publisher(self.args.puppet_arm_left_cmd_topic, JointState, queue_size=10)
        self.puppet_arm_right_publisher = rospy.Publisher(
            self.args.puppet_arm_right_cmd_topic, JointState, queue_size=10
        )
        self.robot_base_publisher = rospy.Publisher(self.args.robot_base_cmd_topic, Twist, queue_size=10)


def get_arguments():
    args = SimpleNamespace()

    args.img_front_topic = "/camera_f/color/image_raw"
    args.img_left_topic = "/camera_l/color/image_raw"
    args.img_right_topic = "/camera_r/color/image_raw"

    args.img_front_depth_topic = "/camera_f/depth/image_raw"
    args.img_left_depth_topic = "/camera_l/depth/image_raw"
    args.img_right_depth_topic = "/camera_r/depth/image_raw"

    args.puppet_arm_left_cmd_topic = "/master/joint_left"
    args.puppet_arm_right_cmd_topic = "/master/joint_right"
    args.puppet_arm_left_topic = "/puppet/joint_left"
    args.puppet_arm_right_topic = "/puppet/joint_right"

    args.robot_base_topic = "/odom_raw"
    args.robot_base_cmd_topic = "/cmd_vel"
    args.use_robot_base = False
    args.publish_rate = 30
    args.ctrl_freq = 25
    args.queue_size = 2000
    args.arm_steps_length = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2]
    args.use_depth_image = False

    return args

"""
Copyright MIT
GNU General Public License v3.0

MIT BWSI Autonomous Drone Racing Course - UAV Neo

File Name: camera_real.py
File Description: Contains the Camera module of the drone_core library
"""

from camera import Camera

# General
import numpy as np
class NDArray:  # stub — no runtime dependency on nptyping
    def __class_getitem__(cls, _): return cls

# ROS2
import rclpy as ros2
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSProfile,
)
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class CameraReal(Camera):
    # ROS topics matching the teleop.launch.py relay names
    __COLOR_TOPIC = "/camera/forward"
    __DEPTH_TOPIC = "/camera/depth"
    __DOWNWARD_TOPIC = "/camera/nadir"

    def __init__(self):
        self.__bridge = CvBridge()

        # ROS node
        self.node = ros2.create_node("image_sub")

        qos_profile = QoSProfile(depth=10)
        qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT
        qos_profile.durability = QoSDurabilityPolicy.VOLATILE

        # subscribe to the color image topic
        self.__color_image_sub = self.node.create_subscription(
            Image, self.__COLOR_TOPIC, self.__color_callback, qos_profile
        )
        self.__color_image = None
        self.__color_image_new = None

        # subscribe to the depth image topic
        self.__depth_image_sub = self.node.create_subscription(
            Image, self.__DEPTH_TOPIC, self.__depth_callback, qos_profile
        )
        self.__depth_image = None
        self.__depth_image_new = None

        # subscribe to the downward (nadir) camera image topic
        self.__downward_sub = self.node.create_subscription(
            Image, self.__DOWNWARD_TOPIC, self.__downward_callback, qos_profile
        )
        self.__downward_image = None
        self.__downward_image_new = None

    def __color_callback(self, data):
        try:
            cv_color_image = self.__bridge.imgmsg_to_cv2(
                data, desired_encoding="bgr8"
            )
        except CvBridgeError as e:
            print(e)
            return

        self.__color_image_new = cv_color_image

    def __depth_callback(self, data):
        try:
            # RealSense publishes 16UC1 in millimeters; convert to cm (float32)
            cv_depth_image = self.__bridge.imgmsg_to_cv2(
                data, desired_encoding="16UC1"
            )
        except CvBridgeError as e:
            print(e)
            return

        self.__depth_image_new = cv_depth_image.astype(np.float32) / 10.0

    def __downward_callback(self, data):
        try:
            cv_downward_image = self.__bridge.imgmsg_to_cv2(
                data, desired_encoding="bgr8"
            )
        except CvBridgeError as e:
            print(e)
            return

        self.__downward_image_new = cv_downward_image

    def __update(self):
        self.__depth_image = self.__depth_image_new
        self.__color_image = self.__color_image_new
        self.__downward_image = self.__downward_image_new

    def get_color_image_no_copy(self) -> NDArray[(480, 640, 3), np.uint8]:
        return self.__color_image

    def get_depth_image(self) -> NDArray[(480, 640), np.float32]:
        return self.__depth_image

    def get_downward_image(self) -> NDArray[(480, 640, 3), np.uint8]:
        return self.__downward_image

    def get_color_image_async(self) -> NDArray[(480, 640, 3), np.uint8]:
        return self.__color_image_new

    def get_depth_image_async(self) -> NDArray[(480, 640), np.float32]:
        return self.__depth_image_new

    def get_downward_image_async(self) -> NDArray[(480, 640, 3), np.uint8]:
        return self.__downward_image_new

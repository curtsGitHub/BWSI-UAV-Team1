"""
Copyright MIT
GNU General Public License v3.0

MIT BWSI Autonomous Drone Racing Course - UAV Neo

File Name: physics_real.py
File Description: Contains the Physics module of the drone_core library
"""

from physics import Physics

# General
from collections import deque
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
from sensor_msgs.msg import Imu, NavSatFix
from geometry_msgs.msg import PoseStamped, TwistStamped


class PhysicsReal(Physics):
    # MAVROS IMU (flight controller — orientation, angular velocity, acceleration)
    __MAVROS_IMU_TOPIC = "/mavros/imu/data"
    # RealSense IMU (camera — high-rate accelerometer and gyroscope)
    __REALSENSE_IMU_TOPIC = "/camera/imu"
    # GPS, velocity, and local position (relayed by teleop.launch.py)
    __NAV_TOPIC = "/nav"
    __VELOCITY_TOPIC = "/velocity"
    __POSE_TOPIC = "/position"

    # Limit on buffer size to prevent memory overflow
    __BUFFER_CAP = 60

    def __init__(self):
        self.node = ros2.create_node("imu_sub")

        qos_profile = QoSProfile(depth=1)
        qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT
        qos_profile.durability = QoSDurabilityPolicy.VOLATILE

        # MAVROS IMU (primary — Pixhawk EKF-fused orientation, accel, gyro)
        # The Pixhawk's internal EKF already fuses its own sensors and provides
        # data in a consistent body frame. This is the authority for flight.
        self.__mavros_imu_sub = self.node.create_subscription(
            Imu, self.__MAVROS_IMU_TOPIC, self.__mavros_imu_callback, qos_profile
        )

        # RealSense IMU is NOT fused here — it uses a different coordinate frame
        # (camera optical: Y=down, Z=forward) vs MAVROS (ENU/NED body frame).
        # It is useful for VIO pipelines but not for direct student access.

        # GPS
        self.__nav_sub = self.node.create_subscription(
            NavSatFix, self.__NAV_TOPIC, self.__nav_callback, qos_profile
        )

        # Velocity
        self.__velocity_sub = self.node.create_subscription(
            TwistStamped, self.__VELOCITY_TOPIC, self.__velocity_callback, qos_profile
        )

        # Local position (EKF origin, set where the EKF initialized — not a fixed world origin)
        self.__pose_sub = self.node.create_subscription(
            PoseStamped, self.__POSE_TOPIC, self.__pose_callback, qos_profile
        )
        
        self.__acceleration = np.array([0.0, 0.0, 0.0])
        self.__acceleration_buffer = deque()
        self.__linear_velocity = np.array([0.0, 0.0, 0.0])
        self.__linear_velocity_buffer = deque()
        self.__angular_velocity = np.array([0.0, 0.0, 0.0])
        self.__angular_velocity_buffer = deque()
        self.__altitude = 0.0
        self.__altitude_buffer = deque()
        self.__attitude = np.array([0.0, 0.0, 0.0])
        self.__orientation_buffer = deque()
        self.__gps = np.array([0.0, 0.0, 0.0])
        self.__gps_buffer = deque()
        self.__position = np.array([0.0, 0.0, 0.0])
        self.__position_buffer = deque()
        self.__position_offset = 0.0

    def __mavros_imu_callback(self, data):
        """IMU callback — Pixhawk EKF data, remapped from body ENU to the library frame."""
        new_accel = _enu_vector_to_library(
            data.linear_acceleration.x,
            data.linear_acceleration.y,
            data.linear_acceleration.z,
        )

        self.__acceleration_buffer.append(new_accel)
        if len(self.__acceleration_buffer) > self.__BUFFER_CAP:
            self.__acceleration_buffer.popleft()

        new_gyro = _enu_rates_to_library(
            data.angular_velocity.x,
            data.angular_velocity.y,
            data.angular_velocity.z,
        )

        self.__angular_velocity_buffer.append(new_gyro)
        if len(self.__angular_velocity_buffer) > self.__BUFFER_CAP:
            self.__angular_velocity_buffer.popleft()

        # Buffer the raw quaternion: averaging Euler angles corrupts any heading that
        # straddles the 0/360 wrap, which is exactly where a north-facing drone sits.
        new_orientation = np.array([
            data.orientation.w, data.orientation.x,
            data.orientation.y, data.orientation.z,
        ])

        self.__orientation_buffer.append(new_orientation)
        if len(self.__orientation_buffer) > self.__BUFFER_CAP:
            self.__orientation_buffer.popleft()

    def __nav_callback(self, data):
        # GPS lat/lon/alt only; AGL now comes from the EKF pose (GPS has no indoor fix).
        new_gps = np.array([data.latitude, data.longitude, data.altitude])

        self.__gps_buffer.append(new_gps)
        if len(self.__gps_buffer) > self.__BUFFER_CAP:
            self.__gps_buffer.popleft()

    def __velocity_callback(self, data):
        new_linear_velocity = _enu_vector_to_library(
            data.twist.linear.x,
            data.twist.linear.y,
            data.twist.linear.z,
        )

        self.__linear_velocity_buffer.append(new_linear_velocity)
        if len(self.__linear_velocity_buffer) > self.__BUFFER_CAP:
            self.__linear_velocity_buffer.popleft()

    def __pose_callback(self, data):
        # Local ENU (east, north, up) -> the sim's world axes (x_east, y_up, z_north).
        new_position = np.array([
            data.pose.position.x,
            data.pose.position.z,
            data.pose.position.y,
        ])

        self.__position_buffer.append(new_position)
        if len(self.__position_buffer) > self.__BUFFER_CAP:
            self.__position_buffer.popleft()

        # AGL from the rangefinder-aided EKF vertical estimate, not GPS: ENU z is up.
        new_altitude = float(data.pose.position.z)

        print(f"New_altitude diff library:{new_altitude}")
        self.__altitude_buffer.append(new_altitude)
        if len(self.__altitude_buffer) > self.__BUFFER_CAP:
            self.__altitude_buffer.popleft()

    def __update(self):
        if len(self.__acceleration_buffer) > 0:
            self.__acceleration = np.mean(self.__acceleration_buffer, axis=0)
            self.__acceleration_buffer.clear()

        if len(self.__linear_velocity_buffer) > 0:
            self.__linear_velocity = np.mean(self.__linear_velocity_buffer, axis=0)
            self.__linear_velocity_buffer.clear()

        if len(self.__angular_velocity_buffer) > 0:
            self.__angular_velocity = np.mean(self.__angular_velocity_buffer, axis=0)
            self.__angular_velocity_buffer.clear()

        if len(self.__altitude_buffer) > 0:
            self.__altitude = float(np.mean(self.__altitude_buffer))
            self.__altitude_buffer.clear()

        if len(self.__orientation_buffer) > 0:
            self.__attitude = _average_orientation_to_attitude(self.__orientation_buffer)
            self.__orientation_buffer.clear()

        if len(self.__gps_buffer) > 0:
            self.__gps = np.mean(self.__gps_buffer, axis=0)
            self.__gps_buffer.clear()

        if len(self.__position_buffer) > 0:
            self.__position = np.mean(self.__position_buffer, axis=0)
            self.__position_buffer.clear()

    def get_linear_acceleration(self) -> NDArray[3, np.float32]:
        return np.array(self.__acceleration)

    def get_linear_velocity(self) -> NDArray[3, np.float32]:
        return np.array(self.__linear_velocity)

    def get_angular_velocity(self) -> NDArray[3, np.float32]:
        return np.array(self.__angular_velocity)

    def get_altitude(self) -> float:
        return float(self.__altitude)

    def get_attitude(self) -> NDArray[3, np.float32]:
        return np.array(self.__attitude)

    def get_gps(self) -> NDArray[3, np.float32]:
        return np.array(self.__gps)

    # def get_position(self) -> NDArray[3, np.float32]:
    #     return np.array(self.__position)

    def get_position(self) -> NDArray[3, np.float32]:
        raw = np.array(self.__position)
        if self.__position_offset is not None:
            return raw - self.__position_offset
        return raw
    
    def zero_position_here(self):
        """Call this once, with the drone sitting at your fixed marked spot,
        before takeoff. Everything after this is relative to that spot."""
        self.__update()  # flush any buffered readings first
        self.__position_offset = np.array(self.__position)
        print(f"Position zeroed. Offset captured: {self.__position_offset}")

    

# MAVROS reports body ENU (x=forward, y=left, z=up); the Physics contract is
# (x=right, y=up, z=forward). Everything below converts between the two.


def _enu_vector_to_library(x, y, z):
    """Body-ENU (forward, left, up) -> library (right, up, forward)."""
    return np.array([-y, z, x])


def _enu_rates_to_library(x, y, z):
    """Body-ENU rates (about forward, left, up) -> library (pitch, yaw, roll) rates,
    signed to match get_attitude: nose-up, clockwise, right-wing-down."""
    return np.array([-y, -z, x])


def quaternion_to_euler_angle_vectorized2(w, x, y, z):
    """Euler angles in degrees about the quaternion's own axes: (roll, pitch, yaw)."""
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = np.degrees(np.arctan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)

    t2 = np.clip(t2, a_min=-1.0, a_max=1.0)
    Y = np.degrees(np.arcsin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = np.degrees(np.arctan2(t3, t4))

    return [X, Y, Z]


def _average_orientation_to_attitude(orientations):
    """Mean of buffered ENU quaternions -> library (pitch, roll, yaw) in degrees.

    Signs are negated against a reference sample first: q and -q are the same rotation,
    so a raw component mean can cancel to garbage when the EKF flips sign mid-buffer.
    """
    quats = np.array(orientations)
    reference = quats[0]
    signs = np.where(quats @ reference < 0.0, -1.0, 1.0)
    mean = np.mean(quats * signs[:, None], axis=0)
    mean = mean / np.linalg.norm(mean)

    roll_enu, pitch_enu, yaw_enu = quaternion_to_euler_angle_vectorized2(*mean)

    # ENU yaw is CCW from east; the contract is CW from north over [0, 360). Python's %
    # rounds a hair-negative operand up to exactly 360.0, which lands on a north heading.
    yaw = (90.0 - yaw_enu) % 360.0
    if yaw >= 360.0:
        yaw = 0.0

    return np.array([
        -pitch_enu,  # ENU pitch is positive nose-down
        roll_enu,    # ENU roll is already positive right-wing-down
        yaw,
    ])

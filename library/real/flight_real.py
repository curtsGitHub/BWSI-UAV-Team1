"""
Copyright MIT
GNU General Public License v3.0

MIT BWSI Autonomous Drone Racing Course - UAV Neo

File Name: flight_real.py
File Description: Contains the Flight module of the drone_core library.
Publishes setpoints directly to MAVROS. Velocity commands (send_pcmd / takeoff)
go to /mavros/setpoint_velocity/cmd_vel; position commands (goto_position) go to
/mavros/setpoint_position/local for PX4's position controller. Only one setpoint
type streams at a time. Does NOT arm or enter OFFBOARD - the safety pilot does
that on the RC transmitter and retains override by leaving OFFBOARD.
"""

from flight import Flight

import rclpy as ros2
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.srv import SetMode


# The mux node used to apply the speed cap. With the autonomy path going straight
# to MAVROS, the library enforces it. These must match neo_lab.REAL_MAX_SPEED and
# config/mux.yaml so a normalized command maps to the same m/s as under teleop.
MAX_SPEED = 2.0        # m/s at a full normalized linear command
MAX_YAW_RATE = 2.5     # rad/s at a full normalized yaw command

# Normalized vertical commands for the takeoff/land velocity setpoints.
TAKEOFF_THROTTLE = 0.5
LAND_THROTTLE = -0.3

# Absolute local setpoint frame; mav_frame in config/mavros_px4.yaml is LOCAL_NED,
# so position targets are anchored to the EKF origin, not offsets from the pose.
SETPOINT_FRAME_ID = "map"

# PX4 mode for autonomous landing, commanded through the set_mode service. Verify
# the service name against `ros2 service list` on the Pi if landing does nothing.
LAND_MODE = "AUTO.LAND"

# 20 Hz; PX4 drops OFFBOARD if setpoints stop for longer than COM_OF_LOSS_T.
_PUBLISH_PERIOD_S = 0.05

_MODE_VELOCITY = "velocity"
_MODE_POSITION = "position"
_MODE_LAND = "land"


class FlightReal(Flight):
    __VELOCITY_TOPIC = "/mavros/setpoint_velocity/cmd_vel"
    __POSITION_TOPIC = "/mavros/setpoint_position/local"
    __SET_MODE_SERVICE = "/mavros/set_mode"

    def __init__(self):
        self.__node = ros2.create_node("flight_pub")
        self.__clock = self.__node.get_clock()

        self.__velocity_pub = self.__node.create_publisher(
            TwistStamped, self.__VELOCITY_TOPIC, qos_profile=10
        )
        self.__position_pub = self.__node.create_publisher(
            PoseStamped, self.__POSITION_TOPIC, qos_profile=10
        )
        self.__set_mode = self.__node.create_client(
            SetMode, self.__SET_MODE_SERVICE
        )

        self.__twist = TwistStamped()
        self.__pose = PoseStamped()
        self.__pose.header.frame_id = SETPOINT_FRAME_ID
        # Identity orientation holds a fixed heading; goto_position does not yaw.
        self.__pose.pose.orientation.w = 1.0

        self.__mode = _MODE_VELOCITY
        self.__land_requested = False

        self.__node.create_timer(_PUBLISH_PERIOD_S, self.__update)

    @property
    def node(self):
        return self.__node

    def send_pcmd(self, pitch: float, roll: float, yaw: float, throttle: float) -> None:
        assert (
            -1.0 <= pitch <= 1.0
        ), f"pitch [{pitch}] must be between -1.0 and 1.0 inclusive."
        assert (
            -1.0 <= roll <= 1.0
        ), f"roll [{roll}] must be between -1.0 and 1.0 inclusive."
        assert (
            -1.0 <= yaw <= 1.0
        ), f"yaw [{yaw}] must be between -1.0 and 1.0 inclusive."
        assert (
            -1.0 <= throttle <= 1.0
        ), f"throttle [{throttle}] must be between -1.0 and 1.0 inclusive."

        # roll/yaw are negated: the MAVROS body-velocity frame is ENU (linear.y
        # left-positive, angular.z CCW-positive) while send_pcmd is documented
        # right-positive and clockwise-positive.
        self.__mode = _MODE_VELOCITY
        self.__twist.twist.linear.x = float(pitch) * MAX_SPEED
        self.__twist.twist.linear.y = float(-roll) * MAX_SPEED
        self.__twist.twist.linear.z = float(throttle) * MAX_SPEED
        self.__twist.twist.angular.z = float(-yaw) * MAX_YAW_RATE

    def takeoff(self) -> None:
        """Send ascending velocity setpoints. The safety pilot must arm and switch
        to OFFBOARD before the drone will actually lift off."""
        self.__mode = _MODE_VELOCITY
        self.__twist.twist.linear.x = 0.0
        self.__twist.twist.linear.y = 0.0
        self.__twist.twist.linear.z = TAKEOFF_THROTTLE * MAX_SPEED
        self.__twist.twist.angular.z = 0.0

    def land(self) -> None:
        """Command PX4's autonomous landing once, then let the flight controller
        own the descent. The safety pilot can override by leaving OFFBOARD."""
        if self.__land_requested:
            return
        self.__land_requested = True

        if self.__set_mode.service_is_ready():
            request = SetMode.Request()
            request.custom_mode = LAND_MODE
            self.__set_mode.call_async(request)
            self.__mode = _MODE_LAND
        else:
            # set_mode not available: fall back to a gentle descent setpoint and
            # leave the actual landing to the safety pilot.
            self.__node.get_logger().warn(
                "set_mode unavailable; descending with a velocity setpoint"
            )
            self.__mode = _MODE_VELOCITY
            self.__twist.twist.linear.x = 0.0
            self.__twist.twist.linear.y = 0.0
            self.__twist.twist.linear.z = LAND_THROTTLE * MAX_SPEED
            self.__twist.twist.angular.z = 0.0

    def goto_position(self, east: float, up: float, north: float) -> None:
        self.__mode = _MODE_POSITION
        self.__pose.pose.position.x = float(east)
        self.__pose.pose.position.y = float(north)
        self.__pose.pose.position.z = float(up)

    def set_max_speed(self, max_speed: float = 0.25) -> None:
        """No-op. The speed cap is the MAX_SPEED constant enforced in this module."""
        pass

    def __update(self):
        try:
            if self.__mode == _MODE_POSITION:
                self.__pose.header.stamp = self.__clock.now().to_msg()
                self.__position_pub.publish(self.__pose)
            elif self.__mode == _MODE_VELOCITY:
                self.__twist.header.stamp = self.__clock.now().to_msg()
                self.__velocity_pub.publish(self.__twist)
            # _MODE_LAND: PX4 owns the descent; stream nothing.
        except Exception as e:
            self.__node.get_logger().error(f'Flight publish failed: {e}')

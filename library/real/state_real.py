"""
Copyright MIT
GNU General Public License v3.0

MIT BWSI Autonomous Drone Racing Course - UAV Neo

File Name: state_real.py
File Description: Contains the State module of the drone_core library.
Subscribes to /mavros/state and /mavros/extended_state for vehicle status.
"""

from state import State

# ROS2
import rclpy as ros2
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSProfile,
)
from mavros_msgs.msg import State as MavrosState, ExtendedState


class StateReal(State):
    __STATE_TOPIC = "/mavros/state"
    __EXTENDED_STATE_TOPIC = "/mavros/extended_state"

    def __init__(self):
        self.node = ros2.create_node("state_sub")

        qos_profile = QoSProfile(depth=1)
        qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT
        qos_profile.durability = QoSDurabilityPolicy.VOLATILE

        self.__state_sub = self.node.create_subscription(
            MavrosState, self.__STATE_TOPIC,
            self.__state_callback, qos_profile,
        )

        self.__extended_state_sub = self.node.create_subscription(
            ExtendedState, self.__EXTENDED_STATE_TOPIC,
            self.__extended_state_callback, qos_profile,
        )

        self.__connected = False
        self.__armed = False
        self.__mode = ""
        self.__landed_state = State.LandedState.UNDEFINED

    def __state_callback(self, msg):
        self.__connected = msg.connected
        self.__armed = msg.armed
        self.__mode = msg.mode

    def __extended_state_callback(self, msg):
        # MAV_LANDED_STATE enum: 0=undefined, 1=on_ground, 2=in_air, 3=takeoff, 4=landing
        try:
            self.__landed_state = State.LandedState(msg.landed_state)
        except ValueError:
            self.__landed_state = State.LandedState.UNDEFINED

    def __update(self):
        pass

    def is_connected(self) -> bool:
        return self.__connected

    def is_armed(self) -> bool:
        return self.__armed

    def get_mode(self) -> str:
        return self.__mode

    def get_landed_state(self) -> State.LandedState:
        return self.__landed_state

    def is_offboard(self) -> bool:
        return self.__mode == "OFFBOARD"

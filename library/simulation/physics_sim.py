"""
Copyright MIT
GNU General Public License v3.0

MIT BWSI Autonomous Drone Racing Course - UAV Neo

File Name: physics_sim.py
File Description: Physics simulation module — queries sensor data from the UAVNeo Simulator via UDP.
"""

import struct
import numpy as np
class NDArray:  # stub — no runtime dependency on nptyping
    def __class_getitem__(cls, _): return cls

from physics import Physics


class PhysicsSim(Physics):
    def __init__(self, drone) -> None:
        self.__drone = drone

    def get_linear_acceleration(self) -> NDArray[3, np.float32]:
        self.__drone._DroneSim__send_header(
            self.__drone.Header.physics_get_linear_acceleration
        )
        values = struct.unpack("fff", self.__drone._DroneSim__receive_data(12))
        return np.array(values, dtype=np.float32)

    def get_linear_velocity(self) -> NDArray[3, np.float32]:
        self.__drone._DroneSim__send_header(
            self.__drone.Header.physics_get_linear_velocity
        )
        values = struct.unpack("fff", self.__drone._DroneSim__receive_data(12))
        return np.array(values, dtype=np.float32)

    def get_angular_velocity(self) -> NDArray[3, np.float32]:
        self.__drone._DroneSim__send_header(
            self.__drone.Header.physics_get_angular_velocity
        )
        values = struct.unpack("fff", self.__drone._DroneSim__receive_data(12))
        return np.array(values, dtype=np.float32)

    def get_altitude(self) -> float:
        self.__drone._DroneSim__send_header(
            self.__drone.Header.physics_get_altitude
        )
        [value] = struct.unpack("f", self.__drone._DroneSim__receive_data(4))
        return value

    def get_attitude(self) -> NDArray[3, np.float32]:
        self.__drone._DroneSim__send_header(
            self.__drone.Header.physics_get_attitude
        )
        values = struct.unpack("fff", self.__drone._DroneSim__receive_data(12))
        return np.array(values, dtype=np.float32)

    def get_gps(self) -> NDArray[3, np.float32]:
        self.__drone._DroneSim__send_header(
            self.__drone.Header.physics_get_gps
        )
        values = struct.unpack("fff", self.__drone._DroneSim__receive_data(12))
        return np.array(values, dtype=np.float32)

    def get_position(self) -> NDArray[3, np.float32]:
        self.__drone._DroneSim__send_header(
            self.__drone.Header.physics_get_position
        )
        values = struct.unpack("fff", self.__drone._DroneSim__receive_data(12))
        return np.array(values, dtype=np.float32)

"""
Copyright MIT
GNU General Public License v3.0

MIT BWSI Autonomous Drone Racing Course - UAV Neo

File Name: flight_sim.py
File Description: Flight simulation module — sends flight commands to the UAVNeo Simulator via UDP.
"""

import math
import struct

from flight import Flight


# PD position controller for goto_position (same structure as the velocity-labs'
# waypoint solution): P on horizontal position error, D on velocity for damping,
# P on altitude. Tilting to move sheds vertical lift, so the altitude gain must be
# stiff enough to hold height while translating.
_GOTO_KP_POS = 0.25         # tilt per meter of horizontal position error
_GOTO_KD_POS = 0.5          # tilt per (m/s) of body velocity (damping)
_GOTO_ALT_KP = 0.45         # throttle per meter of altitude error
_GOTO_TILT_LIMIT = 0.3      # horizontal tilt cap
_GOTO_THROTTLE_LIMIT = 0.5


def _clamp(value, limit):
    return max(-limit, min(limit, value))


class FlightSim(Flight):
    def __init__(self, drone) -> None:
        self.__drone = drone

    def send_pcmd(
        self, pitch: float, roll: float, yaw: float, throttle: float
    ) -> None:
        assert -1.0 <= pitch <= 1.0, f"pitch [{pitch}] must be between -1.0 and 1.0 inclusive."
        assert -1.0 <= roll <= 1.0, f"roll [{roll}] must be between -1.0 and 1.0 inclusive."
        assert -1.0 <= yaw <= 1.0, f"yaw [{yaw}] must be between -1.0 and 1.0 inclusive."
        assert -1.0 <= throttle <= 1.0, f"throttle [{throttle}] must be between -1.0 and 1.0 inclusive."

        self.__drone._DroneSim__send_data(
            struct.pack(
                "Bffff",
                self.__drone.Header.flight_send_pcmd.value,
                pitch, roll, yaw, throttle,
            )
        )

    def set_max_speed(self, max_speed: float = 0.25) -> None:
        assert (
            0.0 <= max_speed <= 1.0
        ), f"max_speed [{max_speed}] must be between 0.0 and 1.0 inclusive."

        self.__drone._DroneSim__send_data(
            struct.pack(
                "Bf",
                self.__drone.Header.flight_set_max_speed.value,
                max_speed,
            )
        )

    def goto_position(self, east: float, up: float, north: float) -> None:
        pos = self.__drone.physics.get_position()      # (east, up, north)
        vel = self.__drone.physics.get_linear_velocity()  # (right, up, forward)
        yaw = self.__drone.physics.get_attitude()[2]   # clockwise from north, deg

        err_east = east - float(pos[0])
        err_up = up - float(pos[1])
        err_north = north - float(pos[2])

        # Rotate the world horizontal error into the body frame. Heading is
        # clockwise from north: forward = (sin, cos) in (east, north), right is
        # 90 deg clockwise of forward.
        psi = math.radians(yaw)
        err_forward = err_east * math.sin(psi) + err_north * math.cos(psi)
        err_right = err_east * math.cos(psi) - err_north * math.sin(psi)

        pitch = _clamp(_GOTO_KP_POS * err_forward - _GOTO_KD_POS * float(vel[2]),
                       _GOTO_TILT_LIMIT)
        roll = _clamp(_GOTO_KP_POS * err_right - _GOTO_KD_POS * float(vel[0]),
                      _GOTO_TILT_LIMIT)
        throttle = _clamp(_GOTO_ALT_KP * err_up, _GOTO_THROTTLE_LIMIT)
        self.send_pcmd(pitch, roll, 0, throttle)

    def takeoff(self) -> bool:
        self.__drone._DroneSim__send_header(
            self.__drone.Header.flight_takeoff
        )
        data = self.__drone._DroneSim__receive_data(1)
        return bool(data[0])

    def land(self) -> bool:
        self.__drone._DroneSim__send_header(
            self.__drone.Header.flight_land
        )
        data = self.__drone._DroneSim__receive_data(1)
        return bool(data[0])

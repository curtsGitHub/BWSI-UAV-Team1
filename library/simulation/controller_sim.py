"""
Copyright MIT
GNU General Public License v3.0

MIT BWSI Autonomous Drone Racing Course - UAV Neo

File Name: controller_sim.py
File Description: Controller simulation module — queries controller input from the UAVNeo Simulator via UDP.
"""

import sys
import struct
from enum import IntEnum

from controller import Controller


class ControllerSim(Controller):
    def __init__(self, drone) -> None:
        self.__drone = drone
        self.__is_down_cache: dict[Controller.Button, bool] = {}
        self.__was_pressed_cache: dict[Controller.Button, bool] = {}
        self.__was_released_cache: dict[Controller.Button, bool] = {}
        self.__get_trigger_cache: dict[Controller.Trigger, float] = {}
        self.__get_joystick_cache: dict[Controller.Joystick, tuple[float, float]] = {}

    def is_down(self, button: Controller.Button) -> bool:
        if button.value not in self.__is_down_cache:
            self.__drone._DroneSim__send_data(
                struct.pack(
                    "BB", self.__drone.Header.controller_is_down.value, button.value
                )
            )
            self.__is_down_cache[button.value] = bool(
                int.from_bytes(
                    self.__drone._DroneSim__receive_data(), sys.byteorder
                )
            )
        return self.__is_down_cache[button.value]

    def was_pressed(self, button: Controller.Button) -> bool:
        if button.value not in self.__was_pressed_cache:
            self.__drone._DroneSim__send_data(
                struct.pack(
                    "BB",
                    self.__drone.Header.controller_was_pressed.value,
                    button.value,
                )
            )
            self.__was_pressed_cache[button.value] = bool(
                int.from_bytes(
                    self.__drone._DroneSim__receive_data(), sys.byteorder
                )
            )
        return self.__was_pressed_cache[button.value]

    def was_released(self, button: Controller.Button) -> bool:
        if button.value not in self.__was_released_cache:
            self.__drone._DroneSim__send_data(
                struct.pack(
                    "BB",
                    self.__drone.Header.controller_was_released.value,
                    button.value,
                )
            )
            self.__was_released_cache[button.value] = bool(
                int.from_bytes(
                    self.__drone._DroneSim__receive_data(), sys.byteorder
                )
            )
        return self.__was_released_cache[button.value]

    def get_trigger(self, trigger: Controller.Trigger) -> float:
        if trigger.value not in self.__get_trigger_cache:
            self.__drone._DroneSim__send_data(
                struct.pack(
                    "BB",
                    self.__drone.Header.controller_get_trigger.value,
                    trigger.value,
                )
            )
            [value] = struct.unpack("f", self.__drone._DroneSim__receive_data())
            self.__get_trigger_cache[trigger.value] = value
        return self.__get_trigger_cache[trigger.value]

    def get_joystick(self, joystick: Controller.Joystick) -> tuple[float, float]:
        if joystick.value not in self.__get_joystick_cache:
            self.__drone._DroneSim__send_data(
                struct.pack(
                    "BB",
                    self.__drone.Header.controller_get_joystick.value,
                    joystick.value,
                )
            )
            self.__get_joystick_cache[joystick.value] = struct.unpack(
                "ff", self.__drone._DroneSim__receive_data(8)
            )
        return self.__get_joystick_cache[joystick.value]

    def __update(self) -> None:
        self.__is_down_cache.clear()
        self.__was_pressed_cache.clear()
        self.__was_released_cache.clear()
        self.__get_trigger_cache.clear()
        self.__get_joystick_cache.clear()

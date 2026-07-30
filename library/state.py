"""
Copyright MIT
GNU General Public License v3.0

MIT BWSI Autonomous Drone Racing Course - UAV Neo

File Name: state.py
File Description: Defines the interface of the State module of the drone_core library.
"""

import abc
from enum import IntEnum


class State(abc.ABC):
    """
    Provides read-only access to the vehicle's current flight state.
    """

    class LandedState(IntEnum):
        """The vehicle's landed state."""
        UNDEFINED = 0
        ON_GROUND = 1
        IN_AIR = 2
        TAKEOFF = 3
        LANDING = 4

    @abc.abstractmethod
    def is_connected(self) -> bool:
        """
        Returns whether the flight controller is connected.

        Example::

            if uav.state.is_connected():
                print("Flight controller online")
        """
        pass

    @abc.abstractmethod
    def is_armed(self) -> bool:
        """
        Returns whether the motors are armed.

        Example::

            if uav.state.is_armed():
                print("Motors armed — stay clear of propellers")
        """
        pass

    @abc.abstractmethod
    def get_mode(self) -> str:
        """
        Returns the current flight mode as a string.

        Common modes: "MANUAL", "STABILIZED", "ALTCTL", "POSCTL",
        "OFFBOARD", "AUTO.LAND", "AUTO.RTL"

        Example::

            mode = uav.state.get_mode()
            if mode == "OFFBOARD":
                print("Pi has control")
        """
        pass

    @abc.abstractmethod
    def get_landed_state(self) -> 'State.LandedState':
        """
        Returns the vehicle's landed state.

        Returns:
            A LandedState enum value: UNDEFINED, ON_GROUND, IN_AIR,
            TAKEOFF, or LANDING.

        Example::

            if uav.state.get_landed_state() == uav.state.LandedState.IN_AIR:
                print("Drone is airborne")
        """
        pass

    @abc.abstractmethod
    def is_offboard(self) -> bool:
        """
        Returns whether the vehicle is in OFFBOARD mode (Pi has control).

        Example::

            if uav.state.is_offboard():
                print("Autonomous control active")
        """
        pass

"""
Copyright MIT
GNU General Public License v3.0

MIT BWSI Autonomous Drone Racing Course - UAV Neo

File Name: drone_core.py
File Description: Contains the Drone class, the top level of the drone_core library.
"""

import abc
import sys
from typing import Callable, Optional

import camera
import controller
import detector
import display
import flight
import led
import physics
import state
import telemetry

import drone_utils as uav_utils


class Drone(abc.ABC):
    """
    The top level drone module containing several submodules which interface
    with and control the different pieces of the drone hardware.
    """

    def __init__(self) -> None:
        self.camera: camera.Camera
        self.controller: controller.Controller
        self.detector: detector.Detector
        self.display: display.Display
        self.flight: flight.Flight
        self.led: led.Led
        self.physics: physics.Physics
        self.state: state.State
        self.telemetry: telemetry.Telemetry

    @abc.abstractmethod
    def go(self, autostart: bool = False) -> None:
        """
        Starts the drone, beginning in default mode.

        Args:
            autostart: If True, enter user program mode immediately instead of
                waiting for the START button. Lets a program run without a game
                controller (stop it with Ctrl-C). On the real drone the safety
                pilot's OFFBOARD switch is still the gate for any motion. Ignored
                by the simulator, which begins on its own ENTER-key signal.

        Note:
            go blocks execution until the program is exited.
        """
        pass

    @abc.abstractmethod
    def set_start_update(
        self,
        start: Callable[[], None],
        update: Callable[[], None],
        update_slow: Optional[Callable[[], None]] = None,
    ) -> None:
        """
        Sets the start and update functions used in user program mode.

        Args:
            start: A function called once when the drone enters user program mode.
            update: A function called every frame in user program mode.
            update_slow: A function called once per fixed time interval in user
                program mode (by default once per second).

        Example::

            drone = Drone()

            def start():
                print("This function is called once")

            def update():
                print("This function is called every frame")

            drone.set_start_update(start, update)
            drone.go()
        """
        pass

    @abc.abstractmethod
    def get_delta_time(self) -> float:
        """
        Returns the number of seconds elapsed in the previous frame.

        Returns:
            The number of seconds between the start of the previous frame and
            the start of the current frame.

        Example::

            counter += uav.get_delta_time()
        """
        pass

    @abc.abstractmethod
    def set_update_slow_time(self, time: float = 1.0) -> None:
        """
        Changes the time between calls to update_slow.

        Args:
            time: The time in seconds between calls to update_slow.

        Example::

            uav.set_update_slow_time(2)
        """
        pass


def create_drone(isSimulation: Optional[bool] = None) -> Drone:
    """
    Generates a drone object based on the isSimulation argument or execution flags.

    Args:
        isSimulation: If True, create a DroneSim. If None, decide based on
            the command line arguments.

    Returns:
        A DroneSim object for use with the Unity simulation.

    Note:
        If isSimulation is None, this function will return a DroneSim if the program
        was executed with the "-s" flag.

        If the program was executed with the "-d" flag, a display window is created.

        If the program was executed with the "-h" flag, it is run in headless mode,
        which disables the display module.
    """
    library_path: str = __file__.replace("drone_core.py", "")
    is_headless: bool = "-h" in sys.argv
    initialize_display: bool = "-d" in sys.argv

    if isSimulation is None:
        isSimulation = "-s" in sys.argv

    drone: Drone
    if isSimulation:
        sys.path.insert(1, library_path + "simulation")
        from drone_core_sim import DroneSim

        drone = DroneSim(is_headless)
    else:
        sys.path.insert(1, library_path + "real")
        from drone_core_real import DroneReal

        drone = DroneReal(is_headless)

    if initialize_display:
        drone.display.create_window()

    uav_utils.print_colored(
        ">> Drone created with the following options:"
        + f"\n    Simulation (-s): [{isSimulation}]"
        + f"\n    Headless (-h): [{is_headless}]"
        + f"\n    Initialize with display (-d): [{initialize_display}]",
        uav_utils.TerminalColor.pink,
    )

    return drone

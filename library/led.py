"""
Copyright MIT
GNU General Public License v3.0

MIT BWSI Autonomous Drone Racing Course - UAV Neo

File Name: led.py
File Description: Defines the interface of the LED module of the drone_core library.
"""

import abc


class Led(abc.ABC):
    """
    Controls the drone's addressable LED strip.

    On the real drone this lights a WS2812 strip; in the simulator it is a no-op,
    so the same code runs in both.
    """

    @abc.abstractmethod
    def fill(self, red: int, green: int, blue: int) -> None:
        """
        Lights the whole strip a single color.

        Args:
            red: Red channel, 0-255.
            green: Green channel, 0-255.
            blue: Blue channel, 0-255.

        Example::

            uav.led.fill(0, 200, 255)
        """
        pass

    def off(self) -> None:
        """
        Turns the strip off.

        Note:
            Equivalent to uav.led.fill(0, 0, 0).

        Example::

            uav.led.off()
        """
        self.fill(0, 0, 0)

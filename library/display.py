"""
Copyright MIT
GNU General Public License v3.0

MIT BWSI Autonomous Drone Racing Course - UAV Neo

File Name: display.py
File Description: Defines the interface of the Display module of the drone_core library.
"""

import abc
import numpy as np
from typing import Any
class NDArray:  # stub — no runtime dependency on nptyping
    def __class_getitem__(cls, _): return cls

import drone_utils as uav_utils


class Display(abc.ABC):
    """
    Allows the user to print images to the screen.
    """

    __BIG_DOT_RADIUS = 8
    __SMALL_DOT_RADIUS = 4

    def __init__(self, isHeadless: bool) -> None:
        self.__isHeadless = isHeadless

    @abc.abstractmethod
    def create_window(self) -> None:
        """
        Creates an empty window into which images will be displayed.
        """
        pass

    @abc.abstractmethod
    def show_color_image(self, image: NDArray) -> None:
        """
        Displays a color image in a window.

        Args:
            image: The color image to display to the screen.

        Example::

            image = uav.camera.get_color_image()
            uav.display.show_color_image(image)
        """
        pass

    def show_depth_image(
        self,
        image: NDArray[(Any, Any), np.float32],
        max_depth: int = 1000,
        points: list[tuple[int, int]] = [],
    ) -> None:
        """
        Displays a depth image in grayscale in a window.

        Args:
            image: The depth image to display to the screen.
            max_depth: The farthest depth to show in the image in cm.
            points: A list of points in (pixel row, pixel column) format to highlight.

        Example::

            depth_image = uav.camera.get_depth_image()
            uav.display.show_depth_image(depth_image)
        """
        if self.__isHeadless:
            return

        assert max_depth > 0, "max_depth must be positive."
        for point in points:
            assert (
                0 <= point[0] < image.shape[0] and 0 <= point[1] < image.shape[1]
            ), f"The point [{point}] is not a valid pixel row and column within image."

        color_image = uav_utils.colormap_depth_image(image, max_depth)

        for point in points:
            uav_utils.draw_circle(
                color_image,
                point,
                uav_utils.ColorBGR.green.value,
                radius=self.__BIG_DOT_RADIUS,
            )
            uav_utils.draw_circle(
                color_image,
                point,
                uav_utils.ColorBGR.blue.value,
                radius=self.__SMALL_DOT_RADIUS,
            )

        self.show_color_image(color_image)

    @abc.abstractmethod
    def set_matrix(self, matrix: NDArray[(8, 24), np.uint8]) -> None:
        """
        Sets the dot matrix display module to the pattern in the argument (2D matrix).

        Args:
            matrix: The 8x24 NumPy array with the pattern to be displayed.

        Example::

            dot_matrix = np.ones((8, 24), dtype=np.uint8)
            uav.display.set_matrix(dot_matrix)
        """
        pass

    @abc.abstractmethod
    def show_text(
        self,
        text: str,
        scroll_speed: float = 2.0
    ) -> None:
        """
        Displays text on the 8x24 matrix. If the text is too long, it scrolls.

        Args:
            text: The string to display.
            scroll_speed: The scrolling speed in characters per second.

        Example::

            uav.display.show_text("Hello, Drone!")
        """
        pass

    @abc.abstractmethod
    def get_matrix(self) -> NDArray[(8, 24), np.uint8]:
        """
        Returns the current configuration of the dot matrix display module.
        """
        pass

    def new_matrix(self) -> NDArray[(8, 24), np.uint8]:
        """
        Returns a new matrix of all zeroes for the dot matrix display module.
        """
        return np.zeros((8, 24), dtype=np.uint8)

    @abc.abstractmethod
    def set_matrix_intensity(self, intensity: float) -> None:
        """
        Sets the intensity of the dot matrix display module.

        Args:
            intensity: The LED intensity (between 0.0 and 1.0) to set.

        Example::

            uav.display.set_matrix_intensity(0.5)
        """
        pass

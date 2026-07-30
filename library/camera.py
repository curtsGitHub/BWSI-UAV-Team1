"""
Copyright MIT
GNU General Public License v3.0

MIT BWSI Autonomous Drone Racing Course - UAV Neo

File Name: camera.py
File Description: Defines the interface of the Camera module of the drone_core library.
"""

import abc
import copy
import numpy as np
class NDArray:  # stub — no runtime dependency on nptyping
    def __class_getitem__(cls, _): return cls


class Camera(abc.ABC):
    """
    Returns color images, depth images, and downward images captured by the drone's cameras.
    """

    # The dimensions of the image in pixels
    _WIDTH: int = 640
    _HEIGHT: int = 480

    # Maximum range of the depth camera (in cm)
    _MAX_RANGE = 1200

    def get_color_image(self) -> NDArray[(480, 640, 3), np.uint8]:
        """
        Returns a deep copy of the current forward color image captured by the camera.

        Returns:
            An array representing the pixels in the image, organized as follows
                0th dimension: pixel rows, indexed from top to bottom.
                1st dimension: pixel columns, indexed from left to right.
                2nd dimension: pixel color channels, in the blue-green-red format.

        Note:
            Each color value ranges from 0 to 255.

        Example::

            image = uav.camera.get_color_image()
            blue = image[3][5][0]
        """
        return copy.deepcopy(self.get_color_image_no_copy())

    @abc.abstractmethod
    def get_color_image_no_copy(self) -> NDArray[(480, 640, 3), np.uint8]:
        """
        Returns a direct reference to the forward color image captured by the camera.

        Returns:
            An array representing the pixels in the image.

        Warning:
            Do not modify the returned image.

        Example::

            image = uav.camera.get_color_image_no_copy()
        """
        pass

    @abc.abstractmethod
    def get_color_image_async(self) -> NDArray[(480, 640, 3), np.uint8]:
        """
        Returns the current forward color image without the drone in "go" mode.

        Warning:
            This function breaks the start-update paradigm and should only be used in
            Jupyter Notebook.
        """
        pass

    @abc.abstractmethod
    def get_depth_image(self) -> NDArray[(480, 640), np.float32]:
        """
        Returns the current depth image captured by the forward camera.

        Returns:
            A two-dimensional array storing the distance of each pixel from
            the drone in cm.

        Example::

            image = uav.camera.get_depth_image()
            distance = image[3][5]
        """
        pass

    @abc.abstractmethod
    def get_depth_image_async(self) -> NDArray[(480, 640), np.float32]:
        """
        Returns the current depth image without the drone in "go" mode.

        Warning:
            This function breaks the start-update paradigm and should only be used in
            Jupyter Notebook.
        """
        pass

    @abc.abstractmethod
    def get_downward_image(self) -> NDArray[(480, 640, 3), np.uint8]:
        """
        Returns the current image from the downward-facing RGB camera.

        Returns:
            An array representing the pixels in the image, organized as follows
                0th dimension: pixel rows, indexed from top to bottom.
                1st dimension: pixel columns, indexed from left to right.
                2nd dimension: pixel color channels, in the blue-green-red format.

        Example::

            down_image = uav.camera.get_downward_image()
        """
        pass

    @abc.abstractmethod
    def get_downward_image_async(self) -> NDArray[(480, 640, 3), np.uint8]:
        """
        Returns the current downward image without the drone in "go" mode.

        Warning:
            This function breaks the start-update paradigm and should only be used in
            Jupyter Notebook.
        """
        pass

    def get_width(self) -> int:
        """
        Returns the pixel width of the color and depth images.
        """
        return self._WIDTH

    def get_height(self) -> int:
        """
        Returns the pixel height of the color and depth images.
        """
        return self._HEIGHT

    def get_max_range(self) -> float:
        """
        Returns the maximum distance in cm which can be detected by the depth camera.
        """
        return self._MAX_RANGE

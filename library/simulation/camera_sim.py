"""
Copyright MIT
GNU General Public License v3.0

MIT BWSI Autonomous Drone Racing Course - UAV Neo

File Name: camera_sim.py
File Description: Camera simulation module — requests images from the UAVNeo Simulator via UDP.
"""

import numpy as np
import cv2 as cv
class NDArray:  # stub — no runtime dependency on nptyping
    def __class_getitem__(cls, _): return cls

from camera import Camera
import drone_utils as uav_utils


class CameraSim(Camera):
    def __init__(self, drone) -> None:
        self.__drone = drone
        self.__color_image: NDArray[(480, 640, 3), np.uint8] = None
        self.__is_color_image_current: bool = False
        self.__depth_image: NDArray[(480, 640), np.float32] = None
        self.__is_depth_image_current: bool = False
        self.__downward_image: NDArray[(480, 640, 3), np.uint8] = None
        self.__is_downward_image_current: bool = False

        self._MAX_DEPTH_WIDTH: int = self._WIDTH // 8
        self._MAX_DEPTH_HEIGHT: int = self._HEIGHT // 8

    def get_color_image_no_copy(self) -> NDArray[(480, 640, 3), np.uint8]:
        if not self.__is_color_image_current:
            self.__color_image = self.__request_color_image(False)
            self.__is_color_image_current = True
        return self.__color_image

    def get_color_image_async(self) -> NDArray[(480, 640, 3), np.uint8]:
        return self.__request_color_image(True)

    def get_depth_image(self) -> NDArray[(480, 640), np.float32]:
        if not self.__is_depth_image_current:
            self.__depth_image = self.__request_depth_image(False)
            self.__is_depth_image_current = True
        return self.__depth_image

    def get_depth_image_async(self) -> NDArray[(480, 640), np.float32]:
        return self.__request_depth_image(True)

    def get_downward_image(self) -> NDArray[(480, 640, 3), np.uint8]:
        if not self.__is_downward_image_current:
            self.__downward_image = self.__request_downward_image(False)
            self.__is_downward_image_current = True
        return self.__downward_image

    def get_downward_image_async(self) -> NDArray[(480, 640, 3), np.uint8]:
        return self.__request_downward_image(True)

    def __update(self) -> None:
        self.__is_color_image_current = False
        self.__is_depth_image_current = False
        self.__is_downward_image_current = False

    def __on_dropped_frame(self, kind: str) -> None:
        uav_utils.print_warning(
            f">> Dropped {kind} frame (UDP loss / sim timeout). Reusing last frame."
        )

    def __request_color_image(self, isAsync: bool) -> NDArray[(480, 640, 3), np.uint8]:
        self.__drone._DroneSim__send_header(
            self.__drone.Header.camera_get_color_image, isAsync
        )

        raw_bytes = self.__drone._DroneSim__receive_fragmented(
            32, self._WIDTH * self._HEIGHT * 4, isAsync
        )
        if raw_bytes is None:
            self.__on_dropped_frame("color")
            if self.__color_image is not None:
                return self.__color_image
            return np.zeros((self._HEIGHT, self._WIDTH, 3), dtype=np.uint8)
        color_image = np.frombuffer(raw_bytes, dtype=np.uint8)
        color_image = np.reshape(color_image, (self._HEIGHT, self._WIDTH, 4), "C")
        color_image = cv.cvtColor(color_image, cv.COLOR_RGB2BGR)
        return color_image

    def __request_depth_image(self, isAsync: bool) -> NDArray[(480, 640), np.float32]:
        self.__drone._DroneSim__send_header(
            self.__drone.Header.camera_get_depth_image, isAsync
        )
        raw_bytes = self.__drone._DroneSim__receive_data(
            self._MAX_DEPTH_WIDTH * self._MAX_DEPTH_HEIGHT * 4,
            timeout=self.__drone._FRAME_TIMEOUT_S,
        )
        if raw_bytes is None:
            # Sim may yet deliver the datagram after our timeout; drain so the
            # next protocol call doesn't read it as its own response.
            self.__drone._DroneSim__drain_socket_quiet()
            self.__on_dropped_frame("depth")
            if self.__depth_image is not None:
                return self.__depth_image
            return np.zeros((self._HEIGHT, self._WIDTH), dtype=np.float32)
        depth_image = np.frombuffer(raw_bytes, dtype=np.float32)

        n: int = (len(depth_image) // 300).bit_length() // 2
        depth_width: int = 20 * 1 << n
        depth_height: int = 15 * 1 << n

        depth_image = np.reshape(depth_image, (depth_height, depth_width), "C")
        depth_image = cv.resize(
            depth_image, (self._WIDTH, self._HEIGHT), interpolation=cv.INTER_AREA
        )
        return depth_image

    def __request_downward_image(self, isAsync: bool) -> NDArray[(480, 640, 3), np.uint8]:
        self.__drone._DroneSim__send_header(
            self.__drone.Header.camera_get_downward_image, isAsync
        )

        raw_bytes = self.__drone._DroneSim__receive_fragmented(
            32, self._WIDTH * self._HEIGHT * 4, isAsync
        )
        if raw_bytes is None:
            self.__on_dropped_frame("downward")
            if self.__downward_image is not None:
                return self.__downward_image
            return np.zeros((self._HEIGHT, self._WIDTH, 3), dtype=np.uint8)
        downward_image = np.frombuffer(raw_bytes, dtype=np.uint8)
        downward_image = np.reshape(downward_image, (self._HEIGHT, self._WIDTH, 4), "C")
        downward_image = cv.cvtColor(downward_image, cv.COLOR_RGB2BGR)
        return downward_image

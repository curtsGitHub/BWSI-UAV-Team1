"""
Copyright MIT
GNU General Public License v3.0

MIT BWSI Autonomous Drone Racing Course - UAV Neo

File Name: detector.py
File Description: Defines the interface of the Detector module of the drone_core library.
"""

import abc
import numpy as np
class NDArray:  # stub — no runtime dependency on nptyping
    def __class_getitem__(cls, _): return cls
from typing import List, Optional


class Detection:
    """
    Represents a single object detection result.

    Attributes:
        class_id: The label or numeric ID of the detected object.
        score: Confidence score between 0.0 and 1.0.
        bbox: Bounding box as (center_x, center_y, width, height) in pixels.
    """

    def __init__(self, class_id: str, score: float, bbox: tuple):
        self.class_id = class_id
        self.score = score
        self.bbox = bbox  # (cx, cy, w, h) in pixels

    def __repr__(self):
        return (
            f"Detection(class_id={self.class_id!r}, score={self.score:.2f}, "
            f"bbox=({self.bbox[0]:.0f}, {self.bbox[1]:.0f}, "
            f"{self.bbox[2]:.0f}, {self.bbox[3]:.0f}))"
        )


class Detector(abc.ABC):
    """
    Provides access to onboard ML object detection via the Coral EdgeTPU.
    """

    @abc.abstractmethod
    def get_detections(self) -> List[Detection]:
        """
        Returns the latest list of object detections from the EdgeTPU.

        Returns:
            A list of Detection objects, each containing a class_id, score,
            and bounding box in pixel coordinates. Empty list if no detections
            or the EdgeTPU node is not running.

        Example::

            detections = uav.detector.get_detections()
            for det in detections:
                print(f"Found {det.class_id} at ({det.bbox[0]}, {det.bbox[1]})")
        """
        pass

    @abc.abstractmethod
    def get_detections_async(self) -> List[Detection]:
        """
        Returns the latest detections without the drone in "go" mode.

        Warning:
            This function breaks the start-update paradigm and should only be
            used in Jupyter Notebook.
        """
        pass

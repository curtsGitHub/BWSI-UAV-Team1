"""
Diagnostic: grab one frame from each camera and run the labs' vision on it, so you can
confirm the real cameras process the same way the simulator's do. It does not fly.

    sim:   drone sim course/camera_check.py       (press ENTER in the sim window)
    drone: python3 camera_check.py                (runs immediately; no flight)

Point the forward camera at a gate and the downward camera at a colored line, then compare the
printed detections and the saved images against a sim run. Watch the image shapes: the labs
assume a 640-wide image (COL_CENTER=320, IMAGE_WIDTH=640, FOCAL_PX=320); a different real
resolution means those per-camera constants need real values.
"""

import os
import sys

import cv2

import drone_core

_d = os.path.dirname(os.path.realpath(__file__))
while os.path.basename(_d) != "labs" and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
if _d not in sys.path:
    sys.path.insert(0, _d)
import neo_lab

OUT_DIR = "/tmp"
CAPTURE_TIMEOUT_FRAMES = 150   # wait this many frames for the camera stream before giving up
_frame = 0
_done = False


if __name__ == "__main__":
    _drone = drone_core.create_drone()

    def start():
        print("Check for button click")
        
    def _update():
        global _frame, _done
        if _done:
            return
        _frame += 1
        forward = _drone.camera.get_color_image()
        downward = _drone.camera.get_downward_image()
        if (forward is not None and downward is not None) or _frame >= CAPTURE_TIMEOUT_FRAMES:
            _report(forward, downward)
            _done = True

    _drone.set_start_update(start, _update)
    _drone.go(not neo_lab._is_sim(_drone))   # real: run without a controller; sim: wait for ENTER

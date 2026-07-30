"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo
GNU General Public License v3.0

Week 2/3 Lab — Step 3: Follow the Edge
Steer the drone to keep the bright edge centered while flying forward.
"""

import drone_core
import drone_utils as uav_utils
import cv2
import numpy as np

# -- Course setup: makes the shared `neo_lab` helper importable.
#    You don't need to read or change this block. --
import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.realpath(__file__))
while _os.path.basename(_d) != "labs" and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
if _d not in _sys.path:
    _sys.path.insert(0, _d)
import neo_lab

# -- Constants --------------------------------------------------------------
V_MIN         = 200
MIN_PIXELS    = 200
FORWARD_PITCH = 0.05 #18     # constant forward speed
MAX_ROLL      = 0.1 #25     # strafe authority for centering
FOLLOW_TIME   = 12.0     # seconds to follow before landing
IMAGE_CENTER  = 320      # 640-wide image -> center column
KP_YAW = 0.5            # yaw command per degree of error
MAX_YAW = .30
MAX_THROTTLE = .05
KP_ALT = 0.1

# -- Module-level state -----------------------------------------------------
_timer = 0.0
_done  = False
_prev_yaw_rate = 0.0
_prev_roll_rate = 0.0

def reset():
    global _timer, _done, _prev_yaw_rate, _prev_roll_rate
    _timer = 0.0
    _done  = False
    _prev_yaw_rate = 0.0
    _prev_roll_rate = 0.0


def update(drone):
    global _timer, _done, _prev_yaw_rate, _prev_roll_rate
    if _done:
        return True
    ##################################
    #### START PUT CODE HERE #########

    # GOAL: fly forward at FORWARD_PITCH while strafing (roll) to keep the bright
    # edge under the middle of the downward camera.
    #
    # Tools: drone.camera.get_downward_image(); neo_lab.bright_mask(image, V_MIN);
    #        np.argwhere(mask) -> bright pixel (row, col); uav_utils.clamp(...);
    #        drone.flight.send_pcmd(pitch, roll, yaw, throttle).
    #
    # The average column of the bright pixels tells you how far off-center the edge
    # is. Turn that pixel offset into a roll command (clamped to MAX_ROLL): an edge
    # right of center means roll right to chase it. If you see too few bright pixels,
    # hold position rather than steering on noise -- but keep the timer running every
    # frame and finish after FOLLOW_TIME regardless, so losing the edge never hangs.

    _timer += drone.get_delta_time()

    img = drone.camera.get_downward_image()
    b_mask = neo_lab.bright_mask(img, V_MIN)
    points = np.argwhere(b_mask)
    if len(points) < MIN_PIXELS:
        drone.flight.send_pcmd(0, 0, 0, 0)
    else:
        offset = (points[:, 1].mean() - IMAGE_CENTER) / IMAGE_CENTER
        roll_rate_raw = uav_utils.clamp(offset * MAX_ROLL, -MAX_ROLL, MAX_ROLL)
        roll_rate = 0.5 * _prev_roll_rate + 0.5 * roll_rate_raw
        _prev_roll_rate = roll_rate

        yaw_rate_raw = uav_utils.clamp(offset * MAX_YAW, -MAX_YAW, MAX_YAW)
        yaw_rate = 0.1 * _prev_yaw_rate + 0.9 * yaw_rate_raw
        _prev_yaw_rate = yaw_rate
        errAlt = 1 - neo_lab.height(drone)
        throttle = uav_utils.clamp(errAlt * KP_ALT, -MAX_THROTTLE, MAX_THROTTLE)
        print(f"Using: pitch:{FORWARD_PITCH} roll:{roll_rate} yaw:{yaw_rate} throttle:{throttle}")
        drone.flight.send_pcmd(FORWARD_PITCH, roll_rate, yaw_rate, throttle)

    if _timer >= FOLLOW_TIME:
        drone.flight.send_pcmd(0, 0, 0, 0)
        print("Finished following line")
        _done = True

    ###### END PUT CODE HERE #########
    ##################################
    return _done


if __name__ == "__main__":
    _drone = drone_core.create_drone()
    _launcher = neo_lab.Launcher(3.0)

    def start():
        _launcher.reset()
        reset()
        print("Step 3: Follow the Edge")

    def _update():
        if not _launcher.done:        # arm + climb to a safe height first
            _launcher.update(_drone)
            return
        if update(_drone):
            _drone.flight.land()

    _drone.set_start_update(start, _update)
    _drone.go()

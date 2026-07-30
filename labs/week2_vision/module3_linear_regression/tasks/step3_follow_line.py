"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo
GNU General Public License v3.0

Week 2/3 Lab — Step 3: Follow the Line
Steer the drone to keep the colored line centered while flying forward.
"""

import math

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
S_MIN         = 100
MIN_PIXELS    = 200
FORWARD_PITCH = 0.03     # slow forward speed so roll can keep up on curves
MAX_ROLL      = 0.20     # lateral centering on the near part of the line
MAX_YAW       = .20     # turn to align heading with the line ahead (primary curve steering)
KP_YAW = 1.0
IMAGE_CENTER  = 320      # 640-wide image -> center column
LOOKAHEAD_FRAC = 0.4     # split the line into a far (yaw) part and a near (roll) part
LOST_TIME     = 3.0      # land after the line has been out of view this long (its end)
MAX_FOLLOW    = 100.0     # safety cap on total follow time
MAX_THROTTLE = .05
KP_ALT = 0.1
TARGET_HEIGHT = 0.5
K_SLOW = 2
MAX_PITCH = 0.1
MIN_PITCH = 0.03

# -- Module-level state -----------------------------------------------------
_timer = 0.0
_lost  = 0.0
_done  = False
_prev_yaw_rate = 0.0
_prev_roll_rate = 0.0

def reset():
    global _timer, _lost, _done, _prev_roll_rate, _prev_yaw_rate, _slow
    _timer = 0.0
    _lost  = 0.0
    _done  = False
    _prev_yaw_rate = 0.0
    _prev_roll_rate = 0.0
    _slow = 0.0


def update(drone):
    global _timer, _lost, _done, _prev_roll_rate, _prev_yaw_rate, _slow
    if _done:
        return True
    ##################################
    #### START PUT CODE HERE #########

    # GOAL: follow the line by turning to face along it. Fly forward at FORWARD_PITCH;
    # split the line pixels into a FAR part (top of the view, where you are heading) and a
    # NEAR part (bottom, under you). Yaw toward the far part to steer through curves; roll
    # toward the near part for fine lateral centering.
    #
    # Tools: drone.camera.get_downward_image(); neo_lab.saturated_mask(image, S_MIN);
    #        np.argwhere(mask) -> line pixel (row, col); uav_utils.clamp(...);
    #        drone.flight.send_pcmd(pitch, roll, yaw, throttle).
    #
    # Split rows at rows.min() + (rows.max()-rows.min())*LOOKAHEAD_FRAC. The far part's mean
    # column vs IMAGE_CENTER gives a yaw error (clamp to MAX_YAW); the near part's gives a
    # roll error (clamp to MAX_ROLL). With too few line pixels, hover and count the time as
    # lost. Finish when the line has been lost for LOST_TIME (its end) or _timer hits MAX_FOLLOW.
    _timer += drone.get_delta_time()
    img = drone.camera.get_downward_image()
    mask = neo_lab.bright_mask(img, S_MIN) > 0
    points = np.argwhere(mask)
    
    if len(points) < MIN_PIXELS:
        drone.flight.stop()
        _lost += drone.get_delta_time()
        
        if _lost >= LOST_TIME:
            _done = True
            print("Lost :(")
        return False
    else:
        _lost = 0.0
        rows = points[:, 0]
        split_row = rows.min() + (rows.max() - rows.min()) * LOOKAHEAD_FRAC
        near = points[rows >= split_row]
        far = points[rows <= split_row]
        farCol = far[:,1].mean()
        nearCol = near[:,1].mean()
        rollErr = (nearCol - IMAGE_CENTER) / IMAGE_CENTER
        yawErr = (farCol - IMAGE_CENTER) / IMAGE_CENTER
        errAlt = TARGET_HEIGHT - neo_lab.height(drone)
        throttle = uav_utils.clamp(errAlt * KP_ALT, -MAX_THROTTLE, MAX_THROTTLE)
        
        m, b = np.polyfit(points[:,0], points[:,1], 1)
        theta = math.atan(-m)
        yaw_rate_raw = uav_utils.clamp(theta * KP_YAW, -MAX_YAW, MAX_YAW)
        # yaw_rate_raw = uav_utils.clamp(yawErr * MAX_YAW, -MAX_YAW, MAX_YAW)
        yaw_rate = 0.1 * _prev_yaw_rate + .90 * yaw_rate_raw
        _prev_yaw_rate = yaw_rate
        print(f"YawErr:{theta}")
        
        offset = (m * rows.max() + b - IMAGE_CENTER) / IMAGE_CENTER
        roll_rate_raw = uav_utils.clamp(offset * 0.3, -MAX_ROLL, MAX_ROLL)
        # roll_rate_raw = uav_utils.clamp(rollErr * MAX_ROLL, -MAX_ROLL, MAX_ROLL)
        roll_rate = 0.5 * _prev_roll_rate + 0.5 * roll_rate_raw
        _prev_roll_rate = roll_rate
        
        # m, b = np.polyfit(far[:,0], far[:,1], 1)
        m, b = np.polyfit(points[:,0], points[:,1], 1)
        resid = np.abs(points[:,1] - (m * rows + b)).mean()
        bend = max(0.0, resid - 20) / IMAGE_CENTER
        slow_raw = uav_utils.clamp(K_SLOW * bend + 0.5 * abs(yaw_rate), 0.0, 1.0)
        # slow_raw = uav_utils.clamp(K_SLOW * bend + 0.5 * abs(yaw_rate), 0.0, 1.0)
        if _slow == 0.0:
            _slow = slow_raw
        else:
            _slow = 0.9 * _slow + 0.1 * slow_raw
        
        # pitch_rate = uav_utils.clamp(1/abs(yawErr) * .02, 0.03, .15)
        pitch_rate = MAX_PITCH - (MAX_PITCH - MIN_PITCH) * _slow #uav_utils.clamp(.15 * (1.0 - bend), 0.03, .15)
        
        print(f"pitch_rate:{pitch_rate} ErrRoll:{rollErr} yawErr:{yawErr} yaw_rate:{yaw_rate} rollRate:{roll_rate} resid:{resid}")
        drone.flight.send_pcmd(pitch_rate, roll_rate, yaw_rate, throttle)
        # drone.flight.send_pcmd(0, roll_rate, 0, 0)
        
    if _timer >= MAX_FOLLOW:
        _done = True
        print("Finished following the line!")
    ###### END PUT CODE HERE #########
    ##################################
    return _done


if __name__ == "__main__":
    _drone = drone_core.create_drone()
    _launcher = neo_lab.Launcher()

    def start():
        _launcher.reset()
        reset()
        print("Step 3: Follow the Line")

    def _update():
        if not _launcher.done:        # arm + climb to a safe height first
            _launcher.update(_drone)
            return
        if update(_drone):
            _drone.flight.land()

    _drone.set_start_update(start, _update)
    _drone.go()

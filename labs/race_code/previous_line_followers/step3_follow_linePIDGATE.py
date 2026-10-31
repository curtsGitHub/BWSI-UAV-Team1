"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo
GNU General Public License v3.0

Week 2/3 Lab — Step 3: Follow the Line
Steer the drone to keep the colored line centered while flying forward.
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
S_MIN         = 200
MIN_PIXELS    = 200
FORWARD_PITCH = 0.20     # slow forward speed so roll can keep up on curves
MAX_ROLL      = 0.10     # lateral centering on the near part of the line
MAX_YAW       = 1.0     # turn to align heading with the line ahead (primary curve steering)
IMAGE_CENTER  = 320      # 640-wide image -> center column
LOOKAHEAD_FRAC = 0.4     # split the line into a far (yaw) part and a near (roll) part
LOST_TIME     = 3.0      # land after the line has been out of view this long (its end)
MAX_FOLLOW    = 100.0     # safety cap on total follow time
MAX_THROTTLE = .05
KP_ALT = 0.1
TARGET_HEIGHT = 1.0
yawKP = 1.5
yawKI = 0.0
yawKD = 0.01
rollKP = 0.3
rollKI = 0.0
rollKD = 0.03

GATES_TOP_TAGS = [36]
GATES_BOTTOM_TAGS = [38]

# -- Module-level state -----------------------------------------------------
_timer = 0.0
_lost  = 0.0
_done  = False
_prev_yaw_rate = 0.0
_prev_roll_rate = 0.0

def reset():
    global _timer, _lost, _done, _prev_roll_rate, _prev_yaw_rate, _yawPrev_err, _rollPrev_err, _prevTags, _prevImg
    _timer = 0.0
    _lost  = 0.0
    _done  = False
    _prev_yaw_rate = 0.0
    _prev_roll_rate = 0.0
    _yawPrev_err = 0.0
    _rollPrev_err = 0.0
    _prevTags = 0
    _prevImg = None

def pid_control(err, err_int, err_dot, kp, ki, kd):
    """Return the PID controller output from the three gain terms (see README, Key terms)."""
    ##################################
    #### START PUT CODE HERE #########
    output = kp * err + ki * err_int + kd * err_dot
    ###### END PUT CODE HERE #########
    ##################################
    return output


def update(drone):
    global _timer, _lost, _done, _prev_roll_rate, _prev_yaw_rate, _yawPrev_err, _rollPrev_err, _prevTags, _prevImg
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
    dt = drone.get_delta_time()
    _timer += dt
    img = drone.camera.get_downward_image()
    stale = _prevImg is not None and np.array_equal(img, _prevImg)
    _prevImg = img.copy()
    # print(img)
    mask = neo_lab.bright_mask(img, S_MIN) > 0
    points = np.argwhere(mask)
    
    if len(points) < MIN_PIXELS:
        drone.flight.stop()
        _lost += dt
        
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
        yawErr = (farCol - IMAGE_CENTER) / IMAGE_CENTER
        yawErr_derv = (yawErr - _yawPrev_err) / dt
        _yawPrev_err = yawErr
        rollErr = (nearCol - IMAGE_CENTER) / IMAGE_CENTER
        rollErr_derv = (rollErr - _rollPrev_err) / dt
        _rollPrev_err = rollErr
        errAlt = TARGET_HEIGHT - neo_lab.height(drone)
        
        throttle = uav_utils.clamp(errAlt * KP_ALT, -MAX_THROTTLE, MAX_THROTTLE)
        yaw_rate_raw = uav_utils.clamp(pid_control(yawErr, uav_utils.clamp(0, -3, 3), yawErr_derv, yawKP, yawKI, yawKD), -MAX_YAW, MAX_YAW)
        yaw_rate = 0.0 * _prev_yaw_rate + 1.0 * yaw_rate_raw
        _prev_yaw_rate = yaw_rate
        
        
        # roll_rate_raw = uav_utils.clamp(rollErr * MAX_ROLL, -MAX_ROLL, MAX_ROLL)
        roll_rate_raw = uav_utils.clamp(pid_control(rollErr, uav_utils.clamp(0, -3, 3), rollErr_derv, rollKP, rollKI, rollKD), -MAX_ROLL, MAX_ROLL)
        roll_rate = 0.0 * _prev_roll_rate + 1.0 * roll_rate_raw
        _prev_roll_rate = roll_rate
        
        pitch_rate = FORWARD_PITCH # * max(0.0, 1.0 - 2.0 * abs(yawErr))
        if neo_lab.height(drone) < 1:
            pitch_rate = 0.1
        if abs(yawErr) >= .5:
            pitch_rate = 0
        
        elif abs(yawErr) > .4 or neo_lab.height(drone) < 0.5:
            pitch_rate = 0.02
        elif abs(yawErr) > .2:
            pitch_rate = 0.04
        
        gateImg = drone.camera.get_color_image()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = neo_lab._detect_gate_markers(gray)
        if ids is not None and len(ids) >= 0:
            print("WORKING OMG GUYS OMGO MOSFIWEIFHSDF")
            if _prevTags >= 2 and len(ids) < 2:
                print("GOOD!")
                pass
            print("not good")
            pitch_rate = 0.0
            # yaw_rate = 0.0
            # roll_rate = 0.0
            topID = False
            bottomID = False
            for id in ids:
                if id in GATES_TOP_TAGS:
                    topID = True
                if id in GATES_BOTTOM_TAGS:
                    bottomID = True
            print(ids)
            if len(ids) >= 2:
                print("Centering!")
                gate = neo_lab.detect_gate(img)
                if gate is not None:
                    errAlt = (gate.cy - IMAGE_CENTER) / IMAGE_CENTER
                    altDerv = (errAlt - _altPrev_err) / dt
                    _altPrev_err = errAlt
                    throttle = uav_utils.clamp(pid_control(errAlt, 0.0, altDerv, KP_THROTTLE, KI_THROTTLE, KD_THROTTLE)
                                                , -MAX_THROTTLE, MAX_THROTTLE)
                    print(f"")
                    if abs(errAlt * IMAGE_CENTER) < 60:
                        print("CENTERED")
                        pitch_rate = FORWARD_PITCH 
            elif topID:
                print("Going down")
                throttle = -0.3
            elif bottomID:
                print("Going up")
                throttle = 0.3
               
        print(f"stale: {stale} throttle:{throttle} errAlt:{errAlt * IMAGE_CENTER} altitude:{neo_lab.height(drone)} pitch:{pitch_rate} ErrRoll:{rollErr} yawErr:{yawErr} yaw_rate:{yaw_rate * 2} rollRate:{roll_rate}")
        print(f"Current yawrate:{drone.physics.get_angular_velocity()}")
        drone.flight.send_pcmd(pitch_rate, roll_rate, yaw_rate, throttle)
        
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

"""
Lock in
"""

import math
import drone_core
import drone_utils as uav_utils
import cv2
import numpy as np
import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.realpath(__file__))
while _os.path.basename(_d) != "labs" and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
if _d not in _sys.path:
    _sys.path.insert(0, _d)
import neo_lab

#GATE Constants
V_MIN = 200
MIN_AREA = 500
COL_CENTER = 320
HOLD_TIME = 1.0

KP_ALT_GATE = 0.3 #original 0.18
KI_ALT_GATE = 0.05
KD_ALT_GATE = 0.02
ALT_INT_CLAMP = 3.0
THROTTLE_LIMIT = 0.5
ALT_TOL = 0.15
ROW_CENTER = 240

HALF_W = 3.7      # gate center -> left/right tag
HALF_H = 3.2      # gate center -> top/bottom tag

#GATE State Vars
_alt_err_int = 0.0
_alt_prev_err = 0.0
_alt_max_err = 0.0
_gate_dict = None
_gate_detector = None

#LINE Constants
S_MIN         = 200
MIN_PIXELS    = 200
FORWARD_PITCH = 0.20     # slow forward speed so roll can keep up on curves
MAX_ROLL      = 0.10     # lateral centering on the near part of the line
MAX_YAW       = 0.5      # real drone: 0.5 * 0.5 rad/s = 0.25 rad/s -> ~0.4 m turn radius
                         # at FORWARD_PITCH speed (0.1 m/s); widen if curves allow
IMAGE_CENTER  = 320      # 640-wide image -> center column
LOOKAHEAD_FRAC = 0.4     # split the line into a far (yaw) part and a near (roll) part
LOST_TIME     = 3.0      # land after the line has been out of view this long (its end)
MAX_FOLLOW    = 100.0     # safety cap on total follow time
MAX_THROTTLE = 0.3       # real drone: 0.3 * 0.5 m/s = 0.15 m/s vertical authority.
                         # The old 0.05 sat inside the sim's deadband, where the sim
                         # itself held altitude -- on real PX4 nothing does that for us.
KP_ALT = 0.3             # retune at the field; sim value 0.1 was masked by the deadband
TARGET_HEIGHT = 1.0
yawKP = 0.6              # yawErr is the line angle in radians, so keep kp*err under MAX_YAW
yawKI = 0.0
yawKD = 0.2              # usable now that the derivative is low-pass filtered
rollKP = 0.4
rollKI = 0.0
rollKD = 0.03

#LINE State Vars
_timer = 0.0
_lost  = 0.0
_done  = False
_prev_yaw_rate = 0.0
_prev_roll_rate = 0.0
_yaw_err_dot_filt = 0.0
_gate_last = None

#Shared State
_throttle_h = 0.0
_throttle_l = 0.0
_throttle_prev = 0.0


# Where each tag sits relative to the gate center: (right, up), in tag sides.
SLOT_OFFSET = {
    "t": ( 0.0,     HALF_H),
    "b": ( 0.0,    -HALF_H),
    "l": (-HALF_W,  0.0),
    "r": ( HALF_W,  0.0),
    "tl": (-HALF_W,  HALF_H),
    "tr": ( HALF_W,  HALF_H),
    "bl": (-HALF_W, -HALF_H),
    "br": ( HALF_W, -HALF_H),
}

# Course map: tag id -> (gate number, slot). Fill this in and let it rip.
TAG_MAP = {
    0: (1, "t"),
    1: (1, "b"),
    2: (1, "l"),
    3: (1, "r"),
}

def pid_control(err, err_int, err_dot, kp, ki, kd):
    p = kp * err
    i = ki * err_int
    d = kd * err_dot
    return p + i + d

#manual aruco
class Gate:
    """A gate located from its corner ArUco tags: image center (cx, cy), inter-tag span
    (gate size proxy), mean tag pixel size (a proximity signal that works with one tag),
    and the decoded corner-tag ids."""

    def __init__(self, cx, cy, span, tag_px, ids):
        self.cx = cx
        self.cy = cy
        self.span = span
        self.tag_px = tag_px
        self.ids = ids
        self.count = len(ids)

def _gate_aruco_dict():
    global _gate_dict
    if _gate_dict is None:
        _gate_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    return _gate_dict

def _detect_gate_markers(gray):
    global _gate_detector
    try:
        if _gate_detector is None:
            _gate_detector = cv2.aruco.ArucoDetector(_gate_aruco_dict(), cv2.aruco.DetectorParameters())
        return _gate_detector.detectMarkers(gray)
    except AttributeError:                   # OpenCV < 4.7 free-function API
        # params must come from _create() here: the direct constructor segfaults detectMarkers
        params = cv2.aruco.DetectorParameters_create()
        return cv2.aruco.detectMarkers(gray, _gate_aruco_dict(), parameters=params)

def _tag_side(quad):
    """Mean of the four edge lengths — steadier than a single edge when tilted."""
    p = quad.reshape(-1, 2)
    edges = [np.linalg.norm(p[i] - p[(i + 1) % 4]) for i in range(4)]
    return float(np.mean(edges))


def _read_tags(gray):
    """Every decoded tag as (id, center_xy, side_px)."""
    corners, ids, _ = _detect_gate_markers(gray)
    if ids is None or len(ids) == 0:
        return []

    tags = []
    for quad, tid in zip(corners, ids.flatten()):
        center = quad.reshape(-1, 2).mean(axis=0)
        tags.append((int(tid), center, _tag_side(quad)))
    return tags


def _nearest_gate(tags):
    """Group the mapped tags by gate, keep the closest gate (biggest tag on screen)."""
    gates = {}
    for tid, center, side in tags:
        if tid not in TAG_MAP:
            continue
        gate_number, slot = TAG_MAP[tid]
        gates.setdefault(gate_number, []).append((tid, slot, center, side))

    if not gates:
        return []
    return max(gates.values(), key=lambda group: max(t[3] for t in group))


def _center_from_tag(slot, center, side):
    """One tag's own guess at where the gate center is, in pixels."""
    right, up = SLOT_OFFSET[slot]
    col = center[0] - right * side
    row = center[1] + up * side       # image rows grow downward, so up flips sign
    return np.array([col, row])


def detect_gate(image):
    """Locate a gate from its DICT_6X6_250 tags. Each visible tag predicts the gate
    center on its own; the predictions are averaged. One tag is enough, two agree."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    group = _nearest_gate(_read_tags(gray))
    if not group:
        return None

    guesses = [_center_from_tag(slot, center, side)
               for _tid, slot, center, side in group]
    cx, cy = np.mean(guesses, axis=0)

    tag_px = float(np.mean([t[3] for t in group]))
    span = 2.0 * HALF_W * tag_px
    ids = [t[0] for t in group]
    return Gate(float(cx), float(cy), float(span), tag_px, ids)

def reset():
    global _rot_err_int, _rot_prev_err, _filtered_rot_err_dot, _target_col, _hold, _done
    global _alt_err_int, _alt_prev_err, _alt_max_err
    global _timer, _lost, _prev_roll_rate, _prev_yaw_rate, _yawPrev_err, _rollPrev_err, _prevTags
    global _yaw_err_dot_filt, _gate_last
    global _throttle_h, _throttle_l, _throttle_prev
    global _fresh, _recorder
    _rot_err_int = 0.0
    _rot_prev_err = 0.0
    _filtered_rot_err_dot = 0.0
    _target_col = None
    _hold = 0.0
    _done = False
    _alt_err_int = 0.0
    _alt_prev_err = 0.0
    _alt_max_err = 0.0

    _timer = 0.0
    _lost  = 0.0
    _prev_yaw_rate = 0.0
    _prev_roll_rate = 0.0
    _yawPrev_err = 0.0
    _rollPrev_err = 0.0
    _prevTags = 0
    _yaw_err_dot_filt = 0.0
    _gate_last = None

    _throttle_h = 0.0
    _throttle_l = 0.0
    _throttle_prev = 0.0
    
    _fresh = 0
    _recorder = None

def update(drone):
    global _rot_err_int, _rot_prev_err, _filtered_rot_err_dot, _target_col, _hold, _done
    global _alt_err_int, _alt_prev_err, _alt_max_err
    global _timer, _lost, _prev_roll_rate, _prev_yaw_rate, _yawPrev_err, _rollPrev_err, _prevTags
    global _yaw_err_dot_filt, _gate_last
    global _throttle_h, _throttle_l, _throttle_prev
    global _fresh
    if _done:
        return True
    
    #General
    dt = drone.get_delta_time()
    _timer += dt
    tick = int(_timer * 2) != int((_timer - dt) * 2)   # rate-limit debug prints to 2 Hz
    img = drone.camera.get_downward_image()
    imgColor = drone.camera.get_color_image()

    #LINE FOLLOW
    if img is None:
        # the real camera returns None until the first frame arrives and on stream
        # dropouts -- hold position, and land via the lost timer if video never returns
        drone.flight.stop()
        _lost += dt
        if _lost >= LOST_TIME:
            _done = True
            print("No video :(")
        return _done

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
        # split at a fixed image row so the far band's physical distance doesn't
        # collapse when the line ahead exits the side of the frame on sharp curves
        split_row = mask.shape[0] * LOOKAHEAD_FRAC
        near = points[rows >= split_row]
        far = points[rows < split_row]
        if len(far) == 0:
            far = points
        if len(near) == 0:
            near = points
        farCol = far[:,1].mean()
        nearCol = near[:,1].mean()

        # yaw error = line angle (near centroid -> far centroid), radians; invariant
        # to lateral offset, so it stays unbiased on curves and leaves offset to roll
        yawErr = math.atan2(farCol - nearCol, max(near[:, 0].mean() - far[:, 0].mean(), 1.0))
        yawErr_derv = (yawErr - _yawPrev_err) / dt if dt > 0 else 0.0
        _yaw_err_dot_filt = 0.8 * _yaw_err_dot_filt + 0.2 * yawErr_derv
        _yawPrev_err = yawErr

        rollErr = (nearCol - IMAGE_CENTER) / IMAGE_CENTER
        rollErr_derv = (rollErr - _rollPrev_err) / dt if dt > 0 else 0.0
        _rollPrev_err = rollErr
        errAlt = TARGET_HEIGHT - neo_lab.height(drone)

        _throttle_l = uav_utils.clamp(errAlt * KP_ALT, -MAX_THROTTLE, MAX_THROTTLE)
        yaw_rate_raw = uav_utils.clamp(pid_control(yawErr, 0.0, _yaw_err_dot_filt, yawKP, yawKI, yawKD), -MAX_YAW, MAX_YAW)
        yaw_rate = 0.0 * _prev_yaw_rate + 1.0 * yaw_rate_raw
        _prev_yaw_rate = yaw_rate


        # roll_rate_raw = uav_utils.clamp(rollErr * MAX_ROLL, -MAX_ROLL, MAX_ROLL)
        roll_rate_raw = uav_utils.clamp(pid_control(rollErr, 0.0, rollErr_derv, rollKP, rollKI, rollKD), -MAX_ROLL, MAX_ROLL)
        roll_rate = 0.0 * _prev_roll_rate + 1.0 * roll_rate_raw
        _prev_roll_rate = roll_rate
        
        pitch_rate = FORWARD_PITCH * max(0.0, 1.0 - 3.0 * abs(yawErr))

    #HEIGHT
    _fresh += 1
    if _fresh % 5 == 0:
        # cache the result so the frames between detections keep the last gate
        # instead of falling into the gate-lost branch (throttle chatter, _hold reset)
        _gate_last = detect_gate(imgColor) if imgColor is not None else None
    gate = _gate_last
    if tick:
        print(f"Gate: {gate}")
    #print(f"id {gate.i}")

    if gate is None:
        # drone.flight.send_pcmd(0, 0, 0, 0)
        _target_col = None
        # _rot_err_int = 0.0
        # _rot_prev_err = 0.0
        _hold = 0.0
        _alt_err_int = 0.0
    
    else:
        pitch_rate = 0.02
        target_row = gate.cy
        _target_col = gate.cx
        if tick:
            print(f"Coords: {_target_col}, {target_row}")
        """
        rot_err = (_target_col - COL_CENTER) / COL_CENTER
        _rot_err_int = uav_utils.clamp(_rot_err_int + rot_err * dt, -1.0, 1.0)
        raw_rot_err_dot = (rot_err - _rot_prev_err) / dt if dt > 0.0 else 0.0 #filtered derivative
        _filtered_rot_err_dot = 0.8 * _filtered_rot_err_dot + 0.2 * raw_rot_err_dot #continuing it
        _rot_prev_err = rot_err
        yaw = uav_utils.clamp(pid_control(rot_err, _rot_err_int, _filtered_rot_err_dot, KP_YAW, KI_YAW, KD_YAW), -MAX_YAW, MAX_YAW)
        """

        #Altitude
        alt_err = (ROW_CENTER - target_row) / ROW_CENTER
        _alt_err_int = uav_utils.clamp(_alt_err_int + alt_err * dt, -ALT_INT_CLAMP, ALT_INT_CLAMP)
        alt_err_dot = (alt_err - _alt_prev_err) / dt if dt > 0 else 0
        _alt_prev_err = alt_err

        feedback = pid_control(alt_err, _alt_err_int, alt_err_dot, KP_ALT_GATE, KI_ALT_GATE, KD_ALT_GATE)
        #feedforward = KFF * r_dot
        _throttle_h = uav_utils.clamp(feedback, -THROTTLE_LIMIT, THROTTLE_LIMIT)
        if tick:
            print(f"Error: {alt_err}")

        if (abs(alt_err) < ALT_TOL):
            _hold += dt
            if tick:
                print(f"holding {_hold:.2f}")
            # drone.flight.send_pcmd(0, 0, 0, 0)
        else:
            _hold = 0.0

    #COMMAND
    #low pass filter
    #throttle = (_throttle_prev * 0.7) + ((_throttle_h + _throttle_l) * 0.3)
    throttle = _throttle_l if gate is None else (_throttle_l + _throttle_h)
    _throttle_prev = throttle
    if tick:
        print(f"throttle:{throttle} errAlt:{errAlt} altitude:{neo_lab.height(drone)} pitch:{pitch_rate} ErrRoll:{rollErr} yawErr:{yawErr} yaw_rate:{yaw_rate} rollRate:{roll_rate}")
        print(f"Current yawrate:{drone.physics.get_angular_velocity()}")
    drone.flight.send_pcmd(pitch_rate, roll_rate, yaw_rate, throttle)
            
    if _timer >= MAX_FOLLOW:
        _done = True
        print("Finished following the line!")

    """
    if _hold >= HOLD_TIME:
        drone.flight.stop()
        print("Done, locked on")
        _done = True
    """
    return _done


if __name__ == "__main__":
    _drone = drone_core.create_drone()
    _launcher = neo_lab.Launcher(1.0)

    def start():
        _launcher.reset()
        reset()
        print("Step 3: Visual Servoing (Vision + PID)")

    def _update():
        if not _launcher.done:        # arm + climb to a safe height first
            _launcher.update(_drone)
            return
        if update(_drone):
            _drone.flight.land()

    _drone.set_start_update(start, _update)
    _drone.go()


def record(z, x, y, yaw, waypointType):
    global _recorder
    path = _os.environ.get("LOG_RACE_RECORD")
    if path is None:
        return
    
    if _recorder is None:
        _recorder = neo_lab.Recorder(path)
    row = {"Z":z, "X":x, "Y":y, "Yaw":yaw, "Type":waypointType}
    _recorder.log(**row)

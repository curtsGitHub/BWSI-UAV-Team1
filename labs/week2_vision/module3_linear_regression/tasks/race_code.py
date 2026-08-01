"""
Lock in
"""

import math
import logging
import time
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


def _make_logger():
    """Two sinks: console (live view) and a timestamped file under <repo>/logs/race/
    (commit it after the flight so the log is findable in git). RACE_LOG_DIR overrides
    the directory; file lines carry a wall-clock timestamp, console stays clean."""
    lg = logging.getLogger("race")
    if lg.handlers:                          # already configured (re-import / re-run)
        return lg
    lg.setLevel(logging.INFO)
    lg.propagate = False
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(message)s"))
    lg.addHandler(console)
    try:
        repo_root = _os.path.dirname(_os.path.dirname(_d))     # labs -> course -> repo
        log_dir = _os.environ.get("RACE_LOG_DIR", _os.path.join(repo_root, "logs", "race"))
        _os.makedirs(log_dir, exist_ok=True)
        path = _os.path.join(log_dir, time.strftime("%Y%m%d_%H%M%S") + "_race.log")
        file_sink = logging.FileHandler(path)
        file_sink.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        lg.addHandler(file_sink)
        lg.info(f"[LOG] writing to {path}")
    except OSError as e:
        lg.info(f"[LOG] file sink disabled: {e}")
    return lg


_log = _make_logger()

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

# !!!!!!!!!!!!!! SIM SETTINGS
HALF_W = 3.7      # gate center -> left/right tag
HALF_H = 3.2      # gate center -> top/bottom tag

# !!!!!!!!!!!!!! SIM SETTINGS
HALF_W = 77./2./7.75     # gate center -> left/right tag
HALF_H = 67./2./7.75     # gate center -> top/bottom tag

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
MAX_ROLL      = 0.30     # lateral centering: 0.3 * 0.5 m/s = 0.15 m/s, enough to chase
                         # the 0.1 m/s forward speed (0.10 pinned all flight in the logs)
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

#DEBUG
DBG_HZ = 4               # telemetry lines per second; [EVT] lines print immediately

#LINE State Vars
_timer = 0.0
_lost  = 0.0
_done  = False
_prev_yaw_rate = 0.0
_prev_roll_rate = 0.0
_yaw_err_dot_filt = 0.0
_yaw_err_filt = 0.0
_roll_err_filt = 0.0
_gate_last = None

#DEBUG State Vars
_dbg_all_tags = []       # every decoded tag (id, side_px) from the last detection attempt
_tags_seen_prev = []     # decoded ids at the previous attempt, for change events
_fb_prev = "--"          # band-fallback state (N=near empty, F=far empty)
_gate_seen_t = None      # _timer at the last successful gate detection
_gate_px_prev = None     # (t, tag_px) of the previous detection, for closure rate
_gate_close_rate = 0.0   # d(tag_px)/dt, px/s -- positive means approaching the gate

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
    48: (1, "t"),
    1: (1, "b"),
    2: (1, "l"),
    46: (1, "r"),
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
    global _dbg_all_tags
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tags = _read_tags(gray)
    _dbg_all_tags = [(tid, round(side, 1)) for tid, _center, side in tags]
    group = _nearest_gate(tags)
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
    global _yaw_err_dot_filt, _yaw_err_filt, _roll_err_filt, _gate_last
    global _dbg_all_tags, _tags_seen_prev, _fb_prev, _gate_seen_t, _gate_px_prev, _gate_close_rate
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
    _yaw_err_filt = 0.0
    _roll_err_filt = 0.0
    _gate_last = None

    _dbg_all_tags = []
    _tags_seen_prev = []
    _fb_prev = "--"
    _gate_seen_t = None
    _gate_px_prev = None
    _gate_close_rate = 0.0

    _throttle_h = 0.0
    _throttle_l = 0.0
    _throttle_prev = 0.0
    
    _fresh = 0
    _recorder = None

def update(drone):
    global _rot_err_int, _rot_prev_err, _filtered_rot_err_dot, _target_col, _hold, _done
    global _alt_err_int, _alt_prev_err, _alt_max_err
    global _timer, _lost, _prev_roll_rate, _prev_yaw_rate, _yawPrev_err, _rollPrev_err, _prevTags
    global _yaw_err_dot_filt, _yaw_err_filt, _roll_err_filt, _gate_last
    global _dbg_all_tags, _tags_seen_prev, _fb_prev, _gate_seen_t, _gate_px_prev, _gate_close_rate
    global _throttle_h, _throttle_l, _throttle_prev
    global _fresh
    if _done:
        return True
    
    #General
    dt = drone.get_delta_time()
    _timer += dt
    tick = int(_timer * DBG_HZ) != int((_timer - dt) * DBG_HZ)
    img = drone.camera.get_downward_image()
    imgColor = drone.camera.get_color_image()

    #LINE FOLLOW
    if img is None:
        # the real camera returns None until the first frame arrives and on stream
        # dropouts -- hold position, and land via the lost timer if video never returns
        if _lost == 0.0:
            _log.info(f"[EVT] t={_timer:.2f} video dropout (downward image is None)")
        drone.flight.stop()
        _lost += dt
        if _lost >= LOST_TIME:
            _done = True
            _log.info(f"[EVT] t={_timer:.2f} no video for {LOST_TIME}s -> landing")
        return _done

    mask = neo_lab.bright_mask(img, S_MIN) > 0
    points = np.argwhere(mask)

    if len(points) < MIN_PIXELS:
        if _lost == 0.0:
            _log.info(f"[EVT] t={_timer:.2f} line lost (px={len(points)} < {MIN_PIXELS})")
        drone.flight.stop()
        _lost += dt

        if _lost >= LOST_TIME:
            _done = True
            _log.info(f"[EVT] t={_timer:.2f} line gone for {LOST_TIME}s -> done")
        return False
    else:
        if _lost > 0.0:
            _log.info(f"[EVT] t={_timer:.2f} line reacquired after {_lost:.2f}s (px={len(points)})")
        _lost = 0.0
        rows = points[:, 0]
        # split at a fixed image row so the far band's physical distance doesn't
        # collapse when the line ahead exits the side of the frame on sharp curves
        split_row = mask.shape[0] * LOOKAHEAD_FRAC
        near = points[rows >= split_row]
        far = points[rows < split_row]
        n_near, n_far = len(near), len(far)
        fb = ("N" if n_near == 0 else "-") + ("F" if n_far == 0 else "-")
        if fb != _fb_prev:
            _log.info(f"[EVT] t={_timer:.2f} bands={fb} (N=near empty, F=far empty; "
                      f"an empty band makes yawErr fall back toward 0)")
            _fb_prev = fb
        if n_far == 0:
            far = points
        if n_near == 0:
            near = points
        farCol = far[:,1].mean()
        nearCol = near[:,1].mean()
        near_row = near[:, 0].mean()
        far_row = far[:, 0].mean()

        # yaw error = line angle (near centroid -> far centroid), radians; invariant
        # to lateral offset, so it stays unbiased on curves and leaves offset to roll
        yawErr_raw = math.atan2(farCol - nearCol, max(near_row - far_row, 1.0))
        rollErr_raw = (nearCol - IMAGE_CENTER) / IMAGE_CENTER

        # low-pass both vision errors: the downward camera is not gimballed, so body
        # tilt shifts the whole line in the image -- flight logs show single tilted
        # frames flipping the raw error sign (yawErr +0.47 -> -0.24 in one tick)
        _yaw_err_filt = 0.7 * _yaw_err_filt + 0.3 * yawErr_raw
        _roll_err_filt = 0.7 * _roll_err_filt + 0.3 * rollErr_raw
        yawErr = _yaw_err_filt
        rollErr = _roll_err_filt

        yawErr_derv = (yawErr - _yawPrev_err) / dt if dt > 0 else 0.0
        _yaw_err_dot_filt = 0.8 * _yaw_err_dot_filt + 0.2 * yawErr_derv
        _yawPrev_err = yawErr

        rollErr_derv = (rollErr - _rollPrev_err) / dt if dt > 0 else 0.0
        _rollPrev_err = rollErr
        errAlt = TARGET_HEIGHT - neo_lab.height(drone)

        # keep the unclamped PID outputs so the telemetry can flag saturation
        _throttle_l = uav_utils.clamp(errAlt * KP_ALT, -MAX_THROTTLE, MAX_THROTTLE)
        yaw_pid = pid_control(yawErr, 0.0, _yaw_err_dot_filt, yawKP, yawKI, yawKD)
        yaw_rate = uav_utils.clamp(yaw_pid, -MAX_YAW, MAX_YAW)
        _prev_yaw_rate = yaw_rate

        roll_pid = pid_control(rollErr, 0.0, rollErr_derv, rollKP, rollKI, rollKD)
        roll_rate = uav_utils.clamp(roll_pid, -MAX_ROLL, MAX_ROLL)
        _prev_roll_rate = roll_rate
        
        # gentle slope with a floor: the old on/off gate stepped pitch 0 <-> full,
        # kicking the airframe (-2.4 rad/s pitch spikes in the logs) and corrupting
        # the camera view with tilt; never stop fully, just slow down for curves
        pitch_rate = FORWARD_PITCH * max(0.3, 1.0 - 1.5 * abs(yawErr))

    #HEIGHT
    _fresh += 1
    if _fresh % 5 == 0:
        # cache the result so the frames between detections keep the last gate
        # instead of falling into the gate-lost branch (throttle chatter, _hold reset)
        if imgColor is not None:
            _gate_last = detect_gate(imgColor)
        else:
            _gate_last = None
            _dbg_all_tags = []
        # event on any change in what's decoded -- unmapped ids here mean TAG_MAP is wrong
        seen = sorted(tid for tid, _px in _dbg_all_tags)
        if seen != _tags_seen_prev:
            mapped = [t for t in seen if t in TAG_MAP]
            unmapped = [t for t in seen if t not in TAG_MAP]
            _log.info(f"[EVT] t={_timer:.2f} tags in view: mapped={mapped} unmapped={unmapped} "
                      f"sizes={_dbg_all_tags}")
            _tags_seen_prev = seen
        if _gate_last is not None:
            if _gate_px_prev is not None and _timer > _gate_px_prev[0]:
                _gate_close_rate = (_gate_last.tag_px - _gate_px_prev[1]) / (_timer - _gate_px_prev[0])
            _gate_px_prev = (_timer, _gate_last.tag_px)
            _gate_seen_t = _timer
    gate = _gate_last

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

        if (abs(alt_err) < ALT_TOL):
            if _hold == 0.0:
                _log.info(f"[EVT] t={_timer:.2f} gate hold started (altE={alt_err:+.3f})")
            _hold += dt
            if _hold >= HOLD_TIME and _hold - dt < HOLD_TIME:
                _log.info(f"[EVT] t={_timer:.2f} gate hold reached {HOLD_TIME}s")
            # drone.flight.send_pcmd(0, 0, 0, 0)
        else:
            if _hold > 0.0:
                _log.info(f"[EVT] t={_timer:.2f} gate hold broken at {_hold:.2f}s (altE={alt_err:+.3f})")
            _hold = 0.0

    #COMMAND
    #low pass filter
    #throttle = (_throttle_prev * 0.7) + ((_throttle_h + _throttle_l) * 0.3)
    throttle = _throttle_l if gate is None else (_throttle_l + _throttle_h)
    _throttle_prev = throttle
    if tick:
        att = drone.physics.get_attitude()             # (roll, pitch, yaw) deg
        gyro = drone.physics.get_angular_velocity()    # body rates (roll, pitch, yaw) rad/s
        sat = (("Y" if abs(yaw_pid) > MAX_YAW else "-")
               + ("R" if abs(roll_pid) > MAX_ROLL else "-")
               + ("T" if abs(errAlt * KP_ALT) > MAX_THROTTLE else "-"))
        _log.info(
            f"[LF] t={_timer:6.2f} dt={dt*1000:4.0f}ms alt={neo_lab.height(drone):.2f}"
            f" att=({att[0]:+5.1f},{att[1]:+5.1f},{att[2]:6.1f})deg"
            f" gyro=({gyro[0]:+.2f},{gyro[1]:+.2f},{gyro[2]:+.2f})"
            f" px={len(points)} n/f={n_near}/{n_far} bands={fb}"
            f" nearC={nearCol:5.1f} farC={farCol:5.1f}"
            f" yawE={yawErr:+.3f}(raw{yawErr_raw:+.3f} d{_yaw_err_dot_filt:+.3f})"
            f" rollE={rollErr:+.3f}(raw{rollErr_raw:+.3f})"
            f" | cmd P={pitch_rate:.3f} R={roll_rate:+.3f} Y={yaw_rate:+.3f} T={throttle:+.3f}"
            f" sat={sat}"
        )
        if gate is not None:
            age = _timer - _gate_seen_t if _gate_seen_t is not None else float("nan")
            _log.info(
                f"[GT] t={_timer:6.2f} ids={gate.ids} n={gate.count}"
                f" cx={gate.cx:5.1f} cy={gate.cy:5.1f} tag_px={gate.tag_px:5.1f}"
                f" span={gate.span:6.1f} close={_gate_close_rate:+5.1f}px/s age={age:.2f}s"
                f" altE={alt_err:+.3f} (P={KP_ALT_GATE*alt_err:+.3f}"
                f" I={KI_ALT_GATE*_alt_err_int:+.3f} D={KD_ALT_GATE*alt_err_dot:+.3f})"
                f" thH={_throttle_h:+.3f} thL={_throttle_l:+.3f} hold={_hold:.2f}"
            )
    drone.flight.send_pcmd(pitch_rate, roll_rate, yaw_rate, throttle)

    if _timer >= MAX_FOLLOW:
        _done = True
        _log.info(f"[EVT] t={_timer:.2f} MAX_FOLLOW ({MAX_FOLLOW}s) reached -> done")

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
        _log.info("Step 3: Visual Servoing (Vision + PID)")
        # snapshot the tuning so every log file records the gains it flew with
        _log.info(f"[CFG] line: yawKP={yawKP} yawKD={yawKD} MAX_YAW={MAX_YAW}"
                  f" rollKP={rollKP} rollKD={rollKD} MAX_ROLL={MAX_ROLL}"
                  f" FORWARD_PITCH={FORWARD_PITCH} KP_ALT={KP_ALT} MAX_THROTTLE={MAX_THROTTLE}"
                  f" S_MIN={S_MIN} LOOKAHEAD_FRAC={LOOKAHEAD_FRAC} TARGET_HEIGHT={TARGET_HEIGHT}")
        _log.info(f"[CFG] gate: KP={KP_ALT_GATE} KI={KI_ALT_GATE} KD={KD_ALT_GATE}"
                  f" THROTTLE_LIMIT={THROTTLE_LIMIT} ALT_TOL={ALT_TOL} HOLD_TIME={HOLD_TIME}"
                  f" TAG_MAP={TAG_MAP}")

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

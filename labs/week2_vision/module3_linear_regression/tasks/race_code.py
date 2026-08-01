"""
Lock in
"""

import atexit
import math
import subprocess
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

# Bump on every code change (date.rev + slug). Printed as the first log line so a
# flight log always identifies the code that actually flew -- a log without a
# matching [VER] line means the drone ran a stale copy.
CODE_VERSION = "2026-08-01.22 max-follow-300s"

# ── Flight log capture ────────────────────────────────────────────────────────────
# Everything written to the console (our prints, neo_lab's, tracebacks) is teed
# into <repo>/flight_logs/<timestamp>_flight.log, flushed line-by-line so a crash
# still leaves a complete file. At program exit an atexit hook commits and pushes
# the flight_logs dir with git in a subprocess (also sweeping up leftover logs
# from crashed runs). RACE_LOG_GIT=0 disables the git step -- set that on dev
# machines so sim runs don't auto-commit; the Pi should leave it unset.

_ORIG_STDOUT, _ORIG_STDERR = _sys.stdout, _sys.stderr
_LOG_REPO = None
_LOG_FILE = None
_LOG_NAME = None
_LOG_DIR = None
try:
    _r = subprocess.run(["git", "-C", _d, "rev-parse", "--show-toplevel"],
                        capture_output=True, text=True, timeout=10)
    if _r.returncode == 0:
        _LOG_REPO = _r.stdout.strip()
        _log_dir = _os.path.join(_LOG_REPO, "flight_logs")
        _os.makedirs(_log_dir, exist_ok=True)
        _LOG_DIR = _log_dir
        # timestamp + pid + counter, created exclusively ("x"): unique even with
        # the Pi's clock resetting between boots or two runs in the same second
        _stamp = time.strftime("%Y%m%d_%H%M%S")
        for _n in range(100):
            _cand = f"{_stamp}_p{_os.getpid()}{'' if _n == 0 else f'_{_n}'}_flight.log"
            try:
                _LOG_FILE = open(_os.path.join(_log_dir, _cand), "x")
                _LOG_NAME = _cand
                break
            except FileExistsError:
                continue
except Exception:
    _LOG_REPO = _LOG_FILE = None


class _Tee:
    """Mirror a console stream into the flight log file."""
    def __init__(self, stream, sink):
        self._stream = stream
        self._sink = sink

    def write(self, data):
        n = self._stream.write(data)
        try:
            self._sink.write(data)
            self._sink.flush()             # every line on disk, crash-proof
        except (OSError, ValueError):
            pass
        return n

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):           # fileno/isatty/encoding for libraries
        return getattr(self._stream, name)


if _LOG_FILE is not None:
    _sys.stdout = _Tee(_ORIG_STDOUT, _LOG_FILE)
    _sys.stderr = _Tee(_ORIG_STDERR, _LOG_FILE)


def _px4_log_summary():
    """Pull the PX4 flight-controller log summary (vibration, clipping, thrust)
    into this flight log via diagnostics/pull_px4_log.py. Runs only when
    PX4_LOG_URL is set (the Pi exports it, e.g. udp:0.0.0.0:14550) and is
    bounded by PX4_LOG_TIMEOUT (default 180 s) so exit can never hang. One pull
    per run at exit: pulling between steps would fight MAVROS for the link.
    The .ulg lands in flight_logs/ and rides the same commit."""
    url = _os.environ.get("PX4_LOG_URL")
    if not url or _LOG_DIR is None:
        return
    script = _os.path.join(_d, "diagnostics", "pull_px4_log.py")
    timeout = float(_os.environ.get("PX4_LOG_TIMEOUT", "180"))
    print(f"[PX4] pulling flight-controller log from {url} (timeout {timeout:.0f}s)")
    try:
        r = subprocess.run([_sys.executable, script, url], cwd=_LOG_DIR,
                           capture_output=True, text=True, timeout=timeout)
        if r.stdout.strip():
            print(r.stdout.rstrip())
        if r.returncode != 0:
            print(f"[PX4] puller exited {r.returncode}: {r.stderr.strip()[:400]}")
    except subprocess.TimeoutExpired:
        print(f"[PX4] pull timed out after {timeout:.0f}s -- run pull_px4_log.py manually")
    except Exception as e:
        print(f"[PX4] pull skipped: {e}")


def _commit_flight_log():
    """atexit: append the PX4 summary while the tee is still attached (so it is
    part of the pushed log), then detach, commit and push. A second Ctrl-C
    during the (potentially slow) PX4 pull cancels just the pull; the log still
    commits. Ctrl-C during the commit itself aborts it -- the file stays on
    disk and is swept into the next run's commit."""
    global _LOG_FILE
    try:
        _px4_log_summary()
    except KeyboardInterrupt:
        print("[PX4] pull cancelled by Ctrl-C -- committing the log without it")
    except Exception as e:
        print(f"[PX4] summary failed: {e}")
    _sys.stdout, _sys.stderr = _ORIG_STDOUT, _ORIG_STDERR
    if _LOG_FILE is not None:
        try:
            _LOG_FILE.close()
        except OSError:
            pass
        _LOG_FILE = None
    if _LOG_REPO is None or _os.environ.get("RACE_LOG_GIT", "1") == "0":
        return

    def _git(*args, timeout):
        return subprocess.run(["git", "-C", _LOG_REPO, *args],
                              capture_output=True, text=True, timeout=timeout)

    try:
        _git("add", "flight_logs", timeout=30)
        # pathspec on the commit too, so nothing else that happens to be staged
        # in the repo ever rides along with a flight log
        c = _git("commit", "-m", f"flight log {_LOG_NAME} ({CODE_VERSION})",
                 "--", "flight_logs", timeout=30)
        if c.returncode != 0:
            print(f"[LOG] flight log commit failed: {(c.stdout + c.stderr).strip()[:200]}")
            return
        print(f"[LOG] flight log committed ({_LOG_NAME})")
        p = _git("push", timeout=60)
        if p.returncode == 0:
            print("[LOG] flight log pushed")
        else:
            print(f"[LOG] PUSH FAILED -- commit is local, `git push` later: "
                  f"{(p.stdout + p.stderr).strip()[:200]}")
    except Exception as e:                 # git missing, timeout, ...
        print(f"[LOG] flight log git step skipped: {e}")


atexit.register(_commit_flight_log)


class _ConsoleLog:
    """Console log; via the stdout tee every line also lands in the flight log."""
    @staticmethod
    def info(msg):
        print(msg, flush=True)


_log = _ConsoleLog()


def _log_ver():
    _log.info(f"[VER] race_code {CODE_VERSION}")


_log_ver()   # at import, before any step runs

#GATE Constants
V_MIN = 200
MIN_AREA = 500
COL_CENTER = 320
HOLD_TIME = 1.0

KP_ALT_GATE = 0.3 #original 0.18
KI_ALT_GATE = 0.05
KD_ALT_GATE = 0.02
GATE_KP_YAW = 0.4        # yaw toward the gate's cx during approach
GATE_NEAR_PX = 40.0      # tag_px at which the gate fully owns yaw (blend weight 1.0);
                         # first detections in the 2026-08-01 log were ~23-28 px,
                         # close approach ~40-53 px
ALT_INT_CLAMP = 3.0
THROTTLE_LIMIT = 0.5
ALT_TOL = 0.15
ROW_CENTER = 240

# Gate geometry in tag-side units -- exactly ONE block active at a time.
# SIM (v1.0.0 gates):
#HALF_W = 3.7             # gate center -> left/right tag
#HALF_H = 3.2             # gate center -> top/bottom tag
# REAL (measured: 77 x 67 cm gate, 7.75 cm tags):
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
FORWARD_PITCH = 0.14     # 0.07 m/s cruise -- conservative for sharp turns. NOTE: yaw
                         # authority (MAX_YAW 0.35) is deliberately NOT reduced; the
                         # v.9 failure showed starving yaw is what loses sharp curves.
MAX_ROLL      = 0.10     # REVERTED from 0.30 after flight 2026-08-01: at 0.3 the roll
                         # loop oscillates through camera-tilt feedback (gyro +-0.8 rad/s,
                         # altitude coupling, ended on the floor). 0.1 keeps the limit
                         # cycle bounded. Don't raise again until yaw authority works or
                         # the vision errors are tilt-compensated.
MAX_YAW       = 0.35     # restored to fbb20c80 (primary curve steering). NOTE: measured
                         # full stick is ~1.8-2 rad/s, so this is ~0.7 rad/s true.
IMAGE_CENTER  = 320      # 640-wide image -> center column
LOOKAHEAD_FRAC = 0.4     # split the line into a far (yaw) part and a near (roll) part
LOST_TIME     = 20.0      # give the climb-and-reacquire recovery room to work: one
                         # recovery in the 2026-08-01 log took 3.2 s, right at the
                         # old 3.0 s limit
MAX_FOLLOW    = 300.0    # total-time cap; at the v.21 cruise (~0.05 m/s effective)
                         # 180 s was no longer enough to finish the course
MAX_THROTTLE = 0.3       # real drone: 0.3 * 0.5 m/s = 0.15 m/s vertical authority.
                         # The old 0.05 sat inside the sim's deadband, where the sim
                         # itself held altitude -- on real PX4 nothing does that for us.
KP_ALT = 0.3             # retune at the field; sim value 0.1 was masked by the deadband
TARGET_HEIGHT = 1.0
MIN_FLY_ALT = 0.25       # below this at startup: climb only (one-way latch, see update)
SEARCH_YAW = 0.1         # gentle yaw toward the side the line last exited, while lost
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
_gate_last = None
_airborne = False
_last_roll_dir = 0.0     # sign of the last seen rollErr; 0.0 until the line is first seen

#DEBUG State Vars
_dbg_all_tags = []       # every decoded tag (id, side_px) from the last detection attempt
_tags_seen_prev = []     # decoded ids at the previous attempt, for change events
_fb_prev = "--"          # band-fallback state (N=near empty, F=far empty)
_gate_seen_t = None      # _timer at the last successful gate detection
_gate_px_prev = None     # (t, tag_px) of the previous detection, for closure rate
_gate_close_rate = 0.0   # d(tag_px)/dt, px/s -- positive means approaching the gate
_color_ok_prev = True    # forward-camera availability, for transition events

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
    # field-verified 2026-08-01: bottom 46, left 47, top 48, right 49.
    # (46 was previously mis-mapped as "r" -- gate cy estimates from it were
    # biased a full HALF_H low, which is why approaches sank toward 0.6 m.)
    46: (1, "b"),
    47: (1, "l"),
    48: (1, "t"),
    49: (1, "r"),
    # 1: (1, "b"),  # sim-era ids, not seen on the real course; re-enable for sim runs
    # 2: (1, "l"),
}

SIDE_SLOTS = ("l", "r")

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
        # field-verified: the real gate tags are 5x5 (the old 6x6 dict only
        # marginally decoded them -- hence the flickering detections in logs).
        # 250-id variant in case later gates use ids >= 50.
        _gate_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
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

    

def _prefer_sides(group):
    """Gates differ in height, so t/b/corner tags guess cy through HALF_H.
    Side tags don't (their vertical offset is 0), so trust them alone if present."""
    sides = [t for t in group if t[1] in SIDE_SLOTS]
    return sides if sides else group


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
    """Locate a gate from its DICT_5X5_250 tags. Each visible tag predicts the gate
    center on its own; the predictions are averaged. One tag is enough, two agree."""
    global _dbg_all_tags
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tags = _read_tags(gray)
    # (id, cx, cy, side_px): positions let a log determine an unmapped tag's slot
    _dbg_all_tags = [(tid, int(center[0]), int(center[1]), round(side, 1))
                     for tid, center, side in tags]
    group = _nearest_gate(tags)
    if not group:
        return None
    group = _prefer_sides(group)

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
    global _yaw_err_dot_filt, _gate_last, _airborne, _last_roll_dir
    global _dbg_all_tags, _tags_seen_prev, _fb_prev, _gate_seen_t, _gate_px_prev, _gate_close_rate, _color_ok_prev
    global _throttle_h, _throttle_l, _throttle_prev
    global _fresh, _recorder
    _log_ver()   # before each step, so every run in a log is attributable
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
    _airborne = False
    _last_roll_dir = 0.0

    _dbg_all_tags = []
    _tags_seen_prev = []
    _fb_prev = "--"
    _gate_seen_t = None
    _gate_px_prev = None
    _gate_close_rate = 0.0
    _color_ok_prev = True

    _throttle_h = 0.0
    _throttle_l = 0.0
    _throttle_prev = 0.0
    
    _fresh = 0
    _recorder = None

def update(drone):
    global _rot_err_int, _rot_prev_err, _filtered_rot_err_dot, _target_col, _hold, _done
    global _alt_err_int, _alt_prev_err, _alt_max_err
    global _timer, _lost, _prev_roll_rate, _prev_yaw_rate, _yawPrev_err, _rollPrev_err, _prevTags
    global _yaw_err_dot_filt, _gate_last, _airborne, _last_roll_dir
    global _dbg_all_tags, _tags_seen_prev, _fb_prev, _gate_seen_t, _gate_px_prev, _gate_close_rate, _color_ok_prev
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

    # One-way airborne latch: until first reaching MIN_FLY_ALT, command climb only.
    # A disarmed-on-ground start otherwise runs the full controller on the floor
    # (27 s of it in the 2026-08-01 log, then it scooted off the line at OFFBOARD
    # engage). One-way on purpose: race gates sit at different heights, so once
    # airborne the drone may legitimately fly below MIN_FLY_ALT again.
    if not _airborne:
        alt_now = neo_lab.height(drone)
        if alt_now < MIN_FLY_ALT:
            drone.flight.send_pcmd(0.0, 0.0, 0.0,
                                   uav_utils.clamp((TARGET_HEIGHT - alt_now) * KP_ALT,
                                                   -MAX_THROTTLE, MAX_THROTTLE))
            return False
        _airborne = True
        _log.info(f"[EVT] t={_timer:.2f} airborne (alt={alt_now:.2f}m), controller active")

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
            _log_ver()   # at the end
        return _done

    mask = neo_lab.bright_mask(img, S_MIN) > 0
    points = np.argwhere(mask)

    if len(points) < MIN_PIXELS:
        # keep closing on TARGET_HEIGHT while lost: a plain stop() deadlocks a low
        # start (footprint too small to see the line, and it stays low because it
        # can't see the line -- flights that began at 0.3 m died here)
        errAlt = TARGET_HEIGHT - neo_lab.height(drone)
        # every real loss in the logs was lateral (line exits the side): once at
        # height, yaw gently toward the side it last exited. Gated on altitude so
        # the low-start climb stays a pure climb (_last_roll_dir is 0.0 until the
        # line has been seen, so a never-seen line also means no blind rotation).
        search_yaw = SEARCH_YAW * _last_roll_dir if abs(errAlt) < 0.2 else 0.0
        if _lost == 0.0:
            _log.info(f"[EVT] t={_timer:.2f} line lost (px={len(points)} < {MIN_PIXELS})"
                      f" search_yaw={search_yaw:+.2f}")
        drone.flight.send_pcmd(0.0, 0.0, search_yaw,
                               uav_utils.clamp(errAlt * KP_ALT, -MAX_THROTTLE, MAX_THROTTLE))
        _lost += dt

        if _lost >= LOST_TIME:
            _done = True
            _log.info(f"[EVT] t={_timer:.2f} line gone for {LOST_TIME}s -> done")
            _log_ver()   # at the end
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
                      f"on an empty band yawErr holds its last value, decaying)")
            _fb_prev = fb
        if n_far == 0:
            far = points
        if n_near == 0:
            near = points
        farCol = far[:,1].mean()
        nearCol = near[:,1].mean()
        near_row = near[:, 0].mean()
        far_row = far[:, 0].mean()

        # fbb20c80 line follow: raw errors into the PIDs, only the yaw derivative
        # is dampened. yaw error = line angle (near -> far centroid), radians;
        # invariant to lateral offset, so offset stays roll's job
        if n_near == 0 or n_far == 0:
            # degenerate band geometry: hold the last valid heading error, decaying,
            # instead of zeroing it -- the v.9 flight lost the line when the far
            # band emptied mid-turn and the old fallback collapsed steering to 0
            yawErr = 0.9 * _yawPrev_err
            _yaw_err_dot_filt = 0.8 * _yaw_err_dot_filt   # decay only, no raw->0 spike
        else:
            yawErr = math.atan2(farCol - nearCol, max(near_row - far_row, 1.0))
            yawErr_derv = (yawErr - _yawPrev_err) / dt if dt > 0 else 0.0
            # clamp: line-geometry flips spike the raw derivative (d=+8.15 in the
            # 2026-08-01 log -> +-1.6 through yawKD, instant saturation); +-1.0
            # caps the D contribution at +-0.2 without touching normal tracking
            _yaw_err_dot_filt = uav_utils.clamp(
                0.8 * _yaw_err_dot_filt + 0.2 * yawErr_derv, -1.0, 1.0)
        _yawPrev_err = yawErr
        rollErr = (nearCol - IMAGE_CENTER) / IMAGE_CENTER
        if rollErr != 0.0:
            _last_roll_dir = 1.0 if rollErr > 0 else -1.0

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
        
        # slow-and-deliberate gate: cut speed for heading error AND lateral offset,
        # down to a full stop (~19 deg misalignment, or ~2/3 image offset, or a
        # mix) -- stop, turn/recentre with full yaw authority, then resume
        pitch_rate = FORWARD_PITCH * max(0.0, 1.0 - 3.0 * abs(yawErr)
                                              - 1.5 * abs(rollErr))

    #HEIGHT
    _fresh += 1
    if _fresh % 5 == 0:
        # cache the result so the frames between detections keep the last gate
        # instead of falling into the gate-lost branch (throttle chatter, _hold reset)
        color_ok = imgColor is not None
        if color_ok != _color_ok_prev:
            _log.info(f"[EVT] t={_timer:.2f} forward camera "
                      + ("frames resumed" if color_ok else
                         "MISSING (get_color_image() is None -> gate detection disabled)"))
            _color_ok_prev = color_ok
        if color_ok:
            _gate_last = detect_gate(imgColor)
        else:
            _gate_last = None
            _dbg_all_tags = []
        # event on any change in what's decoded -- unmapped ids here mean TAG_MAP is wrong
        seen = sorted(tid for tid, *_rest in _dbg_all_tags)
        if seen != _tags_seen_prev:
            mapped = [t for t in seen if t in TAG_MAP]
            unmapped = [t for t in seen if t not in TAG_MAP]
            _log.info(f"[EVT] t={_timer:.2f} tags in view: mapped={mapped} unmapped={unmapped} "
                      f"tags(id,cx,cy,px)={_dbg_all_tags}")
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

        # lateral centering: yaw toward the gate's cx, blended in by proximity.
        # 2026-08-01 flight: span grew to 523 px while cx marched to -148 -- the
        # drone slid past the gate sideways because only the line steered. Roll
        # stays on the line; yaw hands over to the gate as tag_px approaches
        # GATE_NEAR_PX. cx can be far outside the image; the clamp bounds it.
        rot_err = (gate.cx - COL_CENTER) / COL_CENTER
        gate_w = uav_utils.clamp(gate.tag_px / GATE_NEAR_PX, 0.0, 1.0)
        yaw_rate = ((1.0 - gate_w) * yaw_rate
                    + gate_w * uav_utils.clamp(GATE_KP_YAW * rot_err, -MAX_YAW, MAX_YAW))
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
        gyro = drone.physics.get_angular_velocity()    # library Y-up frame: [1] is the
                                                       # YAW rate (verified vs heading
                                                       # sweep in 2026-08-01 logs)
        sat = (("Y" if abs(yaw_pid) > MAX_YAW else "-")
               + ("R" if abs(roll_pid) > MAX_ROLL else "-")
               + ("T" if abs(errAlt * KP_ALT) > MAX_THROTTLE else "-"))
        _log.info(
            f"[LF] t={_timer:6.2f} dt={dt*1000:4.0f}ms alt={neo_lab.height(drone):.2f}"
            f" att=({att[0]:+5.1f},{att[1]:+5.1f},{att[2]:6.1f})deg"
            f" gyro(x,yaw,z)=({gyro[0]:+.2f},{gyro[1]:+.2f},{gyro[2]:+.2f})"
            f" px={len(points)} n/f={n_near}/{n_far} bands={fb}"
            f" nearC={nearCol:5.1f} farC={farCol:5.1f}"
            f" yawE={yawErr:+.3f}(d{_yaw_err_dot_filt:+.3f})"
            f" rollE={rollErr:+.3f}"
            f" | cmd P={pitch_rate:.3f} R={roll_rate:+.3f} Y={yaw_rate:+.3f} T={throttle:+.3f}"
            f" sat={sat}"
        )
        if gate is not None:
            age = _timer - _gate_seen_t if _gate_seen_t is not None else float("nan")
            _log.info(
                f"[GT] t={_timer:6.2f} ids={gate.ids} n={gate.count}"
                f" cx={gate.cx:5.1f} cy={gate.cy:5.1f} tag_px={gate.tag_px:5.1f}"
                f" span={gate.span:6.1f} close={_gate_close_rate:+5.1f}px/s age={age:.2f}s"
                f" rotE={rot_err:+.3f} w={gate_w:.2f}"
                f" altE={alt_err:+.3f} (P={KP_ALT_GATE*alt_err:+.3f}"
                f" I={KI_ALT_GATE*_alt_err_int:+.3f} D={KD_ALT_GATE*alt_err_dot:+.3f})"
                f" thH={_throttle_h:+.3f} thL={_throttle_l:+.3f} hold={_hold:.2f}"
            )
        else:
            # gate-status heartbeat (the original code printed "Gate: None" every
            # frame): confirms at a glance whether the gate is seen right now, and
            # which raw tags the last detection attempt decoded
            _log.info(f"[GT] t={_timer:6.2f} no gate (decoded tags: {_dbg_all_tags})")
    drone.flight.send_pcmd(pitch_rate, roll_rate, yaw_rate, throttle)

    if _timer >= MAX_FOLLOW:
        _done = True
        _log.info(f"[EVT] t={_timer:.2f} MAX_FOLLOW ({MAX_FOLLOW}s) reached -> done")
        _log_ver()   # at the end

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
        _log.info(f"Step 3: Visual Servoing (Vision + PID) -- race_code {CODE_VERSION}")
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

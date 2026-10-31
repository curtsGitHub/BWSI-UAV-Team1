"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo
GNU General Public License v3.0

Minimum Snap Trajectory Calculator and Executor
"""

#Imports
import drone_core
import drone_utils as uav_utils
import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.realpath(__file__))
while _os.path.basename(_d) != "labs" and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
if _d not in _sys.path:
    _sys.path.insert(0, _d)
import neo_lab
import numpy as np

#Constants
MAP_SIZE = 10.0
LAUNCH_HEIGHT = 3.0
START = (0.0, 0.0, 3.0)
GATES = [(0.0, 1.0, 1.0)]  # (forward, right, up) = (Z, X, Y)

def read():
    path = _os.environ.get("MAP_RECORD")
    map = []
    if path is not None:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                map.append([float(row["Z"]),
                    float(row["X"]),
                    float(row["Y"]),
                    float(row["Yaw"])])
                
    return map

def _add_approach(gates, lead=1):
    out = []
    pts = [START] + gates
    for i, g in enumerate(gates):
        prev = np.array(pts[i], dtype=float)
        here = np.array(g, dtype=float)
        nxt  = np.array(gates[i+1], dtype=float) if i < len(gates)-1 else here

        d_in  = here - prev
        d_out = (nxt - here) if i < len(gates)-1 else d_in

        n_in, n_out = np.linalg.norm(d_in), np.linalg.norm(d_out)
        s_in  = d_in / n_in if n_in > 1e-6 else np.zeros(3)
        s_out = d_out / n_out if n_out > 1e-6 else s_in
        d = s_in + s_out
        n = np.linalg.norm(d)
        d = d / n if n > 1e-6 else (s_in if np.linalg.norm(s_in) > 1e-6 else s_out)

        out.append(tuple(here - d*lead))
        out.append(tuple(here))
        out.append(tuple(here + d*lead))
    return out

REAL_GATES = list(GATES)
GATES = _add_approach(GATES, lead=0.8)
WAYPOINTS = [START] + GATES  

KP_POS = 0.9 #Translational
ALT_KP = 1.5 #Vertical
KP_YAW = 2.0 #Rotational
YAW_SPEED_GATE = 0.5 #m/s below which heading is held and not recalculated
MAX_YAW_RATE = 1.0 #Limited by sim
G = 9.81 #Gravity Constant
N_SEG = len(WAYPOINTS) - 1
_LEG = [np.linalg.norm(np.subtract(WAYPOINTS[i+1], WAYPOINTS[i])) for i in range(N_SEG)]

if not np.all(np.isfinite(_LEG)) or min(_LEG) < 1e-6:
    raise ValueError(f"bad leg at index {int(np.argmin(np.nan_to_num(_LEG, nan=-1)))}")

V_CRUISE = 1     #says 3.0 in the secret notes in wsl, but 6.0 is faster for sim
_DURATION = np.array([d / V_CRUISE for d in _LEG])
_ORDER = 8            # coeffs per segment (order-7 poly = minimum-snap Euler-Lagrange solution)

#Functions
def _basis(t, deriv, n=_ORDER):
    """Row [1, t, t^2, ... t^7] differentiated `deriv` times, evaluated at local time t."""
    row = np.zeros(n)
    for i in range(deriv, n):
        c = 1.0
        for j in range(deriv):
            c *= (i - j)    # i! / (i-deriv)!
        row[i] = c * t ** (i - deriv)
    return row

def _build_min_snap(pos, times):
    """Solve one axis. pos = positions at each waypoint, times = arrival time at each.
    Returns (n_seg, 8) coeffs, each segment parameterized in LOCAL time 0..dur.
    Square system (8*m equations, 8*m unknowns): 2m endpoint + 6 rest-to-rest + 6(m-1) continuity.
    """
    m = len(pos) - 1
    duration = [times[k + 1] - times[k] for k in range(m)]
    A = np.zeros((_ORDER * m, _ORDER * m))
    b = np.zeros(_ORDER * m)
    r = 0

    # 1. each segment hits its two waypoints (C0 continuity is automatic from this)
    for i in range(m):
        col = _ORDER * i
        A[r, col:col + _ORDER] = _basis(0.0, 0);       b[r] = pos[i];     r += 1
        A[r, col:col + _ORDER] = _basis(duration[i], 0);    b[r] = pos[i + 1]; r += 1

    # 2. rest-to-rest: vel/acc/jerk = 0 at the very start and very end
    for i in (1, 2, 3):
        A[r, 0:_ORDER] = _basis(0.0, i);                               r += 1
        A[r, _ORDER * (m - 1):_ORDER * m] = _basis(duration[-1], i);        r += 1

    # 3. continuity of derivatives 1..6 across each interior seam
    for i in range(m - 1):
        for d in range(1, 7):
            A[r, _ORDER * i:_ORDER * i + _ORDER] = _basis(duration[i], d)
            A[r, _ORDER * (i + 1):_ORDER * (i + 1) + _ORDER] = -_basis(0.0, d)
            r += 1

    return np.linalg.solve(A, b).reshape(m, _ORDER)

""" Module-level State Variables"""
_t = 0.0
_p = np.array([0.0, 0.0, 0.0])
_max_err = 0.0
_wp = 0
_done = False
_FWD = [w[0] for w in WAYPOINTS]
_RIGHT = [w[1] for w in WAYPOINTS]
_UP = [w[2] for w in WAYPOINTS]
_psi_des = 0.0 #desired heading (psi desired)
_prev_yaw_rate = 0.0
_init = False
_prev_v = None
_peak_a = 0.0
_peak_v = 0.0
_a_hist = []
_started = False

""" Functions Continued """
def _sample(coef, t, deriv=0):
    # Evaluate the `deriv`-th derivative of one axis at global time t.
    i = 0
    while i < N_SEG - 1 and t >= CUMULATIVE[i + 1]:
        i += 1
    tau = min(max(t - CUMULATIVE[i], 0.0), _DURATION[i])
    return _basis(tau, deriv) @ coef[i]

def path_position(t):
    # Position (forward, right) on the trajectory at time t seconds.
    return _sample(_COEF_FWD, t, 0), _sample(_COEF_RIGHT, t, 0)

def trajectory(t):
    # (fwd_pos, right_pos, up_pos, fwd_vel, right_vel, up_vel)
    return (_sample(_COEF_FWD, t, 0), _sample(_COEF_RIGHT, t, 0), _sample(_COEF_UP, t, 0),
            _sample(_COEF_FWD, t, 1), _sample(_COEF_RIGHT, t, 1), _sample(_COEF_UP, t, 1))

def wrap_pi(a):
    return np.arctan2(np.sin(a), np.cos(a)) #Wraps angle to [-pi, pi]

def desired_yaw(vel_f, vel_r, hold):
    if np.hypot(vel_f, vel_r) > YAW_SPEED_GATE:
        return np.arctan2(vel_r, vel_f)
    return hold

def body_to_world(v_bf, v_br, psi):
    """Inverse: body (forward, right) -> world (forward, right)."""
    c, s = np.cos(psi), np.sin(psi)
    return  v_bf * c - v_br * s,  v_bf * s + v_br * c

def world_to_body(v_f, v_r, psi):
    """World (forward, right) -> body (forward, right)
    psi = drone heading measured from world-forward, CCW positive.
    Expressing a world vector in a frame rotated by +psi means rotating
    the vector by -psi."""
    c, s = np.cos(psi), np.sin(psi)
    return  v_f * c + v_r * s, -v_f * s + v_r * c

def reset():
    global _t, _p, _max_err, _wp, _done, _psi_des, _prev_yaw_rate, _init
    global _prev_v, _peak_a, _peak_v, _a_hist
    global _started
    _t = 0.0
    _p = np.array([0.0, 0.0, 0.0])
    _max_err = 0.0
    _wp = 0
    _done = False
    _psi_des = 0.0
    _prev_yaw_rate = 0.0
    _init = False
    _prev_v = None
    _peak_a = 0.0
    _peak_v = 0.0
    _a_hist = []
    _started = False
    
A_MAX = 0.5 #from tilt limit: a_max ~= g*tan(theta_max)

def _rest_segment_time(L, V, A):
    """Min time for a segment starting or ending at rest (trapezoidal
    profile, collapsing to triangular if L is too short to reach V)."""
    t_accel = V / A
    d_accel = 0.5 * V * t_accel
    if d_accel >= L:
        return np.sqrt(2.0 * L / A)
    return t_accel + (L - d_accel) / V

def _rebuild():
    global CUMULATIVE, TOTAL_TIME, _COEF_FWD, _COEF_RIGHT, _COEF_UP
    CUMULATIVE = np.concatenate([[0.0], np.cumsum(_DURATION)])
    TOTAL_TIME = CUMULATIVE[-1]
    _COEF_FWD   = _build_min_snap(_FWD,   CUMULATIVE)
    _COEF_RIGHT = _build_min_snap(_RIGHT, CUMULATIVE)
    _COEF_UP    = _build_min_snap(_UP,    CUMULATIVE)

_DURATION[0]  = max(_DURATION[0],  _rest_segment_time(_LEG[0],  V_CRUISE, A_MAX))
_DURATION[-1] = max(_DURATION[-1], _rest_segment_time(_LEG[-1], V_CRUISE, A_MAX))
_BASE_DURATION = _DURATION.copy()
_rebuild()

_DURATION[0]  = max(_DURATION[0],  _rest_segment_time(_LEG[0],  V_CRUISE, A_MAX))
_DURATION[-1] = max(_DURATION[-1], _rest_segment_time(_LEG[-1], V_CRUISE, A_MAX))
_rebuild()

for _ in range(3):
    _samples = np.linspace(0.0, TOTAL_TIME, 1500)
    PEAK_A = max(np.linalg.norm([_sample(_COEF_FWD, t, 2),
                                 _sample(_COEF_RIGHT, t, 2),
                                 _sample(_COEF_UP, t, 2)]) for t in _samples)
    _DURATION = _DURATION * max(1.0, np.sqrt(PEAK_A / A_MAX))
    _rebuild()

print(f"PEAK_A={PEAK_A:.2f}, A_MAX={A_MAX}, TOTAL_TIME={TOTAL_TIME:.2f}")
print(f"durations: {np.round(_DURATION, 2)}")

def update(drone):
    global _t, _p, _max_err, _wp, _done, _psi_des, _prev_yaw_rate, _init
    global _prev_v, _peak_a, _peak_v, _a_hist
    global _started
    if _done:
        return True
    
    # Controller
    if _started == False:
        neo_lab.send_velocity(drone, 0, 0, 0)
        if drone.controller.was_pressed(drone.controller.Button.A):
            _started = True

    if _started:
        dt = drone.get_delta_time()
        _t += dt

        pos_f, pos_r, pos_u, vel_f, vel_r, vel_u = trajectory(_t)
        h = neo_lab.height(drone)

        att = drone.physics.get_attitude()
        psi = np.radians(att[2])
        _psi_des = desired_yaw(vel_f, vel_r, _psi_des) # <<< uses FF velocity, not corrected v_forward
        yaw_err = wrap_pi(_psi_des - psi) 
        yaw_rate_raw = KP_YAW * yaw_err
        yaw_cmd = uav_utils.clamp(0.5 * _prev_yaw_rate + 0.5 * yaw_rate_raw, -MAX_YAW_RATE, MAX_YAW_RATE)
        _prev_yaw_rate = yaw_cmd

        vz, vx, vy = np.asarray(drone.physics.get_linear_velocity())[[2, 0, 1]]
        _v_now = np.asarray(drone.physics.get_linear_velocity())
        _peak_v = max(_peak_v, np.hypot(_v_now[0], _v_now[2]))
        if _prev_v is not None and dt > 0:
            _a = (_v_now - _prev_v) / dt
            _a_hist.append(np.hypot(_a[0], _a[2]))
            if len(_a_hist) > 10:
                _a_hist.pop(0)
            _peak_a = max(_peak_a, sum(_a_hist) / len(_a_hist))
        _prev_v = _v_now
        vz_world, vx_world = body_to_world(vz, vx, psi)
        _p[0] += vz_world * dt
        _p[1] += vx_world * dt
        _p[2] += vy * dt

        v_forward = vel_f  + KP_POS * (pos_f - _p[0])
        v_right   = vel_r + KP_POS * (pos_r - _p[1])
        v_up = vel_u + ALT_KP * (pos_u - h)                

        v_body_forward, v_body_right = world_to_body(v_forward, v_right, psi) 

        neo_lab.send_velocity(drone, v_body_right, v_up, v_body_forward, yaw_cmd)

        # Checks
        err = np.linalg.norm([pos_f - _p[0], pos_r - _p[1], pos_u - h])
        _max_err = max(_max_err, err)
        reached = sum(_t >= c for c in CUMULATIVE[1:])

        #print(f"{_t:.2f},{pos_u:.3f},{h:.3f},{vel_u:.3f},{v_up:.3f},{att[0]:.1f},{att[1]:.1f},{att[2]:.1f}")


        if reached > _wp:
            _wp = reached
            #Report Real Coords
            print(f"Estimated Coordinates: ({_p[0]:.3f}, {_p[1]:.3f}, {h})")
            #print(f"Real Coordinates: ({z_r:.3f}, {x_r:.3f}, {y_r:.3f})")
            print(f"Target Coordinates: ({WAYPOINTS[_wp][0]}, {WAYPOINTS[_wp][1]}, {WAYPOINTS[_wp][2]})")
            print(f"Reached Waypoint: {_wp}/{N_SEG}")
            print(f"Max Current Error: {_max_err:.3f} m")
            print("")
        if _t >= TOTAL_TIME:
            drone.flight.stop()
            print(f"Total Time: {_t}")
            print(f"Finished Tour: ({_p[1]:.3f}, {_p[0]:.3f}), "
                f"Max Tracking Error: {_max_err:.3f} m")
            print(f"Peak horizontal speed: {_peak_v:.2f} m/s")
            print(f"Peak horizontal accel (10-frame avg): {_peak_a:.2f} m/s^2")
            _done = True
        return _done

# if __name__ == "__main__":
#     _drone = drone_core.create_drone()
#     _launcher = neo_lab.Launcher(LAUNCH_HEIGHT)

#     def start():
#         _launcher.reset()
#         reset()
#         print("Step 2: Fly a Smooth Course (minimum snap)")

#     def _update():
#         if not _launcher.done:
#             _launcher.update(_drone)
#             return
#         if update(_drone):
#             _drone.flight.land()

#     _drone.set_start_update(start, _update)
#     _drone.go()

    
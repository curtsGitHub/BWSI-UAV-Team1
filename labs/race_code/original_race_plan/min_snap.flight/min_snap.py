"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo
GNU General Public License v3.0

Geometric Spline Trajectory (arc-length parameterized) + Curvature Speed Profile

Pipeline:
  1. gates -> approach waypoints (unchanged)
  2. spline r(u) solved in a pure geometry parameter u (centripetal:
     du = chord**0.5). Shape depends ONLY on the waypoints - no timing
     enters the solve, so there is nothing to compress or blow up.
  3. dense sample the spline: arc length s(u), curvature kappa(s)
  4. speed profile: v_cap = min(V_MAX, sqrt(A_MAX/kappa)), then
     accel-limited forward/backward passes with v = 0 at both ends.
     Rest-to-rest lives HERE now, not in the spline.
  5. time map t(s) by integrating ds/v. The executor samples r and
     r'*u_dot at time t exactly like before.
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
#GATES = [(1.0*MAP_SIZE, 0.0, 3.0), (2.0*MAP_SIZE, 1.0*MAP_SIZE, 3.0), (1.0*MAP_SIZE, 2.0*MAP_SIZE, 3.0), (0.0*MAP_SIZE, 2.0*MAP_SIZE, 3.0), (0.0, 0.0, 3.0)]  # (forward, right, up) = (Z, X, Y)
"""
GATES = [
        (4,4,1),
        (8,0,3),
        (12,4,5),
        (16,0,7),
        (20,4,5),
        (24,0,3),

        (28,6,1),
        (24,10,3),
        (20,15,5),
        (12,12,7),

        (5,15,5),
        (0,10,3),

        (0,0,1)
]
"""

""" #Super Hard
GATES = [
    ( 3.5,  0.0, 2.5),   # 1  launch, straight
    ( 7.0, -6.0, 5.8),   # 2  hard left, 3.3 m climb in 6.9 m
    (10.5, -7.0, 1.3),   # 3  4.5 m dive, floor-scraping
    (14.0, -2.5, 5.2),   # 4  3.9 m pop-up mid-turn
    (17.5, -6.5, 1.5),   # 5  reversal + dive
    (21.5, -3.0, 5.5),   # 6  climb toward far corner
    (25.5,  1.0, 6.0),   # 7  U-turn apex, ceiling
    (22.0,  5.0, 5.8),   # 8  STACK upper - level entry off the turn
    (22.0,  5.0, 1.6),   # 9  STACK lower - 4.2 m straight down
    (17.0,  3.0, 4.8),   # 10 climb back out, switchback left
    (17.0, -2.0, 1.4),   # 11 switchback right + dive, same forward as 10
    (12.0,  4.0, 5.5),   # 12 diagonal climb, crosses 3-4 well above it
    ( 8.0,  6.5, 1.2),   # 13 lowest gate
    ( 4.5,  2.0, 4.5),   # 14 final climb
    ( 1.5, -2.5, 2.0),   # 15 last corner
    ( 0.0,  0.0, 1.5),   # 16 finish over the pad
]
"""

# GATES = [(1,0,1), (1,1,1), (0,1,1), (0,0,1)]
GATES = [(1,1,1), (0,0,1)]

# GATES = [(0,1,1), (1,1,1), (1,0,1), (0,0,1)]

def _add_approach(gates, lead=2):
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
#GATES = _add_approach(GATES, lead=1)   # stays until per-gate tangent constraints replace it
WAYPOINTS = [START] + GATES  

KP_POS = 0.9 #Translational
ALT_KP = 1.5 #Vertical
KP_YAW = 4.0 #Rotational
YAW_SPEED_GATE = 0.5 #m/s below which heading is held and not recalculated
MAX_YAW_RATE = 1.0 #Limited by sim
G = 9.81 #Gravity Constant
N_SEG = len(WAYPOINTS) - 1
_LEG = [np.linalg.norm(np.subtract(WAYPOINTS[i+1], WAYPOINTS[i])) for i in range(N_SEG)]

if not np.all(np.isfinite(_LEG)) or min(_LEG) < 1e-6:
    raise ValueError(f"bad leg at index {int(np.argmin(np.nan_to_num(_LEG, nan=-1)))}")

A_MAX = 1.4  
V_MAX = 1.0  
AGGRO = 1.0 

_DU = np.sqrt(np.asarray(_LEG))
U_KNOTS = np.concatenate([[0.0], np.cumsum(_DU)])
U_TOTAL = U_KNOTS[-1]

_ORDER = 8          

#Functions
def _basis(u, deriv, n=_ORDER):
    """Row [1, u, u^2, ... u^7] differentiated `deriv` times, evaluated at local parameter u."""
    row = np.zeros(n)
    for i in range(deriv, n):
        c = 1.0
        for j in range(deriv):
            c *= (i - j)    # i! / (i-deriv)!
        row[i] = c * u ** (i - deriv)
    return row

def _build_spline(pos, d_start, d_end):
    m = N_SEG
    A = np.zeros((_ORDER * m, _ORDER * m))
    b = np.zeros(_ORDER * m)
    r = 0

    for i in range(m):
        col = _ORDER * i
        A[r, col:col + _ORDER] = _basis(0.0, 0);       b[r] = pos[i];     r += 1
        A[r, col:col + _ORDER] = _basis(_DU[i], 0);    b[r] = pos[i + 1]; r += 1

    A[r, 0:_ORDER] = _basis(0.0, 1);                          b[r] = d_start; r += 1
    A[r, _ORDER * (m - 1):_ORDER * m] = _basis(_DU[-1], 1);   b[r] = d_end;   r += 1
    for i in (2, 3):
        A[r, 0:_ORDER] = _basis(0.0, i);                               r += 1
        A[r, _ORDER * (m - 1):_ORDER * m] = _basis(_DU[-1], i);        r += 1

    for i in range(m - 1):
        for d in range(1, 7):
            A[r, _ORDER * i:_ORDER * i + _ORDER] = _basis(_DU[i], d)
            A[r, _ORDER * (i + 1):_ORDER * (i + 1) + _ORDER] = -_basis(0.0, d)
            r += 1

    return np.linalg.solve(A, b).reshape(m, _ORDER)

def _sample(coef, u, deriv=0):
    i = 0
    while i < N_SEG - 1 and u >= U_KNOTS[i + 1]:
        i += 1
    tau = min(max(u - U_KNOTS[i], 0.0), _DU[i])
    return _basis(tau, deriv) @ coef[i]

_FWD = [w[0] for w in WAYPOINTS]
_RIGHT = [w[1] for w in WAYPOINTS]
_UP = [w[2] for w in WAYPOINTS]

_P_FIRST = np.asarray(WAYPOINTS[1], float) - np.asarray(WAYPOINTS[0], float)
_P_LAST  = np.asarray(WAYPOINTS[-1], float) - np.asarray(WAYPOINTS[-2], float)
_TAN_START = _P_FIRST / _DU[0]    # dr/du ~ chord/du, so |dr/du| matches local ds/du
_TAN_END   = _P_LAST / _DU[-1]

_COEF_FWD   = _build_spline(_FWD,   _TAN_START[0], _TAN_END[0])
_COEF_RIGHT = _build_spline(_RIGHT, _TAN_START[1], _TAN_END[1])
_COEF_UP    = _build_spline(_UP,    _TAN_START[2], _TAN_END[2])

_N_GRID = 3000
U_GRID = np.linspace(0.0, U_TOTAL, _N_GRID)

def _grid_eval(deriv):
    return np.array([[_sample(_COEF_FWD, u, deriv),
                      _sample(_COEF_RIGHT, u, deriv),
                      _sample(_COEF_UP, u, deriv)] for u in U_GRID])

_R1 = _grid_eval(1)                        # dr/du
_R2 = _grid_eval(2)                        # d2r/du2
_DS_DU = np.linalg.norm(_R1, axis=1)       # ds/du
S_GRID = np.concatenate([[0.0],
        np.cumsum(0.5 * (_DS_DU[1:] + _DS_DU[:-1]) * np.diff(U_GRID))])
KAPPA = np.linalg.norm(np.cross(_R1, _R2), axis=1) / np.maximum(_DS_DU, 1e-9) ** 3

_V_CAP = np.minimum(V_MAX, np.sqrt(A_MAX / np.maximum(KAPPA, 1e-9)))
_DS = np.diff(S_GRID)

V_PROF = _V_CAP.copy()
V_PROF[0] = 0.0
for _i in range(_N_GRID - 1):              
    _a_lat = KAPPA[_i] * V_PROF[_i] ** 2
    _a_lon = np.sqrt(max(A_MAX ** 2 - _a_lat ** 2, 0.0))
    V_PROF[_i + 1] = min(V_PROF[_i + 1], np.sqrt(V_PROF[_i] ** 2 + 2.0 * _a_lon * _DS[_i]))
V_PROF[-1] = 0.0
for _i in range(_N_GRID - 2, -1, -1):    
    _a_lat = KAPPA[_i + 1] * V_PROF[_i + 1] ** 2
    _a_lon = np.sqrt(max(A_MAX ** 2 - _a_lat ** 2, 0.0))
    V_PROF[_i] = min(V_PROF[_i], np.sqrt(V_PROF[_i + 1] ** 2 + 2.0 * _a_lon * _DS[_i]))
V_PROF = V_PROF * AGGRO                 

_DT = 2.0 * _DS / np.maximum(V_PROF[:-1] + V_PROF[1:], 1e-6)
T_GRID = np.concatenate([[0.0], np.cumsum(_DT)])
TOTAL_TIME = T_GRID[-1]

WP_TIME = np.interp(np.interp(U_KNOTS, U_GRID, S_GRID), S_GRID, T_GRID)

_A_LON_PROF = np.diff(V_PROF ** 2) / (2.0 * np.maximum(_DS, 1e-9))
_V_MID = 0.5 * (V_PROF[:-1] + V_PROF[1:])
_K_MID = 0.5 * (KAPPA[:-1] + KAPPA[1:])
PLAN_PEAK_A = float(np.max(np.hypot(_A_LON_PROF, _K_MID * _V_MID ** 2)))

print(f"course length={S_GRID[-1]:.2f} m, TOTAL_TIME={TOTAL_TIME:.2f} s")
print(f"plan: peak |a|={PLAN_PEAK_A:.2f} (budget {A_MAX}), peak v={V_PROF.max():.2f} (cap {V_MAX}), "
      f"min corner v={V_PROF[5:-5].min():.2f}")

""" Module-level State Variables"""
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

""" Functions Continued """
def path_position(t):
    pos_f, pos_r, *_rest = trajectory(t)
    return pos_f, pos_r

def trajectory(t):
    t = min(max(t, 0.0), TOTAL_TIME)
    s = np.interp(t, T_GRID, S_GRID)
    u = np.interp(s, S_GRID, U_GRID)
    v = np.interp(t, T_GRID, V_PROF)
    pos_f = _sample(_COEF_FWD, u, 0)
    pos_r = _sample(_COEF_RIGHT, u, 0)
    pos_u = _sample(_COEF_UP, u, 0)
    d = np.array([_sample(_COEF_FWD, u, 1),
                  _sample(_COEF_RIGHT, u, 1),
                  _sample(_COEF_UP, u, 1)])
    u_dot = v / max(np.linalg.norm(d), 1e-9)
    return pos_f, pos_r, pos_u, d[0] * u_dot, d[1] * u_dot, d[2] * u_dot

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

        vz, vx, vy = np.asarray(drone.physics.get_linear_velocity())[[0, 2, 1]]
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
        # _p[0] += vz_world * dt    # forward
        # _p[1] += vx_world * dt    # right
        # _p[2] += vy * dt    # up

        _p[0] = drone.physics.get_position()[2]
        _p[1] = drone.physics.get_position()[0]
        _p[2] = neo_lab.height(drone)

        v_forward = vel_f  + KP_POS * (pos_f - _p[0])
        v_right   = vel_r + KP_POS * (pos_r - _p[1])
        v_up = vel_u + ALT_KP * (pos_u - h)                

        v_body_forward, v_body_right = world_to_body(v_forward, v_right, psi) 

        neo_lab.send_velocity(drone, v_body_right, 0, v_body_forward, yaw_cmd)

        # Checks
        err = np.linalg.norm([pos_f - _p[0], pos_r - _p[1], pos_u - h])
        _max_err = max(_max_err, err)
        reached = sum(_t >= c for c in WP_TIME[1:])
        print(f"err:{err}")
        # print(f"{_t:.2f},{pos_u:.3f},{h:.3f},{vel_u:.3f},{v_up:.3f},{att[0]:.1f},{att[1]:.1f},{att[2]:.1f}")

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
#         print("Step 2: Fly a Smooth Course (geometric spline + speed profile)")

#     def _update():
#         if not _launcher.done:
#             _launcher.update(_drone)
#             return
#         if update(_drone):
#             _drone.flight.land()

#     _drone.set_start_update(start, _update)
#     _drone.go()
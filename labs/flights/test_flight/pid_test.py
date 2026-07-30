"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo
GNU General Public License v3.0
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
START = (0.0, 0.0, 3.0, 0.0)
GATES = [(1,0,1,0), (1,1,1,0), (0,1,1,0), (0,0,1,0)]  # (forward, right, up, rotation) = (Z, X, Y, R)

GATES = [(0,1,1,0), (1,1,1,0), (1,0,1,0), (0,0,1,0)] 

REAL_GATES = list(GATES)
WAYPOINTS = [START] + GATES
N_SEG = len(WAYPOINTS) - 1

KP_POS = 0.3 #Translational
ALT_KP = 1.5 #Vertical
KP_YAW = 2.0 #Rotational
MAX_YAW_RATE = 1.0 #Limited by sim

#Waypoint-follower knobs
V_MAX = 3.0 #clamp on the horizontal P output. step setpoints demand KP_POS*leg on the first frame after a switch
HIT_RADIUS = 0.25 #3D proximity gate on the estimate, m
TIMEOUT = 60.0 #wall clock for the whole course, s. run fails past this
YAW_FROM_WAYPOINT = True #True: hold each waypoint's rotation. False: face the direction of travel
YAW_ERR_GATE = 0.5 #m of remaining error below which heading is held, YAW_FROM_WAYPOINT=False only
TELEMETRY = True
_started = False
if len(set(len(w) for w in WAYPOINTS)) != 1:
    raise ValueError("waypoints must all be (forward, right, up, rotation)")

def wrap_pi(a):
    return np.arctan2(np.sin(a), np.cos(a)) #Wraps angle to [-pi, pi]

def desired_yaw(target, err_f, err_r, hold):
    """Commanded heading in rad. No trajectory dependence.

    YAW_FROM_WAYPOINT: face the rotation stored on the waypoint (deg in,
    rad out), so the drone squares up to each gate.
    Otherwise: face the direction of travel, which is now the direction of
    the remaining position error, held once inside YAW_ERR_GATE where that
    direction goes noisy.
    """
    if YAW_FROM_WAYPOINT:
        return wrap_pi(np.radians(target[3]))
    if np.hypot(err_f, err_r) > YAW_ERR_GATE:
        return np.arctan2(err_r, err_f)
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
    global _t, _p, _max_err, _wp, _done, _psi_des, _prev_yaw_rate
    global _prev_v, _peak_a, _peak_v, _a_hist
    global _started
    _t = 0.0
    _p = np.array([0.0, 0.0, 0.0])
    _max_err = 0.0
    _wp = 1                       # index 0 is START, we are already there
    _done = False
    _psi_des = np.radians(START[3])
    _prev_yaw_rate = 0.0
    _prev_v = None
    _peak_a = 0.0
    _peak_v = 0.0
    _a_hist = []
    _started = False

def _finish(drone, ok, h):
    global _done
    drone.flight.stop()
    if ok:
        print("Course complete")
    else:
        print(f"FAILED: timeout parked on waypoint {_wp}/{N_SEG}")
    print(f"Total Time: {_t:.2f}")
    print(f"Finished Tour: ({_p[1]:.3f}, {_p[0]:.3f}, {h:.3f}), "
          f"Max Tracking Error: {_max_err:.3f} m")
    print(f"Peak horizontal speed: {_peak_v:.2f} m/s")
    print(f"Peak horizontal accel (10-frame avg): {_peak_a:.2f} m/s^2")
    _done = True

def update(drone):
    global _t, _p, _max_err, _wp, _done, _psi_des, _prev_yaw_rate
    global _prev_v, _peak_a, _peak_v, _a_hist
    global _started
    if _done:
        return True

    if _started == False:
            neo_lab.send_velocity(drone, 0, 0, 0)
            if drone.controller.was_pressed(drone.controller.Button.A):
                _started = True

    if _started:
        
        dt = drone.get_delta_time()
        _t += dt

        h = neo_lab.height(drone)
        att = drone.physics.get_attitude()
        psi = np.radians(att[2])

        # State estimate
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
        _p[0] += vz_world * dt    # forward
        _p[1] += vx_world * dt    # right
        _p[2] += vy * dt          # up, reported only, control uses measured h

        # Error to the current target
        target = WAYPOINTS[_wp]
        err_f = target[0] - _p[0]
        err_r = target[1] - _p[1]
        err_u = target[2] - h
        err = np.linalg.norm([err_f, err_r, 0])
        _max_err = max(_max_err, err)

        # Yaw
        _psi_des = desired_yaw(target, err_f, err_r, _psi_des)
        yaw_err = wrap_pi(_psi_des - psi)
        yaw_rate_raw = KP_YAW * yaw_err
        yaw_cmd = uav_utils.clamp(0.5 * _prev_yaw_rate + 0.5 * yaw_rate_raw,
                                -MAX_YAW_RATE, MAX_YAW_RATE)
        _prev_yaw_rate = yaw_cmd

        # Position P
        v_forward = KP_POS * err_f
        v_right   = KP_POS * err_r
        speed = np.hypot(v_forward, v_right)
        if speed > V_MAX:
            v_forward *= V_MAX / speed
            v_right   *= V_MAX / speed
        v_up = ALT_KP * err_u

        v_body_forward, v_body_right = world_to_body(v_forward, v_right, psi)

        neo_lab.send_velocity(drone, v_body_right, v_up, v_body_forward, yaw_cmd)

        if TELEMETRY:
            print(f"{_t:.2f},{target[2]:.3f},{h:.3f},{v_up:.3f},"
                f"{err:.3f},{att[0]:.1f},{att[1]:.1f},{att[2]:.1f},"
                f"{np.degrees(_psi_des):.1f}")

        # Checks
        if err < HIT_RADIUS:
            print("")
            print(f"Reached Waypoint: {_wp}/{N_SEG} at t={_t:.2f}")
            print(f"Estimated Coordinates: ({_p[0]:.3f}, {_p[1]:.3f}, {h:.3f})")
            print(f"Target Coordinates: ({target[0]}, {target[1]}, {target[2]})")
            print(f"Max Current Error: {_max_err:.3f} m")
            print("")
            _wp += 1
            if _wp > N_SEG:
                _finish(drone, True, h)
                return _done

        if _t >= TIMEOUT:
            _finish(drone, False, h)
        return _done


# if __name__ == "__main__":
#     _drone = drone_core.create_drone()
#     _launcher = neo_lab.Launcher(LAUNCH_HEIGHT)

#     def start():
#         _launcher.reset()
#         reset()
#         print("Step 2: Fly a Smooth Course (bare PID waypoint follower)")

#     def _update():
#         if not _launcher.done:
#             _launcher.update(_drone)
#             return
#         if update(_drone):
#             _drone.flight.land()

#     _drone.set_start_update(start, _update)
#     _drone.go()
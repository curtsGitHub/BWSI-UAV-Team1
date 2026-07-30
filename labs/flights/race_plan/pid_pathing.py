"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo
GNU General Public License v3.0

Bare PID waypoint follower - no trajectory generator.

Baseline for measuring closed-loop tracking accuracy on its own, before
layering the geometric spline + speed profile back on top.

Pipeline:
  1. target = WAYPOINTS[_wp], a step setpoint. No feedforward anywhere.
  2. horizontal: P on dead-reckoned position error, command clamped to V_MAX
  3. vertical:   P on measured height error
  4. yaw:        commanded from the waypoint's own rotation field, then
                 P + smoothing + rate limit. Nothing here reads a
                 trajectory or a velocity setpoint.
  5. progression: 3D proximity gate on the estimate. Run FAILS on timeout
                 rather than hovering forever or crediting a miss.
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
import csv

def read():
    path = _os.environ.get("MAP_RECORD")
    map = []
    if path is not None:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                map.append([(float(row["Z"]),
                    float(row["X"]),
                    float(row["Y"])),
                    float(row["Yaw"])]
                )
    return map

#Constants
LAUNCH_HEIGHT = 3.0
WAYPOINTS = read() # [(1,0,1,0)] #(0, 0, LAUNCH_HEIGHT, 0)] #, (2,0,1,0), (2,2,1,0), (0,2,1,0), (0,0,1,0)]  # (forward, right, up, rotation) = (Z, X, Y, R)
print(f"WAYPOINTS:{WAYPOINTS}")
KP_POS = 0.3 #Translational
KI_POS = 0
KD_POS = 0
MAX_PITCH_RATE = 1
MAX_ROLL_RATE = 1

KP_ALT = .4 #Vertical
KI_ALT = 0.1
KD_ALT = 0.1

KP_YAW = 2.0 #Rotational
KI_YAW = 0.1
KD_YAW = 0.1
MAX_YAW_RATE = 1.0 #Limited by sim

#Waypoint-follower knobs
V_MAX = 3.0 
TOLERANCE = 0.25
YAW_FROM_WAYPOINT = True #True: hold each waypoint's rotation. False: face the direction of travel
YAW_ERR_GATE = 0.5 #m of remaining error below which heading is held, YAW_FROM_WAYPOINT=False only
TELEMETRY = True
_started = False

def wrap_pi(a):
    return np.arctan2(np.sin(a), np.cos(a)) #Wraps angle to [-pi, pi]

def desired_yaw(target, err_f, err_r, hold):
    if YAW_FROM_WAYPOINT:
        return wrap_pi(np.radians(target[3]))
    if np.hypot(err_f, err_r) > YAW_ERR_GATE:
        return np.arctan2(err_r, err_f)
    return hold

def body_to_world(v_bf, v_br, psi):
    c, s = np.cos(psi), np.sin(psi)
    return  v_bf * c - v_br * s,  v_bf * s + v_br * c

def world_to_body(v_f, v_r, psi):
    c, s = np.cos(psi), np.sin(psi)
    return  v_f * c + v_r * s, -v_f * s + v_r * c

def reset():
    global _t, _p, _max_err, _wp, _done, _psi_des, _prev_yaw_rate
    global _started
    _t = 0.0
    _p = np.array([0.0, 0.0, 0.0])
    _max_err = 0.0
    _wp = 0                      # index 0 is START, we are already there
    _done = False
    _psi_des = np.radians(WAYPOINTS[0][3])
    _prev_yaw_rate = 0.0
    _started = False

def update(drone):
    global _t, _p, _max_err, _wp, _done, _psi_des, _prev_yaw_rate
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

        h = neo_lab.height(drone)
        att = drone.physics.get_attitude()
        psi = np.radians(att[2])

        # State estimate
        vz, vx, vy = np.asarray(drone.physics.get_linear_velocity())[[0, 2, 1]]
        vz_world, vx_world = body_to_world(vz, vx, psi)
        # _p[0] += vz_world * dt    # forward
        # _p[1] += vx_world * dt    # right
        # _p[2] += vy * dt          # up, reported only, control uses measured h
        _p[0] = drone.physics.get_position()[2]
        _p[1] = drone.physics.get_position()[0]
        _p[2] = neo_lab.height(drone)
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
        v_up = KP_ALT * err_u

        v_body_forward, v_body_right = world_to_body(v_forward, v_right, psi)

        neo_lab.send_velocity(drone, v_body_right, v_up, v_body_forward, yaw_cmd)
        print(f"Real Height:{neo_lab.height(drone)} Estimated Height:{_p[2]} Target:{WAYPOINTS[0]}")
        if TELEMETRY:
            print(f"{_t:.2f},{target[2]:.3f},{h:.3f},{v_up:.3f},"
                f"{err:.3f},{att[0]:.1f},{att[1]:.1f},{att[2]:.1f},"
                f"{np.degrees(_psi_des):.1f}")

        # Checks
        if err < TOLERANCE:
            print("")
            print(f"Reached Waypoint: {_wp}/{len(WAYPOINTS)} at t={_t:.2f}")
            print(f"Estimated Coordinates: ({_p[0]:.3f}, {_p[1]:.3f}, {h:.3f})")
            print(f"Target Coordinates: ({target[0]}, {target[1]}, {target[2]})")
            print(f"Max Current Error: {_max_err:.3f} m")
            print("")
            _wp += 1
    if _wp >= len(WAYPOINTS):
                print("Finished Tour")
                return _done
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
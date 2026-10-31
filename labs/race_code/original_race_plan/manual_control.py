import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.realpath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
_d = _here
while _os.path.basename(_d) != "labs" and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
if _d not in _sys.path:
    _sys.path.insert(0, _d)
import neo_lab

from controller import Controller

def reset():
    global _posZ, _posY, _posX, _recorder
    _recorder = None
    _posZ = 0.0
    _posY = 0.0
    _posX = 0.0
    
def update(drone):
    global _posZ, _posY, _posX
    dt = drone.get_delta_time()
    vx, vy, vz = drone.physics.get_linear_velocity()
    # _posZ += vz * dt
    # _posY += vy * dt
    # _posX += vx * dt
    _posX, _posY, _posZ = drone.physics.get_position() # (x_east, y_up, z_north)
    # joyRX, joyRY = drone.controller.get_joystick(drone.controller.Joystick.RIGHT)
    # joyLX, joyLY = drone.controller.get_joystick(drone.controller.Joystick.LEFT)
    # pitch = joyRY * 0.6
    # roll = joyRX * 0.5
    # yaw = joyLX * 0.6
    # throttle = joyLY * 0.5
    # drone.flight.send_pcmd(pitch, roll, yaw, throttle)
    
    if drone.controller.was_pressed(drone.controller.Button.A):
        record(_posZ, _posX, _posY, drone.physics.get_attitude()[2], "gate")
        print(f"Recorded point! self calculated (z,y,x):{(_posZ, _posX, _posY)}")

    if drone.controller.was_pressed(drone.controller.Button.B):
        record(_posZ, _posX, _posY, drone.physics.get_attitude()[2], "waypoint")
        print(f"Recorded point! self calculated (z,y,x):{(_posZ, _posX, _posY)}")

def record(z, x, y, yaw, waypointType):
    global _recorder
    path = _os.environ.get("MAP_RECORD")
    if path is None:
        return
    
    if _recorder is None:
        _recorder = neo_lab.Recorder(path)
    row = {"Z":z, "X":x, "Y":y, "Yaw":yaw, "Type":waypointType}
    _recorder.log(**row)

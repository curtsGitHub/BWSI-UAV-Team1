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

import labs.race_code.race_plan.manual_control as manual_control

neo_lab.run_module("ManualFlight", [("ManualFlight", manual_control)], launch_height=0, autostart=True, led_color=(255, 255, 255), launch=False)



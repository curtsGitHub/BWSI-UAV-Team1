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

import pid_test

neo_lab.run_module("PIDtest", [("PIDtest", pid_test)], launch_height=1.0, autostart=True, led_color=(255, 255, 255), launch=False)



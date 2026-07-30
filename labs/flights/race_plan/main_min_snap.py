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

import min_snap

neo_lab.run_module("MinSnap", [("MinSnap", min_snap)], launch_height=1, autostart=True, led_color=(255, 255, 255), launch=False)



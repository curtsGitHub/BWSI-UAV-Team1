"""
Copyright MIT
GNU General Public License v3.0

MIT BWSI Autonomous Drone Racing Course - UAV Neo

File Name: led_sim.py
File Description: Simulator LED module. The simulator has no LED strip, so the
calls are no-ops - this keeps flight code that uses the strip portable.
"""

from led import Led


class LedSim(Led):
    def fill(self, red: int, green: int, blue: int) -> None:
        pass

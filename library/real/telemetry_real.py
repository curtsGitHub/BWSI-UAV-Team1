"""
Copyright MIT
GNU General Public License v3.0

MIT BWSI Autonomous Drone Racing Course - UAV Neo

File Name: telemetry_real.py
File Description: Contains the Telemetry module of the drone_core library
"""

import os
import time

import pandas as pd
from matplotlib import pyplot as plt
from telemetry import Telemetry, _resolve_log_paths

# ROS2
import rclpy as ros2
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from diagnostic_msgs.msg import DiagnosticArray


class TelemetryReal(Telemetry):

    def __init__(self):
        self.variable_names = None
        self.log_file = None
        self.start_time = None
        self._LOG_FILE_NAME = None
        self._PLOT_FILE_NAME = None

        # Diagnostics subscription
        self.node = ros2.create_node("telemetry_sub")
        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT
        self.__diag_sub = self.node.create_subscription(
            DiagnosticArray, "/diagnostics",
            self.__diagnostics_callback, qos_profile,
        )
        self.__diagnostics = {}

    def __diagnostics_callback(self, msg):
        """Store the latest diagnostics keyed by status name."""
        for status in msg.status:
            entry = {
                "level": int.from_bytes(status.level, byteorder='big'),
                "message": status.message,
                "hardware_id": status.hardware_id,
            }
            for kv in status.values:
                entry[kv.key] = kv.value
            self.__diagnostics[status.name] = entry

    def get_diagnostics(self) -> dict:
        """
        Returns the latest diagnostics from the /diagnostics topic.

        Returns:
            A dict keyed by diagnostic name, each value is a dict of
            key-value pairs including level, message, and hardware_id.

        Example::

            diag = uav.telemetry.get_diagnostics()
            if "EdgeTPU Inference" in diag:
                print(diag["EdgeTPU Inference"]["avg_inference_ms"])
        """
        return dict(self.__diagnostics)

    def declare_variables(self, *names: str) -> None:
        if self.variable_names is None:
            self.variable_names = names
            self.start_time = time.time()
            self._LOG_FILE_NAME, self._PLOT_FILE_NAME = _resolve_log_paths()
            self.log_file = open(self._LOG_FILE_NAME, "w+")
            header = ["time", *names]
            print(','.join(map(str, header)), file=self.log_file)

    def record(self, *values: any) -> None:
        assert self.variable_names is not None, (
            "declare_variables() must be called before record()"
        )
        assert (
            len(self.variable_names) == len(values)
        ), f"{len(self.variable_names)} variables were declared, but {len(values)} values were provided"

        time_since_start = time.time() - self.start_time
        row = [time_since_start, *values]
        print(','.join(map(str, row)), file=self.log_file)

    def visualize(self) -> None:
        if self.variable_names is None or self.log_file is None:
            return

        # Seek to beginning of file before reading,
        # then seek to the end afterwards in case we want to write more data
        self.log_file.seek(0)
        frame = pd.read_csv(self.log_file)
        frame["time"] *= 1000

        fig, ax = plt.subplots()
        ax.set_xlabel("Time (ms)")
        ax.set_title("Telemetry Variables vs. Time")
        for variable in frame.columns:
            if variable != "time":
                ax.plot(frame["time"], frame[variable], label=variable)
        ax.legend()

        plt.savefig(self._PLOT_FILE_NAME)
        self.log_file.seek(0, 2)

    def __update(self):
        pass

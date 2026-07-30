"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Manages communication with the UAVNeo Simulator.
"""

import struct
import socket
import sys
import select
import time
from enum import IntEnum
from signal import signal, SIGINT
from typing import Callable, Optional

import camera_sim
import controller_sim
import display_sim
import flight_sim
import led_sim
import physics_sim
import telemetry_sim

from drone_core import Drone
import drone_utils as uav_utils


class DroneSim(Drone):
    # Resolve the simulator IP: use DRONE_SIM_IP env var if set,
    # otherwise auto-detect the WSL2 Windows host gateway, falling back to localhost.
    @staticmethod
    def __resolve_sim_ip() -> str:
        import os
        env_ip = os.environ.get("DRONE_SIM_IP") or os.environ.get("RACECAR_SIM_IP")
        if env_ip:
            return env_ip
        # Check if we are running in WSL2
        is_wsl2 = False
        try:
            with open("/proc/version") as f:
                is_wsl2 = "microsoft" in f.read().lower()
        except OSError:
            pass
        if not is_wsl2:
            return "127.0.0.1"
        # In WSL2 NAT mode the Windows host is the default gateway
        # (stored little-endian in /proc/net/route).
        # In WSL2 mirrored mode the default gateway is the real router,
        # and localhost (127.0.0.1) reaches the Windows host directly.
        try:
            with open("/proc/net/route") as f:
                for line in f:
                    fields = line.strip().split()
                    if fields[1] == "00000000":  # default route
                        hex_ip = fields[2]
                        ip = ".".join(
                            str(int(hex_ip[i:i+2], 16))
                            for i in range(6, -1, -2)
                        )
                        # Hyper-V NAT gateways live in 172.16.0.0/12
                        first_octet = int(hex_ip[6:8], 16)
                        second_octet = int(hex_ip[4:6], 16)
                        if first_octet == 172 and 16 <= second_octet <= 31:
                            return ip  # NAT mode — gateway is the Windows host
                        # Gateway is outside Hyper-V range — likely mirrored mode
                        return "127.0.0.1"
        except (OSError, IndexError, ValueError):
            pass
        return "127.0.0.1"

    __IP = __resolve_sim_ip.__func__()
    __UNITY_PORT = (__IP, 5065)
    __UNITY_ASYNC_PORT = (__IP, 5064)
    __VERSION = 1

    class Header(IntEnum):
        """
        The packet headers of the communication protocol with the UAVNeo Simulator.
        """

        error = 0
        connect = 1
        unity_start = 2
        unity_update = 3
        unity_exit = 4
        python_finished = 5
        python_send_next = 6
        python_exit = 7
        drone_go = 8
        drone_set_start_update = 9
        drone_get_delta_time = 10
        drone_set_update_slow_time = 11
        camera_get_color_image = 12
        camera_get_depth_image = 13
        camera_get_width = 14
        camera_get_height = 15
        controller_is_down = 16
        controller_was_pressed = 17
        controller_was_released = 18
        controller_get_trigger = 19
        controller_get_joystick = 20
        display_show_image = 21
        # 22-26 reserved (removed: drive and lidar)
        physics_get_linear_acceleration = 27
        physics_get_angular_velocity = 28
        flight_send_pcmd = 29
        flight_stop = 30
        flight_set_max_speed = 31
        physics_get_linear_velocity = 32
        camera_get_downward_image = 33
        physics_get_altitude = 34
        physics_get_attitude = 35
        physics_get_gps = 36
        flight_takeoff = 37
        flight_land = 38
        physics_get_position = 39

    class Error(IntEnum):
        """
        The error codes defined in the communication protocol.
        """

        generic = 0
        timeout = 1
        python_exception = 2
        no_free_car = 3
        python_outdated = 4
        sim_outdated = 5
        fragment_mismatch = 6

    def __send_header(self, function_code: Header, is_async: bool = False) -> None:
        self.__send_data(struct.pack("B", function_code.value), is_async)

    def __send_error(self, error: Error, is_async: bool = False) -> None:
        self.__send_data(struct.pack("BB", self.Header.error, error), is_async)

    def __send_data(self, data: bytes, is_async: bool = False) -> None:
        if is_async:
            self.__socket.sendto(data, self.__UNITY_ASYNC_PORT)
        else:
            self.__socket.sendto(data, self.__UNITY_PORT)

    # Whole-frame deadline for fragmented reads. Must satisfy:
    #   normal_frame_latency < _FRAME_TIMEOUT_S < sim_watchdog_timeout
    # Picked so that a lost UDP fragment fails the current frame in time for
    # python_finished to still reach the sim before its watchdog tears the
    # session down.
    _FRAME_TIMEOUT_S = 0.2

    def __receive_data(
        self, buffer_size: int = 8, timeout: Optional[float] = None
    ) -> Optional[bytes]:
        if timeout is not None:
            ready, _, _ = select.select([self.__socket], [], [], timeout)
            if not ready:
                return None
        data, _ = self.__socket.recvfrom(buffer_size)
        return data

    def __receive_fragmented(
        self, num_fragments: int, total_bytes: int, is_async: bool = False
    ) -> Optional[bytes]:
        raw_bytes = bytearray()
        fragment_size = total_bytes // num_fragments
        deadline = time.monotonic() + self._FRAME_TIMEOUT_S
        received = 0
        for _ in range(num_fragments):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            chunk = self.__receive_data(fragment_size, timeout=remaining)
            if chunk is None:
                break
            raw_bytes += chunk
            self.__send_header(self.Header.python_send_next, is_async)
            received += 1
        if received < num_fragments:
            # Fragment lost / timed out. The sim is mid-send and waiting on
            # our remaining acks before it returns to idle. If we don't send
            # them, the sim stays stuck *and* later delivers fragments that
            # the next protocol call (e.g. get_delta_time) misreads as its
            # own response. Send the remaining acks so the sim finishes its
            # send loop, then drain everything it dumps back.
            for _ in range(num_fragments - received):
                self.__send_header(self.Header.python_send_next, is_async)
            self.__drain_socket_quiet()
            return None
        return bytes(raw_bytes)

    def __drain_socket_quiet(self, quiet_window: float = 0.05) -> None:
        """Drain until the socket has been silent for ``quiet_window`` seconds.

        Bounded by ``_FRAME_TIMEOUT_S`` so a misbehaving sim can't stall us
        indefinitely.
        """
        deadline = time.monotonic() + self._FRAME_TIMEOUT_S
        while time.monotonic() < deadline:
            ready, _, _ = select.select([self.__socket], [], [], quiet_window)
            if not ready:
                return
            try:
                self.__socket.recvfrom(65535)
            except OSError:
                return

    def __init__(self, isHeadless: bool = False) -> None:
        self.camera = camera_sim.CameraSim(self)
        self.controller = controller_sim.ControllerSim(self)
        self.display = display_sim.DisplaySim(isHeadless)
        self.flight = flight_sim.FlightSim(self)
        self.led = led_sim.LedSim()
        self.physics = physics_sim.PhysicsSim(self)
        self.telemetry = telemetry_sim.TelemetrySim()

        self.__start: Callable[[], None]
        self.__update: Callable[[], None]
        self.__update_slow: Optional[Callable[[], None]]
        self.__update_slow_time: float = 1
        self.__update_slow_counter: float = 0
        self.__delta_time: float = -1

        self.__socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.__in_call: bool = False

        signal(SIGINT, self.__handle_sigint)

    def go(self, autostart: bool = False) -> None:
        # autostart is accepted for parity with the real drone but ignored: the
        # simulator starts user program mode on its own ENTER-key signal.
        print(">> Python script loaded, awaiting connection from UAVNeo Simulator.")

        # Repeatedly try to connect (async) until we receive a response
        while True:
            self.__send_data(
                struct.pack("BB", self.Header.connect, self.__VERSION), True
            )
            ready = select.select([self.__socket], [], [], 0.25)
            if ready[0]:
                data, _ = self.__socket.recvfrom(2)
                header = int(data[0])
                if header == self.Header.connect.value:
                    drone_index = int(data[1])
                    uav_utils.print_colored(
                        f">> Connection established with UAVNeo Simulator (assigned to drone {drone_index}). Enter user program mode to begin...",
                        uav_utils.TerminalColor.green,
                    )
                    break
                elif header == self.Header.error.value:
                    self.__handle_error(int(data[1]))
                else:
                    uav_utils.print_error(
                        ">> Invalid handshake with simulator, closing script..."
                    )
                    self.__send_header(self.Header.error)
                    return

        # Respond to start/update commands (sync) until we receive exit or error
        while True:
            data, _ = self.__socket.recvfrom(8)
            header = int(data[0])

            if header == self.Header.unity_start.value:
                try:
                    self.__in_call = True
                    self.set_update_slow_time()
                    self.__start()
                    self.__in_call = False
                except SystemExit:
                    raise
                except:
                    self.__send_error(self.Error.python_exception)
                    raise
            elif header == self.Header.unity_update.value:
                try:
                    self.__in_call = True
                    self.__handle_update()
                    self.__in_call = False
                except SystemExit:
                    raise
                except:
                    self.__send_error(self.Error.python_exception)
                    raise
            elif header == self.Header.unity_exit.value:
                uav_utils.print_warning(
                    ">> Exit command received from simulator, closing script..."
                )
                break
            elif header == self.Header.error:
                error = int(data[1]) if len(data) > 1 else self.Error.generic
                self.__handle_error(error)
            else:
                uav_utils.print_error(
                    f">> Error: unexpected packet with header [{header}] received from simulator, closing script..."
                )
                self.__send_header(self.Header.error)
                break

            self.__send_header(self.Header.python_finished)

        self.telemetry.visualize()

    def set_start_update(
        self,
        start: Callable[[], None],
        update: Callable[[], None],
        update_slow: Optional[Callable[[], None]] = None,
    ) -> None:
        self.__start = start
        self.__update = update
        self.__update_slow = update_slow

    def get_delta_time(self) -> float:
        if self.__delta_time < 0:
            self.__send_header(self.Header.drone_get_delta_time)
            [value] = struct.unpack("f", self.__receive_data())
            self.__delta_time = value
        return self.__delta_time

    def set_update_slow_time(self, update_slow_time: float = 1.0) -> None:
        self.__update_slow_time = update_slow_time

    def __handle_update(self) -> None:
        self.__update()

        self.__delta_time = -1
        if self.__update_slow is not None:
            self.__update_slow_counter -= self.get_delta_time()
            if self.__update_slow_counter < 0:
                self.__update_slow()
                self.__update_slow_counter = self.__update_slow_time

        self.camera._CameraSim__update()
        self.controller._ControllerSim__update()

    def __handle_sigint(self, signal_received: int, frame) -> None:
        is_async = not self.__in_call
        label = "async" if is_async else "sync"

        uav_utils.print_warning(
            f">> CTRL-C (SIGINT) detected. Sending exit command to Unity ({label})..."
        )
        self.__send_header(self.Header.python_exit, is_async)

        print(">> Closing script...")
        exit(0)

    def __handle_error(self, error: Error):
        text = ">> Error: "
        if error == self.Error.generic:
            text += "An unknown error has occurred when communicating with the simulator."
        elif error == self.Error.timeout:
            text = "The Python script took too long to respond. If this issue persists, make sure that your script does not block execution."
        elif error == self.Error.no_free_car:
            text += "Unable to connect because every drone already has a connected Python script."
        elif error == self.Error.python_outdated:
            text = "drone_core is out of date and incompatible with the simulator. Please update your Python drone libraries to the newest version."
        elif error == self.Error.sim_outdated:
            text = "The simulator is out of date and incompatible with drone_core. Please download the newest version."
        elif error == self.Error.fragment_mismatch:
            text = "The simulator and Python became out of sync while sending a block message."

        uav_utils.print_error(text)
        print(">> Closing script...")
        exit(0)

"""
Copyright MIT
GNU General Public License v3.0

MIT BWSI Autonomous Drone Racing Course - UAV Neo

File Name: physics.py
File Description: Defines the interface of the Physics module of the drone_core library.
"""

import abc
import numpy as np
class NDArray:  # stub — no runtime dependency on nptyping
    def __class_getitem__(cls, _): return cls


class Physics(abc.ABC):
    """
    Returns IMU, velocity, altitude, attitude, and GPS data from the drone's sensors.
    """

    @abc.abstractmethod
    def get_linear_acceleration(self) -> NDArray[3, np.float32]:
        """
        Returns the drone's linear acceleration from the IMU.

        Returns:
            A 3-element array of acceleration along (x, y, z) in m/s^2.

        Note:
            x-axis points right, y-axis points up, z-axis points forward.

        Example::

            accel = uav.physics.get_linear_acceleration()
            forward_accel = accel[2]
        """
        pass

    @abc.abstractmethod
    def get_linear_velocity(self) -> NDArray[3, np.float32]:
        """
        Returns the drone's linear velocity.

        Returns:
            A 3-element array of velocity along (x, y, z) in m/s.

        Note:
            x-axis points right, y-axis points up, z-axis points forward.

        Example::

            vel = uav.physics.get_linear_velocity()
            forward_speed = vel[2]
        """
        pass

    @abc.abstractmethod
    def get_angular_velocity(self) -> NDArray[3, np.float32]:
        """
        Returns the drone's angular velocity from the IMU.

        Returns:
            A 3-element array of angular velocity along (x, y, z) in rad/s.

        Note:
            x-axis is pitch, y-axis is yaw, z-axis is roll.
            Uses right-hand rule for sign convention.

        Example::

            ang_vel = uav.physics.get_angular_velocity()
            yaw_rate = ang_vel[1]
        """
        pass

    @abc.abstractmethod
    def get_altitude(self) -> float:
        """
        Returns the drone's altitude above ground level.

        Returns:
            Altitude in meters.

        Example::

            alt = uav.physics.get_altitude()
            if alt < 2.0:
                print("Low altitude warning")
        """
        pass

    @abc.abstractmethod
    def get_attitude(self) -> NDArray[3, np.float32]:
        """
        Returns the drone's attitude (orientation) as Euler angles.

        Returns:
            A 3-element array of (pitch, roll, yaw) in degrees.
            Pitch: nose up is positive, range [-180, 180].
            Roll: right wing down is positive, range [-180, 180].
            Yaw: clockwise from north, range [0, 360).

        Example::

            attitude = uav.physics.get_attitude()
            pitch, roll, yaw = attitude[0], attitude[1], attitude[2]
        """
        pass

    @abc.abstractmethod
    def get_gps(self) -> NDArray[3, np.float32]:
        """
        Returns the drone's GPS coordinates.

        Returns:
            A 3-element array of (latitude, longitude, altitude) where
            latitude and longitude are in degrees, and altitude is in
            meters above sea level.

        Example::

            gps = uav.physics.get_gps()
            lat, lon, alt_msl = gps[0], gps[1], gps[2]
        """
        pass

    def get_position(self) -> NDArray[3, np.float32]:
        """
        Returns the drone's true world position (simulation only).

        Returns:
            A 3-element array (x_east, y_up, z_north) in meters.

        Note:
            Available in simulation only; the real drone has no ground-truth position.

        Example::

            pos = uav.physics.get_position()
            forward = pos[2]
        """
        raise NotImplementedError("get_position is available in simulation only")

    @abc.abstractmethod
    def zero_position_here(self):
        pass

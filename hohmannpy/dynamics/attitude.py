from typing import TYPE_CHECKING

from . import quaternions

import numpy as np

if TYPE_CHECKING:
    from ..astro import perturbations
    from .. import spacecraft


class Orientation:
    def __init__(
            self,
            quaternion: quaternions.Quaternion,
            angular_velocity: np.ndarray,
            track_euler: bool = False,
            *,
            _default: bool = True
    ):
        self.angular_velocity = angular_velocity
        self._quaternion = quaternion / quaternions.quaternion_norm(quaternion)
        self._track_euler = track_euler

        if _default:
            if self._track_euler:
                self.update_euler()

    @property
    def quaternion(self) -> quaternions.Quaternion:
        return self._quaternion

    @quaternion.setter
    def quaternion(self, value: quaternions.Quaternion):
        self._quaternion = value / quaternions.quaternion_norm(value)

    @classmethod
    def from_quaternion(
            cls,
            quaternion: quaternions.Quaternion,
            angular_velocity: np.ndarray,
            track_euler: bool = False
    ) -> Orientation:
        return cls(quaternion, angular_velocity, track_euler, _default=True)

    # TODO: These methods.
    @classmethod
    def from_euler(cls, roll: float, pitch: float, yaw: float, angular_velocity: np.ndarray):
        pass

    @classmethod
    def from_dcm(cls, dcm: np.ndarray, angular_velocity: np.ndarray):
        pass

    def _update_roll(self):
        self.roll = np.arctan2(
            2 * (self.quaternion[0] * self.quaternion[1] + self.quaternion[2] * self.quaternion[3]),
            1 - 2 * (self.quaternion[1] ** 2 + self.quaternion[2] ** 2)
        )

    def _update_pitch(self):
        self.pitch = -np.pi / 2 + np.arctan2(
            np.sqrt(1 + 2 * (self.quaternion[0] * self.quaternion[2] - self.quaternion[1] * self.quaternion[3])),
            np.sqrt(1 - 2 * (self.quaternion[0] * self.quaternion[2] - self.quaternion[1] * self.quaternion[3]))
        )

    def _update_yaw(self):
        self.yaw = np.arctan2(
            2 * (self.quaternion[0] * self.quaternion[3] + self.quaternion[1] * self.quaternion[2]),
            1 - 2 * (self.quaternion[2] ** 2 + self.quaternion[3] ** 2)
        )

    def _update_roll_rate(self):
        self.roll_rate = (
                self.angular_velocity[0]
                    + np.sin(self.roll) * np.tan(self.pitch) * self.angular_velocity[1]
                    + np.cos(self.roll) * np.tan(self.pitch) * self.angular_velocity[2]
        )

    def _update_pitch_rate(self):
        self.pitch_rate = np.cos(self.roll) * self.angular_velocity[1] - np.sin(self.roll) * self.angular_velocity[2]

    def _update_yaw_rate(self):
        self.yaw_rate = (
                np.sin(self.roll) / np.cos(self.pitch) * self.angular_velocity[1]
                + np.cos(self.roll) / np.cos(self.pitch) * self.angular_velocity[2]
        )

    def update_euler(self):
        self._update_roll()
        self._update_pitch()
        self._update_yaw()
        self._update_roll_rate()
        self._update_pitch_rate()
        self._update_yaw_rate()


# REMINDER:
#   (x, y, z)        = (y0, y1, y2)      Positions
#   (vx, vy, vz)     = (y3, y4, y5)      Velocities
#   (q0, q1, q2, q3) = (y6, y7, y8, y9)  Quaternions
#   (wx, wy, wz)     = (y10, y11, y12)   Angular velocities

def attitude_eom(t: float, y: np.ndarray) -> tuple[float, float, float, float]:
    attitude = quaternions.Quaternion(y[6:10])
    angular_vel = quaternions.Quaternion(np.concatenate((np.array([0]), y[10:]), -1))
    attitude_dot = 0.5 * attitude * angular_vel

    return attitude_dot[0], attitude_dot[1], attitude_dot[2], attitude_dot[3]

def rates_eom(
        t: float,
        y: np.ndarray,
        satellite: spacecraft.Satellite,
        perturbing_torques: list[perturbations.Perturbation],
        **kwargs
) -> tuple[float, float, float]:
    angular_vel = np.array([[y[10]], [y[11]], [y[12]]])
    angular_vel_skew = np.array(
        [
            [0, -y[12], y[11]],
            [y[12], 0, -y[10]],
            [-y[11], y[10], 0]
        ]
    )

    net_torque = np.zeros((3, 1))
    if perturbing_torques is not None:
        for torque in perturbing_torques:
            net_torque += torque.evaluate(t, y, satellite)

    omega_dot = np.linalg.inv(satellite.inertia) @ (net_torque - angular_vel_skew @ satellite.inertia @ angular_vel)

    return omega_dot[0, 0], omega_dot[1, 0], omega_dot[2, 0]

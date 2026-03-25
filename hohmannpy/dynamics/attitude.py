from typing import TYPE_CHECKING

from . import quaternions

import numpy as np

if TYPE_CHECKING:
    from ..astro import spacecraft, perturbations


# REMINDER:
#   (x, y, z)        = (y0, y1, y2)      Positions
#   (vx, vy, vz)     = (y3, y4, y5)      Velocities
#   (q0, q1, q2, q3) = (y6, y7, y8, y9)  Quaternions
#   (wx, wy, wz)     = (y10, y11, y12)   Angular velocities

def attitude_eom(t: float, y: np.ndarray) -> tuple[float, float, float, float]:
    attitude = quaternions.Quaternion(y[6:10])
    angular_vel = quaternions.Quaternion(np.vstack((0, y[10:])))
    attitude_dot = 0.5 * attitude * angular_vel

    return attitude_dot[0], attitude_dot[1], attitude_dot[2], attitude_dot[3]

def rates_eom(
        t: float,
        y: np.ndarray,
        satellite: spacecraft.Satellite,
        perturbing_torques: list[perturbations.Perturbation],
        **kwargs
) -> tuple[float, float, float]:
    angular_vel = np.array([[y[10]], y[11], y[12]])
    angular_vel_skew = np.array(
        [
            [0, -y[12], y[11]],
            [y[12], 0, -y[10]],
            [-y[11], y[10], 0]
        ]
    )

    net_torque = 0
    if perturbing_torques is not None:
        for torque in perturbing_torques:
            net_torque += torque.evaluate(t, y, satellite)

    omega_dot = np.linalg.inv(satellite.inertia) @ (net_torque - angular_vel_skew @ satellite.inertia @ angular_vel)

    return omega_dot[0], omega_dot[1], omega_dot[2]

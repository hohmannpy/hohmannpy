from typing import TYPE_CHECKING

import numpy as np

from ..dynamics import dcms

if TYPE_CHECKING:
    from . import spacecraft


# TODO: Documentation.
class Groundtrack:
    def __init__(self, satellite: spacecraft.Satellite, initial_gmst, solver_tol=1e-8):
        self.initial_gmst = initial_gmst
        self.latitude_history = np.zeros([1, satellite.time_history.size])
        self.longitude_history = np.zeros([1, satellite.time_history.size])

        earth_rot = 7.292115e-5  # Mean rotation rate of the Earth in rad/s.
        for i in range(satellite.time_history.size):
            gmst = self.initial_gmst + earth_rot * satellite.time_history[0, i]

            position = satellite.position_history[:, i]
            position = dcms.euler_2_dcm(gmst, 3) @ position

            earth_radius = 6378.1363e3
            earth_eccentricity = 0.081819221456

            # The geodetic latitude can be found as a function of the satellite's current position, however this
            # function is transcendental in latitude and hence must be solved numerically. Fixed-point iteration is
            # used.
            x = np.arctan2(position[2], np.sqrt(position[0] ** 2 + position[1] ** 2))  # Initial guess.
            x_old = 100  # Dummy value to ensure error is initially above tolerance.

            while abs(x - x_old) > solver_tol:  # Fixed point iteration.
                x_old = x
                radius_of_curvature = earth_radius / np.sqrt((1 - earth_eccentricity ** 2 * np.sin(x) ** 2))
                x = np.arctan2(
                    position[2] + radius_of_curvature * earth_eccentricity ** 2 * np.sin(x),
                    np.sqrt(position[0] ** 2 + position[1] ** 2)
                )
            self.latitude_history[0, i] = x

            longitude = np.arctan2(position[1], position[0])
            if longitude < 0:
                longitude += 2 * np.pi
            self.longitude_history[0, i] = longitude
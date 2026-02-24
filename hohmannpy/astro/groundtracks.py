from typing import TYPE_CHECKING

import numpy as np

from ..dynamics import dcms

if TYPE_CHECKING:
    from . import spacecraft


class Groundtrack:
    r"""
    The groundtrack of a :class:`~hohmannpy.astro.Satellite` projected onto an ellipsoidal Earth.

    These are stored as geodetic latitude and longitude history arrays. Note that for this class to work, the
    ``satellite`` must already have had its orbit propagated via :class:`~hohmannpy.astro.Propagator` or one of its
    child classes. In addition, while the GMST of the Earth is initially computed accurately it is then assumed to
    evolve linear at the Earth's mean rotation rate (ignoring nutation).

    Parameters
    ----------
    satellite : :class:`~hohmannpy.astro.Satellite`
        Satellite whose ``orbit`` should be used to generate a groundtrack.
    initial_gmst : float
        GMST of the Earth at the satellite's initial position.
    solver_tol : float
        Root-finding is used to compute the geodetic latitude of the satellite, this is the error tolerance.

    Attributes
    ----------
    initial_gmst : float
        GMST of the Earth at the satellite's initial position.
    latitude_history : np.ndarray
        The geodetic latitude of the satellite at each timestep stored in its ``time_history`` attribute.
    longitude_history : np.ndarray
        The longitude of the satellite at each timestep stored in its ``time_history`` attribute.
    """

    def __init__(self, satellite: spacecraft.Satellite, initial_gmst: float, solver_tol: float = 1e-8):
        self.initial_gmst = initial_gmst
        self.latitude_history = np.zeros([1, satellite.time_history.size])
        self.longitude_history = np.zeros([1, satellite.time_history.size])

        earth_rot = 7.292115e-5  # Mean rotation rate of the Earth in rad/s.

        # For each discrete time in the satellite's time_history array calculate the corresponding latitude and
        # longitude using its position.
        for i in range(satellite.time_history.size):
            gmst = self.initial_gmst + earth_rot * satellite.time_history[0, i]

            try:
                position = satellite.position_history[:, i]
            except AttributeError:  # Safeguard to make sure propagation has occurred first.
                raise AttributeError("The satellite must have its orbit propagated before generating a groundtrack.")

            # Convert position from ECI to ECEF frame.
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

            # Calculate the longitude.
            longitude = np.arctan2(position[1], position[0])
            self.longitude_history[0, i] = longitude
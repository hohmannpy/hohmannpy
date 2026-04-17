from __future__ import annotations
import importlib.resources
from typing import TYPE_CHECKING

import numpy as np
import scipy as sp

from ..dynamics import conversions
from . import base

if TYPE_CHECKING:
    from .. import spacecraft


class NonSphericalEarth(base.Perturbation):
    r"""
    Perturbation caused by the deviations of the Earth's math distribution from a point-mass.

    The zonal, tesseral, and sectorial geopotential coefficients are taken from the 1985 Earth Gravitational Model
    (EGM84).

    Parameters
    ----------
    degree : int
        Maximum degree of harmonics to include.
    zonal : bool
        Disable sectoral and tesseral harmonics to only look at zonal ones (such as J2). Does this by capping the
        maximum order summed to when computing the acceleration terms to 0.

    Notes
    -----
    The following assumptions are made for this implementation:

    1) The gravitational potential field of the Earth is given by the solution to the geopotential partial-differential equation.

    2) The first-order zonal harmonic is not needed since point-mass gravity is implemented by the propagator.

    3) The GMST of the Earth is initially accurately computed wrt. the Vernal equinox (ignoring nutation) and is then said to linearly rotate at the Earth's mean rotation rate without precession.
    """

    def __init__(self, degree: int, zonal: bool = False):
        super().__init__()

        self._degree = degree
        self._zonal = zonal
        self._initial_gmst = None

        # Import the harmonic coefficients.
        with importlib.resources.files("hohmannpy.resources").joinpath("egm84_c_coeffs.csv").open() as f:
            self._c_coeffs = np.loadtxt(f, delimiter=",")  # n rows, m columns, from [0, 180]

        with importlib.resources.files("hohmannpy.resources").joinpath("egm84_s_coeffs.csv").open() as f:
            self._s_coeffs = np.loadtxt(f, delimiter=",")  # n rows, m columns, from [0, 180]

        # Setup finished in finalize__init__() which is called by the Mission.
        self._initial_gmst = None

    def _finalize__init__(self, initial_gmst: float):
        """
        Record the initial GMST of the Earth which is used to correctly orient it for geopotential modeling.

        This is needed by :meth:`evaluate()` but can't be passed to the base ``__init__()``. This is called during
        :class:`~hohmannpy.Mission`'s instantiation.

        Parameters
        ----------
        initial_gmst : float
            Initial angle of the Greenwich meridian in :math:`rad`.
        """

        self._initial_gmst = initial_gmst

    def evaluate(self, time: float, state: np.ndarray, satellite: spacecraft.Satellite) -> np.ndarray:
        r"""
        Computes the perturbing acceleration using a geopotential model of the Earth's gravitational field.

        Parameters
        ----------
        time : float
            Current time in seconds since propagation began.
        state : np.ndarray
            Current translational state in ECI coordinates given as (position, velocity) or
            (position, velocity, quaternion, angular_velocity).
        satellite : :class:`~hohmannpy.astro.Satellite`
            Unused parameter simply based due to ``@abstractmethod`` requirements.

        Returns
        -------
        acceleration : np.ndarray
            Current translational acceleration in ECI coordinates.
        """

        earth_radius = 6378137
        grav_param = 3.986004418e14
        earth_rot = 7.292115e-5  # Mean rotation rate of the Earth in rad/s.

        # Compute the colatitude and longitude.
        radius = np.sqrt(state[0] ** 2 + state[1] ** 2 + state[2] ** 2)
        colatitude, longitude = self._compute_colat_and_long(time, state[:3])

        # Compute the needed Legendre functions and their derivatives.
        legendre_funcs = sp.special.assoc_legendre_p_all(self._degree, self._degree, np.cos(colatitude), diff_n=1)

        # Compute the acceleration in curvilinear coordinates. Since the potential field and hence acceleration is an
        # infinite series in order and degree, iterate through both of these for all three components of the
        # acceleration.
        radial_accel = 0
        longitudinal_accel = 0
        colatitudinal_accel = 0

        for n in range(2, self._degree + 1):  # Degree 0 is point-mass, degree 1 is always 0, so skip.
            if self._zonal:  # Custom order limiter to allow for only inspecting zonal harmonics.
                m_lim = 1
            else:
                m_lim = n + 1
            for m in range(0, m_lim):
                radial_accel += (
                    -(n + 1) * grav_param / radius ** 2 * (earth_radius / radius) ** n
                        * (-1) ** m * legendre_funcs[0, n, m]
                        * (self._c_coeffs[n, m] * np.cos(m * longitude) + self._s_coeffs[n, m] * np.sin(m * longitude))
                )
                longitudinal_accel += (
                    1 / (radius ** 2 * np.sin(colatitude))
                        * grav_param * (earth_radius / radius) ** n
                        * (-1) ** m * legendre_funcs[0, n, m]
                        * m
                        * (self._c_coeffs[n, m] * -np.sin(m * longitude) + self._s_coeffs[n, m] * np.cos(m * longitude))
                )
                colatitudinal_accel += (
                    -1 / radius ** 2
                        * grav_param * (earth_radius / radius) ** n
                        * (-1) ** m * legendre_funcs[1, n, m] * np.sin(colatitude)
                        * (self._c_coeffs[n, m] * np.cos(m * longitude) + self._s_coeffs[n, m] * np.sin(m * longitude))
                )

        # Use a DCM to convert back to rectilinear coordinates.
        curvilinear_accel = np.array([colatitudinal_accel, longitudinal_accel, radial_accel])
        curvilinear_2_rectilinear = conversions.euler_2_dcm(longitude, 3).T @ conversions.euler_2_dcm(colatitude, 2).T
        acceleration = curvilinear_2_rectilinear @ curvilinear_accel

        # Acceleration is still fixed to the Earth, need to now convert to an inertial basis.
        gmst = self._initial_gmst + earth_rot * time
        earth_2_inertial_dcm = conversions.euler_2_dcm(gmst, 3).T
        acceleration = earth_2_inertial_dcm @ acceleration

        if len(state) == 3:
            return acceleration
        else:
            return np.concatenate((acceleration, np.zeros(3)), axis=-1)

    def _compute_colat_and_long(self, time, position):
        r"""
        Computes the colatitude and longitude of the satellite wrt. the Greenwich meridian from the ECI position and
        time.

        Parameters
        ----------
        time : float
            Current time in seconds since propagation began.
        position : np.ndarray
            Current ECI position.

        Returns
        -------
        colatitude : float
            Angle between the Earth-centered Earth-fixed (ECEF) 3-axis and position vector.
        longitude : float
            Angle between the ECEF 1-axis and the projection of the position vector in to the 1-2 ECEF plane.
        """

        earth_rot = 7.292115e-5  # Mean rotation rate of the Earth in rad/s.

        # Update GMST using simplified precession-free rotation of the Earth.
        gmst = self._initial_gmst + earth_rot * time

        # Transform position to the Earth-centered-Earth-fixed frame.
        inertial_2_earth_dcm = conversions.euler_2_dcm(gmst, 3)
        position = inertial_2_earth_dcm @ position

        # Compute longitude and colatitude.
        longitude = np.arctan2(position[1], position[0])
        colatitude = np.pi / 2 - np.arctan2(position[2], np.sqrt(position[0] ** 2 + position[1] ** 2))

        return colatitude, longitude


class J2(base.Perturbation):
    r"""
    Perturbation caused by Earth's equatorial bulge, known as the J2 effect.

    This is a simplified version of :class:`~hohmannpy.astro.perturbations.NonSphericalEarth` intended for us in
    modeling purely the J2 effect. The J2-acceleration is computed explicitly using general perturbation theory.

    Attributes
    ----------
    _initial_gmst : float
        Initial angle of the Greenwich meridian in :math:`rad` when propagation began.

    Notes
    -----
    The following assumptions are made for this implementation:

    1) The gravitational potential field of the Earth is given by the solution to the geopotential partial-differential equation.

    2) Only the J2 zonal harmonic is considered.

    3) The GMST of the Earth is initially accurately computed wrt. the Vernal equinox (ignoring nutation) and is then said to linearly rotate at the Earth's mean rotation rate without precession.

    See Also
    --------
    :class:`~hohmannpy.astro.NonSphericalEarth` : Generalized version of this perturbation which can model
        N-order zonal harmonic effects as well as tesseral and sectoral ones.
    """

    def __init__(self):
        super().__init__()

        self._initial_gmst = None

    def _finalize__init__(self, initial_gmst: float):
        """
        Record the initial GMST of the Earth which is used to correctly orient it for geopotential modeling.

        This is needed by :meth:`evaluate()` but can't be passed to the base ``__init__()``. This is called during
        :class:`~hohmannpy.Mission`'s instantiation.

        Parameters
        ----------
        initial_gmst : float
            Initial angle of the Greenwich meridian in :math:`rad`.
        """

        self._initial_gmst = initial_gmst

    def evaluate(self, time: float, state: np.ndarray, satellite: spacecraft.Satellite) -> np.ndarray:
        r"""
        Computes the perturbing acceleration due to the J2 effect.

        Parameters
        ----------
        time : float
            Current time in seconds since propagation began.
        state : np.ndarray
            Current translational state in ECI coordinates given as (position, velocity) or
            (position, velocity, quaternion, angular_velocity).
        satellite : :class:`~hohmannpy.astro.Satellite`
            Unused parameter simply based due to ``@abstractmethod`` requirements.

        Returns
        -------
        acceleration : np.ndarray
            Current translational acceleration in ECI coordinates.
        """

        earth_radius = 6378.1363e3
        earth_rot = 7.292115e-5  # Mean rotation rate of the Earth in radians.
        grav_param = 3.986004418e14
        j2 = 1.08e-3

        radius = np.linalg.norm(state[:3])

        # Convert the position vector from the ECI to ECEF frame.
        gmst = self._initial_gmst + earth_rot * time
        inertial_2_earth_dcm = conversions.euler_2_dcm(gmst, 3)
        position = inertial_2_earth_dcm @ state[:3]

        # Compute the J2 acceleration.
        acceleration = -3 * j2 * grav_param * earth_radius ** 2 / (2 * radius ** 5) * np.array(
            [
                position[0] * (1 - 5 * position[2] ** 2 / radius ** 2),
                position[1] * (1 - 5 * position[2] ** 2 / radius ** 2),
                position[2] * (3 - 5 * position[2] ** 2 / radius ** 2)
            ]
        )

        # Convert the acceleration vector from the ECEF back to the ECI frame.
        acceleration = inertial_2_earth_dcm.T @ acceleration

        if len(state) == 3:
            return acceleration
        else:
            return np.concatenate((acceleration, np.zeros(3)), axis=-1)

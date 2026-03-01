from __future__ import annotations
import importlib.resources
from typing import TYPE_CHECKING

import numpy as np
import scipy as sp

from ...dynamics import dcms
from . import base

if TYPE_CHECKING:
    from .. import spacecraft


class AtmosphericDrag(base.Perturbation):
    r"""
    Perturbation caused by drag due to Earth's atmosphere.

    Atmospheric density is found using interpolation of the 2012 Committee on Space Research (COSPAR) International
    Reference Atmosphere (CIRA-12) model. Three different density tables are provided, each representing a varying level
    of solar and geomagnetic activity, selected between using the ``solar_activity`` parameter.

    Parameters
    ----------
    solar_activity : str
        Which CIRA-12 reference atmosphere model to use for the density. Can select between "low", "medium", and "high".
        See the CIRA-12 offical report [1]_ for more details on how to select between these.
    solver_tol : float
        Tolerance to use when solving for the geodetic latitude via fixed-point iteration.

    Attributes
    ----------
    initial_gmst : float
        Initial angle of the Greenwich meridian in :math:`rad` when propagation began.
    solver_tol : float
        Tolerance to use when solving for the geodetic latitude via fixed-point iteration.
    densities : scipy.BSpline
        Piece-wise linear spline generated from a density curve where the independent variable is altitudes in
        :math:`km` and the dependent variable is densities in :math:`kg/m^3`.
    exosphere_bound : float
        Upper limit of the exosphere in :math:`km` above which the density is assumed to be zero and hence there is no
        drag.

    Notes
    -----
    The following assumptions are made for this implementation:

    1) Density changes due to solar and geomagnetic activity are ignored apart from determining the mean density distribution.

    2) Cubic spline interpolation of an atmospheric density table is a sufficient representation of the true density wrt. time.

    3) The GMST of the Earth is initially accurately computed wrt. the Vernal equinox (ignoring nutation) and is then said to linearly rotate at the Earth's mean rotation rate without precession.

    4) The drag experienced by a satellite is attitude-independent and simply a function of its ballistic coefficient.

    5) When computing the geodetic altitude the Earth is assumed to have a uniform ellipsoidal shape. This is less accurate than the true altitude found using a series expansion (similar to the non-spherical Earth's geopotential) but the accuracy loss is small.

    6) The velocity of a satellite wrt. the atmosphere is simply its velocity minus the rotation rate of the Earth cross the satellite's position.

    Additionally, the  altitude above an ellipsoid Earth is found using Algorithm 12 in Vallado [2]_.

    .. [1] COSPAR, COSPAR International Reference Atmosphere – CIRA-2012, Version: 1.0, spacewx.com, 2012.
    .. [2] Vallado, D. A., Fundamentals of Astrodynamics and Applications, 3rd ed., Microcosm Press/Springer, 2007.
    """

    def __init__(
            self,
            solar_activity: str = "moderate",
            solver_tol: float = 1e-8
    ):
        super().__init__()

        self.initial_gmst = None
        self.solver_tol = solver_tol

        # Import the density table to use based on the chosen solar and geomagnetic activity level.
        if solar_activity == "low":
            with importlib.resources.files("hohmannpy.resources").joinpath("cira_12_low_activity.csv").open() as f:
                density_curve = np.loadtxt(f, delimiter=",")  # altitude (km), density (kg/m^3)
                self.densities = sp.interpolate.make_interp_spline(
                    density_curve[:, 0].squeeze(),
                    density_curve[:, 1].squeeze(),
                    k=3
                )
        elif solar_activity == "moderate":
            with importlib.resources.files("hohmannpy.resources").joinpath("cira_12_moderate_activity.csv").open() as f:
                density_curve = np.loadtxt(f, delimiter=",")  # altitude (km), density (kg/m^3)
                self.densities = sp.interpolate.make_interp_spline(
                    density_curve[:, 0].squeeze(),
                    density_curve[:, 1].squeeze(),
                    k=3
                )
        elif solar_activity == "high":
            with importlib.resources.files("hohmannpy.resources").joinpath("cira_12_high_activity.csv").open() as f:
                density_curve = np.loadtxt(f, delimiter=",")  # altitude (km), density (kg/m^3)
                self.densities = sp.interpolate.make_interp_spline(
                    density_curve[:, 0].squeeze(),
                    density_curve[:, 1].squeeze(),
                    k=3
                )
        else:
            raise ValueError(f"{solar_activity} is not a valid setting for the solar activity, please choose from "
                             f"'low', 'medium', or 'high'.")
        self.exosphere_bound = density_curve[-1, 0]

    def finalize__init__(self, initial_gmst: float):
        """
        Record the initial GMST of the Earth which is used to correctly orient it for geopotential modeling.

        This is needed by :meth:`evaluate()` but can't be passed to the base ``__init__()``. This is called during
        :class:`~hohmannpy.astro.Mission`'s instantiation.

        Parameters
        ----------
        initial_gmst : float
            Initial angle of the Greenwich meridian in :math:`rad`.
        """

        self.initial_gmst = initial_gmst

    def evaluate(self, time: float, state: np.ndarray, satellite: spacecraft.Satellite) -> np.ndarray:
        r"""
        Computes the perturbing acceleration using a model for the drag caused by the Earth's atmosphere.

        Parameters
        ----------
        time : float
            Current time in seconds since propagation began.
        state : np.ndarray
            Current translational state in ECI coordinates given as (position, velocity).
        satellite : :class:`~hohmannpy.astro.Satellite`
            Satellite object. Passed so that its ``ballistic_coefficient`` may be accessed.

        Returns
        -------
        acceleration : np.ndarray
            Current translational acceleration in ECI coordinates.
        """

        earth_rot = 7.292115e-5  # Mean rotation rate of the Earth in radians.

        # Update GMST using simplified precession-free rotation of the Earth.
        gmst = self.initial_gmst + earth_rot * time

        # Transform position to the Earth-centered-Earth-fixed frame and then compute the geodetic altitude.
        inertial_2_earth_dcm = dcms.euler_2_dcm(gmst, 3)
        position = inertial_2_earth_dcm @ state[:3]
        altitude = self.compute_altitude(position)

        # If above the exosphere_bound, there is effectively no atmosphere.
        if altitude / 1000 > self.exosphere_bound:
            return np.array([0, 0, 0])

        # Compute density.
        density = self.densities(altitude / 1000)  # Need to convert m -> km

        # Compute the velocity of the satellite wrt. the atmosphere.
        velocity = state[3:] - np.cross(np.array([0, 0, earth_rot]), state[:3])

        # Compute perturbing acceleration.
        acceleration = -0.5 * 1 / satellite.ballistic_coeff * density * np.linalg.norm(velocity) * velocity

        return acceleration

    def compute_altitude(self, position: np.ndarray) -> float:
        """
        Compute the altitude above the surface of an ellipsoid Earth.

        Parameters
        ----------
        position : np.ndarray
            Current position in PCI coordinates.

        Returns
        -------
        altitude : float
            Current height above sea level of an ellipsoid Earth.
        """

        earth_radius = 6378.1363e3
        earth_eccentricity = 0.081819221456

        # The geodetic latitude can be found as a function of the satellite's current position, however this function is
        # transcendental in latitude and hence must be solved numerically. Fixed-point iteration is used. Once the
        # latitude is known the ellipsoidal altitude may be computed.
        x = np.arctan2(position[2], np.sqrt(position[0] ** 2 + position[1] ** 2))  # Initial guess.
        x_old = 100  # Dummy value to ensure error is initially above tolerance.

        while abs(x - x_old) > self.solver_tol:  # Fixed point iteration.
            x_old = x
            radius_of_curvature = earth_radius / np.sqrt((1 - earth_eccentricity ** 2 * np.sin(x) ** 2))
            x = np.arctan2(
                position[2] + radius_of_curvature * earth_eccentricity ** 2 * np.sin(x),
                np.sqrt(position[0] ** 2 + position[1] ** 2)
            )

        # Using the latitude compute the ellipsoidal altitude.
        geodetic_latitude = x
        radius_of_curvature = earth_radius / np.sqrt((1 - earth_eccentricity ** 2 * np.sin(x) ** 2))
        altitude = np.sqrt(position[0] ** 2 + position[1] ** 2) / np.cos(geodetic_latitude) - radius_of_curvature

        return altitude

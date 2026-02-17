from __future__ import annotations
import copy
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import scipy as sp

from . import orbit, logging, maneuvers

if TYPE_CHECKING:
    from . import time


# TODO: Update to include burns.
class Satellite:
    r"""
    Basic spacecraft whose motion can be simulated using :class:`~hohmannpy.astro.Mission`.

    Parameters
    ----------
    name : str
        Unique identifier of the spacecraft. Repeats are not allowed.
    starting_orbit : :class:`~hohmannpy.astro.Orbit`
        The orbit the spacecraft is in at the start of the perturbation.
    color: str
        The color of the orbit and spacecraft to display in renderings.
    mass: float
        Mass of the spacecraft in :math:`kg`. Needed for missions where the perturbation
        :class:`~hohmannpy.astro.SolarRadiation` is enabled.
    ballistic_coeff: float
        Dimensionless parameter proportional to the drag effects experienced by a spacecraft. Needed for missions where
        the perturbation :class:`~hohmannpy.astro.AtmosphericDrag` is enabled.
    mean_reflective_area : float
        Average area exposed to solar radiation pressure in :math:`m^2`. Needed for missions where the perturbation
        :class:`~hohmannpy.astro.SolarRadiation` is enabled.
    reflectivity : float
        Dimensionless parameter proportional to how much solar radiation is reflected by the ``mean_reflective_area``.
        0 = transparent, 1 = full absorption, and 2 = full reflection. Needed for missions where the perturbation
        :class:`~hohmannpy.astro.SolarRadiation` is enabled.

    Attributes
    ----------
    name : str
        Unique identifier of the spacecraft. Repeats are not allowed.
    starting_orbit : :class:`~hohmannpy.astro.Orbit`
        The orbit the spacecraft is in at the start of the perturbation.
    orbit : :class:`~hohmannpy.astro.Orbit`
        Current orbit of the spacecraft. Starts as a deep copy of the ``starting_orbit`` and then is updated on each
        timestep during propagation.
    loggers: list[:class:`~hohmannpy.astro.Logger`]
        Loggers which record data on each timestep during propagation. This attribute is initially set to ``None`` and
        is filled in by the ``__init__()`` of ``Mission``.
    color: str
        The color of the orbit and spacecraft to display in renderings.
    mass: float
        Mass of the spacecraft in :math:`kg`. Needed for missions where the perturbation
        :class:`~hohmannpy.astro.SolarRadiation` is enabled.
    ballistic_coeff: float
        Dimensionless parameter proportional to the drag effects experienced by a spacecraft. Needed for missions where
        the perturbation :class:`~hohmannpy.astro.AtmosphericDrag` is enabled.
    mean_reflective_area : float
        Average area exposed to solar radiation pressure in :math:`m^2`. Needed for missions where the perturbation
        :class:`~hohmannpy.astro.SolarRadiation` is enabled.
    reflectivity : float
        Dimensionless parameter proportional to how much solar radiation is reflected by the ``mean_reflective_area``.
        0 = transparent, 1 = full absorption, and 2 = full reflection. Needed for missions where the perturbation
        :class:`~hohmannpy.astro.SolarRadiation` is enabled.

    Notes
    -----
    Once a satellite's orbit is propagator, the recorded data (attributes) of any :class:`~hohmannpy.astro.Logger`
    attached to this satellite can be access from the satellite using ``__getattr_()``.
    """

    def __init__(
            self,
            name: str,
            starting_orbit: orbit.Orbit,
            color: str = "#FF073A",
            burns: Optional[list[Union[maneuvers.ImpulsiveBurn, maneuvers.ContinuousBurn]]] = None,
            mass: float = None,
            ballistic_coeff: float = None,
            mean_reflective_area: float = None,
            reflectivity: float = None
    ):
        self.name = name
        self.starting_orbit = starting_orbit
        self.color = color

        self.impulsive_burns = []
        self.continuous_burns = []
        if burns is not None:
            for burn in burns:
                if isinstance(burn, maneuvers.ImpulsiveBurn):
                    self.impulsive_burns.append(burn)
                else:
                    self.continuous_burns.append(burn)
            self.impulsive_burn_index = 0

        # Perturbation-specific parameters.
        self.mass = mass
        self.ballistic_coeff = ballistic_coeff
        self.mean_reflective_area = mean_reflective_area
        self.reflectivity = reflectivity

        self.orbit: orbit.Orbit = copy.deepcopy(starting_orbit)  # This will be updated over time by the propagator.
        self.loggers: Optional[list[logging.Logger]] = None  # Filled in by the __init__() of Mission.

    def __getattr__(self, name):
        r"""
        Access data from ``Loggers`` assigned to this object as if they were assigned to this class.
        """

        # Need a safeguard here because can't call self.(some attribute) inside __getattr__() because this can break
        # during the pickling which occurs during parallel processing.
        loggers = object.__getattribute__(self, "__dict__").get("loggers", None)

        if loggers is not None:
            for logger in loggers:
                if hasattr(logger, name):
                    return getattr(logger, name)
        raise AttributeError(f"This satellite has not logged data for {name}.")


class Moon(Satellite):
    r"""
    Special "spacecraft" which represents the Earth's moon.

    Used by :class:`~hohmannpy.astro.LunarGravity` for simulating lunar gravity third-body perturbing effects.

    Parameters
    ----------
    initial_true_anomaly : float
        The starting true anomaly of the Moon in :math:`rad`.
    """

    def __init__(self, initial_true_anomaly: float):
        name = "Moon"
        starting_orbit = orbit.Orbit.from_classical_elements(
            sm_axis=3.844e8,
            eccentricity=0.0549,
            inclination=np.deg2rad(5.145),
            raan=np.deg2rad(125.08),
            argp=np.deg2rad(318.15),
            true_anomaly=initial_true_anomaly,
        )
        super().__init__(name, starting_orbit)
        self.loggers = [logging.StateLogger()]


class Earth(Satellite):
    r"""
    Special "spacecraft" which represents the Earth. Alternatively, can represent the Sun orbiting the Earth if you
    invert the position vector.

    Used by :class:`~hohmannpy.astro.SolarGravity` for simulating solar gravity third-body perturbing effects.

    Parameters
    ----------
    initial_global_time: :class:`~hohmannpy.astro.Time`
        Gregorian date and UT1 time at which simulation begins. This is used to locate the Earth via
        :meth:`compute_initial_true_anomaly()`.
    solver_tol : float
        Error tolerance when performing root-finding to solver Kepler's equation in ``compute_initial_true_anomaly()``.
    """

    def __init__(self, initial_global_time: time.Time, solver_tol: float = 1e-8):
        name = "Earth"

        initial_true_anomaly = self.compute_initial_true_anomaly(initial_global_time, solver_tol)
        starting_orbit = orbit.Orbit.from_classical_elements(
            sm_axis=149597870.7e3,
            eccentricity=0.0167086,
            inclination=0,
            raan=0,
            argp=np.deg2rad(102.937),
            true_anomaly=initial_true_anomaly,
            grav_param=1.32712440018e20
        )
        super().__init__(name, starting_orbit)
        self.loggers = [logging.StateLogger()]

    def compute_initial_true_anomaly(self, initial_global_time: time.Time, solver_tol: float):
        r"""
        Calculates the true anomaly of the Earth at the initial date.

        Parameters
        ----------
        initial_global_time : :class:`~hohmannpy.astro.Time`
            Gregorian date and UT1 time at which the Earth is initially located.
        solver_tol : float
            Error tolerance to use when solving Kepler's equation.

        Returns
        -------
        initial_true_anomaly : float
            True anomaly of the earth corresponding to ``initial_global_time``.
        """

        earth_mean_motion = np.deg2rad(0.98560028)
        earth_eccentricity = 0.0167086
        j2000_mean_anomaly = np.deg2rad(357.5277233)
        j2000_julian_time = 2451545

        # Compute the initial mean anomaly wrt. J2000 and then solve Kepler's equation for the corresponding initial
        # eccentric anomaly.
        initial_mean_anomaly = (
            j2000_mean_anomaly
                + earth_mean_motion * ((initial_global_time.julian_date - j2000_julian_time) * 86400)
        ) % 2 * np.pi

        eq = lambda x: initial_mean_anomaly - x + earth_eccentricity * np.sin(x)
        initial_eccentric_anomaly = sp.optimize.newton(eq, initial_mean_anomaly, tol=solver_tol)

        # Use Gauss' equation to compute the initial eccentric anomaly to the initial true anomaly.s
        return  (
            2 * np.arctan(
                np.sqrt((1 + earth_eccentricity) / (1 - earth_eccentricity)) * np.tan(initial_eccentric_anomaly / 2)
            )
        )

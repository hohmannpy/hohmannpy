from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import scipy as sp

from hohmannpy.astro import orbits
from hohmannpy import logging
from .. import spacecraft

if TYPE_CHECKING:
    from hohmannpy.astro import time


class Moon(spacecraft.Satellite):
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
        starting_orbit = orbits.Orbit.from_classical_elements(
            sm_axis=3.844e8,
            eccentricity=0.0549,
            inclination=np.deg2rad(5.145),
            raan=np.deg2rad(125.08),
            argp=np.deg2rad(318.15),
            true_anomaly=initial_true_anomaly,
        )
        super().__init__(name, starting_orbit)
        self.loggers: list[logging.Logger] = [logging.TimeLogger(), logging.StateLogger()]


class Earth(spacecraft.Satellite):
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

        initial_true_anomaly = self._compute_initial_true_anomaly(initial_global_time, solver_tol)
        starting_orbit = orbits.Orbit.from_classical_elements(
            sm_axis=149597870.7e3,
            eccentricity=0.0167086,
            inclination=0,
            raan=0,
            argp=np.deg2rad(102.937),
            true_anomaly=initial_true_anomaly,
            grav_param=1.32712440018e20
        )
        super().__init__(name, starting_orbit)
        self.loggers: list[logging.Logger] = [logging.TimeLogger(), logging.StateLogger()]

    def _compute_initial_true_anomaly(self, initial_global_time: time.Time, solver_tol: float) -> float:
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
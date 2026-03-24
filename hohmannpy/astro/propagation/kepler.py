from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import scipy as sp

from . import base

if TYPE_CHECKING:
    from .. import spacecraft, perturbations


class KeplerPropagator(base.Propagator):
    r"""
    Propagates orbits using an f and g functions as well as Kepler's equation.

    If eccentricity is greater than 1 automatically switches over to using the hyperbolic eccentric anomaly. The
    parabolic case is not included. Be aware that for near-parabolic orbits propagation accuracy will greatly decrease.

    Parameters
    ----------
    step_size : float
        Time interval between propagation steps. If one is not provided by the user it will be set in ``propagate()`` to
        60 :math:`s`.
    solver_tol: float
        Error tolerance when performing root-finding to solver Kepler's equation.
    fg_constraint : bool
        Whether to compute derivative of the g function manually or to instead use a constraint.
    """

    name = "Kepler"
    energy_conserving = True

    def __init__(
            self,
            step_size: float = 60,
            solver_tol: float = 1e-8,
            fg_constraint=True,
            **kwargs
    ):
        self._solver_tol = solver_tol
        self._fg_constraint = fg_constraint

        # Empty dicts containing initial conditions that get filled in propagate().
        self._initial_times = {}
        self._initial_positions = {}
        self._initial_velocities = {}
        self._initial_eccentric_anomalies = {}

        super().__init__(step_size=step_size, **kwargs)

    def _set_initial_conditions(self, satellite: spacecraft.Satellite):
        self._initial_times[satellite.name] = satellite.orbit.time
        self._initial_positions[satellite.name] = satellite.orbit.position.copy()  # Copy to prevent mutation.
        self._initial_velocities[satellite.name] = satellite.orbit.velocity.copy()

        # Run Gauss' equation to get the initial eccentric anomaly of each orbit. This is needed so that logging can
        # being because the user might have passed astro.EccentricAnomalyLogger().
        self._initial_eccentric_anomalies[satellite.name] = (
            self._gauss_equation(
                eccentricity=satellite.orbit.eccentricity,
                true_anomaly=satellite.orbit.true_anomaly
            )
        )
        satellite.orbit.eccentric_anomaly = self._initial_eccentric_anomalies[satellite.name]

    def _step(self, satellite: spacecraft.Satellite, time_change: float):
        # First retrieve the orbit. Next, use  Kepler's equation to solve for the eccentric anomaly at the next
        # timestep, and then use that to form the f and g functions and their derivatives. These can be used to
        # construct the position and velocity.
        orbit = satellite.orbit

        orbit.eccentric_anomaly = self._kepler_equation(
            time=orbit.time,
            eccentricity=orbit.eccentricity,
            sm_axis=orbit.sm_axis,
            grav_param=orbit.grav_param,
            initial_eccentric_anomaly=self._initial_eccentric_anomalies[satellite.name],
            initial_guess=orbit.eccentric_anomaly,
            initial_time=self._initial_times[satellite.name]
        )

        f_func, g_func = self._compute_fg_funcs(
            initial_time=self._initial_times[satellite.name],
            initial_position=self._initial_positions[satellite.name],
            initial_eccentric_anomaly=self._initial_eccentric_anomalies[satellite.name],
            time=orbit.time,
            eccentric_anomaly=orbit.eccentric_anomaly,
            sm_axis=orbit.sm_axis,
            grav_param=orbit.grav_param,
            eccentricity=orbit.eccentricity
        )
        orbit.position = (
                f_func * self._initial_positions[satellite.name] + g_func * self._initial_velocities[satellite.name]
        )

        fdot_func, gdot_func = self._compute_fg_dot_funcs(
            position=orbit.position,
            initial_position=self._initial_positions[satellite.name],
            initial_eccentric_anomaly=self._initial_eccentric_anomalies[satellite.name],
            eccentric_anomaly=orbit.eccentric_anomaly,
            sm_axis=orbit.sm_axis,
            grav_param=orbit.grav_param,
            f_func=f_func,
            g_func=g_func,
            eccentricity=orbit.eccentricity
        )
        orbit.velocity = (
                fdot_func * self._initial_positions[satellite.name] + gdot_func * self._initial_velocities[satellite.name]
        )

    def _compute_fg_funcs(
            self,
            initial_time: float,
            initial_position: np.ndarray,
            initial_eccentric_anomaly: np.ndarray,
            time: float,
            eccentric_anomaly: float,
            sm_axis: float,
            grav_param: float,
            eccentricity: float,
    ) -> tuple[float, float]:
        """
        Computes the f and g functions.
        """

        if eccentricity < 1:  # Elliptic case.
            f_func = (
                    1 - sm_axis / np.linalg.norm(initial_position)
                    * (1 - np.cos(eccentric_anomaly - initial_eccentric_anomaly))
            )
            g_func = (
                    time - initial_time
                    - 1 / np.sqrt(grav_param / sm_axis ** 3)
                    * (eccentric_anomaly - initial_eccentric_anomaly
                       - np.sin(eccentric_anomaly - initial_eccentric_anomaly))
            )
        else:  # Hyperbolic case.
            f_func = (
                    1 - sm_axis / np.linalg.norm(initial_position)
                    * (1 - np.cosh(eccentric_anomaly - initial_eccentric_anomaly))
            )
            g_func = (
                    time - initial_time
                    - 1 / np.sqrt(grav_param / (-sm_axis) ** 3)
                    * (np.sinh(eccentric_anomaly - initial_eccentric_anomaly)
                       - (eccentric_anomaly - initial_eccentric_anomaly))
            )

        return f_func, g_func

    def _compute_fg_dot_funcs(
            self,
            initial_position: np.ndarray,
            initial_eccentric_anomaly: float,
            position: np.ndarray,
            eccentric_anomaly: float,
            sm_axis: float,
            grav_param: float,
            f_func: float,
            g_func: float,
            eccentricity: float,
    ) -> tuple[float, float]:
        """
        Computes the f and g functions' derivatives.
        """

        if eccentricity < 1:  # Elliptic case.
            fdot_func = (
                    -np.sqrt(grav_param * sm_axis)
                    / (np.linalg.norm(initial_position) * np.linalg.norm(position))
                    * np.sin(eccentric_anomaly - initial_eccentric_anomaly)
            )
            if self._fg_constraint:  # Only compute gdot function manually if constraint usage is disabled.
                gdot_func = (g_func * fdot_func + 1) / f_func
            else:
                gdot_func = (
                        1 - sm_axis / np.linalg.norm(position)
                        * (1 - np.cos(eccentric_anomaly - initial_eccentric_anomaly))
                )
        else:  # Hyperbolic case.
            fdot_func = (
                    -np.sqrt(grav_param * -sm_axis)
                    / (np.linalg.norm(initial_position) * np.linalg.norm(position))
                    * np.sinh(eccentric_anomaly - initial_eccentric_anomaly)
            )
            if self._fg_constraint:
                gdot_func = (g_func * fdot_func + 1) / f_func
            else:
                gdot_func = (
                        1 - sm_axis / np.linalg.norm(position)
                        * (1 - np.cosh(eccentric_anomaly - initial_eccentric_anomaly))
                )

        return fdot_func, gdot_func

    def _gauss_equation(self, eccentricity: float, true_anomaly: float) -> float:
        r"""
        Converts true anomaly to eccentric anomaly.
        """

        if eccentricity < 1:  # Elliptic case.
            return (
                    2 * np.arctan(np.sqrt((1 - eccentricity) / (1 + eccentricity))
                        * np.tan(true_anomaly / 2))
            )
        else:  # Hyperbolic case.
            return (
                    2 * np.arctanh(np.sqrt((eccentricity - 1) / (eccentricity + 1))
                                  * np.tan(true_anomaly / 2))
            )

    # TODO: Provide derivative to speed this up.
    def _kepler_equation(
            self,
            time: float,
            eccentricity: float,
            sm_axis: float,
            grav_param: float,
            initial_eccentric_anomaly: float,
            initial_guess: float,
            initial_time: float,
    ) -> float:
        r"""
        Function used to compute the new eccentric anomaly given the current eccentric anomaly and the desired time
        increment.

        Kepler's equation is transcendental wrt. eccentric anomaly so root-finding via :func:`scipy.optimize.newton()`
        is used to solve for it. The ideal initial guess is just the eccentric anomaly on the previous timestep.
        """

        # Set up Kepler's equation as a lambda expression of the eccentric anomaly and then pass it to
        # sp.optimize.newton() for root-finding.
        if eccentricity < 1:  # Elliptic case.
            eq = lambda x: (
                    np.sqrt(grav_param / sm_axis ** 3) * (time - initial_time)
                        + initial_eccentric_anomaly - eccentricity * np.sin(initial_eccentric_anomaly)
                        - x + eccentricity * np.sin(x)
            )
        else:  # Hyperbolic case.
            eq = lambda x: (
                    np.sqrt(grav_param / (-sm_axis) ** 3) * (time - initial_time)
                    + eccentricity * np.sinh(initial_eccentric_anomaly) - initial_eccentric_anomaly
                    - eccentricity * np.sinh(x) + x
            )
        eccentric_anomaly = sp.optimize.newton(eq, initial_guess, tol=self._solver_tol)

        return eccentric_anomaly

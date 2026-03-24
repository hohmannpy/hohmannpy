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
    fg_constraint: bool
        Flag which indicates whether to compute the derivative of the g function (``False``) or to use a constraint to
        eliminate it (``True``).

    """

    name = "Kepler"

    def __init__(
            self,
            step_size: float = 60,
            solver_tol: float = 1e-8,
            fg_constraint: bool = True,
    ):
        self._fg_constraint = fg_constraint
        self._solver_tol = solver_tol

        # Empty dicts containing initial conditions that get filled in propagate().
        self._initial_times = {}
        self._initial_positions = {}
        self._initial_velocities = {}
        self._initial_eccentric_anomalies = {}

        super().__init__(step_size)

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
        r"""
        One step in the propagation loop.

        Parameters
        ----------
        satellite: :class:`~hohmannpy.astro.Satellite`
            Satellite being propagated. Holds the orbit to propagate as an attribute named ``orbit``.
        time_change: float
            How much time has passed since the last propagation step.
        """

        # First retrieve the orbit. Then determine if the orbit is elliptic or hyperbolic based on
        # its eccentricity. The form of Kepler's equation and the f and g functions changes based on this. Next, use
        # Kepler's equation to solve for the eccentric anomaly at the next timestep, and then use that to form the f and
        # g functions and their derivatives. These can be used to construct the position and velocity.
        orbit = satellite.orbit

        # -------------
        # ELLIPTIC CASE
        # -------------
        if orbit.eccentricity < 1:  # Elliptical case.
            # Compute new eccentric anomaly. Use the previous eccentric anomaly as the initial guess for the
            # root-finder.
            orbit.eccentric_anomaly = self._kepler_equation(
                time=orbit.time,
                eccentricity=orbit.eccentricity,
                sm_axis=orbit.sm_axis,
                grav_param=orbit.grav_param,
                initial_eccentric_anomaly=self._initial_eccentric_anomalies[satellite.name],
                initial_guess=orbit.eccentric_anomaly,
                initial_time=self._initial_times[satellite.name]
            )

            # Compute the f and g functions.
            f_func = (
                    1 - orbit.sm_axis / np.linalg.norm(self._initial_positions[satellite.name])
                    * (1 - np.cos(orbit.eccentric_anomaly - self._initial_eccentric_anomalies[satellite.name]))
            )
            g_func = (
                    orbit.time - self._initial_times[satellite.name]
                    - 1 / np.sqrt(orbit.grav_param / orbit.sm_axis ** 3)
                    * (orbit.eccentric_anomaly - self._initial_eccentric_anomalies[satellite.name]
                       - np.sin(orbit.eccentric_anomaly - self._initial_eccentric_anomalies[satellite.name]))
            )

            # Compute new position (and true anomaly). Only need to update fast variables because the other
            # orbital elements are constant for Keplerian orbits.
            orbit.position = (
                    f_func * self._initial_positions[satellite.name] + g_func * self._initial_velocities[satellite.name]
            )

            # Compute fdot and gdot functions.
            fdot_func = (
                    -np.sqrt(orbit.grav_param * orbit.sm_axis)
                    / (np.linalg.norm(self._initial_positions[satellite.name]) * np.linalg.norm(orbit.position))
                    * np.sin(orbit.eccentric_anomaly - self._initial_eccentric_anomalies[satellite.name])
            )
            if self._fg_constraint:  # Only compute gdot function manually if constraint usage is disabled.
                gdot_func = (g_func * fdot_func + 1) / f_func
            else:
                gdot_func = (
                        1 - orbit.sm_axis / np.linalg.norm(orbit.position)
                        * (1 - np.cos(orbit.eccentric_anomaly - self._initial_eccentric_anomalies[satellite.name]))
                )

        # ---------------
        # HYPERBOLIC CASE
        # ---------------
        else:
            # This is the same as the elliptic case except the equations are changed to use a negative
            # semi-major axis and the hyperbolic version of the eccentric anomaly.
            orbit.eccentric_anomaly = self._kepler_equation(
                time=orbit.time,
                eccentricity=orbit.eccentricity,
                sm_axis=orbit.sm_axis,
                grav_param=orbit.grav_param,
                initial_eccentric_anomaly=self._initial_eccentric_anomalies[satellite.name],
                initial_guess=orbit.eccentric_anomaly,
                initial_time=self._initial_times[satellite.name]
            )

            f_func = (
                    1 - orbit.sm_axis / np.linalg.norm(self._initial_positions[satellite.name])
                    * (1 - np.cosh(orbit.eccentric_anomaly - self._initial_eccentric_anomalies[satellite.name]))
            )
            g_func = (
                    orbit.time - self._initial_times[satellite.name]
                    - 1 / np.sqrt(orbit.grav_param / (-orbit.sm_axis) ** 3)
                    * (np.sinh(orbit.eccentric_anomaly - self._initial_eccentric_anomalies[satellite.name])
                       - (orbit.eccentric_anomaly - self._initial_eccentric_anomalies[satellite.name]))
            )

            orbit.position = (
                    f_func * self._initial_positions[satellite.name] + g_func * self._initial_velocities[satellite.name]
            )

            fdot_func = (
                    -np.sqrt(orbit.grav_param * -orbit.sm_axis)
                    / (np.linalg.norm(self._initial_positions[satellite.name]) * np.linalg.norm(orbit.position))
                    * np.sinh(orbit.eccentric_anomaly - self._initial_eccentric_anomalies[satellite.name])
            )
            if self._fg_constraint:
                gdot_func = (g_func * fdot_func + 1) / f_func
            else:
                gdot_func = (
                        1 - orbit.sm_axis / np.linalg.norm(orbit.position)
                        * (1 - np.cosh(orbit.eccentric_anomaly - self._initial_eccentric_anomalies[satellite.name]))
                )

        # Compute the new velocity.
        orbit.velocity = (
                fdot_func * self._initial_positions[satellite.name] + gdot_func * self._initial_velocities[satellite.name]
        )

    def _gauss_equation(self, eccentricity: float, true_anomaly: float) -> float:
        r"""
        Converts true anomaly to eccentric anomaly.

        Parameters
        ----------
        eccentricity : float
            Eccentricity of the orbit.
        true_anomaly : float
            Current true anomaly.

        Returns
        -------
        eccentric_anomaly : float
            Eccentric anomaly corresponding to the given true anomaly.
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

        Parameters
        ----------
        time : float
            Current time.
        eccentricity : float
            Eccentricity of the orbit.
        sm_axis : float
            Semi-major axis of the orbit.
        grav_param : float
            Gravitational parameter of the orbit.
        initial_eccentric_anomaly : float
            Base point of the eccentric anomaly from when propagation began.
        initial_guess : float
            Initial guess for the eccentric anomaly.
        initial_time : float
            Base point for time at which propagation began.

        Returns
        -------
        eccentric_anomaly : float
            Eccentric anomaly at the next time step.
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

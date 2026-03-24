from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import scipy as sp

from . import base

if TYPE_CHECKING:
    from .. import spacecraft, perturbations


class UniversalVariablePropagator(base.Propagator):
    r"""
    Propagates orbits using an f and g functions as well as a universal variable formulation of Kepler's equation.

    Unlike with the standard form of Kepler's equation the inclusion of the universal variable allows propagation of
    parabolic orbits in addition to elliptical and hyperbolic ones.

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
    stumpff_tol: float
        The universal variable is not an angular quantity, so it is used to compute a variable known as the Stumpff
        parameter whose root is an angle. The Stumpff parameter is used to compute two hypergeometric series, termed as
        Stumpff series, from which the f and g functions may be assembled. For most values of the Stumpff parameter
        these series converge absolutely to either trigonometric or hyper-trigonometric functions, but when it is small
        the Stumpff series must be computed via summation. "Small" is defined here as the absolute value of the Stumpff
        parameter being under ``stumpff_tol``.
    stumpff_series_length : int
        When the Stumpff series are computed via summation, how many terms to include.
    """

    name = "Universal Variable"
    energy_conserving = True

    def __init__(
            self,
            step_size: float = 60,
            solver_tol: float = 1e-8,
            stumpff_tol: float = 1e-8,
            stumpff_series_length: int = 10,
            fg_constraint: bool = True
    ):
        self._fg_constraint = fg_constraint
        self._solver_tol = solver_tol
        self._stumpff_tol = stumpff_tol
        self._stumpff_series_length = stumpff_series_length

        # Empty dicts containing initial conditions that get filled in propagate().
        self._initial_times = {}
        self._initial_positions = {}
        self._initial_velocities = {}

        super().__init__(step_size)

    def _set_initial_conditions(self, satellite: spacecraft.Satellite):
        for name, satellite in self._satellites.items():
            self._initial_times[name] = satellite.orbit.time
            self._initial_positions[name] = satellite.orbit.position.copy()
            self._initial_velocities[name] = satellite.orbit.velocity.copy()

            # Conveniently, both the universal variable and Stumpff parameter start at 0.
            satellite.orbit.universal_variable = 0  # Needed for logging purposes.
            satellite.orbit.stumpff_param = 0

            # For parabolic orbits the semi-major axis is infinite so in order for the solver to handle elliptic,
            # parabolic, and hyperbolic orbits using one set of equations it is replaced with the inverse semi-major
            # axis.
            satellite.orbit.inverse_sm_axis = (
                    (2 * satellite.orbit.grav_param / np.linalg.norm(self._initial_positions[name])
                     - np.linalg.norm(self._initial_velocities[name]) ** 2)
                    / satellite.orbit.grav_param
            )

    def _step(self, satellite: spacecraft.Satellite, time_change: float):
        # For each satellite, first retrieve the orbit. Then, compute the universal variable on the next time step from
        # Kepler's equation. From there the Stumpff series can be computed and in turn used to assemble the f and g
        # functions and their derivatives. Finally, from these the position and velocity may be found at the next
        # timestep.
        orbit = satellite.orbit

        orbit.universal_variable = self._kepler_equation(
            inverse_sm_axis=orbit.inverse_sm_axis,
            grav_param=orbit.grav_param,
            time=orbit.time,
            initial_time=self._initial_times[satellite.name],
            initial_position=self._initial_positions[satellite.name],
            initial_velocity=self._initial_velocities[satellite.name],
            initial_guess=orbit.universal_variable,
        )

        # Compute the Stumpff (c and s) functions.
        orbit.stumpff_param = orbit.inverse_sm_axis * orbit.universal_variable ** 2
        s_func, c_func = self._stumpff_funcs(orbit.stumpff_param)

        f_func, g_func = self._compute_fg_funcs(
            initial_time=self._initial_times[satellite.name],
            initial_position=self._initial_positions[satellite.name],
            time=orbit.time,
            universal_variable=orbit.universal_variable,
            grav_param=orbit.grav_param,
            s_func=s_func,
            c_func=c_func,
        )
        orbit.position = (
                f_func * self._initial_positions[satellite.name] + g_func * self._initial_velocities[satellite.name]
        )

        fdot_func, gdot_func = self._compute_fg_dot_funcs(
            initial_position=self._initial_positions[satellite.name],
            position=orbit.position,
            universal_variable=orbit.universal_variable,
            stumpff_param=orbit.stumpff_param,
            grav_param=orbit.grav_param,
            s_func=s_func,
            c_func=c_func,
            f_func=f_func,
            g_func=g_func
        )
        orbit.velocity = (
                fdot_func * self._initial_positions[satellite.name] + gdot_func * self._initial_velocities[
            satellite.name]
        )

    def _stumpff_funcs(self, stumpff_param) -> tuple[float, float]:
        r"""
        Computes the Stumpff functions/series for a given value of the Stumpff parameter.

        The form of the Stumpff series is not based off the type of orbit, instead it is based of the sign and magnitude
        of the Stumpff parameter. Large and positive = trigonometric, small = summation, large and negative = hyper-
        trigonometric.
        """

        if np.abs(stumpff_param) < self._stumpff_tol:  # Summation form.
            s_func = 0
            c_func = 0
            for i in range(self._stumpff_series_length):
                s_func += (-stumpff_param) ** i / sp.special.factorial(2 * i + 3)
                c_func += (-stumpff_param) ** i / sp.special.factorial(2 * i + 2)
        elif stumpff_param > 0:  # Trigonometric form.
            s_func = (
                    (np.sqrt(stumpff_param) - np.sin(np.sqrt(stumpff_param))) / np.sqrt(stumpff_param ** 3)
            )
            c_func = (
                    (1 - np.cos(np.sqrt(stumpff_param))) / stumpff_param
            )
        else:  # Hyper-trigonometric form.
            s_func = (
                    (np.sinh(np.sqrt(-stumpff_param)) - np.sqrt(-stumpff_param)) / np.sqrt(-stumpff_param ** 3)
            )
            c_func = (
                    (1 - np.cosh(np.sqrt(-stumpff_param))) / stumpff_param
            )

        return s_func, c_func

    def _compute_fg_funcs(
            self,
            initial_time: float,
            initial_position: np.ndarray,
            time: float,
            universal_variable: float,
            grav_param: float,
            s_func: float,
            c_func: float,
    ) -> tuple[float, float]:
        """
        Computes the f and g functions.
        """

        f_func = 1 - universal_variable ** 2 / np.linalg.norm(initial_position) * c_func
        g_func = (
                time - initial_time
                - universal_variable ** 3 / np.sqrt(grav_param) * s_func
        )

        return f_func, g_func

    def _compute_fg_dot_funcs(
            self,
            initial_position: np.ndarray,
            position: np.ndarray,
            universal_variable: float,
            stumpff_param: float,
            grav_param: float,
            s_func: float,
            c_func: float,
            f_func,
            g_func,
    ) -> tuple[float, float]:
        """
        Computes the f and g functions' derivatives.
        """

        fdot_func = (
                np.sqrt(grav_param)
                / (np.linalg.norm(position) * np.linalg.norm(initial_position))
                * universal_variable * (stumpff_param * s_func - 1)
        )
        if self._fg_constraint:  # Only compute gdot function manually if constraint usage is disabled.
            gdot_func = (g_func * fdot_func + 1) / f_func
        else:
            gdot_func = 1 - universal_variable ** 2 / np.linalg.norm(position) * c_func

        return fdot_func, gdot_func

    def _kepler_equation(
            self,
            inverse_sm_axis: float,
            grav_param: float,
            time: float,
            initial_time: float,
            initial_position: np.ndarray,
            initial_velocity: np.ndarray,
            initial_guess: float,
    ) -> float:
        r"""
        Function used to compute the new universal variable directly as a function of time.

        Kepler's equation is transcendental wrt. universal variable so root-finding via :func:`scipy.optimize.newton()`
        is used to solve for it. The ideal initial guess is just the universal variable on the previous timestep.
        """

        # Create the function to use in root-finding.
        def eq(x):
            stumpff_param = inverse_sm_axis * x ** 2
            s_func, c_func = self._stumpff_funcs(stumpff_param)

            return (
                    x ** 3 * s_func
                        + np.dot(initial_position, initial_velocity) / np.sqrt(grav_param)
                        * x ** 2 * c_func
                        + np.linalg.norm(initial_position) * x * (1 - stumpff_param * s_func)
                        - np.sqrt(grav_param) * (time - initial_time)
            )

        # Root-finding.
        universal_variable = sp.optimize.newton(eq, initial_guess, tol=self._solver_tol)

        return universal_variable

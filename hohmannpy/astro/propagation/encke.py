from __future__ import annotations

import numpy as np
import scipy as sp

from . import universal_variable, cowell
from .. import orbits, spacecraft


class EnckePropagator(universal_variable.UniversalVariablePropagator, cowell.CowellPropagator):
    r"""
    Non-Keplerian propagator which uses a modified set of equations of motion where the position is given by::

        true positon = Keplerian position + perturbation from Keplerian position

    The Keplerian position comes from what is known as the "reference" orbit and is propagated using the universal
    variable formulation of Kepler's equation. The perturbation is the difference between the true position and
    Keplerian position and this is found via numerical integration using a 4th-order Runge-Kutta method. These are
    summed to get the true position, and all together this is known as Encke's method.

    Like other non-Keplerian methods, it can handle perturbing forces like :class:`~hohmannpy.astro.NonSphericalEarth`.
    However, in addition the accuracy of the propagation decreases over time as opposed to a Keplerian propagator which
    has a fixed accuracy. However, unlike :class:`~hohmannpy.astro.CowellPropagator` this is partially mitigated by only
    numerically integrating the deviation of the true orbit from the Keplerian reference orbit. The idea is that
    integration errors are smaller when they compound for a smaller value. If accuracy is still an issue, reduce step
    size.

    step_size : float
        Time interval between propagation steps. If one is not provided by the user it will be set in
        :meth:`propagate()` to 60 :math:`s`.
    rectification_tol : float
        When the deviation between the true and reference orbits grows large enough (represented by the ratio of their
        positions' magnitudes being greater than this tolerance), reset the rectified orbit by setting it equal to the
        current true orbit.
    encke_tol : float
        When the Encke parameter is close to zero (defined by this tolerance) the Encke function, which is used to
        compute the position, is undefined so must switch to an infinite series definition of it.
    encke_series_length : int
        How many terms to include when using the infinite series definition of the Encke function.
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

    name = "Encke"

    def __init__(
            self,
            step_size: float = 60,
            rectification_tol: float = 0.01,
            encke_tol: float = 1e-8,
            encke_series_length: int = 10,
            solver_tol: float = 1e-8,
            stumpff_tol: float = 1e-8,
            stumpff_series_length: int = 10,
            fg_constraint: bool = True,
            **kwargs
    ):
        self._rectification_tol = rectification_tol
        self._encke_tol = encke_tol
        self._encke_series_length = encke_series_length

        # Dict containing the reference orbits (cached in satellites) propagated via UniversalVariablePropagator
        # in propagate().
        self._reference_sats = {}

        # Unlike other propagations which just inherit from base.Propagate(), this class instead inherits from
        # UniversalVariablePropagator and CowellPropagator so we also need to instantiate them with all their
        # parameters.
        super().__init__(
            step_size=step_size,
            solver_tol=solver_tol,
            stumpff_tol=stumpff_tol,
            stumpff_series_length=stumpff_series_length,
            fg_constraint=fg_constraint,
            **kwargs
        )

    def _set_initial_conditions(self, satellite: spacecraft.Satellite):
        self._reference_sats[satellite.name] = (
            spacecraft.Satellite(
                name=f"{satellite.name}",
                starting_orbit=orbits.Orbit.from_state(
                    position=satellite.orbit.position.copy(),
                    velocity=satellite.orbit.velocity.copy(),
                    grav_param=satellite.orbit.grav_param
                )
            )
        )

        universal_variable.UniversalVariablePropagator._set_initial_conditions(
            self,
            self._reference_sats[satellite.name]
        )

    def _step(self, satellite: spacecraft.Satellite, time_change: float):
        # For each satellite, first retrieve the orbit and reference orbit. Then the reference orbit is propagated
        # analytically to the next timestep via the universal variable formulation of Kepler's equation by calling
        # reference_step(). Afterwards, the difference between the true and reference orbits is propagated by using RK4
        # to numerically integrate this difference's EOMs. The way this difference is propagated when using RK4 is a
        # little bit unintuitive, so see the comments rk4() for more details. Then, sum these to get the true positions
        # and velocities. Finally, check if the deviation between the true and reference orbits is large enough to
        # require rectification of the reference orbit (setting reference orbit = true orbit).
        orbit = satellite.orbit
        reference_orbit = self._reference_sats[satellite.name].orbit

        # Calculate the reference state (position, velocity) at the old time and then propagate it to the new time. Also
        # propagate it at a halfway point between them, this is needed by RK4 integration.
        old_reference_state = np.concatenate(
            [
                    reference_orbit.position.copy(),
                    reference_orbit.velocity.copy()
            ], axis=0
        )
        self._reference_step(satellite, time_change / 2)
        intermediate_reference_state = np.concatenate(
            [
                reference_orbit.position.copy(),
                reference_orbit.velocity.copy()
            ], axis=0
        )
        self._reference_step(satellite, time_change / 2)

        # Perform numerical integration to get the state difference.
        del_state = self._rk4(
            t0=orbit.time-time_change,
            delt=time_change,
            y0=np.concatenate((orbit.position, orbit.velocity)),
            satellite=satellite,
            y0_ref=old_reference_state,
            y1_ref=intermediate_reference_state,
            y2_ref=np.concatenate(
            [
                    reference_orbit.position,
                    reference_orbit.velocity
                ], axis=0
                ),
        )
        del_position = np.array(del_state[:3])
        del_velocity = np.array(del_state[3:])

        # Compute the true position and velocity.
        orbit.position = reference_orbit.position + del_position
        orbit.velocity = reference_orbit.velocity + del_velocity

        # Perform rectification if needed.
        if np.linalg.norm(del_position) / np.linalg.norm(orbit.position) > self._rectification_tol:
            self._set_initial_conditions(satellite)

    def _reference_step(self, satellite: spacecraft.Satellite, time_change: float):
        self._reference_sats[satellite.name].orbit.time += time_change
        universal_variable.UniversalVariablePropagator._step(
            self,
            self._reference_sats[satellite.name],
            time_change
        )

    def _velocity_eom(
            self, t: float, y: np.ndarray, satellite: spacecraft.Satellite, **kwargs
    ) -> tuple[float, float, float]:
        """
        Overwrites the _velocity_eom() defined in CowellPropagator to compute the differential velocity between the true
        and osculating orbit instead of the true velocity.
        """

        y_ref = kwargs.get("y_ref")
        y_true = y_ref + y[:6]
        ref_radius = np.sqrt(y_ref[0] ** 2 + y_ref[1] ** 2 + y_ref[2] ** 2)

        # Compute the Encke parameter and function. If the absolute value of the Encke parameter is smaller than
        # encke_tol use an infinite series definition of the Encke function, otherwise use its analytic form.
        encke_param = -1 / ref_radius ** 2 * (
                y[0] * (y_ref[0] + 0.5 * y[0])
                + y[1] * (y_ref[1] + 0.5 * y[1])
                + y[2] * (y_ref[2] + 0.5 * y[2])
        )

        if abs(encke_param) < self._encke_tol:
            encke_func = 0
            for i in range(1, self._encke_series_length):
                encke_func += (-sp.special.factorial2(2 * i + 3)
                               / (sp.special.factorial(i + 1)) * encke_param ** i
                               )
        else:
            encke_func = 1 / encke_param * (1 - (1 - 2 * encke_param) ** -1.5)

        # Compute derivative of the velocity difference.
        y3_dot = satellite.orbit.grav_param / ref_radius ** 3 * (encke_func * encke_param * y_true[0] - y[0])
        y4_dot = satellite.orbit.grav_param / ref_radius ** 3 * (encke_func * encke_param * y_true[1] - y[1])
        y5_dot = satellite.orbit.grav_param / ref_radius ** 3 * (encke_func * encke_param * y_true[2] - y[2])

        return y3_dot, y4_dot, y5_dot

    def _attitude_eom(
            self, t: float, y: np.ndarray, satellite: spacecraft.Satellite, **kwargs
    ) -> tuple[float, float, float, float, float, float, float]:
        y_ref = kwargs.get("y_ref")
        y = np.stack((y[:6] + y_ref, y[6:].copy()), axis=-1)  # Attitude EOM expect true position.

        y6_dot, y7_dot, y8_dot, y9_dot, y10_dot, y11_dot, y12_dot = super()._attitude_eom(t, y, satellite, **kwargs)
        return y6_dot, y7_dot, y8_dot, y9_dot, y10_dot, y11_dot, y12_dot

    def _rk4(
            self,
            t0: float,
            y0: np.ndarray,
            delt: float,
            satellite: spacecraft.Satellite,
            **kwargs
    ) -> np.ndarray:
        """
        Modified version of _rk4() from CowellPropagator.

        Note that this is not integration of the true state but rather the state difference. However, the EOM for the
        state difference requires the true state, so it must be reconstructed from the sum of the reference state and
        RK4 extrapolation of the state difference at every point at which RK4 integration is performed.
        """

        y0 = y0 - kwargs.get("y0_ref")  # State -> state difference.
        x1 = super()._eom_compiler(t0, y0, satellite, y_ref=kwargs.get("y0_ref"))
        x2 = super()._eom_compiler(t0 + delt / 2, y0 + delt / 2 * x1, satellite, y_ref=kwargs.get("y1_ref"))
        x3 = super()._eom_compiler(t0 + delt / 2, y0 + delt / 2 * x2, satellite, y_ref=kwargs.get("y1_ref"))
        x4 = super()._eom_compiler(t0 + delt, y0 + delt * x3, satellite, y_ref=kwargs.get("y2_ref"))

        return y0 + delt / 6 * (x1 + 2 * x2 + 2 * x3 + x4)

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from . import base

if TYPE_CHECKING:
    from .. import spacecraft, perturbations


class CowellPropagator(base.Propagator):
    r"""
    Simplest non-Keplerian propagate which numerically integrates the equations of motion of a satellite using a
    4th-order Runge-Kutta method. This is known as Cowell's method by astrodynamicists.

    Two things set this apart from Keplerian methods. First, it can handle perturbing forces like
    :class:`~hohmannpy.astro.NonSphericalEarth`. However, in addition the accuracy of the propagation decreases over
    time as opposed to a Keplerian propagator which has a fixed accuracy. To mitigate this decrease the step size.

    step_size : float
        Time interval between propagation steps. If one is not provided by the user it will be set in
        :meth:`propagate()` to 60 :math:`s`.
    """

    def __init__(
            self,
            step_size: float = 60,
    ):
        super().__init__(step_size)

    def propagate(
            self,
            satellites: dict[str, spacecraft.Satellite],
            runtime: float,
            perturbing_forces: list[perturbations.Perturbation] = None
    ):
        r"""
        Perform orbit propagation using Cowell's method.

        Parameters
        ----------
        satellites : dict[str, :class:`~hohmannpy.astro.Satellite`]
            Dictionary which hold the orbits to propagate as an attribute named ``orbit`` attached to each satellite.
            Satellites are indexed by their name.
        runtime : float
            How many :math:`s` to run the propagation for.
        perturbing_forces : list[:class:`~hohmannpy.astro.Perturbation`]
            Perturbations to add to the mission to increase the fidelity of orbital simulation.
        """

        super().propagate(satellites, runtime, perturbing_forces)

        # Get initial values used for propagation and set up logging capabilities.
        for name, satellite in self.satellites.items():
            for logger in satellite.loggers:
                logger.setup(initial_orbit=satellite.orbit, timesteps=self.timesteps)

        # Begin the actual propagation loop. This is made of two loops: timesteps (outer), satellites (inner).
        # For each satellite, first retrieve the orbit. Then use Runge-Kutta 4-th order integration to compute the
        # position and velocity at the next timestep.
        for timestep in range(1, self.timesteps + 1):
            for name, satellite in self.satellites.items():
                orbit = satellite.orbit
                state = self.rk4(
                    t0=orbit.time,
                    y0=np.concatenate((orbit.position, orbit.velocity)),
                    satellite=satellite,
                )
                orbit.time += self.step_size  # Advance time.
                orbit.position = np.array(state[:3])
                orbit.velocity = np.array(state[3:])

                # Use the new position and velocity to update all the orbital elements.
                orbit.update_classical()
                if orbit.track_equinoctial:
                    orbit.update_equinoctial()

            # Save results from this timestep.
            self.log(timestep)

    def eom(
            self,
            t: float,
            y: np.ndarray,
            satellite: spacecraft.Satellite
    ) -> np.ndarray:
        r"""
        Equations of motion for a spacecraft in first order form where the state is given as (position, velocity).

        The default acceleration is the two-body acceleration due to the point mass acceleration of the central body.
        The perturbing accelerations are then added by calling :class:`~hohmannpy.astro.Perturbation` .
        :class:`~hohmannpy.astro.Perturbation.evaluate()` for each perturbation in ``perturbing_forces``.

        Parameters
        ----------
        t: float
            Current time since propagation began,
        y : np.ndarray
            (6, ) array representing the satellite's current state as (position, velocity).
        satellite : :class:`~hohmannpy.astro.Satellite`
            The satellite whose orbit is being propagated. Do not access the position and velocity of the satellite
            through its ``orbit`` attribute. Only use this to access static properties like ``orbit.grav_param``.

        Returns
        -------
        acceleration: np.ndarray
            (6, ) array corresponding the derivative of the satellite's current state as (velocity, acceleration).
        """

        radius = np.sqrt(y[0] ** 2 + y[1] ** 2 + y[2] ** 2)

        # Compute derivative of the position.
        y0_dot = y[3]
        y1_dot = y[4]
        y2_dot = y[5]

        # Compute derivative of velocity.
        y3_dot = -satellite.orbit.grav_param / radius ** 3 * y[0]
        y4_dot = -satellite.orbit.grav_param / radius ** 3 * y[1]
        y5_dot = -satellite.orbit.grav_param / radius ** 3 * y[2]

        # Append perturbing forces.
        if self.perturbing_forces is not None:
            for perturbing_force in self.perturbing_forces:
                y3_perturb, y4_perturb, y5_perturb = perturbing_force.evaluate(t, y, satellite)
                y3_dot += y3_perturb
                y4_dot += y4_perturb
                y5_dot += y5_perturb

        return np.array([y0_dot, y1_dot, y2_dot, y3_dot, y4_dot, y5_dot])

    def rk4(
            self,
            t0: float,
            y0: np.ndarray,
            satellite: spacecraft.Satellite
    ) -> np.ndarray:
        r"""
        Perform one step of 4th-order Runge Kutta integration.

        Parameters
        ----------
        t0 : float
            Base time point at which to start integration step.
        y0 : np.ndarray
            Base state point at which to start integration step.
        satellite : :class:`~hohmannpy.astro.Satellite`
            The satellite whose orbit is being propagated. Do not access the position and velocity of the satellite
            through its ``orbit`` attribute. Only use this to access static properties like ``orbit.grav_param``.

        Returns
        -------
        y: np.ndarray
            Approximated state at time t0 + step_size.

        """

        x1 = self.eom(t0, y0, satellite)
        x2 = self.eom(t0 + self.step_size / 2, y0 + self.step_size / 2 * x1, satellite)
        x3 = self.eom(t0 + self.step_size / 2, y0 + self.step_size / 2 * x2, satellite)
        x4 = self.eom(t0 + self.step_size, y0 + self.step_size * x3, satellite)

        return y0 + self.step_size / 6 * (x1 + 2 * x2 + 2 * x3 + x4)

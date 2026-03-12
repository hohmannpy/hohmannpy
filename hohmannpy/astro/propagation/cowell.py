from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from . import base

if TYPE_CHECKING:
    from .. import spacecraft, perturbations


# TODO: Investigating implementing functools.cache for Taylor series.
class CowellPropagator(base.Propagator):
    r"""
    Simplest non-Keplerian propagate which numerically integrates the equations of motion of a satellite using a
    4th-order Runge-Kutta method. This is known as Cowell's method by astrodynamicists.

    Two things set this apart from Keplerian methods. First, it can handle perturbing forces like
    :class:`~hohmannpy.astro.NonSphericalEarth`. However, in addition the accuracy of the propagation decreases over
    time as opposed to a Keplerian propagator which has a fixed accuracy. To mitigate this decrease the step size.

    Parameters
    ----------
    step_size : float
        Time interval between propagation steps. If one is not provided by the user it will be set in
        :meth:`propagate()` to 60 :math:`s`.

    Attributes
    ----------
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
            # Setup the loggers.
            burns = len(satellite.impulsive_burns) + len(satellite.continuous_burns)

            for logger in satellite.loggers:
                logger.setup(initial_orbit=satellite.orbit, timesteps=self.timesteps, burns=burns)

        # Begin the actual propagation loop. This is made of two loops: timesteps (outer), satellites (inner).
        # This involves a lot of logic surrounding burns which boils down to just determining when to call step(). The
        # steps taken (for a given satellite on a given timestep) are as follows:
        #   1) Set the next "standard time" of propagation to be the current time + timestep.
        #   2) Start a true loop that iterates through all events scheduled between the current time and the next
        #       "standard time". An event can be one of three things: an impulsive burn, a continuous burn starting, or
        #       a continuous burn ending.
        #   3) For each of these event types, determine when the next will occur (if any). Then, out of these determine
        #       which event will occur next.
        #   4) For each iteration of the loop, take a mini-timestep from the current time to the time of the next event.
        #       Then, propagate over this mini-timestep.
        #   5) The next action depends on the type of event.
        #       Impulsive burn:
        #           i) Add the change in velocity.
        #           ii) Update the orbital elements after the impulse and log the results manually.
        #       Continuous burn start:
        #           i) Increment the satellite's continuous_burn_start_index by 1.
        #       Continuous burn end:
        #           i) Increment the satellite's continuous_burn_end_index by 1.
        #       Note that the actual application acceleration due to the continuous burn is handled independently of
        #       this loop by eom(). This loop simply ensures that the discrete time grid includes the exact times at
        #       which a continuous.rst burn starts and stops to prevent discontinuities in integration.
        #   6) Repeat 3-5 until all events scheduled before the next standard time are completed.
        #   7) Take a mini-timestep from the time of the last event till the next standard time. Then, propagate over
        #       this mini-timestep.
        for timestep in range(1, self.timesteps + 1):
            for name, satellite in self.satellites.items():
                if satellite.impulsive_burns or satellite.continuous_burns:  # Skip this step if no burns are scheduled.
                    next_std_time = satellite.orbit.time + self.step_size

                    # Event loop.
                    while True:
                        # For each event type fetch the next event's time. Each time a burn happens the event's index is
                        # incremented, and if this is equivalent to the number of events of that type than all events of
                        # that type are complete. If this is true set the event time to None.
                        if satellite.impulsive_burn_index < len(satellite.impulsive_burns):
                            impulsive_burn = satellite.impulsive_burns[satellite.impulsive_burn_index]
                            next_impulsive_time = impulsive_burn.start_time
                        else:
                            next_impulsive_time = None

                        if satellite.continuous_burn_start_index < len(satellite.continuous_burns):
                            continuous_burn = satellite.continuous_burns[satellite.continuous_burn_start_index]
                            next_continuous_start_time = continuous_burn.start_time
                        else:
                            next_continuous_start_time = None

                        if satellite.continuous_burn_end_index < len(satellite.continuous_burns):
                            continuous_burn = satellite.inverted_continuous_burns[satellite.continuous_burn_end_index]
                            next_continuous_end_time = continuous_burn.start_time
                        else:
                            next_continuous_end_time = None

                        # Construct a dict out of the soonest upcoming times of each type. Note that at this point
                        # these events may occur after next_std_time. We then use list comprehension to remove None
                        # values. Then select the event that will occur soonest.
                        candidate_events = [
                            ("impulsive", next_impulsive_time),
                            ("continuous_start", next_continuous_start_time),
                            ("continuous_end", next_continuous_end_time),
                        ]
                        valid_events = [(name, time) for name, time in candidate_events if time is not None]
                        if not valid_events:
                            break
                        event_type, next_event_time = min(valid_events, key=lambda x: x[1])

                        # If this event would occur before next_std_time, propagate to its event time and then perform
                        # the necessary logic.
                        if next_std_time >= next_event_time:
                            if event_type == "impulsive":
                                self.step(name, satellite, next_event_time - satellite.orbit.time)
                                impulsive_burn.evaluate(satellite)
                                satellite.orbit.update_classical()
                                if satellite.orbit.track_equinoctial:
                                    satellite.orbit.update_equinoctial()
                                self.log(satellite)  # Log this data because it isn't logged in evaluate().
                            elif event_type == "continuous_start":
                                self.step(name, satellite, next_event_time - satellite.orbit.time)
                                satellite.continuous_burn_start_index += 1  # No evaluate() so update index manually.
                            elif event_type == "continuous_end":
                                self.step(name, satellite, next_event_time - satellite.orbit.time)
                                satellite.continuous_burn_end_index += 1
                        else:
                            break

                    # After all events, increment to the next_std_time and perform normal propagation.
                    self.step(name, satellite, next_std_time - satellite.orbit.time)

                # No events, so simply propagate to next standard time.
                else:
                    self.step(name, satellite, self.step_size)

    def step(self, name, satellite, time_change):
        r"""
        One step in the propagation loop.

        Parameters
        ----------
        name : str
            Name of the satellite being propagated.
        satellite: :class:`~hohmannpy.astro.Satellite`
            Satellite being propagated. Holds the orbit to propagate as an attribute named ``orbit``.
        time_change : float
            Change in time to propagate over.
        """

        # For each satellite, first retrieve the orbit. Then step the state and position forward by one timestep using
        # RK4 integration of the satellite's EOM.
        orbit = satellite.orbit

        state = self.rk4(
            t0=orbit.time,
            delt=time_change,
            y0=np.concatenate((orbit.position, orbit.velocity)),
            satellite=satellite,
        )
        orbit.time += time_change
        orbit.position = np.array(state[:3])
        orbit.velocity = np.array(state[3:])

        # Use the new position and velocity to update all the orbital elements.
        orbit.update_classical()
        if orbit.track_equinoctial:
            orbit.update_equinoctial()

        # Save results from this timestep.
        self.log(satellite)

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
        :class:`~hohmannpy.astro.Perturbation.evaluate()` for each perturbation in ``perturbing_forces`` as well as any
        :class:`~hohmannpy.astro.ContinuousBurns` from ``satellite.continuous_burns``.

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
        y_dot: np.ndarray
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

        # Append active continuous burns. Do these first because they can change masses.
        for burn in satellite.continuous_burns:
            if burn.start_time <= t <= burn.end_time:  # Check if burn is active.
                y3_perturb, y4_perturb, y5_perturb = burn.evaluate(t, y, satellite)
                y3_dot += y3_perturb
                y4_dot += y4_perturb
                y5_dot += y5_perturb

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
            delt: float,
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
        delt : float
            Time increment to propagate over.
        satellite : :class:`~hohmannpy.astro.Satellite`
            The satellite whose orbit is being propagated. Do not access the position and velocity of the satellite
            through its ``orbit`` attribute. Only use this to access static properties like ``orbit.grav_param``.

        Returns
        -------
        y: np.ndarray
            Approximated state at time t0 + step_size.
        """

        x1 = self.eom(t0, y0, satellite)
        x2 = self.eom(t0 + delt / 2, y0 + delt / 2 * x1, satellite)
        x3 = self.eom(t0 + delt / 2, y0 + delt / 2 * x2, satellite)
        x4 = self.eom(t0 + delt, y0 + delt * x3, satellite)

        return y0 + delt / 6 * (x1 + 2 * x2 + 2 * x3 + x4)

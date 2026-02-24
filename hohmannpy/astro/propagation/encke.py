from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import scipy as sp

from . import base, universal_variable
from ..orbit import Orbit

if TYPE_CHECKING:
    from .. import spacecraft, perturbations


# TODO: Investigating implementing functools.cache for Taylor series.
class EnckePropagator(universal_variable.UniversalVariablePropagator):
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

    Parameters
    ----------
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

    Attributes
    ----------
    rectification_tol : float
        When the deviation between the true and reference orbits grows large enough (represented by the ratio of their
        positions' magnitudes being greater than this tolerance), reset the rectified orbit by setting it equal to the
        current true orbit.
    encke_tol : float
        When the Encke parameter is close to zero (defined by this tolerance) the Encke function, which is used to
        compute the position, is undefined so must switch to an infinite series definition of it.
    encke_series_length : int
        How many terms to include when using the infinite series definition of the Encke function.
    reference_orbits : dict[str, :class:`~hohmannpy.astro.Orbit`]
        Keplerian orbits used as references to measure deviations of the true orbit from the Keplerian approximation.
    """

    def __init__(
            self,
            step_size: float = 60,
            rectification_tol: float = 0.01,
            encke_tol: float = 1e-8,
            encke_series_length: int = 10,
            solver_tol: float = 1e-8,
            stumpff_tol: float = 1e-8,
            stumpff_series_length: int = 10,
            fg_constraint: bool = True
    ):
        self.rectification_tol = rectification_tol
        self.encke_tol = encke_tol
        self.encke_series_length = encke_series_length

        # Empty dict containing initial conditions that get filled in propagate()
        self.reference_orbits = {}

        # Unlike other propagations which just inherit from base.Propagate(), this class instead inherits from
        # UniversalVariablePropagator so we also need to instantiate this with all its parameters.
        super().__init__(step_size, solver_tol, stumpff_tol, stumpff_series_length, fg_constraint)

    def propagate(
            self,
            satellites: dict[str, spacecraft.Satellite],
            runtime: float,
            perturbing_forces: list[perturbations.Perturbation] = None
    ):
        r"""
        Perform orbit propagation using Encke's method.

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

        # Don't want to call super() because UniversalVariablePropagator.propagate() includes a bunch of extra logic we
        # don't want to run.
        base.Propagator.propagate(self, satellites, runtime, perturbing_forces)

        # Get initial values used for propagation and set up logging capabilities. Note that the entire first code block
        # matches UniversalVariablePropagator.propagate() exactly, so look there for more details.
        for name, satellite in self.satellites.items():
            # Setup from UniversalVariablePropagator.propagate().
            self.initial_times[name] = satellite.orbit.time
            self.initial_positions[name] = satellite.orbit.position.copy()
            self.initial_velocities[name] = satellite.orbit.velocity.copy()

            self.reference_orbits[name] = Orbit.from_state(
                position=satellite.orbit.position.copy(),
                velocity=satellite.orbit.velocity.copy(),
                grav_param=satellite.orbit.grav_param
            )
            self.reference_orbits[name].universal_variable = 0
            self.reference_orbits[name].stumpff_param = 0
            self.reference_orbits[name].inverse_sm_axis = (
                    (2 * satellite.orbit.grav_param / np.linalg.norm(self.initial_positions[name])
                     - np.linalg.norm(self.initial_velocities[name]) ** 2)
                    / satellite.orbit.grav_param
            )

            # Setup the loggers.
            burns = len(satellite.impulsive_burns) + len(satellite.continuous_burns)

            for logger in satellite.loggers:
                logger.setup(initial_orbit=satellite.orbit, timesteps=self.timesteps, burns=burns)

        # Begin the actual propagation loop. This is made of two loops: timesteps (outer), satellites (inner).
        # This involves a lot of logic surrounding burns which boils down to just determining when to call step(). The
        # steps taken (for a given satellite on a given timestep) are as follows:
        #   1) Set the next "standard time" of propagation to be the current time + timestep.
        #   2) Start a true loop that iterates through all events scheduled between the current time and the next
        #       "standard time". An event can be one of three things: an impulsive burn, a continuous.rst burn starting, or
        #       a continuous.rst burn ending.
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
        #       Note that the actual application acceleration due to the continuous.rst burn is handled independently of
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

                                # Unlike with CowellPropagator, there is a bunch of extra logic needed to reset all the
                                # initial conditions used by the reference orbits. See
                                # UniversalVariablePropagator.propagate() for more information.
                                self.reference_orbits[name].position = satellite.orbit.position.copy()
                                self.reference_orbits[name].velocity = satellite.orbit.velocity.copy()
                                self.reference_orbits[name].update_classical()
                                self.initial_times[name] = satellite.orbit.time
                                self.initial_positions[name] = satellite.orbit.position.copy()
                                self.initial_velocities[name] = satellite.orbit.velocity.copy()
                                self.reference_orbits[name].universal_variable = 0
                                self.reference_orbits[name].stumpff_param = 0
                                self.reference_orbits[name].inverse_sm_axis = (
                                        (2 * satellite.orbit.grav_param / np.linalg.norm(self.initial_positions[name])
                                         - np.linalg.norm(self.initial_velocities[name]) ** 2)
                                        / satellite.orbit.grav_param
                                )

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

        # For each satellite, first retrieve the orbit and reference orbit. Then the reference orbit is propagated
        # analytically to the next timestep via the universal variable formulation of Kepler's equation by calling
        # reference_step(). Afterwards, the difference between the true and reference orbits is propagated by using RK4
        # to numerically integrate this difference's EOMs. The way this difference is propagated when using RK4 is a
        # little bit unintuitive, so see the comments rk4() for more details. Then, sum these to get the true positions
        # and velocities. Finally, check if the deviation between the true and reference orbits is large enough to
        # require rectification of the reference orbit (setting reference orbit = true orbit).
        orbit = satellite.orbit

        # Calculate the reference state (position, velocity) at the old time and then propagate it to the new time. Also
        # propagate it at a halfway point between them, this is needed by RK4 integration.
        old_reference_state = np.concatenate(
            [self.reference_orbits[name].position.copy(), self.reference_orbits[name].velocity.copy()], axis=0
        )
        self.reference_step(name, time_change / 2)
        intermediate_reference_state = np.concatenate(
            [self.reference_orbits[name].position.copy(), self.reference_orbits[name].velocity.copy()], axis=0
        )
        self.reference_step(name, time_change / 2)

        # Perform numerical integration to get the state difference.
        del_state = self.rk4(
            t0=orbit.time,
            delt=time_change,
            y0=np.concatenate((orbit.position, orbit.velocity)),
            y0_ref=old_reference_state,
            y1_ref=intermediate_reference_state,
            y2_ref=np.concatenate(
            [self.reference_orbits[name].position, self.reference_orbits[name].velocity], axis=0
                ),
            satellite=satellite,
        )
        del_position = np.array(del_state[:3])
        del_velocity = np.array(del_state[3:])
        orbit.time += time_change

        # Compute the true position and velocity.
        orbit.position = self.reference_orbits[name].position + del_position
        orbit.velocity = self.reference_orbits[name].velocity + del_velocity

        # Perform rectification if needed.
        if np.linalg.norm(del_position) / np.linalg.norm(orbit.position) > self.rectification_tol:
            self.reference_orbits[name].position = orbit.position.copy()
            self.reference_orbits[name].velocity = orbit.velocity.copy()
            self.reference_orbits[name].update_classical()

            # If rectification is performed, need to reset all the initial conditions for the reference orbit because
            # universal variable propagation over changes in angular momentum.
            self.initial_times[name] = orbit.time
            self.initial_positions[name] = orbit.position.copy()
            self.initial_velocities[name] = orbit.velocity.copy()
            self.reference_orbits[name].universal_variable = 0
            self.reference_orbits[name].stumpff_param = 0
            self.reference_orbits[name].inverse_sm_axis = (
                    (2 * satellite.orbit.grav_param / np.linalg.norm(self.initial_positions[name])
                     - np.linalg.norm(self.initial_velocities[name]) ** 2)
                    / satellite.orbit.grav_param
            )

        # Use the new position and velocity to update all the orbital elements.
        orbit.update_classical()
        if orbit.track_equinoctial:
            orbit.update_equinoctial()

        # Save results from this timestep.
        self.log(satellite)

    def reference_step(self, name, time_change):
        r"""
        Perform propagation using the universal variable form of Kepler's method for the reference orbit.

        Parameters
        ----------
        name : str
            Name of the satellite being propagated.
        time_change : float
            Change in time to propagate over.
        """

        # For information on this, see UniversalVariablePropagator.step()
        orbit = self.reference_orbits[name]
        orbit.time += time_change

        orbit.universal_variable = self.kepler_equation(
            inverse_sm_axis=orbit.inverse_sm_axis,
            grav_param=orbit.grav_param,
            time=orbit.time,
            initial_time=self.initial_times[name],
            initial_position=self.initial_positions[name],
            initial_velocity=self.initial_velocities[name],
            initial_guess=orbit.universal_variable,
        )
        orbit.stumpff_param = orbit.inverse_sm_axis * orbit.universal_variable ** 2
        s_func, c_func = self.stumpff_funcs(orbit.stumpff_param)
        f_func = 1 - orbit.universal_variable ** 2 / np.linalg.norm(self.initial_positions[name]) * c_func
        g_func = (
                orbit.time - self.initial_times[name]
                - orbit.universal_variable ** 3 / np.sqrt(orbit.grav_param) * s_func
        )
        orbit.position = f_func * self.initial_positions[name] + g_func * self.initial_velocities[name]
        fdot_func = (
                np.sqrt(orbit.grav_param)
                / (np.linalg.norm(orbit.position) * np.linalg.norm(self.initial_positions[name]))
                * orbit.universal_variable * (orbit.stumpff_param * s_func - 1)
        )
        if self.fg_constraint:
            gdot_func = (g_func * fdot_func + 1) / f_func
        else:
            gdot_func = 1 - orbit.universal_variable ** 2 / np.linalg.norm(orbit.position) * c_func
        orbit.velocity = fdot_func * self.initial_positions[name] + gdot_func * self.initial_velocities[name]

    def eom(
            self,
            t: float,
            del_y: np.ndarray,
            y_ref: np.ndarray,
            satellite: spacecraft.Satellite
    ) -> np.ndarray:
        r"""
        Equations of motion for the difference between the true and reference states of a spacecraft in first order form
        where the state is given as (position, velocity).

        The default acceleration is the difference in the two-body accelerations due to the point mass acceleration of
        the central body for the true and reference orbits. However, to avoid numerical issues when the difference in
        these values are small an alternative formulation using what are known as the Encke function and parameter are
        instead used.

        The perturbing accelerations are then added by calling
        :class:`~hohmannpy.astro.Perturbation` . :meth:`~hohmannpy.astro.Perturbation.evaluate()` for each perturbation
        in ``perturbing_forces`` as well as any :class:`~hohmannpy.astro.ContinuousBurns` from
        ``satellite.continuous_burns``.

        Parameters
        ----------
        t: float
            Current time since propagation began,
        del_y : np.ndarray
            (6, ) array representing the difference between satellite's current state as (position, velocity) and the
            reference orbit's state.
        y_ref : np.ndarray
            (6, ) array representing the reference orbit's state (position, velocity).
        satellite : :class:`~hohmannpy.astro.Satellite`
            The satellite whose orbit is being propagated. Do not access the position and velocity of the satellite
            through its ``orbit`` attribute. Only use this to access static properties like ``orbit.grav_param``.

        Returns
        -------
        del_y_dot: np.ndarray
            (6, ) array corresponding the derivative of the satellite's current state difference as (velocity,
            acceleration).
        """

        # Calculate the true state by adding the state difference to the reference state.
        y = y_ref + del_y

        ref_radius = np.sqrt(y_ref[0] ** 2 + y_ref[1] ** 2 + y_ref[2] ** 2)

        # Compute the Encke parameter and function. If the absolute value of the Encke parameter is smaller than
        # encke_tol use an infinite series definition of the Encke function, otherwise use its analytic form.
        encke_param = -1 / ref_radius ** 2 * (
                del_y[0] * (y_ref[0] + 0.5 * del_y[0])
                    + del_y[1] * (y_ref[1] + 0.5 * del_y[1])
                    + del_y[2] * (y_ref[2] + 0.5 * del_y[2])
        )

        if abs(encke_param) < self.encke_tol:
            encke_func = 0
            for i in range(self.encke_series_length):
                encke_func += (-sp.special.factorial2(2 * i + 3)
                                    / (sp.special.factorial(i + 1) * 2 ** (i + 1)) * encke_param ** i
                               )
        else:
            encke_func = 1 / encke_param * (1 - (1 - 2 * encke_param) ** -1.5)

        # Compute derivative of the position difference.
        del_y0_dot = del_y[3]
        del_y1_dot = del_y[4]
        del_y2_dot = del_y[5]

        # Compute derivative of the velocity difference.
        del_y3_dot = satellite.orbit.grav_param / ref_radius ** 3 * (encke_func * encke_param * y[0] - del_y[0])
        del_y4_dot = satellite.orbit.grav_param / ref_radius ** 3 * (encke_func * encke_param * y[1] - del_y[1])
        del_y5_dot = satellite.orbit.grav_param / ref_radius ** 3 * (encke_func * encke_param * y[2] - del_y[2])

        # Append perturbing forces.
        if self.perturbing_forces is not None:
            for perturbing_force in self.perturbing_forces:
                y3_perturb, y4_perturb, y5_perturb = perturbing_force.evaluate(t, y, satellite)
                del_y3_dot += y3_perturb
                del_y4_dot += y4_perturb
                del_y5_dot += y5_perturb

        # Append active continuous.rst burns.
        for burn in satellite.continuous_burns:
            if burn.start_time <= t <= burn.end_time:  # Check if burn is active.
                y3_perturb, y4_perturb, y5_perturb = burn.evaluate(t, y, satellite)
                del_y3_dot += y3_perturb
                del_y4_dot += y4_perturb
                del_y5_dot += y5_perturb

        return np.array([del_y0_dot, del_y1_dot, del_y2_dot, del_y3_dot, del_y4_dot, del_y5_dot])

    def rk4(
            self,
            t0: float,
            y0: np.ndarray,
            y0_ref: np.ndarray,
            y1_ref: np.ndarray,
            y2_ref: np.ndarray,
            delt: float,
            satellite: spacecraft.Satellite
    ) -> np.ndarray:
        r"""
        Perform one step of 4th-order Runge Kutta integration.

        Note that this is not integration of the true state but rather the state difference. However, the EOM for the
        state difference requires the true state, so it must be reconstructed from the sum of the reference state and
        RK4 extrapolation of the state difference at every point at which RK4 integration is performed.

        Parameters
        ----------
        t0 : float
            Base time point at which to start integration step.
        y0 : np.ndarray
            Base state point at which to start integration step.
        y0_ref : np.ndarray
            Reference state at the first RK4 step.
        y1_ref : np.ndarray
            Reference state at the second and third RK4 steps.
        y2_ref : np.ndarray
            Reference state at the third RK4 step.
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

        del_y0 = y0 - y0_ref
        x1 = self.eom(t0, del_y0, y0_ref, satellite)
        x2 = self.eom(t0 + delt / 2, del_y0 + delt / 2 * x1, y1_ref, satellite)
        x3 = self.eom(t0 + delt / 2, del_y0 + delt / 2 * x2, y1_ref, satellite)
        x4 = self.eom(t0 + delt, del_y0 + delt * x3, y2_ref, satellite)

        return del_y0 + delt / 6 * (x1 + 2 * x2 + 2 * x3 + x4)
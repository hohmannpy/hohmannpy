from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import scipy as sp

from . import kepler

if TYPE_CHECKING:
    from .. import spacecraft, perturbations


class UniversalVariablePropagator(kepler.KeplerPropagator):
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

    def _propagate(
            self,
            satellites: dict[str, spacecraft.Satellite],
            runtime: float,
            perturbing_forces: list[perturbations.Perturbation] = None
    ):
        r"""
        Perform orbit propagation using the universal variable form of Kepler's method.

        Parameters
        ----------
        satellites : dict[str, :class:`~hohmannpy.astro.Satellite`]
            Dictionary which hold the orbits to propagate as an attribute named ``orbit`` attached to each satellite.
            Satellites are indexed by their name.
        runtime : float
            How many :math:`s` to run the propagation for.
        perturbing_forces : list[:class:`~hohmannpy.astro.Perturbation`]
            Perturbations to add to the mission to increase the fidelity of orbital simulation. Note that if any are
            added a non-Keplerian propagator such as ``CowellPropagator`` must be used.
        """

        super()._propagate(satellites, runtime, perturbing_forces)

        # Get initial values used for propagation and set up logging capabilities. This involves iterating through each
        # satellite and extracting attributes of their orbits. Like the satellites themselves these are stored as
        # dictionaries where the satellite name is the key and the property itself is the value.
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

            # Setup the loggers.
            burns = len(satellite.impulsive_burns)

            for logger in satellite.loggers:
                logger.setup(initial_orbit=satellite.orbit, timesteps=self._timesteps, burns=burns)

        # Begin the actual propagation loop. This is made of two loops: timesteps (outer), satellites (inner).
        # This involves a lot of logic surrounding burns which boils down to just determining when to call step(). The
        # steps taken (for a given satellite on a given timestep) are as follows:
        #   1) Set the next "standard time" of propagation to be the current time + timestep.
        #   2) Start a true loop that iterates through all burns scheduled between the current time and the next
        #       "standard time".
        #   3) For each iteration of the loop, take a mini-timestep from the current time to the time of the next burn.
        #       Then, propagate over this mini-timestep.
        #   4) Only impulsive burns possible (Keplerian propagator), so add the change in velocity.
        #   5) Update the orbital elements after the impulse and log the results manually.
        #   6) Reset the initial values used for propagation to match the new orbit.
        #   7) Repeat 3-6 until all burns scheduled before the next standard time are completed.
        #   8) Take a mini-timestep from the time of the last burn till the next standard time. Then, propagate over
        #       this mini-timestep.
        for timestep in range(1, self._timesteps + 1):
            for name, satellite in self._satellites.items():
                if satellite.impulsive_burns:  # Skip this step if no burns are scheduled.
                    next_std_time = satellite.orbit.time + self._step_size

                    # Burn loop.
                    while True:
                        # Fetch next burn. Each time a burn happens the impulsive_burn_index is incremented, and if this
                        # is equivalent to the number of scheduled burns than all burns are complete, and we can break
                        # from the loop.
                        if satellite.impulsive_burn_index < len(satellite.impulsive_burns):
                            burn = satellite.impulsive_burns[satellite.impulsive_burn_index]
                        else:
                            break

                        # If this burn would occur before next_std_time, propagate to its burn time and then perform the
                        # burn.
                        if next_std_time >= burn.start_time:
                            satellite.orbit.time = burn.start_time

                            self._step(name, satellite)

                            # Update elements because evaluate() does not automatically change orbital parameters.
                            burn.evaluate(satellite)
                            satellite.orbit.update_classical()
                            if satellite.orbit.track_equinoctial:
                                satellite.orbit.update_equinoctial()

                            # Keplerian propagation is not possible over changes in angular momentum, so need to restart
                            # propagation (hence find new initial conditions) at the point immediately after the burn
                            # occurs.
                            self._initial_times[name] = satellite.orbit.time
                            self._initial_positions[name] = satellite.orbit.position.copy()
                            self._initial_velocities[name] = satellite.orbit.velocity.copy()

                            satellite.orbit.universal_variable = 0
                            satellite.orbit.stumpff_param = 0
                            satellite.orbit.inverse_sm_axis = (
                                    (2 * satellite.orbit.grav_param / np.linalg.norm(self._initial_positions[name])
                                     - np.linalg.norm(self._initial_velocities[name]) ** 2)
                                    / satellite.orbit.grav_param
                            )

                            self._log(satellite)  # Log this data because it isn't logged in evaluate().
                        else:
                            break

                    # After all burns, increment to the next_std_time and perform normal propagation.
                    satellite.orbit.time = next_std_time
                    self._step(name, satellite)

                # No burns, so simply propagate to next standard time.
                else:
                    satellite.orbit.time += self._step_size
                    self._step(name, satellite)

    def _step(self, name, satellite):
        r"""
        One step in the propagation loop.

        Parameters
        ----------
        name : str
            Name of the satellite being propagated.
        satellite: :class:`~hohmannpy.astro.Satellite`
            Satellite being propagated. Holds the orbit to propagate as an attribute named ``orbit``.
        """

        # For each satellite, first retrieve the orbit. Then, compute the universal variable on the next time step from
        # Kepler's equation. From there the Stumpff series can be computed and in turn used to assemble the f and g
        # functions and their derivatives. Finally, from these the position and velocity may be found at the next
        # timestep.
        orbit = satellite.orbit

        # Compute new universal variable. Use the previous universal variable as the initial guess for the
        # root-finder.
        orbit.universal_variable = self._kepler_equation(
            inverse_sm_axis=orbit.inverse_sm_axis,
            grav_param=orbit.grav_param,
            time=orbit.time,
            initial_time=self._initial_times[name],
            initial_position=self._initial_positions[name],
            initial_velocity=self._initial_velocities[name],
            initial_guess=orbit.universal_variable,
        )

        # Compute the Stumpff (c and s) functions.
        orbit.stumpff_param = orbit.inverse_sm_axis * orbit.universal_variable ** 2
        s_func, c_func = self._stumpff_funcs(orbit.stumpff_param)

        # Compute the f and g functions.
        f_func = 1 - orbit.universal_variable ** 2 / np.linalg.norm(self._initial_positions[name]) * c_func
        g_func = (
                orbit.time - self._initial_times[name]
                    - orbit.universal_variable ** 3 / np.sqrt(orbit.grav_param) * s_func
        )

        # Compute new position (and true anomaly). Only need to update fast variables because the other
        # orbital elements are constant for Keplerian orbits.
        orbit.position = f_func * self._initial_positions[name] + g_func * self._initial_velocities[name]
        orbit._update_true_anomaly()
        orbit._update_argl()
        orbit._update_true_latitude()

        # Compute fdot and gdot functions.
        fdot_func = (
                np.sqrt(orbit.grav_param)
                    / (np.linalg.norm(orbit.position) * np.linalg.norm(self._initial_positions[name]))
                    * orbit.universal_variable * (orbit.stumpff_param * s_func - 1)
        )
        if self._fg_constraint:  # Only compute gdot function manually if constraint usage is disabled.
            gdot_func = (g_func * fdot_func + 1) / f_func
        else:
            gdot_func = 1 - orbit.universal_variable ** 2 / np.linalg.norm(orbit.position) * c_func

        # Compute the new velocity.
        orbit.velocity = fdot_func * self._initial_positions[name] + gdot_func * self._initial_velocities[name]

        # Save results from this timestep.
        self._log(satellite)

    def _stumpff_funcs(self, stumpff_param) -> tuple[float, float]:
        r"""
        Computes the Stumpff functions/series for a given value of the Stumpff parameter.

        The form of the Stumpff series is not based off the type of orbit, instead it is based of the sign and magnitude
        of the Stumpff parameter. Large and positive = trigonometric, small = summation, large and negative = hyper-
        trigonometric.

        Parameters
        ----------
        stumpff_param : float
            The current Stumpff parameter.

        Returns
        -------
        s_func : float
            The "sine" Stumpff function/series.
        c_func : float
            The "cosine" Stumpff function/series.
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

        Parameters
        ----------
        time : float
            Current time.
        inverse_sm_axis : float
            Inverse of the semi-major axis of the orbit.
        grav_param : float
            Gravitational parameter of the orbit.
        initial_time : float
            Base point for time at which propagation began.
        initial_position : np.ndarray
            Base point for position when propagation began.
        initial_velocity : np.ndarray
            Base point for velocity when propagation began.
        initial_guess : float
            Initial guess for the universal variable.

        Returns
        -------
        universal_variable : float
            Universal_variable at the next time step.
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

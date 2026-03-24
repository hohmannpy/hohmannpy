from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from abc import ABC, abstractmethod

import numpy as np

if TYPE_CHECKING:
    from .. import perturbations, spacecraft, maneuvers


# TODO:
#  - (Post-alpha) There is a lot of redundant code between propagators, eliminate as much of it as possible via mixins
#        and dependency injection. Particularly wrt. UniversalVariablePropagator and EnckePropagator.
class Propagator(ABC):
    r"""
    Base class for all orbit propagators.

    This gets passed to :class:`~hohmannpy.astro.Mission` and used to simulate spacecrafts' orbits via
    :class:`~hohmannpy.astro.Mission` . :class:`~hohmannpy.astro.Mission.simulate()`. This class' :meth:`propagate()`
    method is then called to step the orbits through time. On each timestep of this process :meth:`log()` is called to
    store data on each orbit.

    Parameters
    ----------
    step_size : float
        Time interval between propagation steps. If one is not provided by the user it will be set in ``propagate()`` to
        60 :math:`s`.
    """

    name = ""  # Set by child classes and used for command line output.
    energy_conserving = False

    def __init__(self, step_size: float = 60):
        self._step_size = step_size

        self.satellites: Optional[dict[str, spacecraft.Satellite]] = None
        self._perturbing_forces: Optional[list[perturbations.base.Perturbation]] = None
        self._timesteps: Optional[int] = None  # How many discrete time points to propagate at.

        self._active_burns: dict[str, list[maneuvers.ContinuousBurn]] = {}

    def _propagate(
            self,
            satellites: dict[str, spacecraft.Satellite],
            runtime: float,
            perturbing_forces: list[perturbations.Perturbation] = None
    ):
        r"""
        Simulate one or more satellites' orbits in time.

        This method is designed to support child classes' implementations of it via a call to it using ``super()``.
        It fills in all the attributes that were set to ``None`` when ``__int__()`` was called.

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

        self._satellites = satellites
        self._perturbing_forces = perturbing_forces

        # Compute number of discrete timesteps to propagate for.
        self._timesteps = int(np.floor(runtime / self._step_size))

        # Propagators store a variety of variables in dictionaries indexed by satellite name. Store these variables now.
        for name, satellite in self._satellites.items():
            for logger in satellite.loggers:  # Also set up the loggers while we're at it.
                logger.setup(initial_orbit=satellite.orbit, timesteps=self._timesteps, events=satellite._num_events)

            self._set_initial_conditions(satellite)
            self._active_burns[name] = []

        # Begin the actual propagation loop. This is made of two loops: timesteps (outer), satellites (inner).
        # This involves a lot of logic surrounding burns which boils down to just determining when to call step(). The
        # steps taken (for a given satellite on a given timestep) are as follows:
        #   1) Set the next "standard time" of propagation to be the current time + timestep.
        #   2) Start a true loop that iterates through all events scheduled between the current time and the next
        #       "standard time". An event can be one of three things: an impulsive burn, a continuous burn starting, or
        #       a continuous burn ending.
        #   3) For each iteration of the loop, take a mini-timestep from the current time to the time of the next event.
        #       Then, propagate over this mini-timestep.
        #   4) The next action depends on the type of event.
        #       Impulsive burn:
        #           i) Add the change in velocity.
        #           ii) Update the orbital elements after the impulse and log the results manually.
        #       Continuous burn start:
        #           i) Activate continuous burn.
        #       Continuous burn end:
        #           i) Deactivate continuous burn.
        #       Note that the actual application acceleration due to the continuous burn is handled independently of
        #       this loop by eom(). This loop simply ensures that the discrete time grid includes the exact times at
        #       which a continuous burn starts and stops to prevent discontinuities in integration.
        #   5) Repeat 3-4 until all events scheduled before the next standard time are completed.
        #   6) Take a mini-timestep from the time of the last event till the next standard time. Then, propagate over
        #       this mini-timestep.
        for timestep in range(1, self._timesteps + 1):
            for name, satellite in self._satellites.items():
                if satellite._events:
                    next_std_time = satellite.orbit.time + self._step_size

                    while True:
                        event = satellite._events[satellite._events_index]

                        if next_std_time >= event[0]:
                            self._step_wrapper(satellite, event[0] - satellite.orbit.time)
                            match event[1]:
                                case "impulsive":
                                    event[2].evaluate(satellite)

                                    # Log this data because it isn't logged in evaluate() and _step wasn't called.
                                    satellite.orbit.update_classical()
                                    if satellite.orbit.track_equinoctial:
                                        satellite.orbit.update_equinoctial()

                                    self._set_initial_conditions(satellite)
                                    self._log(satellite)

                                    satellite._events_index += 2
                                case "continuous_start":
                                    self._active_burns[name].append(event[2])
                                    satellite._events_index += 1
                                case "continuous_end":
                                    self._active_burns[name][:] = [
                                        x for x in self._active_burns[name] if x is not event[2]
                                    ]
                                    satellite._events_index += 1
                        else:
                            break
                    self._step_wrapper(satellite, next_std_time - satellite.orbit.time)
                else:
                    self._step_wrapper(satellite, self._step_size)

    def _step_wrapper(self, satellite: spacecraft.Satellite, time_change: float):
        satellite.orbit.time += time_change
        self._step(satellite, time_change)

        # Update the needed orbital elements based on the propagation algorithm used.
        if not self.energy_conserving:
            satellite.orbit.update_classical()
            if satellite.orbit.track_equinoctial:
                satellite.orbit.update_equinoctial()
        else:
            satellite.orbit._update_true_anomaly()
            satellite.orbit._update_argl()
            satellite.orbit._update_true_latitude()

        # Save results from this timestep.
        self._log(satellite)

    def _log(self, satellite: spacecraft.Satellite):
        r"""
        For a satellite being propagated access their stored :class:`~hohmannpy.astro.Logger`'s and log data.

        Parameters
        ----------
        satellite : spacecraft.Satellite
            Spacecraft to log data for.
        """

        for logger in satellite.loggers:
            logger.log(current_orbit=satellite.orbit)

    @abstractmethod
    def _set_initial_conditions(self, satellite: spacecraft.Satellite):
        pass

    @abstractmethod
    def _step(self, satellite: spacecraft.Satellite, del_time: float):
        pass
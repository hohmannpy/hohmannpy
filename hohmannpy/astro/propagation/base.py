from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from abc import ABC, abstractmethod

import numpy as np
from ... import logging

if TYPE_CHECKING:
    from .. import perturbations, maneuvers
    from ... import spacecraft


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

    def __init__(self, step_size: float = 60, **kwargs):
        self._step_size = step_size

        self.satellites: Optional[dict[str, spacecraft.Satellite]] = None
        self._perturbing_forces: Optional[list[perturbations.base.Perturbation]] = None
        self._timesteps: Optional[int] = None  # How many discrete time points to propagate at.

        self._active_burns: dict[str, list[maneuvers.ContinuousBurn]] = {}

    def _propagate(
            self,
            satellites: dict[str, spacecraft.Satellite],
            runtime: float,  # Total length of the mission in seconds.
            include_rotation: bool,
            perturbing_forces: list[perturbations.Perturbation] = None,
            perturbing_torques: list[perturbations.Perturbation] = None,
    ):
        """
        Simulate one or more satellites' orbits in time.

        This method is designed to support child classes' implementations of it via a call to it using ``super()``.
        It fills in all the attributes that were set to ``None`` when ``__int__()`` was called.
        """

        self._satellites = satellites
        self._perturbing_forces = perturbing_forces
        self._perturbing_torques = perturbing_torques
        self._include_rotation = include_rotation

        if self._include_rotation and self._step_size > 1:
            raise AttributeError("When modeling attitude select a step size of no larger than 1 second.")

        # Compute number of discrete timesteps to propagate for.
        self._timesteps = int(np.floor(runtime / self._step_size))

        # Propagators store a variety of variables in dictionaries indexed by satellite name. Store these variables now.
        for name, satellite in self._satellites.items():
            for logger in satellite.loggers:  # Also set up the loggers while we're at it.
                if isinstance(logger, (logging.AttitudeLogger, logging.EulerLogger)):
                    logger.setup(
                        initial_obj=satellite.orientation, timesteps=self._timesteps, events=satellite._num_events
                    )
                else:
                    logger.setup(
                        initial_obj=satellite.orbit, timesteps=self._timesteps, events=satellite._num_events
                    )

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
                if satellite._events and satellite._event_index < len(satellite._events):
                    next_std_time = satellite.orbit.time + self._step_size

                    while True:
                        if satellite._event_index >= len(satellite._events):
                            break

                        event = satellite._events[satellite._event_index]

                        if next_std_time >= event[0]:
                            self._step_wrapper(satellite, event[0] - satellite.orbit.time)
                            match event[1]:
                                case "impulsive":
                                    event[2].evaluate(satellite)

                                    # Log this data because it isn't logged in evaluate() and _step wasn't called.
                                    satellite.orbit.update_classical()
                                    if satellite.orbit._track_equinoctial:
                                        satellite.orbit.update_equinoctial()

                                    self._set_initial_conditions(satellite)
                                    self._log(satellite)
                                case "continuous_start":
                                    self._active_burns[name].append(event[2])
                                case "continuous_end":
                                    self._active_burns[name][:] = [
                                        x for x in self._active_burns[name] if x is not event[2]
                                    ]
                            satellite._event_index += 1
                        else:
                            break
                    self._step_wrapper(satellite, next_std_time - satellite.orbit.time)
                else:
                    self._step_wrapper(satellite, self._step_size)

    def _step_wrapper(self, satellite: spacecraft.Satellite, time_change: float):
        """
        Called each timestep to propagate the orbit.

        This is a wrapper around child classes' step() implementation which handles shared code.
        """

        satellite.orbit.time += time_change
        self._step(satellite, time_change)

        # Update the needed orbital elements based on the propagation algorithm used.
        if not self.energy_conserving:
            satellite.orbit.update_classical()
            if satellite.orbit._track_equinoctial:
                satellite.orbit.update_equinoctial()
        else:
            satellite.orbit._update_true_anomaly()
            satellite.orbit._update_argl()
            satellite.orbit._update_true_latitude()

        if self._include_rotation:
            if satellite.orientation._track_euler:
                satellite.orientation.update_euler()

        # Save results from this timestep.
        self._log(satellite)

    def _log(self, satellite: spacecraft.Satellite):
        """
        For a satellite being propagated access their stored :class:`~hohmannpy.astro.Logger`'s and log data.
        """

        for logger in satellite.loggers:
            if isinstance(logger, (logging.AttitudeLogger, logging.EulerLogger)):
                logger.log(obj=satellite.orientation)
            else:
                logger.log(obj=satellite.orbit)

    @abstractmethod
    def _set_initial_conditions(self, satellite: spacecraft.Satellite):
        """
        Set the initial conditions of the satellite as needed by the propagation algorithm.

        This is used by Keplerian propagators which need an unchanging base point to propagate from in order to maintain
        energy conservation.
        """

        pass

    @abstractmethod
    def _step(self, satellite: spacecraft.Satellite, del_time: float):
        """
        One step in the propagation loop.

        All logic which isn't included in _step_wrapper() because it varies from class to class.
        """

        pass
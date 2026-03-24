from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from abc import ABC, abstractmethod

import numpy as np

if TYPE_CHECKING:
    from .. import perturbations, spacecraft


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

    def __init__(self, step_size: float = 60):
        self._step_size = step_size

        self.satellites: Optional[dict[str, spacecraft.Satellite]] = None
        self._perturbing_forces: Optional[list[perturbations.base.Perturbation]] = None
        self._timesteps: Optional[int] = None  # How many discrete time points to propagate at.

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

        for name, satellite in self._satellites.items():
            # Setup the loggers.
            burns = len(satellite.impulsive_burns) + len(satellite.continuous_burns)
            for logger in satellite.loggers:
                logger.setup(initial_orbit=satellite.orbit, timesteps=self._timesteps, burns=burns)

            self._set_initial_conditions(satellite)

        for timestep in range(1, self._timesteps + 1):
            for name, satellite in self._satellites.items():
                if satellite.events:
                    pass

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
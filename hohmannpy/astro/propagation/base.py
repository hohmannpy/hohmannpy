from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from .. import perturbations, spacecraft


class Propagator:
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

    Attributes
    ----------
    step_size : float
        Time interval between propagation steps. If one is not provided by the user it will be set in ``propagate()`` to
        60 :math:`s`.
    timesteps: Optional[int]
        How many discrete timepoints to propagate at. This is based purely on the :attr:`step_size` and doesn't include
        additional timesteps injected to account for events like burns.
    satellites : Optional[dict[str, :class:`~hohmannpy.astro.Satellite`]]
        Dictionary which hold the orbits to propagate as an attribute named ``orbit`` attached to each satellite.
        Satellites are indexed by their name.
    perturbing_forces : Optional[list[:class:`~hohmannpy.astro.Perturbation`]]
        Perturbations to add to the mission to increase the fidelity of orbital simulation. Note that if any are added
        a non-Keplerian propagator such as :class:`~hohmannpy.astro.CowellPropagator` must be used.
    """

    def __init__(self, step_size: float = 60):
        self.step_size = step_size

        self.satellites: Optional[dict[str, spacecraft.Satellite]] = None
        self.perturbing_forces: Optional[list[perturbations.base.Perturbation]] = None
        self.timesteps: Optional[int] = None

    def propagate(
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

        self.satellites = satellites
        self.perturbing_forces = perturbing_forces

        # Compute number of discrete timesteps to propagate for.
        self.timesteps = int(np.floor(runtime / self.step_size))

    def log(self, satellite: spacecraft.Satellite):
        r"""
        For a satellite being propagated access their stored :class:`~hohmannpy.astro.Logger`'s and log data.

        Parameters
        ----------
        satellite : spacecraft.Satellite
            Spacecraft to log data for.
        """

        for logger in satellite.loggers:
            logger.log(current_orbit=satellite.orbit)

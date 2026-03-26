from __future__ import annotations
import copy
from typing import Optional, Union

import numpy as np

from hohmannpy.astro import orbits, maneuvers
from hohmannpy.logging import logging
from hohmannpy.dynamics import attitude


class Satellite:
    r"""
    Basic spacecraft whose motion can be simulated using :class:`~hohmannpy.Mission`.

    Parameters
    ----------
    name : str
        Unique identifier of the spacecraft. Repeats are not allowed.
    starting_orbit : :class:`~hohmannpy.astro.Orbit`
        The orbit the spacecraft is in at the start of the mission.
    starting_orbit : Optional[:class:`~hohmannpy.dynamics.Orientation`]
        The orientation of the spacecraft at the start of the mission.
    color: str
        The color of the orbit and spacecraft to display in renderings.
    burns : Optional[list[Union[:class:`~hohmannpy.astro.ImpulsiveBurn`, :class:`~hohmannpy.astro.ContinuousBurn`]]]
        The set of impulsive and continuous.rst burns to schedule for this spacecraft.
    mass: Optional[float]
        Mass of the spacecraft in :math:`kg`. Needed for missions where the perturbation
        :class:`~hohmannpy.astro.SolarRadiation` is enabled.
    inertia : Optional[np.ndarray]
        The rotational inertia of the spacecraft stored in a (3, 3) numpy array. The coordination of this matrix
        is up to the user. A common choice is the principal axes of the spacecraft.
    ballistic_coeff: Optional[float]
        Dimensionless parameter proportional to the drag effects experienced by a spacecraft. Needed for missions where
        the perturbation :class:`~hohmannpy.astro.AtmosphericDrag` is enabled.
    mean_reflective_area : Optional[float]
        Average area exposed to solar radiation pressure in :math:`m^2`. Needed for missions where the perturbation
        :class:`~hohmannpy.astro.SolarRadiation` is enabled.
    reflectivity : Optional[float]
        Dimensionless parameter proportional to how much solar radiation is reflected by the ``mean_reflective_area``.
        0 = transparent, 1 = full absorption, and 2 = full reflection. Needed for missions where the perturbation
        :class:`~hohmannpy.astro.SolarRadiation` is enabled.

    Attributes
    ----------
    name : str
        Unique identifier of the spacecraft. Repeats are not allowed.
    starting_orbit : :class:`~hohmannpy.astro.Orbit`
        The orbit the spacecraft is in at the start of the perturbation.
    orbit : :class:`~hohmannpy.astro.Orbit`
        Current orbit of the spacecraft. Starts as a deep copy of the ``starting_orbit`` and then is updated on each
        timestep during propagation.
    loggers: list[:class:`~hohmannpy.astro.Logger`]
        Loggers which record data on each timestep during propagation. This attribute is initially set to ``None`` and
        is filled in by the ``__init__()`` of ``Mission``.
    color: str
        The color of the orbit and spacecraft to display in renderings.
    mass: Optional[float]
        Mass of the spacecraft in :math:`kg`. Needed for missions where the perturbation
        :class:`~hohmannpy.astro.SolarRadiation` is enabled.
    inertia : Optional[np.ndarray]
        The rotational inertia of the spacecraft stored in a (3, 3) numpy array. The coordination of this matrix
        is up to the user. A common choice is the principal axes of the spacecraft.
    ballistic_coeff: Optional[float]
        Dimensionless parameter proportional to the drag effects experienced by a spacecraft. Needed for missions where
        the perturbation :class:`~hohmannpy.astro.AtmosphericDrag` is enabled.
    mean_reflective_area : Optional[float]
        Average area exposed to solar radiation pressure in :math:`m^2`. Needed for missions where the perturbation
        :class:`~hohmannpy.astro.SolarRadiation` is enabled.
    reflectivity : Optional[float]
        Dimensionless parameter proportional to how much solar radiation is reflected by the ``mean_reflective_area``.
        0 = transparent, 1 = full absorption, and 2 = full reflection. Needed for missions where the perturbation
        :class:`~hohmannpy.astro.SolarRadiation` is enabled.

    Notes
    -----
    Once a satellite's orbit is propagator, the recorded data (attributes) of any :class:`~hohmannpy.astro.Logger`
    attached to this satellite can be access from the satellite using ``__getattr_()``.
    """

    def __init__(
            self,
            name: str,
            starting_orbit: orbits.Orbit,
            starting_orientation: Optional[attitude.Orientation] = None,
            color: str = "#FF073A",
            burns: Optional[list[Union[maneuvers.ImpulsiveBurn, maneuvers.ContinuousBurn]]] = None,
            mass: Optional[float] = None,
            inertia: Optional[np.ndarray] = None,
            ballistic_coeff: Optional[float] = None,
            mean_reflective_area: Optional[float] = None,
            reflectivity: Optional[float] = None
    ):
        self.name = name
        self.starting_orbit = starting_orbit
        self.starting_orientation = starting_orientation
        self.color = color

        # Form a list of all events that will need mandatory propagation steps. Sorting will then take place back in the
        # Mission class.
        self._events: list[tuple[float, str, Union[maneuvers.ImpulsiveBurn, maneuvers.ContinuousBurn]]] = []
        self._num_events: int = 0
        self._event_index: int = 0
        if burns is not None:
            for burn in burns:
                if isinstance(burn, maneuvers.ImpulsiveBurn):
                    self._events.append((burn.start_time, "impulsive", burn))
                else:
                    self._events.append((burn.start_time, "continuous_start", burn))
                    self._events.append((burn.end_time, "continuous_end", burn))
                self._num_events += 2

        # Perturbation-specific parameters.
        self.mass = mass
        self.inertia = inertia
        self.ballistic_coeff = ballistic_coeff
        self.mean_reflective_area = mean_reflective_area
        self.reflectivity = reflectivity

        # Other parameters.
        self.orbit: orbits.Orbit = copy.deepcopy(starting_orbit)  # This will be updated over time by the propagator.
        self.orientation: Optional[attitude.Orientation] = copy.deepcopy(starting_orientation)  # Same as above.
        self.loggers: Optional[list[logging.Logger]] = None  # Filled in by the __init__() of Mission.

    def __getattr__(self, name):
        r"""
        Access data from ``Loggers`` assigned to this object as if they were assigned to this class.
        """

        # Need a safeguard here because can't call self.(some attribute) inside __getattr__() because this can break
        # during the pickling which occurs during parallel processing.
        loggers = object.__getattribute__(self, "__dict__").get("loggers", None)

        if loggers is not None:
            for logger in loggers:
                if hasattr(logger, name):
                    return getattr(logger, name)
        raise AttributeError(f"This satellite has not logged data for {name}.")

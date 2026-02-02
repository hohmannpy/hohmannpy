from __future__ import annotations
import copy

import pandas as pd
import numpy as np

from . import propagation, perturbations, time, logging, spacecraft
from ..ui import rendering


class Mission:
    r"""
    Master class for all orbital simulations.

    Contains the ability to propagate the orbits of a set of :class:`~hohmannpy.astro.Satellite` and then render and
    propagate the results.

    Parameters
    ----------
    satellites : list of :class:`~hohmannpy.astro.Satellite`
        List of satellites whose orbits the Mission will propagate.
    initial_global_time : :class:`~hohmannpy.astro.Time`
        The Gregorian date and UT1 time to start the mission at.
    final_global_time : :class:`~hohmannpy.astro.Time`
        The Gregorian date and UT1 time to end the mission at.
    loggers : list[:class:`~hohmannpy.astro.Logger`]
        Loggers determine which data to record for each satellite during propagation. To see what data each logger
        records, check the attributes labeled ``..._history`` in their respective documentation. For example,
        :class:`~hohmannpy.astro.StateLogger` records the time, position, and velocity of each ``Satellite``. After
        ``simulate`` has been called, these values can also be accessed as attributes of each ``Satellite``.
    propagator : :class:`~hohmannpy.astro.Propagator`
        Propagation technique to use to simulate the orbits of each ``Satellite``.
    perturbing_forces : list[:class:`~hohmannpy.astro.Perturbation`]
        Perturbations to add to the mission to increase the fidelity of orbital simulation. Note that if any are added
        a non-Keplerian propagator such as :class:`~hohmannpy.astro.CowellPropagator` must be used.

    Attributes
    ----------
    satellites : dict[str, :class:`~hohmannpy.astro.Satellite`]
        Dictionary of satellites created using the ``name`` parameter of each ``Satellite`` as the key and the object
        itself as the value.
    initial_global_time : :class:`~hohmannpy.astro.Time`
        The Gregorian date and UT1 time to start the mission at.
    final_global_time : :class:`~hohmannpy.astro.Time`
        The Gregorian date and UT1 time to end the mission at.
    propagator : :class:`~hohmannpy.astro.Propagator`
        Propagation technique to use to simulate the orbits of each ``Satellite``. If none was passed during
        initialization, it will default to :class:`~hohmannpy.astro.UniversalVariablePropagator` unless any
        ``Perturbation`` in which case a ``CowellPropagator`` will be used.
    perturbing_forces : list[:class:`~hohmannpy.astro.Perturbation`]
        Perturbations to add to the mission to increase the fidelity of orbital simulation. Note that if any are added
        a non-Keplerian propagator such as :class:`~hohmannpy.astro.CowellPropagator` must be used.
    """

    def __init__(
            self,
            satellites: list[spacecraft.Satellite],
            initial_global_time: time.Time,
            final_global_time: time.Time,
            loggers: list[logging.Logger] = None,
            propagator: propagation.base.Propagator = None,
            perturbing_forces: list[perturbations.Perturbation] = None,
    ):
        # Instantiate all the passed-in attributes.
        self.perturbing_forces: list[perturbations.Perturbation] = perturbing_forces
        self.initial_global_time: time.Time = initial_global_time
        self.final_global_time: time.Time = final_global_time
        self.global_time: time.Time = initial_global_time

        # If the user did not pass in a propagator we need to assign one for them. If no perturbations are used we can
        # use the best-in-class Keplerian propagator, UniversalVariablePropagator(). If a perturbation is used instead
        # use CowellPropagator() to account for non-Keplerian effects.
        if propagator is None:
            if perturbing_forces is None:
                self.propagator: propagation.base.Propagator = (
                    propagation.universal_variable.UniversalVariablePropagator()
                )
            else:
                self.propagator: propagation.base.Propagator = propagation.cowell.CowellPropagator()
        else:
            self.propagator: propagation.base.Propagator = propagator

        # If the user did not pass in a logger default to recording the time and state.
        if loggers is None:
            loggers = [logging.StateLogger()]

        # Setup satellite data logging. For easy access the satellites are stored in a dictionary where their name is
        # the key and the object itself is the value. Each satellite is initialized with a logger attribute set to None,
        # we need to copy the logger list the user passed in and assign it to each satellite.
        self.satellites: dict[str, spacecraft.Satellite] = {}
        for satellite in satellites:
            self.satellites[satellite.name] = satellite
            satellite.loggers = copy.deepcopy(loggers)

            # There are a bunch of optional parameters for each satellite only needed for specific perturbations. We
            # want to make sure that if a perturbation is enabled that the user has input value for all the needed
            # optional parameters for each satellite.
            if self.perturbing_forces is not None:
                for perturbation in self.perturbing_forces:
                    if isinstance(perturbation, perturbations.AtmosphericDrag) and satellite.ballistic_coeff is None:
                        raise AttributeError("If AtmosphericDrag is enabled as a perturbation all satellites must have "
                                             "a value for the attribute 'ballistic coefficient'.")
                    if isinstance(perturbation, perturbations.SolarRadiation) and satellite.mass is None:
                        raise AttributeError("If SolarRadiation is enabled as a perturbation all satellites must have "
                                             "a value for the attribute 'mass'.")
                    if isinstance(perturbation, perturbations.SolarRadiation) and satellite.mean_reflective_area is None:
                        raise AttributeError("If SolarRadiation is enabled as a perturbation all satellites must have "
                                             "a value for the attribute 'mean reflective area'.")
                    if isinstance(perturbation, perturbations.SolarRadiation) and satellite.reflectivity is None:
                        raise AttributeError("If SolarRadiation is enabled as a perturbation all satellites must have"
                                             "a value for the attribute 'reflectivity'.")

    def simulate(self):
        r"""
        Propagate the orbits of all stored ``Satellite``.
        """

        self.propagator.propagate(
            satellites=self.satellites,
            perturbing_forces=self.perturbing_forces,
            runtime=(self.final_global_time.julian_date - self.initial_global_time.julian_date) * 86400,
        )  # Propagation uses units of seconds, so convert Gregorian/UT1 -> Julian Date -> seconds.

    def display(self, display_flag = "dynamic"):
        r"""
        Display the orbits of all satellites using :class:`~hohmannpy.ui.RenderEngine` or
        :class:`~hohmannpy.ui.DynamicRenderEngine`.

        This should only be called after :meth:`simulate()` is run.

        Parameters
        ----------
        display_flag : str
            Flag which indicates what type of rendering to use for the mission. If set to "dynamic" a real-time
            rendering will launch. Otherwise, a static rendering will be used.
        """

        if display_flag == "dynamic":
            engine = rendering.DynamicRenderEngine(
                satellites=self.satellites,
                runtime=(self.final_global_time.julian_date - self.initial_global_time.julian_date) * 86400,
                initial_global_time=self.initial_global_time,
            )
        else:
            engine = rendering.RenderEngine(
                satellites=self.satellites,
            )

        engine.render()  # Command which actually launches the graphical application.

    def save(self, target_directory: str, fp_accuracy: int):
        r"""
        Save all the data logged over the course of the mission. For each satellite, all logged data is stored as a .csv
        where each column represents a different variable and each row a timestep of propagation.

        This should only be called after :meth:`simulate()` is run.

        Parameters
        ----------
        target_directory : str
            The folder path to store all the result .csv's to.
        fp_accuracy : int
            How many digits past the decimal to record for each data point.
        """

        # Iterate through each Satellite in satellites and convert its data to a .csv. First, iterate through
        # each logger and call its concatenate() method. This returns all the data the logger stored as a (N, M) array
        # for N timesteps and M unique variables. Then access the loggers labels class attribute, a list whose ordering
        # corresponds to the columns returned by concatenate. Concatenate all of these together and then convert the
        # resulting variables to a pandas.DataFrame.
        for name, satellite in self.satellites.items():
            data = None
            labels = []

            for logger in satellite.loggers:
                local_data = logger.concatenate()
                local_labels = logger.labels

                if data is None:
                    data = local_data
                else:
                    data = np.hstack((data, local_data))
                labels.extend(local_labels)

            data_df = pd.DataFrame(data, columns=labels)
            data_df.to_csv(
                f"{target_directory}\\{name}.csv",
                index=False,
                float_format=f"%.{fp_accuracy}f"
            )

from __future__ import annotations
import copy
import concurrent.futures
from typing import Optional
import pickle
import time as python_time

import pandas as pd
import numpy as np

from hohmannpy.astro import propagation, perturbations, time, maneuvers
from hohmannpy.logging import logging
from hohmannpy import spacecraft
from hohmannpy.viewer import viewing


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
    include_rotation : bool
        If rotational dynamics should be simulated alongside translational ones.
    loggers : list[:class:`~hohmannpy.logging.Logger`]
        Loggers determine which data to record for each satellite during propagation. To see what data each logger
        records, check the attributes labeled ``..._history`` in their respective documentation. For example,
        :class:`~hohmannpy.logging.StateLogger` records the position, and velocity of each ``Satellite``. After
        :meth:`simulate()` has been called, these values can also be accessed as attributes of each ``Satellite``.
    propagator : :class:`~hohmannpy.astro.Propagator`
        Propagation technique to use to simulate the orbits of each ``Satellite``.
    perturbing_forces : list[:class:`~hohmannpy.astro.Perturbation`]
        Perturbations to add to the mission to increase the fidelity of orbital simulation. Note that if any are added
        a non-Keplerian propagator such as :class:`~hohmannpy.astro.CowellPropagator` must be used.
    perturbing_torques : list[:class:`~hohmannpy.astro.Perturbation`]
        Perturbations to add to the mission to increase the fidelity of rotational simulation.
    verbose: bool
        Whether to print information about propagation.
    cores : int
        How many cores to use propagation. If this is greater than 1 parallel computing will be enabled.

    Attributes
    ----------
    satellites : dict[str, :class:`~hohmannpy.astro.Satellite`]
        Dictionary of satellites created using the ``name`` parameter of each ``Satellite`` as the key and the object
        itself as the value. All data post-propagation regarding each satellite is stored here.
    """

    def __init__(
            self,
            satellites: list[spacecraft.Satellite],
            initial_global_time: time.Time,
            final_global_time: time.Time,
            include_rotation: bool = False,
            loggers: Optional[list[logging.Logger]] = None,
            propagator: Optional[propagation.base.Propagator] = None,
            perturbing_forces: Optional[list[perturbations.Perturbation]] = None,
            perturbing_torques: Optional[list[perturbations.Perturbation]] = None,
            verbose: bool = True,
            cores: int = 1
    ):
        # Instantiate all the passed-in attributes.
        self._perturbing_forces = perturbing_forces
        self._perturbing_torques = perturbing_torques
        self._initial_global_time = initial_global_time
        self._final_global_time = final_global_time
        self._verbose = verbose
        self._cores = cores
        self._include_rotation = include_rotation

        # If the user did not pass in a propagator we need to assign one for them. If no perturbations are used we can
        # use the best-in-class Keplerian propagator, UniversalVariablePropagator(). If a perturbation is used instead
        # use CowellPropagator() to account for non-Keplerian effects.
        if propagator is None:
            if self._perturbing_forces is None:
                self._propagator = (
                    propagation.universal_variable.UniversalVariablePropagator()
                )
            else:
                self._propagator = propagation.cowell.CowellPropagator()
        else:
            self._propagator = propagator

        # If the user did not pass in a logger default to recording the time and state.
        if loggers is None:
            loggers = [logging.TimeLogger(), logging.StateLogger()]

            if self._include_rotation:
                loggers.append(logging.AttitudeLogger())

        # Setup satellite data logging. For easy access the satellites are stored in a dictionary where their name is
        # the key and the object itself is the value. Each satellite is initialized with a logger attribute set to None,
        # we need to copy the logger list the user passed in and assign it to each satellite.
        self.satellites: dict[str, spacecraft.Satellite] = {}
        for satellite in satellites:
            self.satellites[satellite.name] = satellite
            satellite.loggers = copy.deepcopy(loggers)

            # Make sure if rotation is enabled the satellite has an inertia matrix.
            if self._include_rotation:
                if satellite.inertia is None:
                    raise AttributeError("If rotational dynamics are enabled all satellites must have a value for the "
                                         "attribute 'inertia'.")
                if satellite.starting_orientation is None:
                    raise AttributeError("If rotational dynamics are enabled all satellites must have a value for the "
                                         "attribute 'starting_orientation'.")

                # Safeguard to ensure Keplerian propagators are not used for orientation simulation.
                if ((isinstance(self._propagator, propagation.KeplerPropagator) or
                     isinstance(self._propagator, propagation.UniversalVariablePropagator))
                        and not isinstance(self._propagator, propagation.EnckePropagator)
                ):
                    raise TypeError(
                        f"Propagators of type {self._propagator} are not supported for rotational dynamics. Please "
                        f"use either CowellPropagator or EnckePropagator instead with a timestep of at most 1 second.")

            # Some satellites may be passed in with maneuvers scheduled. If these maneuvers are set to fire at times
            # determined by Time objects, we now convert those to relative seconds since mission start. This couldn't be
            # done sooner because when the Burn objects were created the start time of the mission may not have been
            # known.
            for event in satellite._events:
                if isinstance(event[2], maneuvers.ImpulsiveBurn):
                    if isinstance(event[2].start_time, time.Time):
                        event[2].start_time = (event[2].start_time.julian_date - initial_global_time.julian_date) * 86400

                        # Safeguard to make sure burn happens after mission start.
                        if event[2].start_time < 0:
                            raise ValueError("Burns may only be scheduled for after the start of the mission.")

                elif isinstance(event[2], maneuvers.ContinuousBurn):
                    if isinstance(event[2].start_time, time.Time):
                        event[2].start_time = (event[2].start_time.julian_date - initial_global_time.julian_date) * 86400
                    if isinstance(event[2].end_time, time.Time):
                        event[2].end_time = (event[2].end_time.julian_date - initial_global_time.julian_date) * 86400

                        # Safeguard to ensure Keplerian propagators are not used with continuous burns.
                        if ((isinstance(self._propagator, propagation.KeplerPropagator) or
                             isinstance(self._propagator, propagation.UniversalVariablePropagator))
                                and not isinstance(self._propagator, propagation.EnckePropagator)
                        ):
                            raise TypeError(
                                f"Propagators of type {self._propagator} are not supported for maneuvers of type "
                                f"ContinuousBurn.")

                        # Safeguard to make sure all satellites have mass attributes.
                        if satellite.mass is None and event[2].masses is None:
                            raise AttributeError(
                                "If a ContinuousBurn is scheduled this satellites must have a value for the "
                                "attribute 'mass' or the burn must have an accompanying mass curve via the "
                                "'masses' attribute.")

                        # Safeguard to make sure burn happens after mission start.
                        if event[2].start_time < 0:
                            raise ValueError("Burns may only be scheduled for after the start of the mission.")

            satellite._events.sort(key=lambda x: x[0])  # Sort from earliest to latest.

            # There are a bunch of optional parameters for each satellite only needed for specific perturbations. We
            # want to make sure that if a perturbation is enabled that the user has input value for all the needed
            # optional parameters for each satellite.
            if self._perturbing_forces is not None:
                # Safeguard to ensure Keplerian propagators are not used with perturbations.
                if ((isinstance(self._propagator, propagation.KeplerPropagator) or
                     isinstance(self._propagator, propagation.UniversalVariablePropagator))
                        and not isinstance(self._propagator, propagation.EnckePropagator)
                ):
                    raise TypeError(
                        f"Propagators of type {self._propagator} are not supported for perturbations. Please use either "
                        f"CowellPropagator or EnckePropagator instead.")

                for perturbation in self._perturbing_forces:
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

        # Perform some QOL assignment of Perturbation object attributes based on the initial and final global times so
        # the user doesn't have to redundantly pass these to both the Mission and these object's __init__()s.
        if self._perturbing_forces is not None:
            for perturbation in self._perturbing_forces:
                if isinstance(perturbation, perturbations.SolarRadiation):
                    perturbation._finalize__init__(self._initial_global_time, self._final_global_time)
                if isinstance(perturbation, perturbations.NonSphericalEarth):
                    perturbation._finalize__init__(self._initial_global_time.gmst)
                if isinstance(perturbation, perturbations.J2):
                    perturbation._finalize__init__(self._initial_global_time.gmst)
                if isinstance(perturbation, perturbations.AtmosphericDrag):
                    perturbation._finalize__init__(self._initial_global_time.gmst)
                if isinstance(perturbation, perturbations.ThirdBodyGravity):
                    perturbation._finalize__init__(self._initial_global_time, self._final_global_time)

    def simulate(self):
        r"""
        Propagate the orbits of all stored ``Satellite``.

        This also contains all the parallel computing logic.
        """

        if self._verbose:
            print("Propagating...\n")
            start_time = python_time.perf_counter()

        # Propagation uses units of seconds, so convert Gregorian/UT1 -> Julian Date -> seconds.
        runtime = (self._final_global_time.julian_date - self._initial_global_time.julian_date) * 86400

        if self._verbose:
            print(f"Satellites: {len(list(self.satellites))}")
            print(f"Algorithm: \t{self._propagator.name}")

        # Single core case.
        if self._cores == 1:
            if self._verbose:
                print(f"Multicore: \tFalse")

            self._propagator._propagate(
                satellites=self.satellites,
                runtime=runtime,
                perturbing_forces=self._perturbing_forces,
                include_rotation=self._include_rotation
            )

        # Multicore case. First, calculate how many satellites should be distributed to each core. These are then split
        # across a series of core_group dicts. Each core_group is just an extension of the satellites dicts with the
        # same {name : satellite} formatting for each item. In theory these are evenly distributed, and then any
        # remainders are given to the last core. Assuming N cores, parallel processing is executed using
        # concurrent.futures.ProcessPoolExecutor() and each instance is passed the Propagator for this mission as well
        # as the satellites to propagate from core_group. After propagation these satellites are and remerged with the
        # main satellites dict.
        else:
            satellites_per_core = int(np.floor(len(self.satellites.items()) / self._cores))
            if satellites_per_core == 0:  # Make sure there aren't more cores than satellites.
                raise ValueError("If M cores are assigned there must be N satellites to simulate, where N >= M.")

            # Create the core groups.
            core_group = {}
            core_groups = []
            core_group_ids = []

            i = 1
            for name, satellite in self.satellites.items():
                core_group[name] = satellite

                # Distribute satellites to core_group until satellites_per_core is reached then move to the next group.
                if len(core_group) == satellites_per_core:
                    core_groups.append(core_group.copy())
                    core_group_ids.append(i)
                    core_group = {}

                    i += 1

            if core_group:
                core_groups.append(core_group.copy())
                core_group_ids.append(i)

            if self._verbose:
                print(f"Multicore: \tTrue")
                print(f"Cores: \t\t{self._cores}")
                print(f"Sat/Core: \t{len(core_groups[0])}\n")

            # Execute the actual multiprocessing. This calls a helper function _parallel_propagate() which actually sets
            # up a clone of the passed in Propagator for parallel processing.
            with concurrent.futures.ProcessPoolExecutor(max_workers=self._cores) as executor:
                for core_group in executor.map(
                        Mission._parallel_propagate,
                        [(self._propagator, core_group, group_id, runtime, self._perturbing_forces, self._verbose) for core_group, group_id in zip(core_groups, core_group_ids)]
                ):
                    for name, satellite in core_group.items():  # Update satellites after propagation.
                        self.satellites[name] = satellite

        end_time = python_time.perf_counter()

        if self._verbose:
            print(f"\nPropagation complete in {end_time - start_time:.2f} seconds!")

    @staticmethod
    def _parallel_propagate(args: tuple[propagation.Propagator, dict[str, spacecraft.Satellite], float, list[perturbations.Perturbation]]) -> None:
        r"""
        Parallel processing helper function.

        This takes in all the values needed by each parallel processing instance in order to call
        Propagator.propagate().

        Parameters
        ----------
        args : tuple[:class:`~hohmannpy.astro.Propagator`, dict[str, :class:`~hohmannpy.Spacecraft], int, float, list[:class:~hohmannpy.astro.Perturbation`, bool]
            Tuple of arguments needed to start propagation on a core. These include the propagator itself, a ``dict`` of
            the satellites to propagate, how long to propagate for, and any perturbations.

        Returns
        -------
        satellites : dict[str, :class:`~hohmannpy.Spacecraft`]
            Propagated satellites to now reintegrate with the main ``satellite`` dict.
        """

        propagator, satellites, group_id, runtime, perturbing_forces, verbose = args
        propagator._propagate(
            satellites=satellites,
            runtime=runtime,
            perturbing_forces=perturbing_forces,
        )

        if verbose:
            print(f"Core {group_id} done")

        return satellites

    def display(self):
        r"""
        Display the orbits of all satellites using a Qt application.

        This should only be called after ``simulate()`` is run.
        """

        # Check to make sure trajectories were logged.
        loggers = next(iter(self.satellites.values())).loggers
        if not any(isinstance(logger, logging.StateLogger) for logger in loggers):
            raise AttributeError("No StateLogger stored for this mission, can not generate trajectories for display.")
        if not any(isinstance(logger, logging.TimeLogger) for logger in loggers):
            raise AttributeError("No TimeLogger stored for this mission, can not generate trajectories for display.")

        sim_manager = viewing.ViewerManager(
            self.satellites,
            self._initial_global_time,
            self._final_global_time,
            self._propagator._step_size
        )
        sim_manager.run()

    def to_csv(self, target_directory: str, fp_accuracy: int):
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
                f"{target_directory}/{name}.csv",
                index=False,
                float_format=f"%.{fp_accuracy}f"
            )

    def save(self, name: str, target_directory: str):
        r"""
        Pickle the ``Mission`` so that it may be loaded later.

        Parameters
        ----------
        name : str
            Name of the pickled ``Mission``.
        target_directory : str
            The folder path to store the pickled ``Mission`` in.
        """

        with open(f"{target_directory}/{name}.pkl", "wb") as f:
            pickle.dump(self, f)

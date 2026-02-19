from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import scipy as sp

from . import base, universal_variable
from ..orbit import Orbit

if TYPE_CHECKING:
    from .. import spacecraft, perturbations


# TODO: Documentation for this class.
class EnckePropagator(universal_variable.UniversalVariablePropagator):
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

        self.reference_orbits = {}

        super().__init__(step_size, solver_tol, stumpff_tol, stumpff_series_length, fg_constraint)

    def propagate(
            self,
            satellites: dict[str, spacecraft.Satellite],
            runtime: float,
            perturbing_forces: list[perturbations.Perturbation] = None
    ):
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
        for timestep in range(1, self.timesteps + 1):
            for name, satellite in self.satellites.items():
                if satellite.impulsive_burns or satellite.continuous_burns:
                    next_std_time = satellite.orbit.time + self.step_size

                    while True:
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

                        candidate_events = [
                            ("impulsive", next_impulsive_time),
                            ("continuous_start", next_continuous_start_time),
                            ("continuous_end", next_continuous_end_time),
                        ]
                        valid_events = [(name, time) for name, time in candidate_events if time is not None]
                        if not valid_events:
                            break
                        event_type, next_event_time = min(valid_events, key=lambda x: x[1])

                        if next_std_time >= next_event_time:
                            if event_type == "impulsive":
                                self.step(name, satellite, next_event_time - satellite.orbit.time)
                                impulsive_burn.evaluate(satellite)

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

                                self.log(satellite)

                            elif event_type == "continuous_start":
                                self.step(name, satellite, next_event_time - satellite.orbit.time)
                                satellite.continuous_burn_start_index += 1
                            elif event_type == "continuous_end":
                                self.step(name, satellite, next_event_time - satellite.orbit.time)
                                satellite.continuous_burn_end_index += 1
                        else:
                            break

                    self.step(name, satellite, next_std_time - satellite.orbit.time)

                else:
                    self.step(name, satellite, self.step_size)

    def step(self, name, satellite, time_change):
        """
        One step in the propagation loop.
        """

        orbit = satellite.orbit

        old_reference_state = np.concatenate(
            [self.reference_orbits[name].position.copy(), self.reference_orbits[name].velocity.copy()], axis=0
        )
        self.reference_step(name, time_change / 2)
        intermediate_reference_state = np.concatenate(
            [self.reference_orbits[name].position.copy(), self.reference_orbits[name].velocity.copy()], axis=0
        )
        self.reference_step(name, time_change / 2)

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
        orbit.position = self.reference_orbits[name].position + del_position
        orbit.velocity = self.reference_orbits[name].velocity + del_velocity

        if np.linalg.norm(del_position) / np.linalg.norm(orbit.position) > self.rectification_tol:
            self.reference_orbits[name].position = orbit.position.copy()
            self.reference_orbits[name].velocity = orbit.velocity.copy()
            self.reference_orbits[name].update_classical()

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
        y = y_ref + del_y

        ref_radius = np.sqrt(y_ref[0] ** 2 + y_ref[1] ** 2 + y_ref[2] ** 2)

        encke_param = -1 / ref_radius ** 2 * (
                del_y[0] * (y_ref[0] + 0.5 * del_y[0])
                    + del_y[1] * (y_ref[1] + 0.5 * del_y[1])
                    + del_y[2] * (y_ref[2] + 0.5 * del_y[2])
        )

        if abs(encke_param) < self.encke_tol:
            encke_func = 0
            for i in range(self.encke_series_length):
                encke_func += -sp.special.factorial2(2 * i + 3) / sp.special.factorial(i + 1) * encke_param ** i
        else:
            encke_func = 1 / encke_param * (1 - (1 - 2 * encke_param) ** -1.5)

        del_y0_dot = del_y[3]
        del_y1_dot = del_y[4]
        del_y2_dot = del_y[5]

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

        for burn in satellite.continuous_burns:
            if burn.start_time <= t <= burn.end_time:
                y3_perturb, y4_perturb, y5_perturb = burn.evaluate(t, y, satellite)
                del_y3_dot += y3_perturb
                del_y4_dot += y4_perturb
                del_y5_dot += y5_perturb

        return np.array([del_y0_dot, del_y1_dot, del_y2_dot, del_y3_dot, del_y4_dot, del_y5_dot])

    def rk4(
            self,
            t0: float,
            delt: float,
            y0: np.ndarray,
            y0_ref: np.ndarray,
            y1_ref: np.ndarray,
            y2_ref: np.ndarray,
            satellite: spacecraft.Satellite
    ) -> np.ndarray:
        del_y0 = y0 - y0_ref
        x1 = self.eom(t0, del_y0, y0_ref, satellite)
        x2 = self.eom(t0 + delt / 2, del_y0 + delt / 2 * x1, y1_ref, satellite)
        x3 = self.eom(t0 + delt / 2, del_y0 + delt / 2 * x2, y1_ref, satellite)
        x4 = self.eom(t0 + delt, del_y0 + delt * x3, y2_ref, satellite)

        return del_y0 + delt / 6 * (x1 + 2 * x2 + 2 * x3 + x4)
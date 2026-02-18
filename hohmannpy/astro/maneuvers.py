from __future__ import annotations
from typing import Union, Callable, TYPE_CHECKING

import numpy as np
import scipy as sp

from . import perturbations

if TYPE_CHECKING:
    from . import spacecraft, time


# TODO: Documentation.
class ImpulsiveBurn:
    def __init__(
            self,
            start_time: Union[float, time.Time],
            velocity_change: np.ndarray,
            inertial: bool = False,
    ):
        self.start_time = start_time
        self.velocity_change = velocity_change
        self.inertial = inertial

    def evaluate(self, satellite):
        if not self.inertial:
            sat_2_inertial_dcm =  self.compute_sat_2_inertial_dcm(satellite)
            velocity_change = sat_2_inertial_dcm @ self.velocity_change.copy()
        else:
            velocity_change = self.velocity_change

        satellite.orbit.velocity += velocity_change
        satellite.impulsive_burn_index += 1

    def compute_sat_2_inertial_dcm(self, satellite):
        radial_uvec = satellite.orbit.position / np.linalg.norm(satellite.orbit.position)
        normal_uvec = satellite.orbit.spf_angular_momentum / np.linalg.norm(satellite.orbit.spf_angular_momentum)
        transverse_uvec = np.cross(normal_uvec, radial_uvec)

        return np.stack((radial_uvec.T, transverse_uvec.T, normal_uvec.T), axis=1)


class ContinuousBurn(perturbations.Perturbation):
    def __init__(
            self,
            start_time: Union[float, time.Time],
            end_time: Union[float, time.Time],
            inertial: bool = False,
    ):
        super().__init__()

        self.start_time = start_time
        self.end_time = end_time
        self.inertial = inertial

    def evaluate(self, time: float, state: np.ndarray, satellite: spacecraft.Satellite) -> np.ndarray:
        pass

    def compute_sat_2_inertial_dcm(self, state):
        position = state[:3]
        velocity = state[3:]
        spf_angular_momentum = np.cross(position, velocity)

        radial_uvec = position / np.linalg.norm(position)
        normal_uvec = spf_angular_momentum / np.linalg.norm(spf_angular_momentum)
        transverse_uvec = np.cross(normal_uvec, radial_uvec)

        return  np.stack((radial_uvec.T, transverse_uvec.T, normal_uvec.T), axis=1)


class ConstantContinuousBurn(ContinuousBurn):
    def __init__(
            self,
            start_time: Union[float, time.Time],
            end_time: Union[float, time.Time],
            thrust: np.ndarray,
            inertial: bool = False,
    ):
        super().__init__(start_time, end_time, inertial)

        self.thrust = thrust

    def evaluate(self, time: float, state: np.ndarray, satellite: spacecraft.Satellite) -> np.ndarray:
        if not self.inertial:
            sat_2_inertial_dcm = self.compute_sat_2_inertial_dcm(state)
            thrust = sat_2_inertial_dcm @ self.thrust.copy()
        else:
            thrust = self.thrust

        return thrust / satellite.mass


class LookupContinuousBurn(ContinuousBurn):
    def __init__(
            self,
            start_time: Union[float, time.Time],
            end_time: Union[float, time.Time],
            times: np.ndarray,
            thrusts: np.ndarray,
            inertial: bool = False,
    ):
        super().__init__(start_time, end_time, inertial)

        self.burn_spline = sp.interpolate.make_interp_spline(
            times.squeeze(),
            thrusts,
            k=3
        )

    def evaluate(self, time: float, state: np.ndarray, satellite: spacecraft.Satellite) -> np.ndarray:
        if not self.inertial:
            sat_2_inertial_dcm = self.compute_sat_2_inertial_dcm(state)
            thrust = sat_2_inertial_dcm @ self.burn_spline(time)
        else:
            thrust = self.burn_spline(time)

        return thrust / satellite.mass


class FunctionContinuousBurn(ContinuousBurn):
    def __init__(
            self,
            start_time: Union[float, time.Time],
            end_time: Union[float, time.Time],
            thrust_function: Callable,
            inertial: bool = False,
    ):
        super().__init__(start_time, end_time, inertial)

        self.thrust_function = thrust_function

    def evaluate(self, time: float, state: np.ndarray, satellite: spacecraft.Satellite) -> np.ndarray:
        if not self.inertial:
            sat_2_inertial_dcm = self.compute_sat_2_inertial_dcm(state)
            thrust = sat_2_inertial_dcm @ self.thrust_function(time)
        else:
            thrust = self.thrust_function(time)

        return thrust / satellite.mass
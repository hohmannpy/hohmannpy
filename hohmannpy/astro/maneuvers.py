from __future__ import annotations
from typing import Union, Callable, TYPE_CHECKING

import numpy as np

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
            radial_uvec = satellite.orbit.position / np.linalg.norm(satellite.orbit.position)
            normal_uvec = satellite.orbit.spf_angular_momentum / np.linalg.norm(satellite.orbit.spf_angular_momentum)
            transverse_uvec = np.cross(normal_uvec, radial_uvec)

            sat_2_inertial_dcm = np.stack((radial_uvec.T, transverse_uvec.T, normal_uvec.T), axis=1)
            velocity_change = sat_2_inertial_dcm @ self.velocity_change.copy()
        else:
            velocity_change = self.velocity_change

        satellite.orbit.velocity += velocity_change
        satellite.impulsive_burn_index += 1


class ConstantContinuousBurn(perturbations.Perturbation):
    def __init__(
            self,
            start_time: Union[float, time.Time],
            end_time: Union[float, time.Time],
            thrust: np.ndarray,
            inertial: bool = False,
    ):
        super().__init__()

        self.start_time = start_time
        self.end_time = end_time
        self.thrust = thrust
        self.inertial = inertial

    def evaluate(self, time: float, state: np.ndarray, satellite: spacecraft.Satellite) -> np.ndarray:
        if not self.inertial:
            radial_uvec = satellite.orbit.position / np.linalg.norm(satellite.orbit.position)
            normal_uvec = satellite.orbit.spf_angular_momentum / np.linalg.norm(satellite.orbit.spf_angular_momentum)
            transverse_uvec = np.cross(normal_uvec, radial_uvec)

            sat_2_inertial_dcm = np.stack((radial_uvec.T, transverse_uvec.T, normal_uvec.T), axis=1)
            thrust = sat_2_inertial_dcm @ self.thrust.copy()
        else:
            thrust = self.thrust

        return thrust / satellite.mass


class LookupContinuousBurn(perturbations.Perturbation):
    def __init__(self, burn_profile: Callable):
        super().__init__()

    def evaluate(self, time: float, state: np.ndarray, satellite: spacecraft.Satellite) -> np.ndarray:
        pass


class FunctionContinuousBurn(perturbations.Perturbation):
    def __init__(self, burn_profile: Callable):
        super().__init__()

    def evaluate(self, time: float, state: np.ndarray, satellite: spacecraft.Satellite) -> np.ndarray:
        pass
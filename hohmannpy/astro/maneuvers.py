from __future__ import annotations
from typing import Union, Callable, TYPE_CHECKING

import numpy as np

from . import perturbations
from ..dynamics import dcms

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
        if isinstance(start_time, float):
            if start_time < 0:
                raise ValueError("Burns may only be scheduled for after the start of the mission.")

        self.start_time = start_time
        self.velocity_change = velocity_change
        self.inertial = inertial

    def evaluate(self, satellite):
        if not self.inertial:
            radial_uvec = satellite.orbit.position / np.linalg.norm(satellite.orbit.position)
            normal_uvec = satellite.orbit.spf_angular_momentum / np.linalg.norm(satellite.orbit.spf_angular_momentum)
            transverse_uvec = np.cross(normal_uvec, radial_uvec)

            sat_2_inertial_dcm = np.stack((radial_uvec.T, transverse_uvec.T, normal_uvec.T), axis=1)
            print(sat_2_inertial_dcm)
            velocity_change = sat_2_inertial_dcm @ self.velocity_change.copy()
        else:
            velocity_change = self.velocity_change

        satellite.orbit.velocity += velocity_change
        satellite.burn_index += 1


class ContinuousBurn(perturbations.Perturbation):
    def __init__(self, burn_profile: Callable):
        super().__init__()
#
#         self.burn_profile = burn_profile
#
#     @staticmethod
#     def constant_2_burn():
#         pass
#
#     @staticmethod
#     def arrays_2_burn():
#         pass
#
#     @staticmethod
#     def function_2_burn():
#         pass
#
#     @staticmethod
#     def csv_2_burn():
#         pass
#
#
    def evaluate(self, time: float, state: np.ndarray, satellite: spacecraft.Satellite) -> np.ndarray:
        pass
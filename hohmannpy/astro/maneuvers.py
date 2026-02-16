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
            true_anomaly = satellite.orbit.true_anomaly
            inclination = satellite.orbit.inclination
            argp = satellite.orbit.argp
            raan = satellite.orbit.raan

            inertial_2_sat_dcm = (
                dcms.euler_2_dcm(raan, 3)
                    @ dcms.euler_2_dcm(inclination, 1)
                    @ dcms.euler_2_dcm(argp, 3)
                    @ dcms.euler_2_dcm(true_anomaly, 3)
            )

            velocity_change = inertial_2_sat_dcm.T @ self.velocity_change.copy()
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
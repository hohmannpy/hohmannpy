from __future__ import annotations
from typing import Callable, TYPE_CHECKING

import numpy as np

from . import perturbations

if TYPE_CHECKING:
    from . import spacecraft


class ImpulsiveBurn:
    def __init__(self, burn_time, velocity_change):
        self.burn_time = burn_time
        self.velocity_change = velocity_change

    def evaluate(self, satellite):
        satellite.orbit.velocity += self.velocity_change

class ContinuousBurn(perturbations.Perturbation):
    def __init__(self, burn_profile: Callable):
        super().__init__()

        self.burn_profile = burn_profile

    @staticmethod
    def constant_2_burn():
        pass

    @staticmethod
    def arrays_2_burn():
        pass

    @staticmethod
    def function_2_burn():
        pass

    @staticmethod
    def csv_2_burn():
        pass


    def evaluate(self, time: float, state: np.ndarray, satellite: spacecraft.Satellite) -> np.ndarray:
        acceleration = self.burn_profile(time)

        return acceleration
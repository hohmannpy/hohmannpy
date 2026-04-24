from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from hohmannpy import spacecraft, perturbations


# TODO: This class.
class GravityGradient(perturbations.Perturbation):
    def __init__(self, attitude_only: bool = True):
        super().__init__()

    def evaluate(self, time: float, state: np.ndarray, satellite: spacecraft.Satellite) -> np.ndarray:
        pass
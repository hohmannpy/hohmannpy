from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from ...dynamics import dcms
from ...astro import perturbations

if TYPE_CHECKING:
    from ... import spacecraft

class GravityGradient(perturbations.Perturbation):
    def __init__(self):
        super().__init__()

    def evaluate(self, time: float, state: np.ndarray, satellite: spacecraft.Satellite) -> np.ndarray:
        pass
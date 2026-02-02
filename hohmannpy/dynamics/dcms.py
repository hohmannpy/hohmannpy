from __future__ import annotations

import numpy as np


def euler_2_dcm(angle, axis):
    if axis == 1:
        dcm = np.array(
            [[1, 0, 0],
            [0, np.cos(angle), np.sin(angle)],
            [0, -np.sin(angle), np.cos(angle)]]
        )
        return dcm
    elif axis == 2:
        dcm = np.array(
            [[np.cos(angle), 0, -np.sin(angle)],
             [0, 1, 0],
             [np.sin(angle), 0, np.cos(angle)]]
        )
        return dcm
    elif axis == 3:
        dcm = np.array(
            [[np.cos(angle), np.sin(angle), 0],
             [-np.sin(angle), np.cos(angle), 0],
             [0, 0, 1]]
        )
        return dcm
    else:
        raise ValueError(f"{axis} is not a valid axis for a Euler angle-based DCM to be generated about.")

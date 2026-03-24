from __future__ import annotations

import numpy as np


def euler_2_dcm(angle: float, axis: int):
    """
    Generate a direction cosine matrix (DCM) about the given axis (1, 2, or 3) using the provided Euler angle.

    Parameters
    ----------
    angle : float
        Angle to rotate by.
    axis : int
        Axis to rotate about.

    Returns
    -------
    dcm : np.ndarray
        A (3, 3) DCM which rotates about the provided axis by the provided angle.
    """

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

def axis_angle_2_dcm(angle, axis):
    pass

def quaternion_2_dcm(quaternion):
    pass
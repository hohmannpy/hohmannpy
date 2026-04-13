from __future__ import annotations

import numpy as np

from . import quaternions


def euler_2_dcm(angle: float, axis: int):
    r"""
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

def quaternion_2_dcm(q: quaternions.Quaternion):
    r"""
    Generate a direction cosine matrix (DCM) from a given quaternion.

    Parameters
    ----------
    q : quaternion.quaternion
        A (4, ) quaternion.

    Returns
    -------
    dcm : np.ndarray
       A (3, 3) DCM which rotates vectors by the provided quaternion.
    """

    q = q / quaternions.quaternion_norm(q)
    dcm = np.array(
        [
            [1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] - q[0] * q[3]), 2 * (q[1] * q[3] + q[0] * q[2])],
            [2 * (q[1] * q[2] + q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] - q[0] * q[1])],
            [2 * (q[1] * q[3] - q[0] * q[2]), 2 * (q[2] * q[3] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)]
        ]
    )

    return dcm


# TODO: These functions.
def vec_2_dcm(vec1: np.ndarray, vec2: np.ndarray, vec3: np.ndarray):
    pass


def dcm_2_euler(dcm: np.ndarray):
    pass


def dcm_2_quaternion(dcm: np.ndarray):
    pass


def dcm_2_vec(dcm: np.ndarray):
    pass

def vec_2_quaternion(vec1: np.ndarray, vec2: np.ndarray, vec3: np.ndarray):
    pass

def quaternion_2_vec(q: quaternions.Quaternion):
    pass

def euler_2_vec():
    pass

def quaternion_2_euler():
    pass

def vec_2_euler():
    pass
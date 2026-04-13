from __future__ import annotations
import numpy as np

from . import quaternions


# DCM GENERATION FUNCTIONS.
def euler_2_dcm(angle: float, axis: int) -> np.ndarray:
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

def quaternion_2_dcm(q: quaternions.Quaternion) -> np.ndarray:
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

def vecs_2_dcm(vec1: np.ndarray, vec2: np.ndarray, vec3: np.ndarray)-> np.ndarray:
    r"""
    Generate a direction cosine matrix (DCM) from a set of three ORTHOGONAL COLUMN vectors.

    Parameters
    ----------
    vec1 : np.ndarray
        A (3, ) vector.
    vec2 : np.ndarray
        Another (3, ) vector.
    vec3 : np.ndarray
        Yet another (3, ) vector.

    Returns
    -------
    dcm : np.ndarray
       A (3, 3) DCM which rotates vectors by the vectors.
    """

    uvec1 = vec1 / np.linalg.norm(vec1)
    uvec2 = vec2 / np.linalg.norm(vec2)
    uvec3 = vec3 / np.linalg.norm(vec3)

    dcm = np.stack([uvec1, uvec2, uvec3], axis=1)

    return dcm


# EULER ANGLE RETRIEVAL FUNCTIONS.
def dcm_2_euler(dcm: np.ndarray, sequence: str) -> tuple[float, float, float]:
    r"""
    Generate the euler1-euler2-euler3 Euler angle sequence from a provided DCM.

    Parameters
    ----------
    dcm : np.ndarray
       A (3, 3) DCM.
    sequence : str
        The Euler angle sequence to retrieve from the DCM (ex. 3-2-1 or 3-1-1) in the form euler1-euler2-euler3.

    Returns
    -------
    euler1 : float
       First angle in a rotation sequence
    euler2 : float
        Second angle in a rotation sequence.
    euler3 : float
        First angle in a rotation sequence.
    """

    match sequence:
        case "321":
            euler3 = np.arctan2(dcm[0, 1], dcm[0, 0])
            euler2 = np.arctan2(dcm[1, 2], dcm[2, 2])
            euler1 = float(np.arcsin(-dcm[1, 3]))
        case "313":
            euler3 = np.arctan2(dcm[2, 0], dcm[2, 1])
            euler2 = np.arctan2(dcm[0, 2], -dcm[1, 2])
            euler1 = np.arccos(dcm[2, 2])
        case _:
            raise NotImplementedError(f"Recovery of Euler angles for the {sequence} sequence is not supported.")

    return euler1, euler2, euler3

def quaternion_2_euler(q: quaternions.Quaternion, sequence: str) -> tuple[float, float, float]:
    r"""
    Generate the euler1-euler2-euler3 Euler angle sequence from a provided quaternion.

    Parameters
    ----------
    q : quaternions.Quaternion
       A (4, ) quaternion.
    sequence : str
        The Euler angle sequence to retrieve from the DCM (ex. 3-2-1 or 3-1-1) in the form euler1-euler2-euler3.

    Returns
    -------
    euler1 : float
       First angle in a rotation sequence
    euler2 : float
        Second angle in a rotation sequence.
    euler3 : float
        First angle in a rotation sequence.
    """

    dcm = quaternion_2_dcm(q)
    return dcm_2_euler(dcm, sequence)

def vecs_2_euler(vec1: np.ndarray, vec2: np.ndarray, vec3: np.ndarray, sequence: str) -> tuple[float, float, float]:
    r"""
    Generate the euler1-euler2-euler3 Euler angle sequence from a set of three ORTHOGONAL COLUMN vectors.

    Parameters
    ----------
    vec1 : np.ndarray
        A (3, ) vector.
    vec2 : np.ndarray
        Another (3, ) vector.
    vec3 : np.ndarray
        Yet another (3, ) vector.
    sequence : str
        The Euler angle sequence to retrieve from the DCM (ex. 3-2-1 or 3-1-1) in the form euler1-euler2-euler3.

    Returns
    -------
    q : np.ndarray
       A (4, ) quaternion.
    """

    dcm = vecs_2_dcm(vec1, vec2, vec3)
    return dcm_2_euler(dcm, sequence)


# QUATERNION RETRIEVAL FUNCTIONS.
def dcm_2_quaternion(dcm: np.ndarray):
    r"""
    Generate the quaternion corresponding to a provided DCM.

    Parameters
    ----------
    dcm : np.ndarray
       A (3, 3) DCM.

    Returns
    -------
    q : quaternion.quaternion
        A (4, ) quaternion.
    """

    trace = dcm.trace()
    if trace > 0:
        q0 = np.sqrt((1 + trace) / 4)
        q1 = (dcm[1, 2] - dcm[2, 1]) / (4 * q0)
        q2 = (dcm[2, 0] - dcm[0, 2]) / (4 * q0)
        q3 = (dcm[0, 1] - dcm[1, 0]) / (4 * q0)
    elif dcm[1, 1] > dcm[2, 2] and dcm[1, 1] > dcm[3, 3]:
        q0 = np.sqrt((1 + dcm[0, 0] - dcm[1, 1] - dcm[2, 2]) / 4)
        q1 = (dcm[1, 2] - dcm[2, 1]) / (4 * q0)
        q2 = (dcm[0, 1] + dcm[1, 0]) / (4 * q0)
        q3 = (dcm[2, 0] + dcm[0, 2]) / (4 * q0)
    elif dcm[2, 2] > dcm[3, 3]:
        q0 = np.sqrt((1 - dcm[0, 0] + dcm[1, 1] - dcm[2, 2]) / 4)
        q1 = (dcm[2, 0] - dcm[0, 2]) / (4 * q0)
        q2 = (dcm[0, 1] + dcm[1, 0]) / (4 * q0)
        q3 = (dcm[1, 2] + dcm[2, 1]) / (4 * q0)
    else:
        q0 = np.sqrt((1 - dcm[0, 0] - dcm[1, 1] + dcm[2, 2]) / 4)
        q1 = (dcm[0, 1] - dcm[1, 0]) / (4 * q0)
        q2 = (dcm[2, 0] + dcm[0, 2]) / (4 * q0)
        q3 = (dcm[1, 2] + dcm[2, 1]) / (4 * q0)

    return quaternions.Quaternion((q0, q1, q2, q3))

def vecs_2_quaternion(vec1: np.ndarray, vec2: np.ndarray, vec3: np.ndarray) -> np.ndarray:
    r"""
    Generate a quaternion from a set of three ORTHOGONAL COLUMN vectors.

    Parameters
    ----------
    vec1 : np.ndarray
        A (3, ) vector.
    vec2 : np.ndarray
        Another (3, ) vector.
    vec3 : np.ndarray
        Yet another (3, ) vector.

    Returns
    -------
    q : np.ndarray
       A (4, ) quaternion.
    """

    dcm = vecs_2_dcm(vec1, vec2, vec3)
    return dcm_2_quaternion(dcm)

def euler_2_quaternion(euler1: float, euler2: float, euler3: float, sequence: str) -> quaternions.Quaternion:
    r"""
    Generate a quaternion from the euler1-euler2-euler3 Euler angle sequence.

    Parameters
    ----------
    euler1 : float
       First angle in a rotation sequence
    euler2 : float
        Second angle in a rotation sequence.
    euler3 : float
        First angle in a rotation sequence.
    sequence : str
        The Euler angle sequence to retrieve from the DCM (ex. 3-2-1 or 3-1-1) in the form euler1-euler2-euler3.

    Returns
    -------
    q : np.ndarray
       A (4, ) quaternion.
    """

    dcm = (
            euler_2_dcm(euler1, int(sequence[2]))
                @ euler_2_dcm(euler2, int(sequence[1]))
                @ euler_2_dcm(euler3, int(sequence[0]))
    )
    return dcm_2_quaternion(dcm)

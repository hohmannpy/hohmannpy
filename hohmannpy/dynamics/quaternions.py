from typing import Union

import numpy as np


# TODO: Documentation.
class Quaternion:
    def __init__(self, data: Union[tuple, list, np.ndarray]):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=float)

        data = data.flatten()
        if len(data) != 4:
            raise ValueError("Quaternion must have 4 elements.")

        self._data: np.ndarray = data

    def __getitem__(self, index: Union[int, slice]) -> Union[float, np.ndarray]:
        return self._data[index]

    def __setitem__(self, index: int, value: float):
        self._data[index] = value

    def __add__(self, other: Quaternion) -> Quaternion:
        return Quaternion(self[:] + other[:])

    def __sub__(self, other: Quaternion) -> Quaternion:
        return Quaternion(self[:] - other[:])

    def __mul__(self, other: Quaternion) -> Quaternion:
        q0 = self[0]
        q1 = self[1]
        q2 = self[2]
        q3 = self[3]

        p0 = other[0]
        p1 = other[1]
        p2 = other[2]
        p3 = other[3]

        return Quaternion(np.array(
            [
                [q0 * p0 - q1 * p1 - q2 * p2 - q3 * p3],
                [q0 * p1 + q1 * p0 + q2 * p3 - q3 * p2],
                [q0 * p2 - q1 * p3 + q2 * p0 + q3 * p1],
                [q0 * p3 + q1 * p2 - q2 * p1 + q3 * p0],
            ]
        ))

    def __rmul__(self, other: Union[int, float]) -> Quaternion:
        return Quaternion(self._data * other)

    def conjugate(self) -> Quaternion:
        return Quaternion(np.array([self[0], -self[1], -self[2], -self[3]]))

    def invert(self) -> Quaternion:
        return Quaternion(
            np.array([self[0], -self[1], -self[2], -self[3]]) / np.linalg.norm(self[:]) ** 2
        )


def norm(quat: Quaternion) -> np.floating:
    return np.linalg.norm(quat[:])


# TODO: These functions.
def quaternion_2_euler(quat: Quaternion):
    pass


def euler_2_quaternion(angles: tuple[float, float, float]):
    pass

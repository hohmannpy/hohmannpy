from typing import TYPE_CHECKING

import numpy as np

from . import conversions

if TYPE_CHECKING:
    from . import quaternions


# TODO: Documentation and all functions involving the body frame.
def eci_2_ecef(vec: np.ndarray, gmst: float) -> np.ndarray:
    return conversions.euler_2_dcm(gmst, 3) @ vec

def eci_2_perifocal(vec: np.ndarray, raan: float, inclination: float, argp: float) -> np.ndarray:
    return conversions.euler_2_dcm(argp, 3) @ conversions.euler_2_dcm(inclination, 1) @ conversions.euler_2_dcm(raan, 3) @ vec

def eci_2_rtn(vec: np.ndarray, raan: float, inclination: float, argp: float, true_anomaly: float) -> np.ndarray:
    return perifocal_2_rtn(eci_2_perifocal(vec, raan, inclination, argp), true_anomaly)

def eci_2_body(
        quaternion: quaternions.Quaternion,
) -> np.ndarray:
    pass

def eci_2_heliocentric(vec: np.ndarray) -> np.ndarray:
    earth_tilt = np.deg2rad(-23.439291115)
    return conversions.euler_2_dcm(earth_tilt, 1).T @ vec

def ecef_2_eci(vec: np.ndarray, gmst: float) -> np.ndarray:
    return conversions.euler_2_dcm(gmst, 3).T @ vec

def perifocal_2_eci(vec: np.ndarray, raan: float, inclination: float, argp: float) -> np.ndarray:
    return (
            (
                conversions.euler_2_dcm(argp, 3)
                    @ conversions.euler_2_dcm(inclination, 1)
                    @ conversions.euler_2_dcm(raan, 3)
            ).T @ vec
    )

def perifocal_2_rtn(vec: np.ndarray, true_anomaly: float) -> np.ndarray:
    return conversions.euler_2_dcm(true_anomaly, 3) @ vec

def perifocal_2_body(vec: np.ndarray, true_anomaly: float, quaternion: quaternions.Quaternion) -> np.ndarray:
    return rtn_2_body(perifocal_2_rtn(vec, true_anomaly), quaternion)

def body_2_eci(
        vec: np.ndarray,
        raan: float,
        argp: float,
        inclination: float,
        true_anomaly: float,
        quaternion: quaternions.Quaternion,
) -> np.ndarray:
    pass

def body_2_perifocal(vec, quaternion: quaternions.Quaternion, true_anomaly: float) -> np.ndarray:
    pass

def body_2_rtn(vec: np.ndarray, quaternion: quaternions.Quaternion) -> np.ndarray:
    pass

def rtn_2_eci(vec: np.ndarray, raan: float, inclination: float, argp: float, true_anomaly: float) -> np.ndarray:
    return perifocal_2_eci(rtn_2_perifocal(vec, true_anomaly), raan, inclination, argp)

def rtn_2_perifocal(vec: np.ndarray, true_anomaly: float) -> np.ndarray:
    return conversions.euler_2_dcm(true_anomaly, 3).T @ vec

def rtn_2_body(vec: np.ndarray, quaternion: quaternions.Quaternion) -> np.ndarray:
    pass

def heliocentric_2_eci(vec: np.ndarray) -> np.ndarray:
    earth_tilt = np.deg2rad(-23.439291115)
    return conversions.euler_2_dcm(earth_tilt, 1) @ vec

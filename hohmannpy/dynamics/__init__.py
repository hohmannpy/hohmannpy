"""
Dynamics utilities designed for use in simulating spacecraft attitudes as well as a few additional utility functions
used by other modules.
"""


from .conversions import (
    euler_2_dcm, quaternion_2_dcm, vecs_2_dcm,
    dcm_2_quaternion, vecs_2_quaternion, euler_2_quaternion,
    dcm_2_euler, vecs_2_euler, quaternion_2_euler

)
from .attitude import Orientation
from .quaternions import Quaternion, quaternion_norm
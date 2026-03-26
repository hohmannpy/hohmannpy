"""
Dynamics utilities designed for use in simulating spacecraft attitudes as well as a few additional utility functions
used by other modules.
"""


from .dcms import euler_2_dcm, quaternion_2_dcm, dcm_2_euler, dcm_2_quaternion
from .attitude import Orientation
from .quaternions import Quaternion, euler_2_quaternion, quaternion_2_euler, quaternion_norm
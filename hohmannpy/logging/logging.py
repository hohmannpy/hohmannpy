from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    from ..astro import orbits
    from ..dynamics import attitude


class Logger(ABC):
    r"""
    A logger is used to store data regarding a :class:`~hohmannpy.astro.Orbit` generated on each timestep by
    :class:`~hohmannpy.astro.Propagator` . :meth:`~hohmannpy.astro.Propagator.propagate()`.
    """

    # Each subclass contains two class variables. They are both lists and are ordered identically to how the history
    # arrays are added to the classes internally.
    labels = []  # Display names for each history array, ex. "RAAN [rad]".
    attributes = []  # Internal list of only history array attributes, ex. "raan_history". Stores attribute names.

    def __init__(self):
        self._current_index: int = 0  # Tracks column-wise position along all history arrays.

    @abstractmethod
    def setup(self, initial_obj: Union[orbits.Orbit, attitude.Orientation], timesteps: int, events: int):
        r"""
        Sets up a logger.

        All child classes must implement this method with the following steps:

        1) Allocate space using :func:`numpy.zeros()` where the data is stored column-wise with N + M columns where N = the number of timesteps stored in the ``Propagator``'s ``timestep`` attribute and M is the number of events scheduled for the :class:`~hohmannpy.astro.Satellite` whose data is being logged.

        2) Fill in the 0th column of each array with the orbit's initial values for the stored data.

        Parameters
        ----------
        initial_obj : Union[:class:`~hohmannpy.astro.Orbit`, :class:`~hohmannpy.dynamics.Attitude`]
            The object which holds the data to log.
        timesteps : int
            How many timesteps to log the data for.
        events : int
            How many events to log data for. For events two points are logged. For impulsive events this is the
            immediately before and after the impulse (two data points on the same timestep). For continuous events this
            is the start and end times of the burn. These additional points are included for precision purposes.

        Notes
        -----
        Can't call this till after the initial values of propagator-unique attributes, such as ``eccentric_anomaly``
        for :class:`~hohmannpy.astro.KeplerPropagator`, have been set. This is typically towards the start of a
        propagators' ``propagate()`` method. This is why ``setup()`` is separated from ``__init__()``.
        """

        pass

    @abstractmethod
    def log(self, obj: Union[orbits.Orbit, attitude.Orientation]):
        r"""
        Fills in the next empty column of each history array with the orbit's current values for each data.

        Parameters
        ----------
        obj : Union[:class:`~hohmannpy.astro.`, :class:`~hohmannpy.dynamics.Attitude`]
            The object which holds the data to log.
        """

        pass

    @abstractmethod
    def concatenate(self) -> np.ndarray:
        """
        Takes the M history array attributes, each of length N timesteps, concatenates them into a (N, M) array, and
        then transposes and returns it.
        """

        pass

class TimeLogger(Logger):
    r"""
    Child of :class:`~hohmannpy.logging.Logger` that logs the time since mission start at N timesteps

    Attributes
    ----------
    time_history : np.ndarray
        (1, N) array of times with the mission start time set to 0. Units: :math:`s`.
    """

    labels = ["Time [s]"]
    attributes = ["time_history"]

    def __init__(self):
        super().__init__()

        self.time_history = None

    def setup(self, initial_obj: orbits.Orbit, timesteps: int, events: int):
        length = timesteps + events + 1

        self.time_history = np.zeros([1, length])

        self.time_history[0, 0] = initial_obj.time

    def log(self, obj: orbits.Orbit):
        self._current_index += 1  # Increment index.

        self.time_history[0, self._current_index] = obj.time

    def concatenate(self) -> np.ndarray:
        data = np.vstack((
            self.time_history,
        ))

        return data.T

class StateLogger(Logger):
    r"""
    Child of :class:`~hohmannpy.logging.Logger` that logs the ECI Cartesian state (position and velocity) of
    an orbit at N timesteps.

    Attributes
    ----------
    position_history : np.ndarray
        (3, N) array of the Cartesian positions in planet-centered inertial coordinates. Units: :math:`m`.
    velocity_history : np.ndarray
        (3, N) array of the Cartesian velocities in planet-centered inertial coordinates. Units: :math:`m/s`.
    """

    labels = [
        "x-Position [m]", "y-Position [m]", "z-Position [m]",
        "x-Velocity [m/s]", "y-Velocity [m/s]", "z-Velocity [m/s]",
    ]
    attributes = [
        "position_history",
        "velocity_history"
    ]

    def __init__(self):
        super().__init__()

        self.position_history = None
        self.velocity_history = None
        self.time_history = None

    def setup(self, initial_obj: orbits.Orbit, timesteps: int, events: int):
        length = timesteps + events + 1

        self.position_history = np.zeros([3, length])
        self.velocity_history = np.zeros([3, length])

        self.position_history[:, 0] = initial_obj.position
        self.velocity_history[:, 0] = initial_obj.velocity

    def log(self, obj: orbits.Orbit):
        self._current_index += 1  # Increment index.

        self.position_history[:, self._current_index] = obj.position
        self.velocity_history[:, self._current_index] = obj.velocity

    def concatenate(self) -> np.ndarray:
        data = np.vstack((
            self.position_history,
            self.velocity_history,
        ))

        return data.T


class ClassicalElementsLogger(Logger):
    r"""
    Child of :class:`~hohmannpy.logging.Logger` that logs the equinoctial orbital elements of an orbit at N
    timesteps.

    Attributes
    ----------
    sm_axis_history : np.ndarray
        (1, N) array of the semi-major axis over time. Units: :math:`m`.
    sl_rectum_history : np.ndarray
        (1, N) array of the semi-latus rectum over time.
        Units: :math:`m`.
    eccentricity_history : np.ndarray
        (1, N) array of the eccentricity over time.
    inclination_history : np.ndarray
        (1, N) array of the inclination over time. Units: :math:`rad`.
    raan_history : np.ndarray
        (1, N) array of the RAAN over time. Units: :math:`rad`.
    argp_history : np.ndarray
        (1, N) array of the argument of periapsis over time.
        Units: :math:`rad`.
    true_anomaly_history : np.ndarray
        (1, N) array of the true anomaly over time.
        Units: :math:`rad`.
    longp_history : np.ndarray
        (1, N) array of the longitude of periapsis over time.
        Units: :math:`rad`.
    argl_history : np.ndarray
        (1, N) array of the argument of latitude over time. Units: :math:`rad`.
    true_latitude_history : np.ndarray
        (1, N) array of the true latitude over time. Units: :math:`rad`.
    """

    labels = [
        "Semi-Axis Axis [m]", "Semi-Latus Rectum [m]",
        "Eccentricity",
        "Inclination [rad]", "RAAN [rad]", "Argument of Periapsis [rad]",
        "True Anomaly [rad]",
        "Longitude of Periapsis [rad]", "Argument of Latitude [rad]", "True Latitude [rad]"
    ]
    attributes = [
        "sm_axis_history",
        "sl_rectum_history",
        "eccentricity_history",
        "inclination_history",
        "raan_history",
        "argp_history",
        "true_anomaly_history",
        "longp_history",
        "argl_history",
        "true_latitude_history"
    ]

    def __init__(self):
        super().__init__()

        self.sm_axis_history = None
        self.sl_rectum_history = None
        self.eccentricity_history = None
        self.inclination_history = None
        self.raan_history = None
        self.argp_history = None
        self.true_anomaly_history = None
        self.longp_history = None
        self.argl_history = None
        self.true_latitude_history = None

    def setup(self, initial_obj: orbits.Orbit, timesteps: int, events: int):
        length = timesteps + events + 1

        self.sm_axis_history = np.zeros([1, length])
        self.sl_rectum_history = np.zeros([1, length])
        self.eccentricity_history = np.zeros([1, length])
        self.inclination_history = np.zeros([1, length])
        self.raan_history = np.zeros([1, length])
        self.argp_history = np.zeros([1, length])
        self.true_anomaly_history = np.zeros([1, length])
        self.longp_history = np.zeros([1, length])
        self.argl_history = np.zeros([1, length])
        self.true_latitude_history = np.zeros([1, length])

        self.sm_axis_history[0, 0] = initial_obj.sm_axis
        self.sl_rectum_history[0, 0] = initial_obj.sl_rectum
        self.eccentricity_history[0, 0] = initial_obj.eccentricity
        self.inclination_history[0, 0] = initial_obj.inclination
        self.raan_history[0, 0] = initial_obj.raan
        self.argp_history[0, 0] = initial_obj.argp
        self.true_anomaly_history[0, 0] = initial_obj.true_anomaly
        self.longp_history[0, 0] = initial_obj.longp
        self.argl_history[0, 0] = initial_obj.argl
        self.true_latitude_history[0, 0] = initial_obj.true_latitude

    def log(self, obj: orbits.Orbit):
        self._current_index += 1  # Increment index.

        self.sm_axis_history[0, self._current_index] = obj.sm_axis
        self.sl_rectum_history[0, self._current_index] = obj.sl_rectum
        self.eccentricity_history[0, self._current_index] = obj.eccentricity
        self.inclination_history[0, self._current_index] = obj.inclination
        self.raan_history[0, self._current_index] = obj.raan
        self.argp_history[0, self._current_index] = obj.argp
        self.true_anomaly_history[0, self._current_index] = obj.true_anomaly
        self.longp_history[0, self._current_index] = obj.longp
        self.argl_history[0, self._current_index] = obj.argl
        self.true_latitude_history[0, self._current_index] = obj.true_latitude

    def concatenate(self) -> np.ndarray:
        data = np.vstack((
            self.sm_axis_history,
            self.sl_rectum_history,
            self.eccentricity_history,
            self.inclination_history,
            self.raan_history,
            self.argp_history,
            self.true_anomaly_history,
            self.longp_history,
            self.argl_history,
            self.true_latitude_history
        ))

        return data.T

class EquinoctialElementsLogger(Logger):
    """
    Child of :class:`~hohmannpy.logging.Logger` that logs the equinoctial orbital elements of an orbit at N
    timesteps.

    Attributes
    ----------
    e_component1_history: np.ndarray
        (1, N) array of the x-component of the projection of the eccentricity vector into the equinoctial frame.
    e_component2_history: np.ndarray
        (1, N) array of the y-component of the projection of the eccentricity vector into the equinoctial frame.
    n_component1_history: np.ndarray
        (1, N) array of the x-component of the projection of the nodal vector into the equinoctial frame.
    n_component2_history: np.ndarray
        (1, N) array of the y-component of the projection of the nodal vector into the equinoctial frame.
    """

    labels = [
        "e-component 1", "e-component 2",
        "n-component 2", "n-component 2",
    ]
    attributes = [
        "e_component1_history",
        "e_component2_history",
        "n_component1_history",
        "n_component2_history",
    ]

    def __init__(self):
        super().__init__()

        self.e_component1_history = None
        self.e_component2_history = None
        self.n_component1_history = None
        self.n_component2_history = None

    def setup(self, initial_obj: orbits.Orbit, timesteps: int, events: int):
        length = timesteps + events + 1

        self.e_component1_history = np.zeros([1, length])
        self.e_component2_history = np.zeros([1, length])
        self.n_component1_history = np.zeros([1, length])
        self.n_component2_history = np.zeros([1, length])

        self.e_component1_history[0, 0] = initial_obj.e_component1
        self.e_component2_history[0, 0] = initial_obj.e_component2
        self.n_component1_history[0, 0] = initial_obj.n_component1
        self.n_component2_history[0, 0] = initial_obj.n_component2

    def log(self, obj: orbits.Orbit):
        self._current_index += 1  # Increment index.

        self.e_component1_history[0, self._current_index] = obj.e_component1
        self.e_component2_history[0, self._current_index] = obj.e_component2
        self.n_component1_history[0, self._current_index] = obj.n_component1
        self.n_component2_history[0, self._current_index] = obj.n_component2

    def concatenate(self) -> np.ndarray:
        data = np.vstack((
            self.e_component1_history,
            self.e_component2_history,
            self.n_component1_history,
            self.n_component2_history,
        ))

        return data.T


class EccentricAnomalyLogger(Logger):
    r"""
    Child of :class:`~hohmannpy.logging.Logger` that logs the eccentric anomaly an orbit at N timesteps.

    Attributes
    ----------
    eccentric_anomaly_history : np.ndarray
        (1, N) array of the eccentric anomaly over time. Units: :math:`rad`.
    """

    labels = ["Eccentric Anomaly [rad]"]
    attributes = ["eccentric_anomaly_history"]

    def __init__(self):
        super().__init__()

        self.eccentric_anomaly_history = None

    def setup(self, initial_obj: orbits.Orbit, timesteps: int, events: int):
        length = timesteps + events + 1

        self.eccentric_anomaly_history = np.zeros([1, length])

        self.eccentric_anomaly_history[0, 0] = initial_obj.eccentric_anomaly

    def log(self, obj: orbits.Orbit):
        self._current_index += 1  # Increment index.

        self.eccentric_anomaly_history[0, self._current_index] = obj.eccentric_anomaly

    def concatenate(self) -> np.ndarray:
        return self.eccentric_anomaly_history.T


class UniversalVariableLogger(Logger):
    r"""
    Child of :class:`~hohmannpy.logging.Logger` that logs the universal variable and Stumpff parameter of an orbit
    at N timesteps.

    Attributes
    ----------
    universal_variable_history : np.ndarray
        (1, N) array of the universal variable over time.
    stumpff_param_history : np.ndarray
        (1, N) array of the Stumpff parameter over time. Units: :math:`rad`.
    """

    labels = ["Universal Variable, Stumpff Parameter [rad]"]
    attributes = ["universal_variable_history", "stumpff_param_history"]

    def __init__(self):
        super().__init__()

        self.universal_variable_history = None
        self.stumpff_param_history = None

    def setup(self, initial_obj: orbits.Orbit, timesteps: int, events: int):
        length = timesteps + events + 1

        self.universal_variable_history = np.zeros([1, length])
        self.stumpff_param_history = np.zeros([1, length])

        self.universal_variable_history[0, 0] = initial_obj.universal_variable
        self.stumpff_param_history[0, 0] = initial_obj.stumpff_param

    def log(self, obj: orbits.Orbit):
        self._current_index += 1  # Increment index.

        self.universal_variable_history[0, self._current_index] = obj.universal_variable
        self.stumpff_param_history[0, self._current_index] = obj.stumpff_param

    def concatenate(self) -> np.ndarray:
        data = np.vstack((
            self.universal_variable_history,
            self.stumpff_param_history,
        ))

        return data.T


# TODO: Finish doc strings for these classes.
class AttitudeLogger(Logger):
    r"""
    Child of :class:`~hohmannpy.logging.Logger` that logs the attitude of a spacecraft wrt. to the ECI-fixed basis
    at N timesteps using quaternions as well as the body-fixed rates of the spacecraft.

    Attributes
    ----------
    quaternion_history : np.ndarray
        (4, N) array of the ...
    angular_velocity_history : np.ndarray
        (3, N) array of the ...
    """

    labels = [
        "w-Quaternion", "x-Quaternion", "y-Quaternion", "z-Quaternion",
        "x-Rate [rad/s]", "y-Rate [rad/s]", "z-Rate [rad/s]",
    ]
    attributes = [
        "quaternion_history",
        "angular_velocity_history",
    ]

    def __init__(self):
        super().__init__()

        self.quaternion_history = None
        self.angular_velocity_history = None

    def setup(self, initial_obj: attitude.Orientation, timesteps: int, events: int):
        length = timesteps + events + 1

        self.quaternion_history = np.zeros([4, length])
        self.angular_velocity_history = np.zeros([3, length])

        self.quaternion_history[:, 0] = initial_obj.quaternion
        self.angular_velocity_history[:, 0] = initial_obj.angular_velocity

    def log(self, obj: attitude.Orientation):
        self._current_index += 1  # Increment index.

        self.quaternion_history[:, self._current_index] = obj.quaternion
        self.angular_velocity_history[:, self._current_index] = obj.angular_velocity

    def concatenate(self) -> np.ndarray:
        data = np.vstack((
            self.quaternion_history,
            self.angular_velocity_history,
        ))

        return data.T


class EulerLogger(Logger):
    r"""
    Child of :class:`~hohmannpy.logging.Logger` that logs the attitude of a spacecraft wrt. to the ECI-fixed basis
    at N timesteps using 3-2-1 Euler angles well as their time derivatives.

    Attributes
    ----------
    roll_history : np.ndarray
        (1, N) array of the ...
    pitch_history : np.ndarray
        (1, N) array of the ...
    yaw_history : np.ndarray
        (1, N) array of the ...
    roll_rate_history : np.ndarray
        (1, N) array of the ...
    pitch_rate_history : np.ndarray
        (1, N) array of the ...
    yaw_rate_history : np.ndarray
        (1, N) array of the ...
    """

    labels = [
        "Roll [rad]", "Pitch [rad]", "Yaw [rad]",
        "Roll Rate [rad/s]", "Pitch Rate [rad/s]", "Yaw Rate [rad/s]",
    ]
    attributes = [
        "roll_history",
        "pitch_history",
        "yaw_history",
        "roll_rate_history",
        "pitch_rate_history",
        "yaw_rate_history",
    ]

    def __init__(self):
        super().__init__()

        self.roll_history = None
        self.pitch_history = None
        self.yaw_history = None
        self.roll_rate_history = None
        self.pitch_rate_history = None
        self.yaw_rate_history = None

    def setup(self, initial_obj: attitude.Orientation, timesteps: int, events: int):
        length = timesteps + events + 1

        self.roll_history = np.zeros([1, length])
        self.pitch_history = np.zeros([1, length])
        self.yaw_history = np.zeros([1, length])
        self.roll_rate_history = np.zeros([1, length])
        self.pitch_rate_history = np.zeros([1, length])
        self.yaw_rate_history = np.zeros([1, length])

        self.roll_history[0, 0] = initial_obj.roll
        self.pitch_history[0, 0] = initial_obj.pitch
        self.yaw_history[0, 0] = initial_obj.yaw
        self.roll_rate_history[0, 0] = initial_obj.roll_rate
        self.pitch_rate_history[0, 0] = initial_obj.pitch_rate
        self.yaw_rate_history[0, 0] = initial_obj.yaw_rate

    def log(self, obj: attitude.Orientation):
        self._current_index += 1  # Increment index.

        self.roll_history[0, self._current_index] = obj.roll
        self.pitch_history[0, self._current_index] = obj.pitch
        self.yaw_history[0, self._current_index] = obj.yaw
        self.roll_rate_history[0, self._current_index] = obj.roll_rate
        self.pitch_rate_history[0, self._current_index] = obj.pitch_rate
        self.yaw_rate_history[0, self._current_index] = obj.yaw_rate

    def concatenate(self) -> np.ndarray:
        data = np.vstack((
            self.roll_history,
            self.pitch_history,
            self.yaw_history,
            self.roll_rate_history,
            self.pitch_rate_history,
            self.yaw_rate_history,
        ))

        return data.T
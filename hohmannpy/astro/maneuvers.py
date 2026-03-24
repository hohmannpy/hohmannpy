from __future__ import annotations
from typing import Any, Callable, Optional, TYPE_CHECKING

import numpy as np
import scipy as sp

from . import perturbations

if TYPE_CHECKING:
    from . import spacecraft, time


class ImpulsiveBurn:
    r"""
    A burn whose impulse is delivered instantaneously, obeying the impulsive thrust assumption.

    This can be passed to :class:`~hohmannpy.astro.Satellite`'s ``burns`` parameter during instantiation to schedule it
    for that satellite.

    Parameters
    ----------
    start_time : Any[float, :class:`~hohmannpy.astro.Time`]
        The time at which the impulse is delivered. Can either be the relative time since mission start in seconds or a
        :class:`~hohmannpy.astro.Time` object.
    velocity_change : np.ndarray
        The desired impulsive change in velocity as a (3, ) array. By default, this is assumed to be in the satellite's
        radial-transverse-normal (RTN) frame unless ``inertial`` is set to ``True``.
    inertial : bool
        Whether the ``velocity_change`` is parameterized in planet-centered inertial coordinates.

    Attributes
    ----------
    start_time : Any[float, :class:`~hohmannpy.astro.Time`]
        The time at which the impulse is delivered. Can either be the relative time since mission start in seconds or a
        ``Time`` object.
    velocity_change : np.ndarray
        The desired impulsive change in velocity as a (3, ) array. By default, this is assumed to be in the satellite's
        radial-transverse-normal (RTN) frame unless ``inertial`` is set to ``True``.
    """

    def __init__(
            self,
            start_time: Any[float, time.Time],
            velocity_change: np.ndarray,
            inertial: bool = False,
    ):
        self.start_time = start_time
        self.velocity_change = velocity_change
        self._inertial = inertial

    def evaluate(self, satellite: spacecraft.Satellite):
        r"""
        Modify the velocity of a ``Satellite``'s :class:`~hohmannpy.astro.Orbit`` attribute.

        Parameters
        ----------
        satellite : class:`~hohmannpy.astro.Satellite`
            Satellite which is performing the burn.
        """

        # Orbit stores an inertial velocity, so if the user supplied an impulse coordinated in the RTN frame use a DCM
        # to transform it to the inertial frame.
        if not self._inertial:
            sat_2_inertial_dcm =  self._compute_sat_2_inertial_dcm(satellite)
            velocity_change = sat_2_inertial_dcm @ self.velocity_change.copy()
        else:
            velocity_change = self.velocity_change

        satellite.orbit.velocity += velocity_change  # Increase velocity.
        satellite.impulsive_burn_index += 1  # Increment this index to indicate this burn fired.

    def _compute_sat_2_inertial_dcm(self, satellite: spacecraft.Satellite) -> np.ndarray:
        r"""
        Generate a DCM which transforms from a satellite's local frame to the inertial frame.

        This can't be done using Euler angles because for some orbit types (i.e. equatorial) some of the standard 3-1-3
        orbital elements are undefined. Instead, form the DCM using a set of unit vectors.

        Parameters
        ----------
        satellite : class:`~hohmannpy.astro.Satellite`
            Satellite which is performing the burn.

        Returns
        -------
        sat_2_inertial_dcm : np.ndarray
            DCM which transforms from the satellite's local RTN frame to the inertial frame.
        """

        radial_uvec = satellite.orbit.position / np.linalg.norm(satellite.orbit.position)
        normal_uvec = satellite.orbit.spf_angular_momentum / np.linalg.norm(satellite.orbit.spf_angular_momentum)
        transverse_uvec = np.cross(normal_uvec, radial_uvec)

        return np.stack((radial_uvec.T, transverse_uvec.T, normal_uvec.T), axis=1)


class ContinuousBurn(perturbations.Perturbation):
    r"""
    The base class for all burns whose acceleration is delivered over a continuous period of time.

    This can be passed to :class:`~hohmannpy.astro.Satellite`'s ``burns`` parameter during instantiation to schedule it
    for that satellite. However, this class doesn't define the burn profile so it should never be directly instantiated.
    Instead, use its children such as :class:`~hohmannpy.astro.FunctionContinuousBurn`.

    This is an extension of :class:`~hohmannpy.astro.Perturbation` so within propagators continuous burns are
    essentially treated as just another form of propagation.

    Parameters
    ----------
    start_time : Any[float, :class:`~hohmannpy.astro.Time`]
        The time at which the burn is to begin. Can either be the relative time since mission start in seconds or a
        :class:`~hohmannpy.astro.Time` object.
    end_time : Any[float, :class:`~hohmannpy.astro.Time`]
        The time at which the burn is to end. Can either be the relative time since mission start in seconds or a
        ``Time`` object.
    inertial : bool
        Whether the ``velocity_change`` is parameterized in planet-centered inertial coordinates.
    masses : Any[Optional[scipy.BSpline, Callable]]
        The mass can optionally change over the course of the burn. If it does, the masses of the satellite over time
        can be provided as either a (N, 2) lookup table of (time, mass) entries where the first row is the mass at
        ``start_time`` and the last is the mass at ``end_time`` or alternatively as a function. Can not be a lambda
        function as this interferes with parallel processing capabilities.

    Attributes
    ----------
    start_time : Any[float, :class:`~hohmannpy.astro.Time`]
        The time at which the burn is to begin. Can either be the relative time since mission start in seconds or a
        ``Time`` object.
    end_time : Any[float, :class:`~hohmannpy.astro.Time`]
        The time at which the burn is to end. Can either be the relative time since mission start in seconds or a
        ``Time`` object.
    masses : Any[Optional[scipy.BSpline, Callable]]
        The mass can optionally change over the course of the burn. If it does, the masses of the satellite over time
        can be provided as either a (N, 2) lookup table of (time, mass) entries where the first row is the mass at
        ``start_time`` and the last is the mass at ``end_time`` or alternatively as a function. Can not be a lambda
        function as this interferes with parallel processing capabilities.
    """

    def __init__(
            self,
            start_time: Any[float, time.Time],
            end_time: Any[float, time.Time],
            masses: Optional[Any[np.ndarray, Callable]] = None,
            inertial: bool = False,
    ):
        super().__init__()

        self.start_time = start_time
        self.end_time = end_time
        self._inertial = inertial

        if masses is not None:
            if masses is Callable:
                self.masses = masses
            else:
                self.masses = sp.interpolate.make_interp_spline(
                    masses[:, 0],
                    masses[:, 1:],
                    k=1
                )  # Interpolate the mass table.

        self.masses = masses

    def evaluate(self, time: float, state: np.ndarray, satellite: spacecraft.Satellite) -> np.ndarray:
        r"""
        Takes in the current time and planet-centered inertial state (the position and velocity) and returns the
        acceleration due to this burn.

        Parameters
        ----------
        time : float
            Current time in seconds since propagation began.
        state : np.ndarray
            Current translational state in planet-centered inertial coordinates given as (position, velocity).
        satellite : :class:`~hohmannpy.astro.Satellite`
            The satellite object experiencing the perturbing acceleration. It is passed so that attributes of the
            satellite can be used in computing perturbing acceleration, such as the ``ballistic_coeff`` by
            :class:`~hohmannpy.astro.AtmosphericDrag`. However, the state of the satellite should never be accessed
            directly, only via the passed ``state`` parameter.

        Returns
        -------
        acceleration : np.ndarray
            Acceleration due to this burn.
        """

        pass

    def _compute_sat_2_inertial_dcm(self, state):
        r"""
        Generate a DCM which transforms from a satellite's local frame to the inertial frame.

        This can't be done using Euler angles because for some orbit types (i.e. equatorial) some of the standard 3-1-3
        orbital elements are undefined. Instead, form the DCM using a set of unit vectors.

        Parameters
        ----------
        state : class:`~hohmannpy.astro.Satellite`
            Current translational state in planet-centered inertial coordinates given as (position, velocity).

        Returns
        -------
        sat_2_inertial_dcm : np.ndarray
            DCM which transforms from the satellite's local RTN frame to the inertial frame.
        """

        position = state[:3]
        velocity = state[3:]
        spf_angular_momentum = np.cross(position, velocity)

        radial_uvec = position / np.linalg.norm(position)
        normal_uvec = spf_angular_momentum / np.linalg.norm(spf_angular_momentum)
        transverse_uvec = np.cross(normal_uvec, radial_uvec)

        return  np.stack((radial_uvec.T, transverse_uvec.T, normal_uvec.T), axis=1)


class ConstantContinuousBurn(ContinuousBurn):
    r"""
    Continuous burn where the supplied thrust is constant.

    Parameters
    ----------
    start_time : Any[float, :class:`~hohmannpy.astro.Time`]
        The time at which the burn is to begin. Can either be the relative time since mission start in seconds or a
        :class:`~hohmannpy.astro.Time` object.
    end_time : Any[float, :class:`~hohmannpy.astro.Time`]
        The time at which the burn is to end. Can either be the relative time since mission start in seconds or a
        ``Time`` object.
    thrust : np.ndarray
        Constant thrust to burn at as a (3, ) array.
    masses : Any[Optional[scipy.BSpline, Callable]]
        The mass can optionally change over the course of the burn. If it does, the masses of the satellite over time
        can be provided as either a (N, 2) lookup table of (time, mass) entries where the first row is the mass at
        ``start_time`` and the last is the mass at ``end_time`` or alternatively as a function. Can not be a lambda
        function as this interferes with parallel processing capabilities.
    inertial : bool
        Whether the ``velocity_change`` is parameterized in planet-centered inertial coordinates.

    Attributes
    ----------
    thrust : np.ndarray
        Constant thrust to burn at as a (3, ) array. By default, this is assumed to be in the satellite's
        radial-transverse-normal (RTN) frame unless ``inertial`` is set to ``True``.
    """

    def __init__(
            self,
            start_time: Any[float, time.Time],
            end_time: Any[float, time.Time],
            thrust: np.ndarray,
            masses: Optional[Any[np.ndarray, Callable]] = None,
            inertial: bool = False,
    ):
        super().__init__(start_time, end_time, masses, inertial)

        self.thrust = thrust

    def evaluate(self, time: float, state: np.ndarray, satellite: spacecraft.Satellite) -> np.ndarray:
        r"""
        See :class:`~hohmannpy.astro.ContinuousBurn`. :meth:`~hohmannpy.astro.ContinuousBurn.evaluate()`.
        """

        if not self._inertial:
            sat_2_inertial_dcm = self._compute_sat_2_inertial_dcm(state)
            thrust = sat_2_inertial_dcm @ self.thrust.copy()
        else:
            thrust = self.thrust

        if self.masses is not None:
            satellite.mass = self.masses(time)

        return thrust / satellite.mass


class LookupContinuousBurn(ContinuousBurn):
    r"""
    Continuous burn where a time-varying thrust is interpolated from a lookup table.

    Parameters
    ----------
    start_time : Any[float, :class:`~hohmannpy.astro.Time`]
        The time at which the burn is to begin. Can either be the relative time since mission start in seconds or a
        :class:`~hohmannpy.astro.Time` object.
    end_time : Any[float, :class:`~hohmannpy.astro.Time`]
        The time at which the burn is to end. Can either be the relative time since mission start in seconds or a
        ``Time`` object.
    thrusts : np.ndarray
        (N, 4) array of (time, (3, ) thrust) at N timesteps ranging from ``start_time`` to ``end_time``. Each row
        consists of the time for that entry followed by the 1, 2, and 3 components of the thrust. By default, this is
        assumed to be in the satellite's radial-transverse-normal (RTN) frame unless ``inertial`` is set to ``True``.
    masses : Any[Optional[scipy.BSpline, Callable]]
        The mass can optionally change over the course of the burn. If it does, the masses of the satellite over time
        can be provided as either a (N, 2) lookup table of (time, mass) entries where the first row is the mass at
        ``start_time`` and the last is the mass at ``end_time`` or alternatively as a function. Can not be a lambda
        function as this interferes with parallel processing capabilities.
    inertial : bool
        Whether the ``velocity_change`` is parameterized in planet-centered inertial coordinates.

    Attributes
    ----------
    burn_spline : scipy.BSpline
        Cubic spline of the thrust. Calling it via ``burn_spline(time)`` returns the interpolated thrust at that time.
    """

    def __init__(
            self,
            start_time: Any[float, time.Time],
            end_time: Any[float, time.Time],
            thrusts: np.ndarray,
            masses: Optional[Any[np.ndarray, Callable]] = None,
            inertial: bool = False,
    ):
        super().__init__(start_time, end_time, masses, inertial)

        self.burn_spline = sp.interpolate.make_interp_spline(
            thrusts[:, 0],
            thrusts[:, 1:],
            k=3
        )  # Interpolate the thrust table.

    def evaluate(self, time: float, state: np.ndarray, satellite: spacecraft.Satellite) -> np.ndarray:
        r"""
        See :class:`~hohmannpy.astro.ContinuousBurn`. :meth:`~hohmannpy.astro.ContinuousBurn.evaluate()`.
        """

        if not self._inertial:
            sat_2_inertial_dcm = self._compute_sat_2_inertial_dcm(state)
            thrust = sat_2_inertial_dcm @ self.burn_spline(time)
        else:
            thrust = self.burn_spline(time)

        if self.masses is not None:
            satellite.mass = self.masses(time)

        return thrust / satellite.mass


class FunctionContinuousBurn(ContinuousBurn):
    r"""
    Continuous burn where a time-varying thrust is interpolated from a lookup table.

    Parameters
    ----------
    start_time : Any[float, :class:`~hohmannpy.astro.Time`]
        The time at which the burn is to begin. Can either be the relative time since mission start in seconds or a
        :class:`~hohmannpy.astro.Time` object.
    end_time : Any[float, :class:`~hohmannpy.astro.Time`]
        The time at which the burn is to end. Can either be the relative time since mission start in seconds or a
        ``Time`` object.
    thrust_function: Callable,
        Function which when passed an input via ``thrust_function(time)`` returns the thrust as a (3, ) numpy array. By
        default, this is assumed to be in the satellite's radial-transverse-normal (RTN) frame unless ``inertial`` is
        set to ``True``. Can not be a lambda function as this interferes with parallel processing capabilities.
    masses : Any[Optional[scipy.BSpline, Callable]]
        The mass can optionally change over the course of the burn. If it does, the masses of the satellite over time
        can be provided as either a (N, 2) lookup table of (time, mass) entries where the first row is the mass at
        ``start_time`` and the last is the mass at ``end_time`` or alternatively as a function. Can not be a lambda
        function as this interferes with parallel processing capabilities.
    inertial : bool
        Whether the ``velocity_change`` is parameterized in planet-centered inertial coordinates.

    Attributes
    ----------
    thrust_function: Callable,
        Function which when passed an input via ``thrust_function(time)`` returns the thrust as a (3, ) numpy array. By
        default, this is assumed to be in the satellite's radial-transverse-normal (RTN) frame unless ``inertial`` is
        set to ``True``.
    """

    def __init__(
            self,
            start_time: Any[float, time.Time],
            end_time: Any[float, time.Time],
            thrust_function: Callable,
            masses: Optional[Any[np.ndarray, Callable]] = None,
            inertial: bool = False,
    ):
        super().__init__(start_time, end_time, masses, inertial)

        self.thrust_function = thrust_function

    def evaluate(self, time: float, state: np.ndarray, satellite: spacecraft.Satellite) -> np.ndarray:
        r"""
        See :class:`~hohmannpy.astro.ContinuousBurn` . :meth:`~hohmannpy.astro.ContinuousBurn.evaluate()`.
        """

        if not self._inertial:
            sat_2_inertial_dcm = self._compute_sat_2_inertial_dcm(state)
            thrust = sat_2_inertial_dcm @ self.thrust_function(time)
        else:
            thrust = self.thrust_function(time)

        if self.masses is not None:
            satellite.mass = self.masses(time)

        return thrust / satellite.mass
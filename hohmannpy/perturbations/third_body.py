from __future__ import annotations
from typing import Union
import copy
import functools

import numpy as np
import scipy as sp

from ..dynamics import conversions
from ..astro import propagation, time, orbits
from .. import logging
from . import base


class ThirdBodyGravity(base.Perturbation):
    r"""
    Perturbation caused by a third body's gravity. This third body can either orbit the central body or orbit another
    arbitrary fixed point.

    Parameters
    ----------
    third_body_grav_param : float
        Gravitational parameter of the third body.
    third_body: :class:`~hohmannpy.astro.Satellite`
        The object which holds the orbit of the third body.
    central_body: :class:`~hohmannpy.astro.Satellite`
        The object which holds the orbit of the central body. If this is not passed it will be assumed the third body
        orbits the central body. If this is used, the central body must orbit the same focus as the third body. A valid
        example would be if the central and third body were two of Jupiter's moons as the focus would then be Jupiter.
        An invalid example would be if the central body was the Earth and the third body was one of Mars' moons because
        the Earth's focus is the Sun and the Mars' moon's focus is Mars.
    dcm: np.ndarray
        If the central or third body's orbit is not propagated in the same frame as that of the satellite the DCM needed
        to transform their positions to the same frame as the satellite must be provided. These can be generated using
        :func:`~hohmannpy.dynamics.euler_2_dcm()`. An example is if the third body is the Sun which "orbits" the Earth.
        The Sun's propagated "orbit" (or more accurately the inverse of the Earth's orbit about the Sun) will be in the
        heliocentric frame and so a DCM is needed to transform the Sun's position into the Earth-centered inertial frame
        in which the satellite is propagated.
    legendre: bool
        Whether to use a Legendre polynomial expansion in the computation of the third body's perturbing effects. Used
        to avoid small difference numerical accuracy losses from the difference between the two position cubics due to
        their potential similarities.
    legendre_series_length: int
        If a Legendre polynomial expansion is used, how many terms to include.

    Attributes
    ----------
    third_body: :class:`~hohmannpy.astro.Satellite`
        The object which holds the orbit of the third body.
    central_body: :class:`~hohmannpy.astro.Satellite`
        The object which holds the orbit of the central body.
    tb_orbit_spline : :class:`scipy.BSpline`
        Cubic spline of the third body's trajectory. Calling it via ``tb_orbit_spline(time)`` returns the interpolated
        orbit at that time.
    cb_orbit_spline : Union[:class:`scipy.BSpline`, func]
        Cubic spline of the central body's trajectory. Calling it via ``cb_orbit_spline(time)`` returns the
        interpolated orbit at that time.

    Notes
    -----
    The following assumptions are made for this implementation:

    1) The third (and potentially central) bodies follow Keplerian orbits.
    """

    def __init__(
            self,
            third_body_grav_param: float,
            third_body: spacecraft.Satellite,
            central_body: spacecraft.Satellite = None,
            dcm: np.ndarray = None,
            legendre: bool = True,
            legendre_series_length: int = 10,
    ):
        super().__init__()

        self._tb_grav_param = third_body_grav_param
        self._legendre = legendre
        self._legendre_series_length = legendre_series_length
        self._dcm = dcm
        self.third_body = third_body
        self.central_body = central_body

        # If no DCM is passed set it to a 3x3 identity matrix.
        if self._dcm is None:
            self._dcm = np.array(([1, 0, 0], [0, 1, 0], [0, 0, 1]))

        # Setup finished in finalize__init__() which is called by the Mission.
        self.tb_orbit_spline = None
        self.cb_orbit_spline = None

    def _finalize__init__(self, initial_global_time: time.Time, final_global_time: time.Time):
        """
        Create a ``np.BSpline`` for the third and central body's orbits.

        Both of these attributes are needed by :meth:`evaluate()` but can't be computed in the base ``__init__()``. This
        is called during :class:`~hohmannpy.Mission`'s instantiation.

        Parameters
        ----------
        initial_global_time: time.Time
            Gregorian date and UT1 time at which simulation begins. Used to compute the time (in days) since the Earth's
            last aphelion passage for the solar irradiance model.
        final_global_time: time.Time
            Gregorian date and UT1 time at which simulation ends.
        """

        # Propagate the orbit's of the third and central bodies and then convert them into cubic splines. If the
        # central body's orbit is not to be propagated instead set it to a function which always returns [0, 0, 0] when
        # passed any time value.
        propagator = propagation.KeplerPropagator()
        self.third_body.loggers = [logging.TimeLogger(), logging.StateLogger()]

        if self.central_body is None:
            propagator._propagate(
                satellites={self.third_body.name: self.third_body},
                runtime=(final_global_time.julian_date - initial_global_time.julian_date) * 86400,
                include_rotation=False
            )

            self.cb_orbit_spline = dummy_spline
        else:
            self.central_body.loggers = [logging.TimeLogger(), logging.StateLogger()]
            propagator._propagate(
                satellites={self.third_body.name: self.third_body, self.central_body.name: self.central_body},
                runtime=(final_global_time.julian_date - initial_global_time.julian_date) * 86400,
                include_rotation=False
            )

            self.cb_orbit_spline = sp.interpolate.make_interp_spline(
                self.central_body.time_history.squeeze(), self.central_body.position_history.T, k=3
            )

        self.tb_orbit_spline = sp.interpolate.make_interp_spline(
            self.third_body.time_history.squeeze(), self.third_body.position_history.T, k=3
        )

    def evaluate(self, time: float, state: np.ndarray, satellite: spacecraft.Satellite) -> np.ndarray:
        """
        Computes the perturbing acceleration due to the third body.

        Parameters
        ----------
        time : float
            Current time in seconds since propagation began.
        state : np.ndarray
            Current translational state in planet-centered inertial coordinates given as (position, velocity) or
            (position, velocity, quaternion, angular_velocity).
        satellite : :class:`~hohmannpy.astro.Satellite`
            Unused parameter simply based due to ``@abstractmethod`` requirements.

        Returns
        -------
        acceleration : np.ndarray
            Current translational acceleration in ECI coordinates.
        """

        # Calculate the position of the third body wrt. the central body and the satellite.
        position_tb_wrt_cb = self._dcm @ (-self.cb_orbit_spline(time) + self.tb_orbit_spline(time))
        position_tb_wrt_sat = position_tb_wrt_cb - state[:3]

        # The third and central body may be in the same relative location wrt. the central body if the central body is
        # very far away, such as if the perturbation is caused by the Sun. Rather than using the N-body equation in this
        # instance we can optionally rewrite it as a Legendre polynomial expansion of the cosine of the phase angle
        # between the satellite and third body wrt. the central body.
        if self._legendre:
            phase_angle_cosine = (
                    np.dot(state[:3], position_tb_wrt_cb)
                        / (np.linalg.norm(state[:3]) * np.linalg.norm(position_tb_wrt_cb))
            )

            # Compute the Legendre polynomial series sum.
            legendre_sum = 0
            position_ratio = np.linalg.norm(state[:3]) / np.linalg.norm(position_tb_wrt_cb)
            for i in range(1, self._legendre_series_length):
                legendre_sum += sp.special.legendre_p(i, phase_angle_cosine) * position_ratio ** i

            # Compute the acceleration.
            acceleration = (
                    -self._tb_grav_param / np.linalg.norm(position_tb_wrt_cb) ** 3
                    * (state[:3]
                           - position_tb_wrt_sat * (3 * legendre_sum + 3 * legendre_sum ** 2 + legendre_sum ** 3)
                           )
            )
        else:  # Alternative acceleration computation if the user just wants the standard N-body equation to be used.
            acceleration = (
                    self._tb_grav_param * (
                    position_tb_wrt_sat / np.linalg.norm(position_tb_wrt_sat) ** 3
                        - position_tb_wrt_cb / np.linalg.norm(position_tb_wrt_cb) ** 3
                )
            )

        if len(state) == 3:
            return acceleration
        else:
            return np.concatenate((acceleration, np.zeros(3)), axis=-1)


class LunarGravity(ThirdBodyGravity):
    r"""
    Perturbation caused by the Moon's gravity.

    This class implements a specialized version of :class:`~hohmannpy.astro.ThirdBodyGravity` adjusted to
    specifically account for the third body perturbations due to the Earth's moon.

    Parameters
    ----------
    legendre: bool
        Whether to use a Legendre polynomial expansion in the computation of the Moon's perturbing effects. Used to
        avoid small difference numerical accuracy losses from the difference between the two position cubics due to
        their potential similarities.
    legendre_series_length: int
        If a Legendre polynomial expansion is used, how many terms to include.

    Notes
    -----
    The following assumptions are made for this implementation:

    1) The Moon's orbit is Keplerian.

    See Also
    --------
    :class:`~hohmannpy.astro.ThirdBodyGravity` : Base version of this class which can be used for any third body.
    """

    def __init__(
            self,
            initial_true_anomaly: float,
            legendre: bool = True,
            legendre_series_length: int = 10,
    ):
        # Initialize the Moon.
        from ..astro import celestial  # Stuffed down here to prevent circular imports.
        moon = celestial.Moon(initial_true_anomaly)

        super().__init__(
            third_body_grav_param=4.9048695e12,
            third_body=moon,
            legendre=legendre,
            legendre_series_length=legendre_series_length
        )


class SolarGravity(ThirdBodyGravity):
    r"""
    Perturbation caused by the Sun's gravity.

    This class implements a specialized version of :class:`~hohmannpy.astro.ThirdBodyGravity` adjusted to
    specifically account for the third body perturbations due to the Sun. The true anomaly of the Earth wrt. to the
    ecliptic plane is computed automatically based on the dates of desired propagation.

    Parameters
    ----------
    legendre: bool
        Whether to use a Legendre polynomial expansion in the computation of the Earth's perturbing effects. Used to
        avoid small difference numerical accuracy losses from the difference between the two position cubics due to
        their potential similarities.
    legendre_series_length: int
        If a Legendre polynomial expansion is used, how many terms to include.

    Notes
    -----
    The following assumptions are made for this implementation:

    1) The Earth's orbit is Keplerian.

    3) The 1-2 plane of the Earth-centered-inertial (ECI) basis is also assumed to be inclined at constant 23.5 :math:`deg` from the ecliptic plane.

    See Also
    --------
    :class:`~hohmannpy.astro.ThirdBodyGravity` : Base version of this class which can be used for any third body.
    """

    def __init__(
            self,
            legendre: bool = True,
            legendre_series_length: int = 10,
    ):
        earth_tilt = np.deg2rad(-23.439291115)  # Rotate from Sun-fixed to Earth-fixed frame via the Earth's axial tilt.

        # Our actually third body is the Earth but to instantiate it we need the initial global time (so we can locate
        # the Earth on its orbit). That isn't passed in till finalize__init__() is called so create a temporary dummy
        # third body. We'll replace it with the Earth in the aforementioned function.
        from .. import spacecraft  # Stuffed down here to prevent circular imports.
        dummy_third_body = spacecraft.Satellite(
            starting_orbit=orbits.Orbit(
                position=np.array([1, 1, 1]),
                velocity=np.array([0, 0, 1]),
            ),
            name="temp"
        )
        super().__init__(
            third_body_grav_param=1.32712440018e20,
            third_body=dummy_third_body,
            legendre=legendre,
            legendre_series_length=legendre_series_length,
            dcm=conversions.euler_2_dcm(earth_tilt, 1)
        )


    def _finalize__init__(self, initial_global_time: time.Time, final_global_time: time.Time):
        """
        Extension of :class:`~hohmannpy.astro.ThirdBodyGravity` .
        :meth:`~hohmannpy.astro.ThirdBodyGravity.finalize__init__()` that creates an Earth object and then calculates
        its orbit around the Sun. This can be inverted to get the Sun's "orbit" about the Earth.

        Parameters
        ----------
        initial_global_time: time.Time
            Gregorian date and UT1 time at which simulation begins. Used to compute the time (in days) since the Earth's
            last aphelion passage for the solar irradiance model.
        final_global_time: time.Time
            Gregorian date and UT1 time at which simulation ends.
        """

        # Initialize the Earth.
        from ..astro import celestial  # Stuffed down here to prevent circular imports.
        earth = celestial.Earth(initial_global_time)
        self.third_body = earth

        # Call parent class' finalize__init__() to create the needed orbit splines.
        super()._finalize__init__(initial_global_time=initial_global_time, final_global_time=final_global_time)

        # Currently we have a spline which returns the Earth as the third body orbiting about the Sun. We want the Sun
        # to be the third body so we wrap the orbit_spline of the Earth so that it always returns the position of the
        # Sun wrt. the Earth instead.
        tb_orbit_spline = copy.deepcopy(self.tb_orbit_spline)

        self.tb_orbit_spline = functools.partial(inverted_spline, func=tb_orbit_spline)


# Helper functions used.
def dummy_spline(x):
    return np.array([0, 0, 0])


def inverted_spline(x, func):
    return -func(x)
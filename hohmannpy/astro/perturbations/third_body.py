from __future__ import annotations
from typing import Union, TYPE_CHECKING
import copy

import numpy as np
import scipy as sp

from ...dynamics import dcms
from .. import propagation, time, spacecraft, logging
from . import base


class ThirdBodyGravity(base.Perturbation):
    r"""
    Perturbation caused by a third body's gravity. This third body can either orbit the central body or orbit another
    arbitrary fixed point.

    Parameters
    ----------
    initial_global_time: :class:`~hohmannpy.astro.Time`
        Gregorian date and UT1 time at which propagation of the third (and optionally central) body orbits should begin.
        Should match the initial and final time passed to the :class:`~hohmannpy.astro.Mission` which holds this
        perturbation.
    final_global_time: :class:`~hohmannpy.astro.Time`
        Gregorian date and UT1 time at which propagation of the third (and optionally central) body orbits should end.
        Should match the initial and final time passed to the ``Mission`` which holds this perturbation.
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
    tb_grav_param : float
        Gravitational parameter of the third body.
    legendre: bool
        Whether to use a Legendre polynomial expansion in the computation of the third body's perturbing effects. Used
        to avoid small difference numerical accuracy losses from the difference between the two position cubics due to
        their potential similarities.
    legendre_series_length: int
        If a Legendre polynomial expansion is used, how many terms to include.
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
            initial_global_time: time.Time,
            final_global_time:  time.Time,
            third_body_grav_param: float,
            third_body: spacecraft.Satellite,
            central_body: spacecraft.Satellite = None,
            dcm: np.ndarray = None,
            legendre: bool = True,
            legendre_series_length: int = 10,
    ):
        super().__init__()

        self.tb_grav_param = third_body_grav_param
        self.legendre = legendre
        self.legendre_series_length = legendre_series_length
        self.dcm = dcm

        # If no DCM is passed set it to a 3x3 identity matrix.
        if self.dcm is None:
            self.dcm = np.array(([1, 0, 0], [0, 1, 0], [0, 0, 1]))

        # Propagate the orbit's of the third and central bodies and then convert them into cubic splines. If the
        # central body's orbit is not to be propagated instead set it to a function which always returns [0, 0, 0] when
        # passed any time value.
        propagator = propagation.UniversalVariablePropagator()
        third_body.loggers = [logging.StateLogger()]

        if central_body is None:
            propagator.propagate(
                satellites={third_body.name: third_body},
                runtime = (final_global_time.julian_date - initial_global_time.julian_date) * 86400,
            )

            def dummy_spline(x):
                return np.array([0, 0, 0])
            self.cb_orbit_spline = dummy_spline
        else:
            central_body.loggers = [logging.StateLogger()]
            propagator.propagate(
                satellites={third_body.name: third_body, central_body.name: central_body},
                runtime=(final_global_time.julian_date - initial_global_time.julian_date) * 86400,
            )

            self.cb_orbit_spline = sp.interpolate.make_interp_spline(
                central_body.time_history.squeeze(), central_body.position_history.T, k=3
            )

        self.tb_orbit_spline = sp.interpolate.make_interp_spline(
            third_body.time_history.squeeze(), third_body.position_history.T, k=3
        )

    def evaluate(self, time: float, state: np.ndarray, satellite: spacecraft.Satellite) -> np.ndarray:
        """
        Computes the perturbing acceleration due to the third body.

        Parameters
        ----------
        time : float
            Current time in seconds since propagation began.
        state : np.ndarray
            Current translational state in ECI coordinates given as (position, velocity).
        satellite : :class:`~hohmannpy.astro.Satellite`
            Unused parameter simply based due to ``@abstractmethod`` requirements.

        Returns
        -------
        acceleration : np.ndarray
            Current translational acceleration in ECI coordinates.
        """

        # Calculate the position of the third body wrt. the central body and the satellite.
        position_tb_wrt_cb = self.dcm @ (-self.cb_orbit_spline(time) + self.tb_orbit_spline(time))
        position_tb_wrt_sat = position_tb_wrt_cb - state[:3]

        # The third and central body may be in the same relative location wrt. the central body if the central body is
        # very far away, such as if the perturbation is caused by the Sun. Rather than using the N-body equation in this
        # instance we can optionally rewrite it as a Legendre polynomial expansion of the cosine of the phase angle
        # between the satellite and third body wrt. the central body.
        if self.legendre:
            phase_angle_cosine = (
                    np.dot(state[:3], position_tb_wrt_cb)
                        / (np.linalg.norm(state[:3]) * np.linalg.norm(position_tb_wrt_cb))
            )

            # Compute the Legendre polynomial series sum.
            legendre_sum = 0
            position_ratio = np.linalg.norm(state[:3]) / np.linalg.norm(position_tb_wrt_cb)
            for i in range(1, self.legendre_series_length):
                legendre_sum += sp.special.legendre_p(i, phase_angle_cosine) * position_ratio ** i

            # Compute the acceleration.
            acceleration = (
                    -self.tb_grav_param / np.linalg.norm(position_tb_wrt_cb) ** 3
                        * (state[:3]
                           - position_tb_wrt_sat * (3 * legendre_sum + 3 * legendre_sum ** 2 + legendre_sum ** 3)
                           )
            )
        else:  # Alternative acceleration computation if the user just wants the standard N-body equation to be used.
            acceleration = (
                self.tb_grav_param * (
                    position_tb_wrt_sat / np.linalg.norm(position_tb_wrt_sat) ** 3
                        - position_tb_wrt_cb / np.linalg.norm(position_tb_wrt_cb) ** 3
                )
            )

        return acceleration


class LunarGravity(ThirdBodyGravity):
    r"""
    Perturbation caused by the Moon's gravity.

    This class implements a specialized version of :class:`~hohmannpy.astro.perturbations.ThirdBodyGravity` adjusted to
    specifically account for the third body perturbations due to the Earth's moon.

    Parameters
    ----------
    initial_global_time: :class:`~hohmannpy.astro.Time`
        Gregorian date and UT1 time at which propagation of the Moon's orbit should begin. Should match the initial and
        final time passed to the :class:`~hohmannpy.astro.Mission` which holds this perturbation.
    final_global_time: :class:`~hohmannpy.astro.Time`
        Gregorian date and UT1 time at which propagation of the Moon's orbit should end. Should match the initial and
        final time passed to the ``~hohmannpy.astro.Mission`` which holds this perturbation.
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
            initial_global_time: time.Time,
            final_global_time: time.Time,
            initial_true_anomaly: float,
            legendre: bool = True,
            legendre_series_length: int = 10,
    ):
        # Initialize the Moon.
        moon = spacecraft.Moon(initial_true_anomaly)

        super().__init__(
            initial_global_time=initial_global_time,
            final_global_time=final_global_time,
            third_body_grav_param=4.9048695e12,
            third_body=moon,
            legendre=legendre,
            legendre_series_length=legendre_series_length
        )


class SolarGravity(ThirdBodyGravity):
    r"""
    Perturbation caused by the Sun's gravity.

    This class implements a specialized version of :class:`~hohmannpy.astro.perturbations.ThirdBodyGravity` adjusted to
    specifically account for the third body perturbations due to the Sun. The true anomaly of the Earth wrt. to the
    ecliptic plane is computed automatically based on the dates of desired propagation.

    Parameters
    ----------
    initial_global_time: :class:`~hohmannpy.astro.Time`
        Gregorian date and UT1 time at which propagation of Earth's orbit should begin. Should match the initial and
        final time passed to the :class:`~hohmannpy.astro.Mission` which holds this perturbation.
    final_global_time: :class:`~hohmannpy.astro.Time`
        Gregorian date and UT1 time at which propagation of the Earth's orbit should end. Should match the initial and
        final time passed to the ``Mission`` which holds this perturbation.
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
            initial_global_time: time.Time,
            final_global_time: time.Time,
            legendre: bool = True,
            legendre_series_length: int = 10,
    ):
        # Initialize the Earth.
        earth = spacecraft.Earth(initial_global_time)
        earth_tilt = np.deg2rad(-23.439291115)  # Rotate from Sun-fixed to Earth-fixed frame via the Earth's axial tilt.

        super().__init__(
            initial_global_time=initial_global_time,
            final_global_time=final_global_time,
            third_body_grav_param=1.32712440018e20,
            third_body=earth,
            legendre=legendre,
            legendre_series_length=legendre_series_length,
            dcm=dcms.euler_2_dcm(earth_tilt, 1)
        )

        # Currently we have a spline which returns the Earth as the third body orbiting about the Sun. We want the Sun
        # to be the third body so we wrap the orbit_spline of the Earth so that it always returns the position of the
        # Sun wrt. the Earth instead.
        tb_orbit_spline = copy.deepcopy(self.tb_orbit_spline)
        def inverted_spline(x):
            return -tb_orbit_spline(x)
        self.tb_orbit_spline = inverted_spline

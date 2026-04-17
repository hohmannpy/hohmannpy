from __future__ import annotations

import numpy as np
import scipy as sp

from ..dynamics import conversions
from ..astro import time, propagation
from .. import spacecraft
from . import base


# TODO: Add rotation to this.
class SolarRadiation(base.Perturbation):
    r"""
    Perturbation caused by solar radiation from the Sun.

    Parameters
    ----------
    irradiance_scale_factor : float
        Constant to scale the solar irradiance by at all timesteps. Useful for representing heightened solar activity
        such as during solar flares.
    shadowing: bool
        Flag which determines whether the Earth's shadow should be modeled.

    Attributes
    ----------
    earth_orbit_spline : :class:`numpy.BSpline`
        Linear spline of the Earth's trajectory. Calling it via ``earth_orbit_spline(time)`` returns the interpolated
        orbit at that time.

    Notes
    -----
    The following assumptions are made for this implementation:

    1) Solar irradiance is given by an equation developed by Wertz [1]_ and is primarily based on the number of days since Earth's last aphelion.

    2) Aphelion is taken to have occurred on the most recent July 4th, 12:00:00 UT1 preceding ``initial_global_time``. It is then taken to occur every 365.25 Julian days after this (leap days are not accounted for).

    3) The 1-2 plane of the Earth-centered-inertial (ECI) basis is also assumed to be inclined at constant 23.5 :math:`deg` from the ecliptic plane.

    4) The reflective area facing the Sun and reflectivity of said area is also assumed to be constant and attitude-independent.

    5) As a consequence of 4, the perturbing acceleration is said to act along a line from the Sun to the satellite.

    6) The Earth's shadow (if enabled) is assumed to be a cylinder of equivalent radius to the Earth along the Earth-Sun line on the shadowed face of the Earth. Solar radiation is fully disabled whenever the satellite lies in this cylinder and otherwise is assumed to act at full strength.

    .. [1] James R. Wertz, Spacecraft Attitude Determination and Control, Astrophysics and Space Science Library, vol.
        73. Dordrecht, The Netherlands: Springer, 1978
    """

    def __init__(
            self,
            irradiance_scale_factor: float = 1,
            shadowing: bool = True
    ):
        super().__init__()

        self._irradiance_scale_factor = irradiance_scale_factor
        self._shadowing = shadowing

        # Setup finished in finalize__init__() which is called by the Mission.
        self.earth_orbit_spline = None
        self._initial_jd_since_aphelion = None

    def _finalize__init__(self, initial_global_time: time.Time, final_global_time: time.Time):
        """
        Create a ``np.BSpline`` for the Earth's orbit and determine the current number of Julian days since the last
        aphelion passage.

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

        # Get a spline corresponding to the Earth's orbit.
        from ..astro import celestial  # Stuffed down here to prevent circular imports.
        earth = celestial.Earth(initial_global_time)

        propagator = propagation.KeplerPropagator()
        propagator._propagate(
            satellites={earth.name: earth},
            runtime=(final_global_time.julian_date - initial_global_time.julian_date) * 86400,
            include_rotation=False
        )

        self.earth_orbit_spline = sp.interpolate.make_interp_spline(
            earth.time_history.squeeze(), earth.position_history.T, k=3
        )

        # Compute the Julian days since the most recent aphelion. First we determine if we are in a month before or
        # after the current year's aphelion (July 4th, 12:00:00 UT1). If after just compute the Julian days since the
        # current years aphelion. If after, compute the Julian days since last year's aphelion.
        initial_date = initial_global_time._date
        initial_month = initial_date[3:5]

        if int(initial_month) > 7:
            initial_year = int(initial_date[6:])
        else:
            initial_year = int(initial_date[6:]) - 1

        aphelion_time = time.Time(date=f"07/04/{initial_year}", time="12:00:00")
        self._initial_jd_since_aphelion = initial_global_time.julian_date - aphelion_time.julian_date

    def evaluate(self, time: float, state: np.ndarray, satellite: spacecraft.Satellite) -> np.ndarray:
        r"""
        Computes the perturbing acceleration due to the Sun's radiation.

        Parameters
        ----------
        time : float
            Current time in seconds since propagation began.
        state : np.ndarray
            Current translational state in ECI coordinates given as (position, velocity) or
            (position, velocity, quaternion, angular_velocity).
        satellite : :class:`~hohmannpy.astro.Satellite`
            Satellite object. Passed so that its ``mass``, ``mean_reflective_area``, and ``reflectivity`` may be
            accessed.

        Returns
        -------
        acceleration : np.ndarray
            Current translational acceleration in ECI coordinates.
        """

        speed_of_light = 3e8
        earth_radius = 6378.1363e3

        # Compute the position of the Sun wrt. the satellite. This involves first transforming the positon of the Earth
        # wrt. the sun from the heliocentric to ECI frame (this position is accessed from the earth_orbit_spline). Then
        # add the ECI position of the satellite to that to get the position of the satellite wrt. the Sun and then
        # finally invert this vector.
        earth_tilt = np.deg2rad(-23.439291115)
        position_earth_wrt_sun = conversions.euler_2_dcm(earth_tilt, 1) @ self.earth_orbit_spline(time)
        position_sun_wrt_sat = -(position_earth_wrt_sun + state[:3])

        # Compute the solar pressure. We can get the irradiance by plugging the days since the Earth's last aphelion
        # passage into the Wertz model. Then that is divided by the speed of light.
        days_since_aphelion = (self._initial_jd_since_aphelion + time / 86400) % 365.25
        irradiance = 1358 / (1.004 + 0.0334 * np.cos(2 * np.pi * days_since_aphelion)) * self._irradiance_scale_factor
        solar_pressure = irradiance / speed_of_light

        # Perform a check to see if the satellite is in the Earth's shadow. First check if the cosine of the angle
        # between the Earth-Sun line and the satellite's position vector is less than 0. If so, the satellite is on the
        # shadowed side of the Earth. However, this alone is not enough because the satellite may be far enough away
        # from the Earth that it is still not eclipsed. To account for this, perform a second check to determine if the
        # satellite lies in the Earth's shade cylinder. This can be done by finding the component of the satellite's
        # position perpendicular to the Earth-Sun line and checking to see if it's magnitude is less than the radius of
        # the cylinder (which is equivalent to Earth's radius).
        if self._shadowing:
            position_sun_wrt_earth = -position_earth_wrt_sun
            sun_angle_check = (
                    np.dot(position_sun_wrt_earth, state[:3])
                        / (np.linalg.norm(position_sun_wrt_earth) * np.linalg.norm(state[:3]))
            )
            if sun_angle_check < 1:
                shade_uvec = position_earth_wrt_sun / np.linalg.norm(position_earth_wrt_sun)
                if np.linalg.norm(state[:3] - np.dot(state[:3], shade_uvec) * shade_uvec) < earth_radius:
                    return np.array([0, 0, 0])  # If eclipsed, disable solar radiation.

        # Compute the acceleration.
        acceleration = (
            -solar_pressure * satellite.reflectivity * satellite.mean_reflective_area / satellite.mass
                * position_sun_wrt_sat / np.linalg.norm(position_sun_wrt_sat)
        )

        if len(state) == 3:
            return acceleration
        else:
            return np.concatenate((acceleration, np.zeros(3)), axis=-1)

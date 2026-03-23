from __future__ import annotations

import numpy as np
import scipy as sp

from .propagation import universal_variable as uv_propagation  # Weird name to avoid circular import.
from . import conversions


class Orbit:
    r"""
    Class which contains all the information needed to define an orbit. This includes a variety of orbit determination
    schemes (from state, from orbital elements, Gibb's method, etc;) as well as functions to compute all the parameters
    which define an orbit.

    The basic ``__init__()`` creates an object from the Cartesian state. As an alternative to this, a variety of
    different instantiations decorated using ``@classmethod`` are available for alternative forms of orbit
    determination. Internally, these all use the input parameters to generate the Cartesian state which is then used to
    call the base ``__init__()``.

    .. caution::

        The Cartesian state, as well as orbital elements (including the classical and equinoctial ones), are all stored
        as attributes of this class. The Cartesian state acts as the source of truth for the orbit and is always kept up
        to date. The other attributes are not updated automatically from said state when they are accessed (such as by
        using the ``@property`` decorator). The rationale for this is that when determining an orbit from say classical
        elements, if these elements were then passed into a Keplerian solver it would be expected that they would not
        change. However, if the elements were constantly recomputed from the Cartesian state it would introduce
        floating-point errors that could accumulate over time. Instead, manual update methods of the form
        update_element() must be called to force a recomputation. These should be used with caution, and ideally only
        :meth:`~hohmannpy.astro.Orbit.update_classical()` and :meth:`~hohmannpy.astro.Orbit.update_equinoctial()`
        should be called externally.

    Parameters
    ----------
    position : np.ndarray
        A (3, ) vector of the satellite's planet-centered inertial position, units of :math:`m`.
    velocity : np.ndarray
        A (3, ) vector of the satellite's planet-centered velocity, units of :math:`m`.
    grav_param : float
        Constant related to the gravitational field strength of the central body, units of :math:`m^3/s^2`. Defaults to
        that of the Earth.
    track_equinoctial : bool
        Flag which indicates whether to track the equinoctial elements. By default, only the Cartesian state and the
        classical orbital elements (as well as their degenerate cases) are tracked.

    Attributes
    ----------
    position : np.ndarray
        A (3, ) vector of the satellite's planet-centered inertial position.
    velocity : np.ndarray
        A (3, ) vector of the satellite's planet-centered velocity.
    spf_angular_momentum : np.ndarray
        A (3, ) vector of the satellite's specific angular momentum.
    eccentricity_vec : np.ndarray
        A (3, ) vector whose magnitude is the eccentricity of the orbit and that points towards the orbit's periapsis.
    nodal_vec : np.ndarray
        A (3, ) vector which points along the line of intersection between the orbital and ecliptic (PCI 1-2) planes
        towards the right ascending node.
    time : float
        Current simulation time. This starts at 0 when propagation begins and is not equivalent to the actual time in
        for example UT1 and Gregorian date.
    grav_param : float
        Constant related to the gravitational field strength of the central body. Defaults to that of the Earth.
    sm_axis : float
        :math:`1/2` the length of the orbit's major axis.
    sl_rectum : float
        :math:`1/2` the length of the orbit's latus rectum (line perpendicular to the major axis whose midpoint is the
        primary foci of the orbit.
    eccentricity : float
        How elliptical the orbit is (<1 = ellipse, >1 = hyperbola).
    raan : float
        (Longitude of the right ascending node) Angle between the PCI 1-axis (approximately the Vernal equinox) and the
        nodal vector.
    argp : float
        (Argument of periapsis) Angle between nodal and eccentricity vectors in the orbital plane.
    inclination : float
        Angle between the orbital and ecliptic (PCI 1-2) planes. Equivalently the angle between the eccentricity and
        specific angular momentum vectors.
    true_anomaly : float
        Angle between the eccentricity and position vector, "fast variable" used to track the satellite's current
        location.
    longp : float
        (Longitude of periapsis) Sum of the RAAN and argument of periapsis, only well-defined when the inclination is
        zero.
    argl : float
        (Argument of latitude) Sum of the argument of periapsis and true anomaly, only well-defined when the orbit is
        circular and used in place of the true anomaly to track the satellite.
    true_latitude : float
        Sum of the RAAN, argument of periapsis, and true anomaly, only well-defined when the inclination is zero and the
        orbit is circular and used in place of the true anomaly to track the satellite.
    e_component1 : float
        1-component of the project of the eccentricity vector into the equinoctial plane.
    e_component2 : float
        2-component of the project of the eccentricity vector into the equinoctial plane.
    n_component1 : float
        1-component of the project of the nodal vector into the equinoctial plane.
    n_component2 : float
        2-component of the project of the eccentricity vector into the equinoctial plane.
    eccentric_anomaly: float
        Eccentric anomaly of the orbit. Not used by default unless orbit is propagated using
        :class:`~hohmannpy.astro.KeplerPropagator`.
    stumpff_param: float
        Stumpff parameter of the orbit. Not used by default unless orbit is propagated using
        :class:`~hohmannpy.astro.UniversalVariablePropagator`.
    universal_variable: float
        Universal variable of the orbit. Not used by default unless orbit is propagated using
        ``UniversalVariablePropagator``.
    inverse_sm_axis: float
        Inverse of the semi-major axis of the orbit. Used by ``UniversalVariablePropagator`` so that it can handle
        parabolic orbits.
    track_equinoctial : bool
        Flag which indicates whether to track the equinoctial elements. By default, only the Cartesian state and the
        classical orbital elements (as well as their degenerate cases) are tracked.

    Notes
    -----
    All vector quantities are given in planet-centered inertial (PCI) coordinates.
    """

    def __init__(
            self,
            position: np.ndarray,
            velocity: np.ndarray,
            grav_param: float = 3.986004418e14,  # Default to Earth in units of m^3/s^2.
            track_equinoctial: bool = False,
            *,  # Hides parameters beneath this from the user.
            _default: bool = True,
    ):
        self.position = position
        self.velocity = velocity
        self.time = 0
        self.grav_param = grav_param
        self.track_equinoctial = track_equinoctial

        # Compute orbital elements along with other useful attributes.
        if _default:
            self.update_classical()
            if self.track_equinoctial:
                self.update_equinoctial()

        # Initialize other attributes which are propagator-specific.
        self.eccentric_anomaly = None
        self.stumpff_param = None
        self.universal_variable = None
        self.inverse_sm_axis = None

    # ---------------------------------
    # ALTERNATE INSTANTIATION FUNCTIONS
    # ---------------------------------
    @classmethod
    def from_state(
            cls,
            position: np.ndarray,
            velocity: np.ndarray,
            grav_param: float = 3.986004418e14,  # Default to Earth in units of m^3/s^2.
            track_equinoctial: bool = False,
    ) -> Orbit:
        r"""
        Identical to ``__init__()``. Returns an ``Orbit`` based on the Cartesian position and velocity.
        """

        return cls(position, velocity, grav_param, track_equinoctial)

    @classmethod
    def from_classical_elements(
            cls,
            sm_axis: float,
            eccentricity: float,
            raan: float,
            inclination: float,
            argp: float,
            true_anomaly: float,
            grav_param: float = 3.986004418e14,  # Default to Earth in units of m^3/s^2.
            track_equinoctial: bool = False,
    ) -> Orbit:
        r"""
        Alternative constructor which performs orbit determination using the classical orbital elements.

        It takes in the classical orbital elements and calls :func:`~hohmannpy.astro.conversions.elements_2_state()` to
        convert the Cartesian position and velocity and from there generate the orbit.

        Parameters
        ----------
        sm_axis : float
            Semi-major axis.
        eccentricity : float
            Eccentricity.
        raan : float
            Right ascension (longitude) of the ascending node.
        argp: float
            Argument of periapsis.
        inclination : float
            Inclination.
        true_anomaly : float
            True anomaly.
        grav_param: float
            Gravitational parameter of the central body (defaults to that of the Earth in :math:`\text{m}^3/\text{s}^2`).
        track_equinoctial : bool
            Flag which indicates whether to track the equinoctial elements. By default, only the Cartesian state and the
            classical orbital elements (as well as their degenerate cases) are tracked.

        Returns
        -------
        orbit : :class:`~hohmannpy.astro.Orbit`
            Call to the base initialization function for this class.
        """

        position, velocity = conversions.classical_2_state(
            sm_axis=sm_axis,
            eccentricity=eccentricity,
            raan=raan,
            inclination=inclination,
            argp=argp,
            true_anomaly=true_anomaly,
            grav_param=grav_param,
        )
        orbit = cls(position, velocity, grav_param, track_equinoctial, _default=False)

        # Store/compute orbital elements. Manually call update functions because update_all() would overwrite some
        # elements passed in by the user.
        orbit._update_spf_angular_momentum()
        orbit._update_eccentricity()
        orbit._update_nodal_vec()
        orbit._update_sl_rectum()
        orbit.sm_axis = sm_axis
        orbit.raan = raan
        orbit.inclination = inclination
        orbit.argp = argp
        orbit.true_anomaly = true_anomaly
        orbit._update_longp()
        orbit._update_argl()
        orbit._update_true_latitude()

        if track_equinoctial:
            orbit.update_equinoctial()

        return orbit

    @classmethod
    def from_classical_elements_p(
        cls,
        sl_rectum: float,
        eccentricity: float,
        raan: float,
        inclination: float,
        argp: float,
        true_anomaly: float,
        grav_param: float = 3.986004418e14,  # Default to Earth in units of m^3/s^2.
        track_equinoctial: bool = False,
    ) -> Orbit:
        r"""
        Same as :meth:`from_classical_elements()` but explicitly for parabolic orbits where semi-major axis is not
        defined and the semi-latus rectum is used instead.

        Parameters
        ----------
        sl_rectum : float
            Semi-major axis.
        eccentricity : float
            Eccentricity.
        raan : float
            Right ascension (longitude) of the ascending node.
        argp: float
            Argument of periapsis.
        inclination : float
            Inclination.
        true_anomaly : float
            True anomaly.
        grav_param: float
            Gravitational parameter of the central body (defaults to that of the Earth in :math:`\text{m}^3/\text{s}^2`).
        track_equinoctial : bool
            Flag which indicates whether to track the equinoctial elements. By default, only the Cartesian state and the
            classical orbital elements (as well as their degenerate cases) are tracked.

        Returns
        -------
        orbit : :class:`~hohmannpy.astro.Orbit`
            Call to the base initialization function for this class.
        """

        position, velocity = conversions.classical_2_state_p(
            sl_rectum=sl_rectum,
            eccentricity=eccentricity,
            raan=raan,
            inclination=inclination,
            argp=argp,
            true_anomaly=true_anomaly,
            grav_param=grav_param
        )
        orbit = cls(position, velocity, grav_param, track_equinoctial, _default=False)

        # Store/compute orbital elements. Manually call update functions because update_all() would overwrite some
        # elements passed in by the user.
        orbit._update_spf_angular_momentum()
        orbit._update_eccentricity()
        orbit._update_nodal_vec()
        orbit.sl_rectum = sl_rectum
        orbit.sm_axis = np.inf
        orbit.raan = raan
        orbit.inclination = inclination
        orbit.argp = argp
        orbit.true_anomaly = true_anomaly
        orbit._update_longp()
        orbit._update_argl()
        orbit._update_true_latitude()

        if track_equinoctial:
            orbit.update_equinoctial()

        return orbit

    @classmethod
    def from_equinoctial_elements(
            cls,
            sl_rectum: float,
            e_component1: float,
            e_component2: float,
            n_component1: float,
            n_component2: float,
            true_latitude: float,
            grav_param: float = 3.986004418e14,  # Default to Earth in units of m^3/s^2.
            track_equinoctial: bool = True,
    ) -> Orbit:
        r"""
        Similar to ``from_classical_elements()`` but using the modified equinoctial elements instead of the classical
        ones.

        Parameters
        ----------
        sl_rectum : float
            Semi-major axis.
        e_component1 : float
            1-component of the project of the eccentricity vector into the equinoctial plane.
        e_component2 : float
            2-component of the project of the eccentricity vector into the equinoctial plane.
        n_component1 : float
            1-component of the project of the nodal vector into the equinoctial plane.
        n_component2 : float
            2-component of the project of the eccentricity vector into the equinoctial plane.
        true_latitude : float
            Sum of the RAAN, argument of periapsis, and true anomaly, only well-defined when the inclination is zero and
            the orbit is circular and used in place of the true anomaly to track the satellite.
        grav_param: float
            Gravitational parameter of the central body (defaults to that of the Earth in :math:`\text{m}^3/\text{s}^2`).
        track_equinoctial : bool
            Flag which indicates whether to track the equinoctial elements. By default, only the Cartesian state and the
            classical orbital elements (as well as their degenerate cases) are tracked.

        Returns
        -------
        orbit : :class:`~hohmannpy.astro.Orbit`
            Call to the base initialization function for this class.
        """

        position, velocity = conversions.equinoctial_2_state(
            sl_rectum=sl_rectum,
            e_component1=e_component1,
            e_component2=e_component2,
            n_component1=n_component1,
            n_component2=n_component2,
            true_latitude=true_latitude,
        )
        orbit = cls(position, velocity, grav_param, track_equinoctial, _default=False)

        # Store/compute orbital elements. Manually call update functions because update_all() would overwrite some
        # elements passed in by the user.
        orbit._update_spf_angular_momentum()
        orbit._update_eccentricity()
        orbit._update_nodal_vec()
        orbit.sl_rectum = sl_rectum
        orbit._update_sm_axis()
        orbit._update_raan()
        orbit._update_inclination()
        orbit._update_argp()
        orbit._update_true_anomaly()
        orbit._update_longp()
        orbit._update_argl()
        orbit.true_latitude = true_latitude

        if track_equinoctial:
            orbit.e_component1 = e_component1
            orbit.e_component2 = e_component2
            orbit.n_component1 = n_component1
            orbit.n_component2 = n_component2

        return orbit

    @classmethod
    def from_gibbs(
            cls,
            position1: np.ndarray,
            position2: np.ndarray,
            position3: np.ndarray,
            current_position_index: int = 2,
            grav_param: float = 3.986004418e14,  # Default to Earth in units of m^3/s^2.
            track_equinoctial: bool = False,
    ) -> Orbit:
        r"""
        Alternative constructor which returns the position and velocity given three co-planar position vectors. The
        process of determining the conic whose origin lies at the center of three co-planar vectors is known as Gibbs'
        method.

        .. caution:
            The three positon vectors must be ordered from earliest the latest in time when they are passed as arguments
            in order for Gibbs' method to encode retrograde/prograde correctly. They should also have true anomaly
            spacings of approximately ~5 :math:`deg` in order to ensure the resultant velocities are not ill-conditioned.

        Parameters
        ----------
        position1 : np.ndarray
            A (3, ) vector of the satellite's planet-centered inertial position at the earliest time sighting.
        position2 : np.ndarray
            A (3, ) vector of the satellite's planet-centered inertial position at the middle time sighting.
        position3 : np.ndarray
            A (3, ) vector of the satellite's planet-centered inertial position at the latest time sighting.
        current_position_index: int
            Which position vector the satellite currently is located at.
        grav_param: float
            Constant related to the gravitational field strength of the central body. Defaults to that of the Earth.
        track_equinoctial : bool
            Flag which indicates whether to track the equinoctial elements. By default, only the Cartesian state and the
            classical orbital elements (as well as their degenerate cases) are tracked.
        """

        # Form the three vectors used in Gibbs' method. The first two correspond to sl_rectum = vec1 / vec2, and the
        # third comes from eccentricity = vec3 / vec2 in the derivation. These vectors can be used to compute the
        # velocity at any of the three positions input by the user.
        gibbs_vec1 = (
                np.linalg.norm(position3) * np.cross(position1, position2)
                    + np.linalg.norm(position1) * np.cross(position2, position3)
                    + np.linalg.norm(position2) * np.cross(position3, position1)
        )
        gibbs_vec2 = np.cross(position1, position2) + np.cross(position2, position3) + np.cross(position3, position1)
        gibbs_vec3 = (
                (np.linalg.norm(position2) - np.linalg.norm(position3)) * position1
                    + (np.linalg.norm(position3) - np.linalg.norm(position1)) * position2
                    + (np.linalg.norm(position1) - np.linalg.norm(position2)) * position3
        )

        # Compute the velocity corresponding to current position.
        if current_position_index == 1:
            position = position1
        elif current_position_index == 2:
            position = position2
        elif current_position_index == 3:
            position = position3
        else:
            raise ValueError(f"{current_position_index} is not a valid position index.")

        velocity = (
                1 / np.linalg.norm(position)
                    * np.sqrt(grav_param / (np.linalg.norm(gibbs_vec1) * np.linalg.norm(gibbs_vec2)))
                    * np.cross(gibbs_vec2, position)
                    + np.sqrt(grav_param / (np.linalg.norm(gibbs_vec1) * np.linalg.norm(gibbs_vec2))) * gibbs_vec3
        )

        return cls(position, velocity, grav_param, track_equinoctial)

    @classmethod
    def from_lambert(
            cls,
            position1: np.ndarray,
            position2: np.ndarray,
            tof: float,
            grav_param: float = 3.986004418e14,  # Default to Earth in units of m^3/s^2.
            track_equinoctial: bool = False,
            current_position_index: int = 1,
            short_transfer: bool = True,
            prograde: bool = True,
            fg_constraint: bool = True,
            solver_tol: float = 1e-8,
            stumpff_tol: float = 1e-8,
            stumpff_series_length: int = 10,
    ) -> Orbit:
        """
        A universal variable implementation of Gauss' method to solving Lambert's problem.

        Lambert's problem involves finding the orbit which corresponds to two position vectors and the time-of-flight
        between them (whether that involves traveling the long or short route between them must be specified by the
        user).

        Parameters
        ----------
        position1 : np.ndarray
            A (3, ) vector of the satellite's planet-centered inertial position at one of the two sightings.
        position2 : np.ndarray
            A (3, ) vector of the satellite's planet-centered inertial position at one of the two sightings.
        tof : float
            Time-of-flight between the two positions.
        current_position_index : int
            Which position vector the satellite currently is located at. Should be either 1 or 2.
        short_transfer : bool
            Flag which indicates the satellite took the long or short arc between the two positions. Set to true for a
            short transfer arc and false for a long arc transfer.
        prograde : bool
            Flag which indicates whether satellite is in a prograde orbit, determines the sign on Lambert's constant.
            True for prograde orbits and false for retrograde ones.
        fg_constraint : bool
            Flag indicating whether to compute the gdot-series independently (increasing computation time) or to
            instead use the series constraint.
        solver_tol : float
            Tolerance to use when solving Kepler's equation.
        stumpff_tol : float
            Minimum absolute value of the Stumpff parameter before switching to the infinite series definition of the
            Stumpff series.
        stumpff_series_length :
            How many terms to evaluate in the Stumpff series when using their infinite series definitions.
        grav_param: float
            Constant related to the gravitational field strength of the central body. Defaults to that of the Earth.
        track_equinoctial : bool
            Flag which indicates whether to track the equinoctial elements. By default, only the Cartesian state and the
            classical orbital elements (as well as their degenerate cases) are tracked.
        """

        # Compute the true anomaly between the two position vectors. This could be the change in true anomaly between
        # the two positions (known as the short transfer case), but it is also possible that the satellite went the long
        # way around (known as the long transfer case). The short_transfer flag is used to decide if the short or large
        # arc solution to Lamber's problem should be used.
        true_anomaly = (
                np.arccos(np.dot(position1, position2) / (np.linalg.norm(position1) * np.linalg.norm(position2)))
        )
        if not short_transfer:
            true_anomaly = 2 * np.pi - true_anomaly
        if not prograde:
            true_anomaly *= -1

        # Define a constant to make carrying terms easier.
        lambert_const = (
                np.sqrt(np.linalg.norm(position1) * np.linalg.norm(position2)) * np.sin(true_anomaly)
                / np.sqrt(1 - np.cos(true_anomaly))
        )

        # The UniversalVariablePropagator class contains a method called stumpff_funcs() which given a value of the
        # Stumpff parameter evaluates the Stumpff series. We need this function so we're going to perform a
        # pseudo-instantiation of this class to get access to it, only passing in attributes relevant to calling
        # stumpff_funcs().
        uv_propagator = uv_propagation.UniversalVariablePropagator(
            stumpff_tol=stumpff_tol,
            stumpff_series_length=stumpff_series_length,
        )

        # Lambert's problem involves solving three equations for three unknowns to get the velocity at the initial
        # position. The three equations used here are the f and g functions and the derivative of the f function
        # (derivative of the g function is not an independent equation because of a constraint). We solve these three
        # equations for three unknowns: the Stumpff parameter, universal variable, and a third term known as the
        # Lambert parameter. These can then be plugged back in to the f and g functions to get the initial and final
        # velocities. Since the equations are transcendental in the Stumpff parameter, a root-finding method is used to
        # solve them in which a value for the Stumpff parameter is guessed, all three equations are solved and then
        # these are plugged into Kepler's equation where time is set equal to the time-of-flight.
        def eq(x):
            s_func, c_func = uv_propagator._stumpff_funcs(x)
            lambert_param = (
                    np.linalg.norm(position1) + np.linalg.norm(position2)
                        - lambert_const * (1 - x * s_func) / np.sqrt(c_func)
            )
            universal_variable = np.sqrt(lambert_param / c_func)

            return (
                tof - (universal_variable ** 3 * s_func + lambert_const * np.sqrt(lambert_param))
                    / np.sqrt(grav_param)
            )
        stumpff_param = sp.optimize.newton(eq, 0, tol=solver_tol)

        # Compute the f and g functions from the resultant change in the Stumpff parameter.
        s_func, c_func = uv_propagator._stumpff_funcs(stumpff_param)
        lambert_param = (
                np.linalg.norm(position1) + np.linalg.norm(position2)
                - lambert_const * (1 - stumpff_param * s_func) / np.sqrt(c_func)
        )
        universal_variable = np.sqrt(lambert_param / c_func)

        f_func = 1 - universal_variable ** 2 / np.linalg.norm(position1) * c_func
        g_func = tof - universal_variable ** 3 / np.sqrt(grav_param) * s_func
        gdot_func = 1 - universal_variable ** 2 / np.linalg.norm(position2) * c_func
        if fg_constraint:
            fdot_func = (f_func * gdot_func - 1) / g_func
        else:
            fdot_func = (
                    -np.sqrt(grav_param) / (np.linalg.norm(position1) * np.linalg.norm(position2))
                        * universal_variable * (1 - stumpff_param * s_func)
            )

        # Compute the velocity corresponding to current position from the f and g functions.
        if current_position_index == 1:
            position = position1
            velocity = (position2 - f_func * position1) / g_func
        elif current_position_index == 2:
            position = position2
            if fg_constraint:  # Implicitly uses fg-constraint to eliminate fdot_func and hence velocity1.
                velocity = (gdot_func * position2 - position1) / g_func
            else:
                velocity1 = (position2 - f_func * position1) / g_func
                velocity = fdot_func * position1 + gdot_func * velocity1
        else:
            raise ValueError(f"{current_position_index} is not a valid position index.")

        return cls(position, velocity, grav_param, track_equinoctial)

    # ------------------------------
    # ORBITAL ELEMENT UPDATE METHODS
    # ------------------------------
    # NOTE: Unless you know what you are doing just call update_classical() because the order these are run in matters.
    def _update_spf_angular_momentum(self):
        self.spf_angular_momentum = np.cross(self.position, self.velocity)

    def _update_eccentricity(self):  # This one updates eccentricity and eccentricity vector.
        self.eccentricity_vec = (
                np.cross(self.velocity, self.spf_angular_momentum)
                / self.grav_param - self.position / np.linalg.norm(self.position)
        )
        self.eccentricity = np.linalg.norm(self.eccentricity_vec)

    def _update_nodal_vec(self):
        unit_vec_3 = np.array([0, 0, 1])
        self.nodal_vec = np.cross(unit_vec_3, self.spf_angular_momentum)

    def _update_sl_rectum(self):
        self.sl_rectum = np.linalg.norm(self.spf_angular_momentum) ** 2 / self.grav_param

    def _update_sm_axis(self):
        self.sm_axis = self.sl_rectum / (1 - self.eccentricity ** 2)

    def _update_raan(self):
        unit_vec_1 = np.array([1, 0, 0])
        unit_vec_2 = np.array([0, 1, 0])
        raan = np.arctan2(
            np.dot(self.nodal_vec, unit_vec_2),
            np.dot(self.nodal_vec, unit_vec_1)
        )
        if raan < 0:  # Wrap to [0, 2pi].
            raan += 2 * np.pi
        self.raan = raan

    def _update_inclination(self):
        unit_vec_3 = np.array([0, 0, 1])
        self.inclination = np.arctan2(
            np.dot(self.spf_angular_momentum, np.cross(self.nodal_vec, unit_vec_3)),
            np.linalg.norm(self.nodal_vec) * np.dot(self.spf_angular_momentum, unit_vec_3)
        )

    def _update_argp(self):
        argp = np.arctan2(
            np.dot(self.eccentricity_vec, np.cross(self.spf_angular_momentum, self.nodal_vec)),
            np.linalg.norm(self.spf_angular_momentum) * np.dot(self.eccentricity_vec, self.nodal_vec)
        )
        if argp < 0:  # Wrap to [0, 2pi].
            argp += 2 * np.pi
        self.argp = argp

    def _update_e_component1(self):
        self.e_component1 = self.eccentricity * np.cos(self.argp + self.raan)

    def _update_e_component2(self):
        self.e_component2 = self.eccentricity * np.sin(self.argp + self.raan)

    def _update_n_component1(self):
        self.n_component1 = np.tan(self.inclination / 2) * np.cos(self.raan)

    def _update_n_component2(self):
        self.n_component2 = np.tan(self.inclination / 2) * np.sin(self.raan)

    def _update_true_anomaly(self):
        true_anomaly = np.arctan2(
            np.dot(self.position, np.cross(self.spf_angular_momentum, self.eccentricity_vec)),
            np.linalg.norm(self.spf_angular_momentum) * np.dot(self.position, self.eccentricity_vec)
        )
        if true_anomaly < 0:  # Wrap to [0, 2pi].
            true_anomaly += 2 * np.pi
        self.true_anomaly = true_anomaly

    def _update_longp(self):
        longp = self.raan + self.argp
        if longp > 2 * np.pi: # Wrap to [0, 2pi]. No need for < 0 wrapping since other angles are already wrapped.
            longp %= 2 * np.pi
        self.longp = longp

    def _update_argl(self):
        argl = self.argp + self.true_anomaly
        if argl > 2 * np.pi: # Wrap to [0, 2pi]. No need for < 0 wrapping since other angles are already wrapped.
            argl %= 2 * np.pi
        self.argl = argl

    def _update_true_latitude(self):
        true_latitude = self.raan + self.argp + self.true_anomaly
        if true_latitude > 2 * np.pi: # Wrap to [0, 2pi]. No need for < 0 wrapping since other angles are already wrapped.
            true_latitude %= 2 * np.pi
        self.true_latitude = true_latitude

    def update_classical(self):
        """
        Master function which updates all the classical orbital elements (as well as intermediate values needed to
        compute them and the additional parameters used in degenerate cases) based on the PCI position and
        velocity.
        """

        self._update_spf_angular_momentum()
        self._update_eccentricity()
        self._update_nodal_vec()
        self._update_sl_rectum()
        self._update_sm_axis()
        self._update_raan()
        self._update_inclination()
        self._update_argp()
        self._update_true_anomaly()
        self._update_longp()
        self._update_argl()
        self._update_true_latitude()

    def update_equinoctial(self):
        """
        Master function which updates all the modified equinoctial orbital parameters based on the given position and
        velocity. Note that this does not update the semi-latus rectum and true latitude as those are already handled in
        ``update_classical()``.
        """

        self._update_e_component1()
        self._update_e_component2()
        self._update_n_component1()
        self._update_n_component2()

from __future__ import annotations

import numpy as np

from ..dynamics import conversions


def classical_2_state(
        sm_axis: float,
        eccentricity: float,
        raan: float,
        argp: float,
        inclination: float,
        true_anomaly: float,
        grav_param: float = 3.986004418e14,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Converts the classical orbital elements (where true anomaly is the fast parameter) into inertial position and
    velocity.

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

    Returns
    -------
    position: np.ndarray
        Position of the satellite in planet-centered inertial coordinates.
    velocity: np.ndarray
        Velocity of the satellite in planet-centered inertial coordinates.

    See Also
    --------
    :func:`classical_2_state_p` : Alternate version of this function which use the semi-latus rectum instead of
        the semi-major axis. Needed for parabolic orbits where the semi-major axis is infinite.
    """

    # Construct the component's of position and velocity in the satellite's local frame. The position comes from the
    # trajectory equation and requires the semi-major axis. The velocity can be assembled from the rate of change of the
    # position's magnitude and true anomaly. The former again comes from the trajectory equation and the latter from
    # conversation of angular momentum.
    sl_rectum = sm_axis * (1 - eccentricity ** 2)
    pos_magnitude = sl_rectum / (1 + eccentricity * np.cos(true_anomaly))
    pos_magnitude_dt = np.sqrt(grav_param / sl_rectum) * eccentricity * np.sin(true_anomaly)
    true_anomaly_dt = np.sqrt(grav_param * sl_rectum) / pos_magnitude ** 2

    # Use these components to assemble the position and velocity vectors in the local frame.
    position = np.array([pos_magnitude, 0, 0])
    velocity = np.array([pos_magnitude_dt, pos_magnitude * true_anomaly_dt, 0])

    # Now, transform them to the perifocal frame. For a 3-1-1 DCM using the RAAN, inclination, and argument of periapsis
    # and then invert it to get the local -> perifocal transformation.
    local_2_perifocal_dcm = conversions.euler_2_dcm(true_anomaly, 3).T
    perifocal_2_inertial_dcm = (
            conversions.euler_2_dcm(raan, 3).T
            @ conversions.euler_2_dcm(inclination, 1).T
            @ conversions.euler_2_dcm(argp, 3).T
    )

    position = perifocal_2_inertial_dcm @ local_2_perifocal_dcm @ position
    velocity = perifocal_2_inertial_dcm @ local_2_perifocal_dcm @ velocity

    return position, velocity

def state_2_classical(
        position: np.ndarray,
        velocity: np.ndarray,
        grav_param: float = 3.986004418e14
) -> tuple[float, float, float, float, float, float]:
    r"""
    Converts inertial position and velocity into the classical orbital elements (where true anomaly is the fast
    parameter).

    Parameters
    ----------
    position: np.ndarray
        Position of the satellite in planet-centered inertial coordinates.
    velocity: np.ndarray
        Velocity of the satellite in planet-centered inertial coordinates.
    grav_param: float
        Gravitational parameter of the central body (defaults to that of the Earth in :math:`\text{m}^3/\text{s}^2`).

    Returns
    -------
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

    See Also
    --------
    :func:`state_2_classical_p` : Alternate version of this function which returns the semi-latus rectum
        instead of the semi-major axis. Useful for parabolic orbits where the semi-major axis is infinite.
    """

    # Define unit vectors in planet-centered inertial (PCI) coordinates.
    unit_vec_1 = np.array([1, 0, 0])
    unit_vec_2 = np.array([0, 1, 0])
    unit_vec_3 = np.array([0, 0, 1])

    # Compute the specific angular momentum, eccentricity, and nodal vector. These will be used in conjunction with
    # vector products to get the RAAN, inclination, argument of periapsis, and true anomaly.
    spf_angular_momentum = np.cross(position, velocity)
    eccentricity_vec = (
            np.cross(velocity, spf_angular_momentum)
            / grav_param - position / np.linalg.norm(position)
    )
    nodal_vec = np.cross(unit_vec_3, spf_angular_momentum)

    # Now do all the vector geometry to get these angles. All of these use np.arctan2() which is defined on
    # [-pi/2, pi/2] but the angles are parameterized on [0, 2pi] (except inclination which is in fact
    # defined on [-pi/2, pi/2]). We can add 2pi to these angles to fix their domains.
    raan = np.arctan2(
        np.dot(nodal_vec, unit_vec_2),
        np.dot(nodal_vec, unit_vec_1)
    )
    inclination = np.arctan2(
        np.dot(spf_angular_momentum, np.cross(nodal_vec, unit_vec_3)),
        np.linalg.norm(nodal_vec) * np.dot(spf_angular_momentum, unit_vec_3)
    )
    argp = np.arctan2(
        np.dot(eccentricity_vec, np.cross(spf_angular_momentum, nodal_vec)),
        np.linalg.norm(spf_angular_momentum) * np.dot(eccentricity_vec, nodal_vec)
    )
    true_anomaly = np.arctan2(
        np.dot(position, np.cross(spf_angular_momentum, eccentricity_vec)),
        np.linalg.norm(spf_angular_momentum) * np.dot(position, eccentricity_vec)
    )

    # Wrap angles (not needed for inclination because it is already defined on [-pi/2, pi/2]).
    if raan < 0:  # Wrap to [0, 2pi].
        raan += 2 * np.pi
    if argp < 0:  # Wrap to [0, 2pi].
        argp += 2 * np.pi
    if true_anomaly < 0:  # Wrap to [0, 2pi].
        true_anomaly += 2 * np.pi

    # Compute the semi-major axis and eccentricity.
    eccentricity = np.linalg.norm(eccentricity_vec)
    sl_rectum = np.linalg.norm(spf_angular_momentum) ** 2 / grav_param
    sm_axis = sl_rectum / (1 - eccentricity ** 2)

    return sm_axis, eccentricity, raan, inclination, argp, true_anomaly

def classical_2_state_p(
        sl_rectum: float,
        eccentricity: float,
        raan: float,
        argp: float,
        inclination: float,
        true_anomaly: float,
        grav_param: float = 3.986004418e14,
):
    r"""
    Parabolic version of :func:`classical_2_state()` which takes in the semi-latus rectum instead of the semi-major
    axis.

    For a parabolic orbit the semi-major axis is infinite. As a result computing the semi-latus rectum (and hence the
    orbital radius) from these is impossible and instead to define a parabolic orbit the semi-latus rectum must be
    defined directly. Note that for parabolic orbits only this is equivalent to the radius of periapsis.

    Parameters
    ----------
    sl_rectum : float
        Semi-latus rectum.
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

    Returns
    -------
    position: np.ndarray
        Position of the satellite in planet-centered inertial coordinates.
    velocity: np.ndarray
        Velocity of the satellite in planet-centered inertial coordinates.

    See Also
    --------
    :func:`classical_2_state` : Standard version of this function which use the semi-major axis instead of the
        semi-latus rectum.

    Notes
    -----
    The semi-latus rectum :math:`p` is computed from the eccentricity :math:`e` and semi-major axis :math:`a` via

    .. math::

        p = a (1 - e^2)

    For a parabolic orbit this results in the indeterminate form :math:`p = \infty (0)` from which the semi-latus rectum
    can not be recovered.
    """

    pos_magnitude = sl_rectum / (1 + eccentricity * np.cos(true_anomaly))  # Trajectory eq.
    pos_magnitude_dt = np.sqrt(grav_param / sl_rectum) * eccentricity * np.sin(true_anomaly)
    true_anomaly_dt = np.sqrt(grav_param * sl_rectum) / pos_magnitude ** 2

    position = np.array([pos_magnitude, 0, 0])
    velocity = np.array([pos_magnitude_dt, pos_magnitude * true_anomaly_dt, 0])

    local_2_perifocal_dcm = conversions.euler_2_dcm(true_anomaly, 3).T
    perifocal_2_inertial_dcm = (
            conversions.euler_2_dcm(raan, 3).T
            @ conversions.euler_2_dcm(inclination, 1).T
            @ conversions.euler_2_dcm(argp, 3).T
    )

    position = perifocal_2_inertial_dcm @ local_2_perifocal_dcm @ position
    velocity = perifocal_2_inertial_dcm @ local_2_perifocal_dcm @ velocity

    return position, velocity

def state_2_classical_p(
        position: np.ndarray,
        velocity: np.ndarray,
        grav_param: float = 3.986004418e14
) -> tuple[float, float, float, float, float, float]:
    r"""
    Parabolic version of :func:`state_2_classical()` which takes in the semi-latus rectum instead of the semi-major
    axis.

    For a parabolic orbit the semi-major axis is infinite. As a result computing the semi-latus rectum (and hence the
    orbital radius) from these is impossible and instead to define a parabolic orbit the semi-latus rectum must be
    defined directly. Note that for parabolic orbits only this is equivalent to the radius of periapsis.

    Parameters
    ----------
    position: np.ndarray
        Position of the satellite in planet-centered inertial coordinates.
    velocity: np.ndarray
        Velocity of the satellite in planet-centered inertial coordinates.
    grav_param: float
        Gravitational parameter of the central body (defaults to that of the Earth in :math:`\text{m}^3/\text{s}^2`).

    Returns
    -------
    sl_rectum : float
        Semi-latus rectum.
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

    See Also
    --------
    :func:`state_2_classical` : Standard version of this function which returns the semi-major axis instead of
        the semi-latus rectum.

    Notes
    -----
    The semi-latus rectum :math:`p` is computed from the eccentricity :math:`e` and semi-major axis :math:`a` via

    .. math::

        p = a (1 - e^2)

    For a parabolic orbit this results in the indeterminate form :math:`p = \infty (0)` from which the semi-latus rectum
    can not be recovered.
    """

    unit_vec_1 = np.array([1, 0, 0])
    unit_vec_2 = np.array([0, 1, 0])
    unit_vec_3 = np.array([0, 0, 1])

    spf_angular_momentum = np.cross(position, velocity)
    eccentricity_vec = (
            np.cross(velocity, spf_angular_momentum)
            / grav_param - position / np.linalg.norm(position)
    )
    nodal_vec = np.cross(unit_vec_3, spf_angular_momentum)

    raan = np.arctan2(
        np.dot(nodal_vec, unit_vec_2),
        np.dot(nodal_vec, unit_vec_1)
    )
    inclination = np.arctan2(
        np.dot(spf_angular_momentum, np.cross(nodal_vec, unit_vec_3)),
        np.linalg.norm(nodal_vec) * np.dot(spf_angular_momentum, unit_vec_3)
    )
    argp = np.arctan2(
        np.dot(eccentricity_vec, np.cross(spf_angular_momentum, nodal_vec)),
        np.linalg.norm(spf_angular_momentum) * np.dot(eccentricity_vec, nodal_vec)
    )
    true_anomaly = np.arctan2(
        np.dot(position, np.cross(spf_angular_momentum, eccentricity_vec)),
        np.linalg.norm(spf_angular_momentum) * np.dot(position, eccentricity_vec)
    )

    if raan < 0:  # Wrap to [0, 2pi].
        raan += 2 * np.pi
    if argp < 0:  # Wrap to [0, 2pi].
        argp += 2 * np.pi
    if true_anomaly < 0:  # Wrap to [0, 2pi].
        true_anomaly += 2 * np.pi

    eccentricity = np.linalg.norm(eccentricity_vec)
    sl_rectum = np.linalg.norm(spf_angular_momentum) ** 2 / grav_param

    return sl_rectum, eccentricity, raan, inclination, argp, true_anomaly

def equinoctial_2_state(
        sl_rectum: float,
        e_component1: float,
        e_component2: float,
        n_component1: float,
        n_component2: float,
        true_latitude: float,
        grav_param: float = 3.986004418e14
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Converts the modified equinoctial orbital elements (where true latitude is the fast parameter) into inertial
    position and velocity.

    Parameters
    ----------
    sl_rectum : float
        Semi-latus rectum.
    e_component1 : float
        x-component of the eccentricity vector in the planet-centered inertial basis.
    e_component2 : float
        y-component of the eccentricity vector in the planet-centered inertial basis.
    n_component1 : float
        x-component of the nodal vector in the planet-centered inertial basis.
    n_component2 : float
        y-component of the nodal vector in the planet-centered inertial basis.
    true_latitude : float
        True latitude.
    grav_param: float
        Gravitational parameter of the central body (defaults to that of the Earth in :math:`\text{m}^3/\text{s}^2`).

    Returns
    -------
    position: np.ndarray
        Position of the satellite in planet-centered inertial coordinates.
    velocity: np.ndarray
        Velocity of the satellite in planet-centered inertial coordinates.
    """

    # The definition of the equinoctial elements is long and involved so to simplify it we can define a set of
    # intermediate variables.
    var1 = n_component1 ** 2 - n_component2 ** 2  # alpha
    var2 = 1 + n_component1 ** 2 + n_component2 ** 2  # s
    var3 = 1 + e_component1 * np.cos(true_latitude) + e_component2 * np.sin(true_latitude)  # w
    var4 = sl_rectum / var3  # r

    # Construct position and velocity from these intermediate variables.
    position = var4 / var2 *  np.array([
        np.cos(true_latitude)
            + var1 * np.cos(true_latitude)
            + 2 * n_component1 * n_component2 * np.sin(true_latitude),
        np.sin(true_latitude)
            - var1 * np.sin(true_latitude)
            + 2 * n_component1 * n_component2 * np.cos(true_latitude),
        2 * (n_component1 * np.sin(true_latitude) - n_component2 * np.cos(true_latitude))
    ])
    velocity = -1 / var2 * np.sqrt(grav_param / sl_rectum) * np.array([
        np.sin(true_latitude) + var1 * np.sin(true_latitude)
            - 2 * n_component1 * n_component2 * np.cos(true_latitude)
            + e_component2 - 2 * e_component1 * n_component1 * n_component2
            + var1 * e_component2,
        -np.cos(true_latitude) + var1 * np.cos(true_latitude)
            + 2 * n_component1 * n_component2 * np.sin(true_latitude)
            - e_component1 + 2 * e_component2 * n_component1 * n_component2
            + var1 * e_component1,
        -2 * (n_component1 * np.cos(true_latitude) + n_component2 * np.sin(true_latitude)
              + e_component1 * n_component1 + e_component2 * n_component2)
    ])

    return position, velocity

def classical_2_equinoctial(
        sm_axis: float,
        eccentricity: float,
        raan: float,
        argp: float,
        inclination: float,
        true_anomaly: float,
) -> tuple[float, float, float, float, float, float]:
    """
    Converts the classical orbital elements into the modified equinoctial orbital elements.

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

    Returns
    -------
    sl_rectum : float
        Semi-latus rectum.
    e_component1 : float
        x-component of the eccentricity vector in the planet-centered inertial basis.
    e_component2 : float
        y-component of the eccentricity vector in the planet-centered inertial basis.
    n_component1 : float
        x-component of the nodal vector in the planet-centered inertial basis.
    n_component2 : float
        y-component of the nodal vector in the planet-centered inertial basis.
    true_latitude : float
        True latitude.

    Notes
    -----
    For a parabolic orbit this function will still run if an infinite semi-major axis is passed in. The resultant
    semi-latus rectum will be NAN but all other parameters will be correct.
    """

    # We can construct the e and n components by projecting the eccentricity and nodal vectors respectively into the
    # equinoctial plane.
    e_component1 = eccentricity * np.cos(argp + raan)
    e_component2 = eccentricity * np.sin(argp + raan)
    n_component1 = np.tan(inclination / 2) * np.cos(raan)
    n_component2 = np.tan(inclination / 2) * np.sin(raan)

    # Semi-latus rectum and true latitude can be easily constructed from the classical orbit elements.
    sl_rectum = sm_axis * (1 - eccentricity ** 2)
    true_latitude = raan + argp + true_anomaly

    return sl_rectum, e_component1, e_component2, n_component1, n_component2, true_latitude

def equinoctial_2_classical(
        sl_rectum: float,
        e_component1: float,
        e_component2: float,
        n_component1: float,
        n_component2: float,
        true_latitude: float,
) -> tuple[float, float, float, float, float, float]:
    """
    Converts the modified equinoctial orbital elements to the classical orbital elements.

    Parameters
    ----------
    sl_rectum : float
        Semi-latus rectum.
    e_component1 : float
        x-component of the eccentricity vector in the planet-centered inertial basis.
    e_component2 : float
        y-component of the eccentricity vector in the planet-centered inertial basis.
    n_component1 : float
        x-component of the nodal vector in the planet-centered inertial basis.
    n_component2 : float
        y-component of the nodal vector in the planet-centered inertial basis.
    true_latitude : float
        True latitude.

    Returns
    -------
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
    """

    # Convert the modified equinoctial orbital elements back to the classical orbital elements.
    sm_axis = sl_rectum / (1 - e_component1 ** 2 - e_component2 ** 2)
    eccentricity = np.sqrt(e_component1 ** 2 + e_component2 ** 2)
    inclination = np.arctan2(
        2 * np.sqrt(n_component1 ** 2 + n_component2 ** 2),
        1 - n_component1 ** 2 - n_component2 ** 2
    )
    argp = np.arctan2(
        e_component2 * n_component1 - e_component1 * n_component2,
        e_component1 * n_component1 + e_component2 * n_component2
    )
    raan = np.arctan2(n_component2, n_component1)
    true_anomaly = true_latitude - np.arctan2(e_component2, e_component1)

    return sm_axis, eccentricity, raan, argp, inclination, true_anomaly

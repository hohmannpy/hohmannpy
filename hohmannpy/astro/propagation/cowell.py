from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from . import base
from ...dynamics import attitude, quaternions

if TYPE_CHECKING:
    from ... import spacecraft


class CowellPropagator(base.Propagator):
    r"""
    Simplest non-Keplerian propagate which numerically integrates the equations of motion of a satellite using a
    4th-order Runge-Kutta method. This is known as Cowell's method by astrodynamicists.

    Two things set this apart from Keplerian methods. First, it can handle perturbing forces like
    :class:`~hohmannpy.astro.NonSphericalEarth`. However, in addition the accuracy of the propagation decreases over
    time as opposed to a Keplerian propagator which has a fixed accuracy. To mitigate this decrease the step size.

    Parameters
    ----------
    step_size : float
        Time interval between propagation steps. If one is not provided by the user it will be set in
        :meth:`propagate()` to 60 :math:`s`.
    """

    name = "Cowell"

    def __init__(
            self,
            step_size: float = 60,
            **kwargs
    ):
        super().__init__(step_size=step_size, **kwargs)

    def _set_initial_conditions(self, satellite: spacecraft.Satellite):
        pass

    def _step(self, satellite, time_change):
        # For each satellite, first retrieve the orbit. Then step the state and position forward by one timestep using
        # RK4 integration of the satellite's EOM.
        orbit = satellite.orbit
        orientation = satellite.orientation

        if self._include_rotation:
            state = self._rk4(
                t0=orbit.time - time_change,  # Need this because time change is added in super()._step_wrapper().
                delt=time_change,
                y0=np.concatenate(
                    (orbit.position, orbit.velocity, orientation.quaternion, orientation.angular_velocity)
                ),
                satellite=satellite,
            )
            orientation.quaternion = quaternions.Quaternion(state[6:10])
            orientation.angular_velocity = np.array(state[10:])
        else:
            state = self._rk4(
                t0=orbit.time-time_change,  # Need this because time change is added in super()._step_wrapper().
                delt=time_change,
                y0=np.concatenate((orbit.position, orbit.velocity)),
                satellite=satellite,
            )
        orbit.position = np.array(state[:3])
        orbit.velocity = np.array(state[3:6])

    def _eom_compiler(
            self,
            t: float,
            y: np.ndarray,  # Should be (3 x position, 3 x velocity, 4 x quaternion, 3 x angular velocity)
            satellite: spacecraft.Satellite,
            **kwargs
    ) -> np.ndarray:
        """
        Method which forms the equations of motion for a spacecraft in first order form where the state is given as
        (position, velocity, attitude).

        The default acceleration is the two-body acceleration due to the point mass acceleration of the central body.
        The perturbing accelerations and thrusts from continuous burns are then added. If enabled attitude is also
        simulated.
        """

        y0_dot, y1_dot, y2_dot = self._positon_eom(t, y)

        # Based on whether attitude dynamics are included or not assemble EOMs.
        y3_dot, y4_dot, y5_dot = self._velocity_eom(t, y, satellite, **kwargs)
        if self._include_rotation:
            y6_dot, y7_dot, y8_dot, y9_dot, y10_dot, y11_dot, y12_dot = self._attitude_eom(t, y, satellite, **kwargs)

        # Append active continuous burns. Do these first because they can change masses.
        for burn in self._active_burns[satellite.name]:
            y3_perturb, y4_perturb, y5_perturb = burn.evaluate(t, y[:6], satellite)
            y3_dot += y3_perturb
            y4_dot += y4_perturb
            y5_dot += y5_perturb

        # Append perturbing forces.
        if self._perturbing_forces is not None:
            for perturbing_force in self._perturbing_forces:
                y3_perturb, y4_perturb, y5_perturb = perturbing_force.evaluate(t, y[:6], satellite)
                y3_dot += y3_perturb
                y4_dot += y4_perturb
                y5_dot += y5_perturb

        # Return state with or without rotation.
        if self._include_rotation:
            return np.array(
                [
                    y0_dot, y1_dot, y2_dot, y3_dot, y4_dot, y5_dot,
                    y6_dot, y7_dot, y8_dot, y9_dot, y10_dot, y11_dot, y12_dot
                ]
            )
        else:
            return np.array([y0_dot, y1_dot, y2_dot, y3_dot, y4_dot, y5_dot])

    def _positon_eom(self, t: float, y: np.ndarray) -> tuple[float, float, float]:
        return y[3], y[4], y[5]

    def _velocity_eom(
            self, t: float, y: np.ndarray, satellite: spacecraft.Satellite, **kwargs
    ) -> tuple[float, float, float]:
        radius = np.sqrt(y[0] ** 2 + y[1] ** 2 + y[2] ** 2)

        y3_dot = -satellite.orbit.grav_param / radius ** 3 * y[0]
        y4_dot = -satellite.orbit.grav_param / radius ** 3 * y[1]
        y5_dot = -satellite.orbit.grav_param / radius ** 3 * y[2]

        return y3_dot, y4_dot, y5_dot

    def _attitude_eom(  # Wrapper used because Encke propagator needs to do some additional logic for these EOM.
            self, t: float, y: np.ndarray, satellite: spacecraft.Satellite, **kwargs
    ) -> tuple[float, float, float, float, float, float, float]:
        y6_dot, y7_dot, y8_dot, y9_dot = attitude.attitude_eom(t, y)
        y10_dot, y11_dot, y12_dot = attitude.rates_eom(t, y, satellite, self._perturbing_torques, **kwargs)

        return y6_dot, y7_dot, y8_dot, y9_dot, y10_dot, y11_dot, y12_dot

    def _rk4(
            self,
            t0: float,
            y0: np.ndarray,
            delt: float,
            satellite: spacecraft.Satellite,
            **kwargs,
    ) -> np.ndarray:
        """
        Perform one step of 4th-order Runge Kutta integration.
        """

        x1 = self._eom_compiler(t0, y0, satellite, **kwargs)
        x2 = self._eom_compiler(t0 + delt / 2, y0 + delt / 2 * x1, satellite, **kwargs)
        x3 = self._eom_compiler(t0 + delt / 2, y0 + delt / 2 * x2, satellite, **kwargs)
        x4 = self._eom_compiler(t0 + delt, y0 + delt * x3, satellite, **kwargs)

        return y0 + delt / 6 * (x1 + 2 * x2 + 2 * x3 + x4)

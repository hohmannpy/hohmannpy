from __future__ import annotations
import time

import numpy as np
import pygfx as gfx
import pylinalg as la
import scipy as sp

from . import base, groundtrack
from ... import astro


# TODO:
#  - Update documentation to include groundtrack.
class DynamicRenderEngine(base.RenderEngine):
    r"""
    An extension of :class:`~hohmannpy.ui.RenderEngine` which renders satellites which move along the simulated orbits.

    Rotation of the Earth is also implemented. The Greenwich mean-sidereal time of the Earth at mission start is used to
    orient the Earth and the rotation is assumed to progress linearly from there for the remainder of the mission in
    accordance with the Earth's mean rotation rate.

    Parameters
    ----------
    satellites : dict[str, :class:`~hohmannpy.astro.Satellite`]
        Dictionary of satellites created using the ``name`` parameter of each ``Satellite`` as the key and the object
        itself as the value. This holds the spacecraft trajectories to draw as well as the necessary temporal
        information to animate them.
    initial_global_time: :class:`~hohmannpy.astro.Time`
        The Gregorian date and  UT1 time at which the mission began.
    runtime : float
        How many simulated seconds the :class:`~hohmannpy.astro.Mission` ran for. This is used to loop the rendering
        animation once then end of the mission is reached.
    draw_basis: bool
        Flag which indicates whether to draw the Earth-centered inertial basis vectors.
    draw_skybox: bool
        Flag which indicates whether space should be a black void or filled with stars.

    Attributes
    ----------
    base_earth_rotation: :class:`pylinalg.Quaternion`
        Quaternion representing the initial rotation of the earth at ``initial_global_time`` with respect to the mean
        Vernal equinox. The Vernal equinox points in the x-direction of the scene coordinates.
    satellites: list[:class:`pygfx.Mesh`]
        Collection of spherical object representing the satellites.
    orbit_splines: list[:class:`scipy.BSpline`]
        Collection of linear splines of the satellite's trajectories. Calling one via ``orbit_spline[i](time)`` returns
        the interpolated orbit at that time.
    initial_global_time: :class:`~hohmannpy.astro.Time`
        The initial Gregorian date and UT1 time at which the mission began.
    initial_local_time: float
        The real-world time at which the engine begins rendering (in seconds).
    local_time: float
        The current real-world time (in seconds).
    sim_time: float
        The time since the mission began (in seconds). This may vary from ``local_time`` because the sim may be sped up.
    final_sim_time: float
        The last timestep propagator for the mission. After this is reached the sim resets.
    speed_factor: float
        How much fast ``sim_time`` is compared to ``local_time``.
    old_speed_factor: float
        The ``speed_factor`` before pausing is saved so that when the sim is unpaused the sim returns to the
        pre-pause ``speed_factor``.

    See Also
    --------
    :class:`~hohmannpy.ui.RenderEngine` : Parent of this class which implements static rendering.
    """

    def __init__(
            self,
            satellites: dict[str, astro.Satellite],
            runtime: float,
            initial_global_time: astro.Time,
            draw_basis: bool = False,
            draw_skybox: bool = True,
            draw_groundtracks: bool = False,
    ):
        # Base installation.
        super().__init__(satellites, draw_basis, draw_skybox)

        self.initial_global_time = initial_global_time
        self.initial_local_time = None  # Set during initial animation.
        self.local_time = None  # Set during initial animation.
        self.sim_time = 0
        self.final_sim_time = runtime
        self.speed_factor = 100
        self.old_speed_factor = 0

        # Orbits stored by satellites are really just a collection of points at a series of discrete timesteps. However,
        # for rendering purposes we want to be able to move to satellites continuously along their orbits. For this, we
        # turn to scipy.interpolate and convert these sets of points into splines. The orbit_splines list holds the
        # orbits in the same order as satellites are held in the satellites attribute.
        self.orbit_splines = []
        self.satellites = []
        for satellite in satellites.values():
            self.orbit_splines.append(
                sp.interpolate.make_interp_spline(
                    satellite.time_history.squeeze(),
                    satellite.position_history.T / 1000,
                    k=3
                )
            )
            self.satellites.append(self.create_satellite(satellite.color))
            self.scene.add(self.satellites[-1])
            self.satellites[-1].local.position = self.orbit_splines[-1](0)  # Set initial location of the satellite.

        # Rotate the Earth to start at the correct GMST, this overwrites any base-class rotation.
        self.base_earth_rotation = la.quat_from_euler(
            (np.pi / 2, self.initial_global_time.gmst, 0), order="XYZ"
        )  # Rotate Earth since texture is 90 deg offset about x-axis, then offset terminator in new body frame.
        self.earth.local.rotation = self.base_earth_rotation

        # Add additional event handling.
        self.canvas.add_event_handler(self.time_event_handler, "key_down")

        # If using UI (currently just groundtracks).
        self.draw_groundtracks = draw_groundtracks
        if self.draw_groundtracks:
            self.gt_engine = groundtrack.GroundtrackRenderEngine(satellites, initial_global_time, _dynamic=True)


    def animate(self):
        r"""
        See :class:`~hohmannpy.ui.RenderEngine` . :meth:`~hohmannpy.ui.RenderEngine.animate()`. This simply adds in
        ``sim_time`` marching as well as rotating the Earth and moving the satellite.
        """

        earth_rot = 7.292115e-5  # Mean rotation rate of the Earth in rad/s.

        # The local time is the real-world seconds since the rendering of the mission began, so we update it using
        # time.perf_counter(). The sim time is then updated using the change in local time scaled by speed_factor.
        old_local_time = self.local_time
        self.local_time = time.perf_counter()
        self.sim_time += (self.local_time - old_local_time) * self.speed_factor

        if self.draw_groundtracks:
            self.gt_engine.sim_time = self.sim_time

        # If the end of the simulation has been reached, reset.
        if self.sim_time > self.final_sim_time:
            self.initial_local_time = time.perf_counter()
            self.sim_time = 0

        # Rotate the Earth and move the satellites.
        self.earth.local.rotation = la.quat_mul(
            self.base_earth_rotation,
            la.quat_from_axis_angle((0, 1, 0), self.sim_time * earth_rot),
        )

        for i in range(len(self.satellites)):
            self.satellites[i].local.position = self.orbit_splines[i](self.sim_time)

        self.camera.orient()
        self.renderer.render(self.scene, self.camera)

        self.canvas.request_draw(self.animate)

    def render(self):
        r"""
        See :class:`~hohmannpy.ui.RenderEngine` . :meth:`~hohmannpy.ui.RenderEngine.render()`.
        """

        self.initial_local_time = time.perf_counter()
        self.local_time = self.initial_local_time

        if self.draw_groundtracks:
            self.gt_engine.render()

        super().render()

    def time_event_handler(self, event):
        r"""
        Extension of the event handler from ``RenderEngine`` that adds in time controls.

        Parameters
        ----------
        event: dict
            Events are dispatched as a dictionary with two main attributes: ``event_type`` and ``key``. The former
            indicates what user action took place (key presses versus key releases) and assuming the action was key
            press the latter indicates which key was actually pressed.
        """

        if event["event_type"] == "key_down":
            key = event["key"].lower()
            if key == "1":  # 1x speed.
                self.speed_factor = 1
            elif key == "2":  # 10x speed.
                self.speed_factor = 10
            elif key == "3":  # 100x speed.
                self.speed_factor = 100
            elif key == "4":  # 1000x speed.
                self.speed_factor = 1000
            elif key == "5":  # 10000x speed.
                self.speed_factor = 10000
            elif key == " ":  # Play/pause.
                if self.speed_factor == 0:
                    self.speed_factor = self.old_speed_factor
                else:
                    self.old_speed_factor = self.speed_factor
                    self.speed_factor = 0

    # --------------
    # OBJECT METHODS
    # --------------
    def create_satellite(self, color: str):
        r"""
        Method which instantiates the satellite object.

        Returns
        -------
        satellite: :class:`pygfx.Mesh`
           Satellite object which moves along the orbit.
        """

        sat_mat =  gfx.MeshPhongMaterial(color=gfx.Color(color), flat_shading=True)
        satellite = gfx.Mesh(
            gfx.sphere_geometry(radius=300, width_segments=64, height_segments=32),
            sat_mat
        )

        return satellite
from __future__ import annotations

import numpy as np
import pygfx as gfx
import rendercanvas.auto
import importlib.resources
import imageio.v3 as iio
import scipy as sp

from ... import astro


# TODO:
#  - Documentation.
#  - Overhaul of UI.
class GroundtrackRenderEngine:
    def __init__(self, satellites, initial_global_time):

        self.canvas = rendercanvas.auto.RenderCanvas(size=(640, 320), title="HohmannPy (Groundtrack Viewer)")
        self.renderer = gfx.renderers.WgpuRenderer(self.canvas)
        self.scene = gfx.Scene()

        self.sim_time = 0

        with importlib.resources.files("hohmannpy.resources").joinpath("earth_texture_map.jpg").open("rb") as f:
            earth_img = iio.imread(f)
            earth_img = np.ascontiguousarray(np.flipud(earth_img))  # Need to flip array.
            img_height, img_width = earth_img.shape[:2]
        self.pixel_wrap = img_width
        earth_img = gfx.Image(
            gfx.Geometry(grid=gfx.Texture(earth_img, dim=2)),
            gfx.ImageBasicMaterial(clim=(0, 255)),
        )
        self.scene.add(earth_img)
        earth_img.local.position = (-img_width / 2, -img_height / 2, 0)

        self.groundtracks = []
        self.satellite_icons = []
        for satellite in satellites.values():
            groundtrack = astro.Groundtrack(satellite=satellite, initial_gmst=initial_global_time.gmst)
            wrapped_groundtrack_x = img_width / (2 * np.pi) * groundtrack.longitude_history
            unwrapped_groundtrack_x = img_width / (2 * np.pi) * np.unwrap(groundtrack.longitude_history)
            groundtrack_y = img_height / np.pi * groundtrack.latitude_history
            groundtrack_z = np.ones([1, groundtrack.longitude_history.shape[1]])

            unwrapped_groundtrack_xy = np.vstack((unwrapped_groundtrack_x, groundtrack_y, groundtrack_z))
            wrapped_groundtrack_xy = np.vstack((wrapped_groundtrack_x, groundtrack_y, groundtrack_z))

            # If there were any impulsive burns their wil be two entries with the same time which breaks
            # scipy.make_interp_spline(). To resolve use numpy.where() to find duplicate times and add 1 ns to them.
            times = satellite.time_history.squeeze().copy()
            for i in range(1, len(times)):
                if times[i] <= times[i - 1]:
                    times[i] = times[i - 1] + 1e-9

            self.groundtracks.append(
                sp.interpolate.make_interp_spline(
                    times,
                    unwrapped_groundtrack_xy.T,
                    k=1
                )
            )
            wrapped_groundtrack_xy = wrapped_groundtrack_xy.astype(np.float32)  # Data type needed by gfx.Geometry.
            self.scene.add(
                gfx.Points(
                    gfx.Geometry(positions=wrapped_groundtrack_xy.T),
                    gfx.PointsMaterial(size=4, color=gfx.Color(satellite.color)),
                )
            )
            satellite_icon = gfx.Points(
                    gfx.Geometry(positions=np.array([[0, 0, 0]], dtype=np.float32)),
                    gfx.PointsMaterial(size=20, color=gfx.Color(satellite.color)),
                )
            self.scene.add(satellite_icon)
            self.satellite_icons.append(
                satellite_icon
            )

        self.camera = gfx.OrthographicCamera(img_width, img_height)
        self.camera.local.position = (0, 0, 10)

    def animate(self):
        for i in range(len(self.satellite_icons)):
            x_position, y_position, _ = self.groundtracks[i](self.sim_time)
            x_position = (x_position + self.pixel_wrap / 2) % self.pixel_wrap - self.pixel_wrap / 2


            self.satellite_icons[i].local.position = (
                x_position,
                y_position,
                1
            )

        self.renderer.render(self.scene, self.camera)
        self.canvas.request_draw(self.animate)

    def render(self):
        self.canvas.request_draw(self.animate)
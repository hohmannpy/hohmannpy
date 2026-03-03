import PySide6.QtWidgets
import rendercanvas.qt
import pygfx as gfx
import importlib.resources
import imageio.v3 as iio
import numpy as np
import scipy as sp

from ...astro import groundtracks


# TODO: Finish lower indexing for all times, and insert point for "past" display mode, and add sat icons.
class GroundtrackRenderer(PySide6.QtWidgets.QWidget):
    """
    Render engine for the groundtrack scene.
    """

    def __init__(self, sim, tabs):
        super().__init__()

        self.sim = sim
        self.initial_gmst = sim.initial_global_time.gmst
        self.objects = {}
        self.tabs = tabs

        self._canvas = rendercanvas.qt.QRenderWidget()
        self._renderer = gfx.WgpuRenderer(self._canvas)
        self._scene = gfx.Scene()

        # Load the projection of the Earth used as the backdrop for the groundtracks and add it to the scene as well as
        # the camera.
        with importlib.resources.files("hohmannpy.resources").joinpath("gfx/earth_texture_map.jpg").open("rb") as f:
            earth_img = iio.imread(f)
            earth_img = np.ascontiguousarray(np.flipud(earth_img))  # Need to flip array.
            self.img_height, self.img_width = earth_img.shape[:2]
        self.pixel_wrap = self.img_width
        self.objects["projection"] = gfx.Image(
            gfx.Geometry(grid=gfx.Texture(earth_img, dim=2)),
            gfx.ImageBasicMaterial(clim=(0, 255)),
        )
        self._scene.add(self.objects["projection"])
        self.objects["projection"].local.position = (-self.img_width / 2, -self.img_height / 2, 0)

        self._camera = gfx.OrthographicCamera(self.img_width, self.img_height)
        self._camera.local.position = (0, 0, 10)

        # Create and render groundtracks. This is similar to the process used by OrbitRenderer so see that for more
        # details. The only main difference is that groundtracks are not continuous when rendered (the longitude
        # repeatedly jumps from pi to -pi). To account for this, these discontinuities must be detected and
        # each groundtrack split up into segments whenever this occurs. These segments are then stitched back together
        # with NaNs as breakpoints in between.
        if len(self.sim.satellites.values()) < 100:
            density = 1
        elif len(self.sim.satellites.values()) < 1000:
            density = 10
        else:
            density = 100

        self.positions = {}
        self.dense_times = np.arange(
            0,
            self.sim.final_sim_time,
            density
        )

        self.gt_chunks = {}
        self.color_chunks = {}
        self.latitude_splines = {}
        self.longitude_splines = {}
        self.gt_lengths = {}
        self.lead_points = {}
        self.lag_points = {}
        gt_buffer_break = np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)
        color_buffer_break = np.array([[0, 0, 0, 0]], dtype=np.float32)

        gt_buffer = []
        color_buffer = []
        for name, satellite in self.sim.satellites.items():
            groundtrack = groundtracks.Groundtrack(
                satellite=satellite,
                initial_gmst=self.initial_gmst,
            )
            latitudes = groundtrack.latitude_history
            longitudes = groundtrack.longitude_history
            unwrapped_longitudes = np.unwrap(longitudes)

            self.longitude_splines[name] = sp.interpolate.make_interp_spline(
                x=satellite.time_history.squeeze(),
                y=unwrapped_longitudes.T,
                k=3
            )
            self.latitude_splines[name] = sp.interpolate.make_interp_spline(
                x=satellite.time_history.squeeze(),
                y=latitudes.T,
                k=3
            )
            dense_unwrapped_longitudes = self.longitude_splines[name](self.dense_times)
            dense_wrapped_longitudes = (dense_unwrapped_longitudes + np.pi) % (2 * np.pi) - np.pi
            dense_latitudes = self.latitude_splines[name](self.dense_times)

            base_index = 0
            self.gt_chunks[name] = []
            self.gt_lengths[name] = []
            self.lead_points[name] = []
            self.lag_points[name] = []
            for i in range(1, len(self.dense_times)):
                if abs(dense_wrapped_longitudes[i] - dense_wrapped_longitudes[i - 1]) > np.pi:
                    partial_gt_chunk = np.hstack(
                        (
                            dense_wrapped_longitudes[base_index:i],
                            dense_latitudes[base_index:i],
                            np.ones_like(dense_latitudes[base_index:i])
                         )
                    )
                    partial_gt_chunk[:, 0] = self.img_width / (2 * np.pi) * partial_gt_chunk[:, 0]
                    partial_gt_chunk[:, 1] = self.img_height / np.pi * partial_gt_chunk[:, 1]
                    self.gt_chunks[name].append(
                        partial_gt_chunk.astype(np.float32)
                    )
                    self.gt_lengths[name].append(partial_gt_chunk.shape[0])

                    if np.sign(dense_wrapped_longitudes[i, 0]) < 0:
                        lead_latitude = np.interp(
                            np.pi,
                            (dense_wrapped_longitudes[i - 1, 0], dense_wrapped_longitudes[i, 0] + 2 * np.pi),
                            (dense_latitudes[i - 1, 0], dense_latitudes[i, 0])
                        )
                    else:
                        lead_latitude = np.interp(
                            -np.pi,
                            (dense_wrapped_longitudes[i - 1, 0], dense_wrapped_longitudes[i, 0] - 2 * np.pi),
                            (dense_latitudes[i - 1, 0], dense_latitudes[i, 0])
                        )

                    lead_point = np.array([[-np.pi, lead_latitude, 1]])
                    lead_point[:, 0] = self.img_width / (2 * np.pi) * lead_point[:, 0]
                    lead_point[:, 1] = self.img_height / np.pi * lead_point[:, 1]
                    self.lead_points[name].append(lead_point.astype(np.float32))

                    lag_point = np.array([[np.pi, lead_latitude, 1]])
                    lag_point[:, 0] = self.img_width / (2 * np.pi) * lag_point[:, 0]
                    lag_point[:, 1] = self.img_height / np.pi * lag_point[:, 1]
                    self.lag_points[name].append(lag_point.astype(np.float32))

                    base_index = i
            partial_gt_chunk = np.hstack(
                (
                    dense_wrapped_longitudes[base_index:],
                    dense_latitudes[base_index:],
                    np.ones_like(dense_latitudes[base_index:])
                )
            )
            partial_gt_chunk[:, 0] = self.img_width / (2 * np.pi) * partial_gt_chunk[:, 0]
            partial_gt_chunk[:, 1] = self.img_height / np.pi * partial_gt_chunk[:, 1]
            self.gt_chunks[name].append(
                partial_gt_chunk.astype(np.float32)
            )
            self.gt_lengths[name].append(partial_gt_chunk.shape[0])

            for i in range(len(self.gt_chunks[name])):
                if i == 0:
                    groundtrack = self.gt_chunks[name][i]
                else:
                    groundtrack = np.vstack(
                        (
                            groundtrack,
                            self.lag_points[name][i - 1],
                            gt_buffer_break,
                            self.lead_points[name][i - 1],
                            self.gt_chunks[name][i]
                        )
                    )

            gt_buffer.append(np.vstack((groundtrack, gt_buffer_break)))

            color = np.array(gfx.Color(satellite.color), dtype=np.float32)
            color_chunk = np.tile(color, (groundtrack.shape[0] + 1, 1))
            self.color_chunks[name] = color_chunk
            color_buffer.append(np.vstack((color_chunk, color_buffer_break)))

        self.gt_buffer = np.vstack(gt_buffer)
        self.color_buffer = np.vstack(color_buffer)
        self.objects["gt_polyline"] = gfx.Line(
            geometry=gfx.Geometry(positions=self.gt_buffer, colors=self.color_buffer),
            material=gfx.LineMaterial(color_mode="vertex")
        )

        self._scene.add(self.objects["gt_polyline"])

        # Draw the finalized canvas and add it to the gui.
        self._canvas.request_draw(self.animate)

        layout = PySide6.QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._canvas)

    def animate(self):
        """
        One frame of the animation loop.
        """

        # Don't render if this tab isn't visible.
        if self.tabs.currentIndex() != 1:
            self._canvas.request_draw(self.animate)  # Buffer a recursive call to start rendering loop.
            return

        self.gt_buffer[:, :] = np.nan  # Flush the buffer.

        buffer_index = 0
        for name in self.sim.satellites.keys():
            if self.sim.orbit_display_mode != "rso":
                if self.sim.satellite_display_flags[name]:
                    match self.sim.horizon_display_mode:
                        case "period":
                            position = self.sim.splines["positions"][name](self.sim.sim_time) * 1000
                            velocity = self.sim.splines["velocities"][name](self.sim.sim_time) * 1000
                            grav_param = self.sim.satellites[name].orbit.grav_param

                            sm_axis = -grav_param / (
                                    np.linalg.norm(velocity) ** 2 / 2 - grav_param / np.linalg.norm(position))
                            if sm_axis < 0:
                                sm_axis *= -1
                            period = 2 * np.pi * np.sqrt(sm_axis ** 3 / grav_param)
                        case "full":
                            temp_index = buffer_index
                            for i, chunk in enumerate(self.gt_chunks[name]):
                                if i != 0:
                                    self.gt_buffer[temp_index - 4] = self.lag_points[name][i - 1]
                                    self.gt_buffer[temp_index - 1] = self.lead_points[name][i - 1]
                                self.gt_buffer[temp_index : temp_index + self.gt_lengths[name][i]] = chunk
                                temp_index += self.gt_lengths[name][i] + 4
                        case "past":
                            dense_index = 0
                            upper_index = np.searchsorted(self.dense_times, self.sim.sim_time)
                            temp_index = buffer_index
                            for i, chunk in enumerate(self.gt_chunks[name]):
                                if dense_index + self.gt_lengths[name][i] < upper_index:
                                    if i < len(self.gt_chunks[name]) - 1:
                                        self.gt_buffer[temp_index + self.gt_lengths[name][i]] = self.lag_points[name][i]
                                        self.gt_buffer[temp_index + self.gt_lengths[name][i] + 3] = self.lead_points[name][i]
                                    self.gt_buffer[temp_index: temp_index + self.gt_lengths[name][i]] = chunk
                                    temp_index += self.gt_lengths[name][i] + 4
                                    dense_index += self.gt_lengths[name][i]
                                else:
                                    self.gt_buffer[temp_index: temp_index + upper_index - dense_index] = (
                                        chunk[: upper_index - dense_index, :]
                                    )
                                    break
                        case "hour":
                            dense_index = 0
                            lower_index = np.searchsorted(self.dense_times, self.sim.sim_time - 60 * 60)
                            upper_index = np.searchsorted(self.dense_times, self.sim.sim_time + 60 * 60)

                            temp_index = buffer_index
                            for i, chunk in enumerate(self.gt_chunks[name]):
                                if sum(self.gt_lengths[name][:i]):
                                    pass
                                if dense_index + self.gt_lengths[name][i] < upper_index:
                                    if i < len(self.gt_chunks[name]) - 1:
                                        self.gt_buffer[temp_index + self.gt_lengths[name][i]] = self.lag_points[name][i]
                                        self.gt_buffer[temp_index + self.gt_lengths[name][i] + 3] = \
                                        self.lead_points[name][i]
                                    self.gt_buffer[temp_index: temp_index + self.gt_lengths[name][i]] = chunk
                                    temp_index += self.gt_lengths[name][i] + 4
                                    dense_index += self.gt_lengths[name][i]
                                else:
                                    self.gt_buffer[temp_index: temp_index + upper_index - dense_index] = (
                                        chunk[: upper_index - dense_index, :]
                                    )
                                    break
                        case "half_day":
                            pass
                        case "day":
                            pass
                        case "custom":
                            pass

            buffer_index += sum(self.gt_lengths[name]) + 4 * (len(self.gt_lengths[name]) - 1) + 2

        # Rebuild buffers.
        self.objects["gt_polyline"].geometry.positions.set_data(
            self.gt_buffer
        )

        self._renderer.render(self._scene, self._camera)
        self._canvas.request_draw(self.animate)  # Buffer a recursive call to start rendering loop.

import PySide6.QtWidgets
import rendercanvas.qt
import pygfx as gfx
import importlib.resources
import imageio.v3 as iio
import numpy as np
import scipy as sp

from ...astro import groundtracks, conversions


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
        gt_buffer_break = np.full((3, 3), np.nan, dtype=np.float32)
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
                delta = dense_wrapped_longitudes[i] - dense_wrapped_longitudes[i - 1]
                if abs(delta) > np.pi:
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

                    if delta < 0:  # Left to right case.
                        lead_latitude = np.interp(
                            np.pi,
                            (dense_wrapped_longitudes[i - 1, 0], dense_wrapped_longitudes[i, 0] + 2 * np.pi),
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
                    else:  # Right to left case.
                        lead_latitude = np.interp(
                            -np.pi,
                            (dense_wrapped_longitudes[i, 0] - 2 * np.pi, dense_wrapped_longitudes[i - 1, 0]),
                            (dense_latitudes[i, 0], dense_latitudes[i - 1, 0])
                        )

                        lead_point = np.array([[np.pi, lead_latitude, 1]])
                        lead_point[:, 0] = self.img_width / (2 * np.pi) * lead_point[:, 0]
                        lead_point[:, 1] = self.img_height / np.pi * lead_point[:, 1]
                        self.lead_points[name].append(lead_point.astype(np.float32))

                        lag_point = np.array([[-np.pi, lead_latitude, 1]])
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

        # Repeat the above process but this time with a pygfx.Points object in order to render the satellites
        # themselves.
        satellite_chunks = []
        color_chunks = []

        for name, satellite in self.sim.satellites.items():
            latitude = self.latitude_splines[name](self.sim.sim_time)[0]
            longitude = self.longitude_splines[name](self.sim.sim_time)[0]
            longitude = (longitude + np.pi) % (2 * np.pi) - np.pi

            longitude = self.img_width / (2 * np.pi) * longitude
            latitude = self.img_height / np.pi * latitude

            satellite_chunks.append(np.array([[latitude, longitude, 2]]).astype(np.float32))
            color_chunks.append(np.array(gfx.Color(satellite.color)).astype(np.float32))

        satellite_buffer = np.vstack(satellite_chunks)
        color_buffer = np.vstack(color_chunks)

        self.objects["satellite_points"] = gfx.Points(
            geometry=gfx.Geometry(positions=satellite_buffer, colors=color_buffer),
            material=gfx.PointsMaterial(size=12, color_mode="vertex")
        )
        self._scene.add(self.objects["satellite_points"])
        self.satellite_buffer = satellite_buffer

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
        self.satellite_buffer[:, :] = np.nan

        buffer_index = 0
        sat_buffer_index = 0
        for name in self.sim.satellites.keys():
            if self.sim.orbit_display_mode != "rso":
                if self.sim.satellite_display_flags[name]:
                    match self.sim.horizon_display_mode:
                        case "period":
                            position = self.sim.splines["positions"][name](self.sim.sim_time) * 1000
                            velocity = self.sim.splines["velocities"][name](self.sim.sim_time) * 1000
                            grav_param = self.sim.satellites[name].orbit.grav_param

                            sm_axis, _, _, _, _, _ = conversions.state_2_classical(position, velocity, grav_param)
                            period = 2 * np.pi * np.sqrt(sm_axis ** 3 / grav_param)

                            horizon = period
                            lower_index = np.searchsorted(self.dense_times, self.sim.sim_time - horizon)
                            upper_index = np.searchsorted(self.dense_times, self.sim.sim_time + horizon)
                        case "full":
                            lower_index = 0
                            upper_index = len(self.dense_times)
                        case "past":
                            lower_index = 0
                            upper_index = np.searchsorted(self.dense_times, self.sim.sim_time)
                        case "hour":
                            horizon = 60 * 60
                            lower_index = np.searchsorted(self.dense_times, self.sim.sim_time - horizon)
                            upper_index = np.searchsorted(self.dense_times, self.sim.sim_time + horizon)
                        case "half_day":
                            horizon = 60 * 60 * 12
                            lower_index = np.searchsorted(self.dense_times, self.sim.sim_time - horizon)
                            upper_index = np.searchsorted(self.dense_times, self.sim.sim_time + horizon)
                        case "day":
                            horizon = 60 * 60 * 24
                            lower_index = np.searchsorted(self.dense_times, self.sim.sim_time - horizon)
                            upper_index = np.searchsorted(self.dense_times, self.sim.sim_time + horizon)
                        case "custom":
                            horizon = self.sim.custom_horizon
                            lower_index = np.searchsorted(self.dense_times, self.sim.sim_time - horizon)
                            upper_index = np.searchsorted(self.dense_times, self.sim.sim_time + horizon)

                    gt_index = buffer_index
                    chunk_start_index = 0
                    chunk_end_index = 0
                    buffering = False
                    for i, chunk in enumerate(self.gt_chunks[name]):
                        chunk_end_index += self.gt_lengths[name][i]
                        last_chunk = i

                        if not buffering:
                            if chunk_start_index <= lower_index < chunk_end_index:
                                flex_lag_index = gt_index + lower_index - chunk_start_index - 1
                                if upper_index <= chunk_end_index:
                                    self.gt_buffer[
                                        gt_index + lower_index - chunk_start_index
                                            : gt_index + upper_index - chunk_start_index,
                                        :
                                    ] = self.gt_chunks[name][i][lower_index - chunk_start_index
                                            : upper_index - chunk_start_index,
                                            :
                                    ]

                                    flex_lead_index = gt_index + upper_index - chunk_start_index
                                    break
                                else:
                                    self.gt_buffer[
                                        gt_index + lower_index - chunk_start_index
                                            : gt_index + self.gt_lengths[name][i],
                                        :
                                    ] = self.gt_chunks[name][i][lower_index - chunk_start_index :, :]
                                first_chunk = i - 1
                                buffering = True
                        else:
                            if i > 0:
                                if upper_index == chunk_start_index:
                                    flex_lead_index = gt_index - 5
                                    break

                                self.gt_buffer[gt_index - 5, :] = self.lag_points[name][i - 1]
                                self.gt_buffer[gt_index - 1, :] = self.lead_points[name][i - 1]
                            if chunk_start_index < upper_index <= chunk_end_index:
                                self.gt_buffer[
                                    gt_index : gt_index + upper_index - chunk_start_index, :
                                ] = self.gt_chunks[name][i][: upper_index - chunk_start_index, :]
                                flex_lead_index = gt_index + upper_index - chunk_start_index
                                break

                            else:
                                self.gt_buffer[
                                    gt_index : gt_index + self.gt_lengths[name][i], :
                                ] = self.gt_chunks[name][i]

                        chunk_start_index += self.gt_lengths[name][i]

                        if i != len(self.gt_lengths[name]) - 1:
                            gt_index += self.gt_lengths[name][i] + 5
                        else:
                            gt_index += self.gt_lengths[name][i]

                    if self.sim.horizon_display_mode == "past":
                        flex_lead_latitude = self.latitude_splines[name](self.sim.sim_time)[0]
                        flex_lead_longitude = self.longitude_splines[name](self.sim.sim_time)[0]
                        flex_lead_longitude = (flex_lead_longitude + np.pi) % (2 * np.pi) - np.pi

                        flex_lead_longitude = self.img_width / (2 * np.pi) * flex_lead_longitude
                        flex_lead_latitude = self.img_height / np.pi * flex_lead_latitude

                        delta = flex_lead_longitude - self.gt_buffer[flex_lead_index - 1, 0]
                        if abs(delta) > self.img_width / 2:
                            self.gt_buffer[flex_lead_index, :] = self.lag_points[name][last_chunk]
                            self.gt_buffer[flex_lead_index + 2, :] = self.lead_points[name][last_chunk]
                            self.gt_buffer[flex_lead_index + 3, :] = (
                                np.array([[flex_lead_longitude, flex_lead_latitude, 1]]).astype(np.float32)
                            )
                        else:
                            self.gt_buffer[flex_lead_index, :] = (
                                np.array([[flex_lead_longitude, flex_lead_latitude, 1]]).astype(np.float32)
                            )
                    elif self.sim.horizon_display_mode == "full":
                        pass
                    else:
                        lag_time = self.sim.sim_time - horizon
                        lead_time = self.sim.sim_time + horizon

                        if lead_time < self.sim.final_sim_time:
                            flex_lead_latitude = self.latitude_splines[name](self.sim.sim_time + horizon)[0]
                            flex_lead_longitude = self.longitude_splines[name](self.sim.sim_time + horizon)[0]
                            flex_lead_longitude = (flex_lead_longitude + np.pi) % (2 * np.pi) - np.pi

                            flex_lead_longitude = self.img_width / (2 * np.pi) * flex_lead_longitude
                            flex_lead_latitude = self.img_height / np.pi * flex_lead_latitude

                            delta = flex_lead_longitude - self.gt_buffer[flex_lead_index - 1, 0]
                            if abs(delta) > self.img_width / 2:
                                self.gt_buffer[flex_lead_index, :] = self.lag_points[name][last_chunk]
                                self.gt_buffer[flex_lead_index + 2, :] = self.lead_points[name][last_chunk]
                                self.gt_buffer[flex_lead_index + 3, :] = (
                                    np.array([[flex_lead_longitude, flex_lead_latitude, 1]]).astype(np.float32)
                                )
                            else:
                                self.gt_buffer[flex_lead_index, :] = (
                                    np.array([[flex_lead_longitude, flex_lead_latitude, 1]]).astype(np.float32)
                                )

                        if lag_time > 0:
                            flex_lag_latitude = self.latitude_splines[name](self.sim.sim_time - horizon)[0]
                            flex_lag_longitude = self.longitude_splines[name](self.sim.sim_time - horizon)[0]
                            flex_lag_longitude = (flex_lag_longitude + np.pi) % (2 * np.pi) - np.pi

                            flex_lag_longitude = self.img_width / (2 * np.pi) * flex_lag_longitude
                            flex_lag_latitude = self.img_height / np.pi * flex_lag_latitude

                            delta = self.gt_buffer[flex_lag_index + 1, 0] - flex_lag_longitude
                            if abs(delta) > self.img_width / 2:
                                self.gt_buffer[flex_lag_index - 3, :] = (
                                    np.array([[flex_lag_longitude, flex_lag_latitude, 1]]).astype(np.float32)
                                )
                                self.gt_buffer[flex_lag_index - 2, :] = self.lag_points[name][first_chunk]
                                self.gt_buffer[flex_lag_index, :] = self.lead_points[name][first_chunk]
                            else:
                                self.gt_buffer[flex_lag_index, :] = (
                                    np.array([[flex_lag_longitude, flex_lag_latitude, 1]]).astype(np.float32)
                                )

            buffer_index += sum(self.gt_lengths[name]) + 5 * (len(self.gt_lengths[name]) - 1) + 2

            # Update the satellite's positions using similar logic as with the orbits.
            if self.sim.orbit_display_mode != "traj":
                if self.sim.satellite_display_flags[name]:
                    latitude = self.latitude_splines[name](self.sim.sim_time)[0]
                    longitude = self.longitude_splines[name](self.sim.sim_time)[0]
                    longitude = (longitude + np.pi) % (2 * np.pi) - np.pi

                    longitude = self.img_width / (2 * np.pi) * longitude
                    latitude = self.img_height / np.pi * latitude

                    self.satellite_buffer[sat_buffer_index, :] = (
                        np.array([[longitude, latitude, 2]]).astype(np.float32)
                    )

                sat_buffer_index += 1

        # Rebuild buffers.
        self.objects["gt_polyline"].geometry.positions.set_data(
            self.gt_buffer
        )
        self.objects["satellite_points"].geometry.positions.set_data(
            self.satellite_buffer
        )

        self._renderer.render(self._scene, self._camera)
        self._canvas.request_draw(self.animate)  # Buffer a recursive call to start rendering loop.

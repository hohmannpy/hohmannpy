import PySide6.QtWidgets
import rendercanvas.qt
import pygfx as gfx
import importlib.resources
import imageio.v3 as iio
import numpy as np
import scipy as sp

from ...astro import groundtracks


class GroundtrackRenderer(PySide6.QtWidgets.QWidget):
    """
    Render engine for the groundtrack scene.
    """

    def __init__(self, sim):
        super().__init__()

        self.sim = sim
        self.objects = {}

        self._canvas = rendercanvas.qt.QRenderWidget()
        self._renderer = gfx.WgpuRenderer(self._canvas)
        self._scene = gfx.Scene()

        # Load the projection of the Earth used as the backdrop for the groundtracks and add it to the scene as well as
        # the camera.
        with importlib.resources.files("hohmannpy.resources").joinpath("gfx/earth_texture_map.jpg").open("rb") as f:
            earth_img = iio.imread(f)
            earth_img = np.ascontiguousarray(np.flipud(earth_img))  # Need to flip array.
            img_height, img_width = earth_img.shape[:2]
        self.pixel_wrap = img_width
        self.objects["projection"] = gfx.Image(
            gfx.Geometry(grid=gfx.Texture(earth_img, dim=2)),
            gfx.ImageBasicMaterial(clim=(0, 255)),
        )
        self._scene.add(self.objects["projection"])
        self.objects["projection"].local.position = (-img_width / 2, -img_height / 2, 0)

        self._camera = gfx.OrthographicCamera(img_width, img_height)
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
        gt_chunks = []
        color_chunks = []
        partial_gt_buffer_break = np.array([[np.nan, np.nan]], dtype=np.float32)
        gt_buffer_break = np.array([[np.nan, np.nan], [np.nan, np.nan]], dtype=np.float32)
        color_buffer_break = np.array([[0, 0, 0, 0]], dtype=np.float32)

        # TODO: This will break if there is a single point in a partial_gt_chunk when splining.
        for name, satellite in self.sim.satellites.items():
            groundtrack = groundtracks.Groundtrack(satellite, self.sim.initial_global_time.gmst)
            partial_gt_chunks = []

            # Iterate through all stored longitudes and look for discontinuities. When one is found, isolate the chunk
            # of the groundtrack up from the last discontinuity (represented by base_index) up to the current one.
            # Spline and densify this chunk individually and then repeat the process.
            base_index = 0
            for i in range(np.shape(groundtrack.longitude_history)[1] - 1) :
                if abs(groundtrack.longitude_history[0, i + 1] - groundtrack.longitude_history[0, i]) > np.pi:
                    latitudes = groundtrack.latitude_history[0, base_index : i + 1]
                    longitudes = groundtrack.longitude_history[0, base_index : i + 1]

                    latitude_spline = sp.interpolate.make_interp_spline(
                        satellite.time_history[0, base_index : i + 1].squeeze(),
                        latitudes.squeeze(),
                        k=1
                    )
                    longitude_spline = sp.interpolate.make_interp_spline(
                        satellite.time_history[0, base_index : i + 1].squeeze(),
                        longitudes.squeeze(),
                        k=1
                    )

                    partial_dense_times = self.dense_times[
                        (self.dense_times >= satellite.time_history[0, base_index])
                        & (self.dense_times < satellite.time_history[0, i + 1])
                        ]  # Use masking to get only the dense_times corresponding to this chunk.

                    dense_latitudes = latitude_spline(partial_dense_times).astype(np.float32)
                    dense_longitudes = longitude_spline(partial_dense_times).astype(np.float32)

                    # Since we don't interpolate over the discontinuity drawing this will cause the groundtracks to not
                    # begin on the left side of the Earth after each jump and instead start at the next sparse point
                    # computed during initial propagation. To fix this, after each jump we compute the average of the
                    # latitude before and after the jump and append this to the start of the trajectory (unless this is
                    # the very first jump).
                    if base_index != 0:
                        dense_latitudes = np.hstack((latitude_left_lead, dense_latitudes)).astype(np.float32)
                        dense_longitudes = np.hstack((-np.pi, dense_longitudes)).astype(np.float32)

                    latitude_left_lead = (
                             groundtrack.latitude_history[0, i + 1] + groundtrack.latitude_history[ 0, i]
                     ) / 2

                    partial_gt = np.stack((dense_longitudes, dense_latitudes), axis=1)
                    partial_gt_chunks.append(np.vstack((partial_gt, partial_gt_buffer_break)))
                    base_index = i + 1

            # Repeat the above process for any remaining groundtrack.
            latitudes = groundtrack.latitude_history[0, base_index:]
            longitudes = groundtrack.longitude_history[0, base_index:]
            latitude_spline = sp.interpolate.make_interp_spline(
                satellite.time_history[0, base_index:].squeeze(),
                latitudes.squeeze(),
                k=1
            )
            longitude_spline = sp.interpolate.make_interp_spline(
                satellite.time_history[0, base_index:].squeeze(),
                longitudes.squeeze(),
                k=1
            )
            partial_dense_times = self.dense_times[
                self.dense_times >= satellite.time_history[0, base_index]
                ]
            dense_latitudes = latitude_spline(partial_dense_times).astype(np.float32)
            dense_longitudes = longitude_spline(partial_dense_times).astype(np.float32)
            if base_index != 0:
                dense_latitudes = np.hstack((latitude_left_lead, dense_latitudes)).astype(np.float32)
                dense_longitudes = np.hstack((-np.pi, dense_longitudes)).astype(np.float32)

            partial_gt = np.stack((dense_longitudes, dense_latitudes), axis=1)
            partial_gt_chunks.append(partial_gt)

            # Assemble all the partial chunks into one full groundtrack chunk.
            gt_chunk = np.vstack(partial_gt_chunks)
            gt_chunks.append(np.vstack((gt_chunk, gt_buffer_break)))

            # Tile colors as well.
            color = np.array(gfx.Color(satellite.color), dtype=np.float32)
            color_chunk = np.tile(color, (gt_chunk.shape[0] + 1, 1))  # +1 to account for flex positions.
            color_chunks.append(np.vstack([color_chunk, color_buffer_break]))

        # Create the full buffer and then line object from the chunk lists. We store the buffer here because it is
        # needed later for dynamic horizon updates. We also need to extend the groundtrack to 3D (we can place it
        # slightly above the Earth's projection at Z=1) as well as transforming the groundtracks from projection
        # coordinates to Cartesian coordinates based on the size of the projection.
        gt_buffer = np.vstack(gt_chunks)
        gt_dummy_indices = np.ones((gt_buffer.shape[0], 1), dtype=np.float32)
        gt_buffer = np.hstack((gt_buffer, gt_dummy_indices))

        gt_buffer[:, 0] = img_width / (2 * np.pi) * gt_buffer[:, 0]
        gt_buffer[:, 1] = img_height / np.pi * gt_buffer[:, 1]

        color_buffer = np.vstack(color_chunks)

        self.objects["gt_polyline"] = gfx.Line(
            geometry=gfx.Geometry(positions=gt_buffer, colors=color_buffer),
            material=gfx.LineMaterial(color_mode="vertex")
        )

        self._scene.add(self.objects["gt_polyline"])
        self.gt_buffer = gt_buffer

        # Draw the finalized canvas and add it to the gui.
        self._canvas.request_draw(self.animate)

        layout = PySide6.QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._canvas)

    def animate(self):
        """
        One frame of the animation loop.
        """

        self._renderer.render(self._scene, self._camera)
        self._canvas.request_draw(self.animate)  # Recursive call to start rendering loop.
import importlib.resources
import imageio.v3 as iio
import PySide6.QtWidgets
import rendercanvas.qt
import pygfx as gfx
import numpy as np
import scipy as sp
import pylinalg as la


class OrbitRenderer(PySide6.QtWidgets.QWidget):
    def __init__(self, sim):
        super().__init__()

        self.display_mode = "full"
        self.custom_horizon = 24 * 3600  # Defaults to one day.
        self.sim = sim
        self.objects = {}
        self.frame = 0

        # Set up the internal pygfx rendering via a QRenderWidget. This is placed inside a normal QWidget so that UI
        # can be overlaid on the rendering.
        self._canvas = rendercanvas.qt.QRenderWidget()
        self._renderer = gfx.WgpuRenderer(self._canvas)
        self._scene = gfx.Scene()

        self._camera = gfx.PerspectiveCamera(fov=50, aspect=16/9)
        self._camera.local.position = (15000, 0, 0)
        self._controller = gfx.OrbitController(
            self._camera,
            register_events=self._renderer,
        )  # Add mouse control to camera.
        self._camera.show_pos((0, 0, 0), up=(0, 0, 1))

        # Add scene lighting, the Earth, and the skybox.
        self._scene.add(gfx.AmbientLight(intensity=0.1))
        sunlight = gfx.DirectionalLight(intensity=1.25)
        sunlight.local.position = (100000, 0, 0)
        self._scene.add(sunlight)

        earth_mat = gfx.MeshPhongMaterial(shininess=5)
        with importlib.resources.files("hohmannpy.resources").joinpath("gfx/earth_texture_map.jpg").open("rb") as f:
            earth_img = iio.imread(f)
            earth_img = np.ascontiguousarray(np.flipud(earth_img))  # Need to flip array.
        earth_mat.map = gfx.Texture(earth_img, dim=2)

        self.objects["earth"] = gfx.Mesh(
            gfx.sphere_geometry(radius=6371, width_segments=64, height_segments=32),
            earth_mat
        )
        self.base_earth_rotation = la.quat_from_euler(
            (np.pi / 2, 0, 0), order="XYZ"
        )  # Rotate Earth since texture is 90 deg offset about x-axis, then offset terminator in new body frame.
        self.objects["earth"].local.rotation = self.base_earth_rotation
        self._scene.add(self.objects["earth"])

        with importlib.resources.files("hohmannpy.resources").joinpath("gfx/skybox/skybox_right1.png").open("rb") as f:
            skybox_right1_img = iio.imread(f)
        with importlib.resources.files("hohmannpy.resources").joinpath("gfx/skybox/skybox_left2.png").open("rb") as f:
            skybox_left2_img = iio.imread(f)
        with importlib.resources.files("hohmannpy.resources").joinpath("gfx/skybox/skybox_top3.png").open("rb") as f:
            skybox_top3_img = iio.imread(f)
        with importlib.resources.files("hohmannpy.resources").joinpath("gfx/skybox/skybox_bottom4.png").open("rb") as f:
            skybox_bottom4_img = iio.imread(f)
        with importlib.resources.files("hohmannpy.resources").joinpath("gfx/skybox/skybox_front5.png").open("rb") as f:
            skybox_front5_img = iio.imread(f)
        with importlib.resources.files("hohmannpy.resources").joinpath("gfx/skybox/skybox_back6.png").open("rb") as f:
            skybox_back6_img = iio.imread(f)

        skybox_img = np.stack(
            [skybox_right1_img, skybox_left2_img, skybox_top3_img,
             skybox_bottom4_img, skybox_front5_img, skybox_back6_img],
            axis=0
        )  # Stack the faces.

        width = skybox_img.shape[1]
        height = skybox_img.shape[2]
        skybox = gfx.Background(
            None,
            gfx.BackgroundSkyboxMaterial(map=gfx.Texture(skybox_img, dim=2, size=(width, height, 6))),
        )

        self._scene.add(skybox)
        self._canvas.request_draw(self.animate)

        # Create the orbit objects.
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
        position_chunks = []
        color_chunks = []

        position_buffer_break = np.array([[np.nan, np.nan, np.nan]], dtype=np.float32)
        color_buffer_break = np.array([[0, 0, 0, 0]], dtype=np.float32)
        for name, satellite in self.sim.satellites.items():
            self.positions[name] = self.sim.orbit_splines[name](self.dense_times)
            self.positions[name] = self.positions[name].astype(np.float32)  # Data type needed by gfx.Geometry.

            color = np.array(gfx.Color(satellite.color), dtype=np.float32)
            color = np.tile(color, (len(self.positions[name]), 1))

            position_chunks.append(np.vstack([self.positions[name], position_buffer_break]))
            color_chunks.append(np.vstack([color, color_buffer_break]))
        orbit_buffer = np.vstack(position_chunks)
        self.orbit_buffer = orbit_buffer
        color_buffer = np.vstack(color_chunks)
        self.orbits = gfx.Line(
            geometry=gfx.Geometry(positions=orbit_buffer, colors=color_buffer),
            material=gfx.LineMaterial(color_mode="vertex")
        )
        self._scene.add(self.orbits)

        layout = PySide6.QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._canvas)

    def animate(self):
        earth_rot = 7.292115e-5  # Mean rotation rate of the Earth in rad/s.
        self.objects["earth"].local.rotation = la.quat_mul(
            self.base_earth_rotation,
            la.quat_from_axis_angle((0, 1, 0), self.sim.sim_time * earth_rot),
        )

        if self.frame % 1 == 0:
            base_index = 0
            self.orbit_buffer[:, :] = np.nan
            for name in self.positions.keys():
                match self.display_mode:
                    case "none":
                        lower_index = 0
                        upper_index = 0
                    case "full":
                        lower_index = 0
                        upper_index = len(self.dense_times)
                    case "past":
                        lower_index = 0
                        upper_index = np.searchsorted(self.dense_times, self.sim.sim_time)
                    case "hour":
                        lower_index = np.searchsorted(self.dense_times, self.sim.sim_time - 60 * 30)
                        upper_index = np.searchsorted(self.dense_times, self.sim.sim_time + 60 * 30)
                    case "half_day":
                        lower_index = np.searchsorted(self.dense_times, self.sim.sim_time - 60 * 60 * 6)
                        upper_index = np.searchsorted(self.dense_times, self.sim.sim_time + 60 * 60 * 6)
                    case "day":
                        lower_index = np.searchsorted(self.dense_times, self.sim.sim_time - 60 * 60 * 12)
                        upper_index = np.searchsorted(self.dense_times, self.sim.sim_time + 60 * 60 * 12)
                    case "custom":
                        lower_index = np.searchsorted(self.dense_times, self.sim.sim_time - self.custom_horizon)
                        upper_index = np.searchsorted(self.dense_times, self.sim.sim_time + self.custom_horizon)

                self.orbit_buffer[base_index + lower_index: base_index + upper_index, :] = self.positions[name][lower_index:upper_index, :]
                base_index += self.positions[name].shape[0] + 1

            self.orbits.geometry.positions.set_data(
               self.orbit_buffer
            )

        self._renderer.render(self._scene, self._camera)
        self._camera.show_pos((0, 0, 0), up=(0, 0, 1))
        self._canvas.request_draw(self.animate)
        self.frame += 1
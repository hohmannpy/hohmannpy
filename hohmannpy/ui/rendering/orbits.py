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
        self.sim = sim

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

        earth = gfx.Mesh(
            gfx.sphere_geometry(radius=6371, width_segments=64, height_segments=32),
            earth_mat
        )
        earth.local.rotation = la.quat_from_euler(
            (np.pi / 2, 0, 0), order="XYZ"
        )  # Rotate Earth since texture is 90 deg offset about x-axis, then offset terminator in new body frame.
        self._scene.add(earth)

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
        self.orbits = {}
        self.positions = {}
        for name, satellite in self.sim.satellites.items():
            sparse_times = satellite.time_history.squeeze()
            sparse_positions = satellite.position_history.T / 1000  # Scale to engine units (km).
            position_spline = sp.interpolate.make_interp_spline(
                    sparse_times,
                    sparse_positions,
                    k=1
                )

            self.positions[name] =  self.positions[name].astype(np.float32)  # Data type needed by gfx.Geometry.
            self.orbits[name] = gfx.Line(
                geometry=gfx.Geometry(positions= self.positions[name]),
                material=gfx.LineMaterial(color=gfx.Color(satellite.color))
            )
            self._scene.add(self.orbits[name])

        layout = PySide6.QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._canvas)

    def animate(self):
        for name, orbit in self.orbits.items():
            match self.display_mode:
                case "full":
                    positions = self.positions[name]
                case "past":
                    times = self.sim.satellites[name].time_history
                    index = np.abs(times - self.sim.sim_time).argmin()
                    positions = self.positions[name][:index]
                case "half_period":
                    pass

            self.orbits[name].geometry = gfx.Geometry(positions=positions)

        self._renderer.render(self._scene, self._camera)
        self._camera.show_pos((0, 0, 0), up=(0, 0, 1))
        self._canvas.request_draw(self.animate)
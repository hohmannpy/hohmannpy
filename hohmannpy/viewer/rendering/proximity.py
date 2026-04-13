import importlib.resources

import imageio.v3 as iio
import PySide6.QtWidgets
import PySide6.QtCore
import PySide6.QtGui
import rendercanvas.qt
import pygfx as gfx
import numpy as np
import pylinalg as la

from ...dynamics import dcms, quaternions


class ProximityRenderer(PySide6.QtWidgets.QWidget):
    """
    Render engine for the local scene around the currently selected spacecraft.
    """

    def __init__(self, sim, tabs):
        super().__init__()

        self.sim = sim
        self.initial_gmst = sim.initial_global_time.gmst
        self.objects = {}
        self.tabs = tabs

        # Set up the internal pygfx rendering via a QRenderWidget. This is placed inside a normal QWidget so that UI
        # can be overlaid on the rendering. The involves the standard pygfx pipeline of canvas -> renderer -> scene ->
        # camera -> camera controller.
        self._canvas = rendercanvas.qt.QRenderWidget()
        self._renderer = gfx.WgpuRenderer(self._canvas)
        self._scene = gfx.Scene()

        self._camera = gfx.PerspectiveCamera(fov=50, aspect=16/9)
        self._camera.local.position = (10, 0, 0)
        self._controller = gfx.OrbitController(
            self._camera,
            register_events=self._renderer,
        )  # Add mouse control to camera.
        self._camera.show_pos((0, 0, 0), up=(0, 0, 1))

        # TODO: Add logic to turn off spacecraft lighting when in eclipse.
        # Add some basic objects to the scene - lighting, the Earth, and a background skybox.
        self._scene.add(gfx.AmbientLight(intensity=0.1))
        sunlight = gfx.DirectionalLight(intensity=1.25)
        sunlight.local.position = (100000, 0, 0)  # Sun points along Vernal equinox.
        self._scene.add(sunlight)

        with importlib.resources.as_file(
                importlib.resources.files("hohmannpy.resources").joinpath("models/default.glb")
        ) as path:
            gltf = gfx.load_gltf(path)
        self.model = gltf.scene
        self._scene.add(self.model)

        earth_mat = gfx.MeshPhongMaterial(shininess=5)
        with importlib.resources.files("hohmannpy.resources").joinpath("gfx/earth_texture_map_high_res.jpg").open("rb") as f:
            earth_img = iio.imread(f)
            earth_img = np.ascontiguousarray(np.flipud(earth_img))  # Need to flip array.
        earth_mat.map = gfx.Texture(earth_img, dim=2)

        self.objects["earth"] = gfx.Mesh(
            gfx.sphere_geometry(radius=6371, width_segments=128, height_segments=64),
            earth_mat
        )
        quat = la.quat_from_euler(
            (np.pi / 2, 0, 0), order="XYZ"
        )
        self.base_earth_rotation = la.quat_from_euler(
            (np.pi / 2, 0, 0), order="XYZ"
        )
        self.objects["earth"].local.rotation = la.quat_mul(
            self.base_earth_rotation,
            la.quat_from_axis_angle((0, 1, 0), self.initial_gmst),
        ) # Rotate Earth since texture is 90 deg offset about x-axis, then offset terminator in new body frame.
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

        # Draw the finalized canvas and add it to the gui.
        self._canvas.request_draw(self.animate)

        layout = PySide6.QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._canvas)

    # TODO: Satellite rotation.
    def animate(self):
        """
        One frame of the animation loop.
        """

        # Don't render if this tab isn't visible.
        if self.tabs.currentIndex() != self.tabs.indexOf(self):
            self._canvas.request_draw(self.animate)  # Buffer a recursive call to start rendering loop.
            return

        earth_rot = 7.292115e-5  # Mean rotation rate of the Earth in rad/s.

        # Update the Earth's rotation. Note that this is a simple mean precession linear progression and doesn't contain
        # nutation.
        self.objects["earth"].local.rotation = la.quat_mul(
            self.base_earth_rotation,
            la.quat_from_axis_angle((0, 1, 0), self.initial_gmst + self.sim.sim_time * earth_rot),
        )

        # Standard rendering code, also orients the camera. If the camera is supposed to be orbiting the Earth we just
        # have it orbit (0, 0, 0). However, if it is orbiting a satellite we instead shift the entire pygfx.Scene (as
        # the scene is itself an object located in the frame of pygfx.Canvas) so that that object lives at (0, 0, 0).
        # This is done because the logic of pygfx.OrbitController breaks when trying to follow a moving target.
        if self.sim.focus is None:
            target = np.array([0, 0, 0])
        else:
            target = self.sim.splines["positions"][self.sim.focus](self.sim.sim_time)

        self.model.local.position = tuple(target)

        if self.sim.focus is not None:
            self.model.local.rotation = self.sim.splines["attitudes"][self.sim.focus](self.sim.sim_time)

        # Must update before render() call.
        self._scene.local.position = tuple(-target)
        self._camera.show_pos((0, 0, 0), up=(0, 0, 1))

        self._renderer.render(self._scene, self._camera)
        self._canvas.request_draw(self.animate)  # Buffer a recursive call to start rendering loop.

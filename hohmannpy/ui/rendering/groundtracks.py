import PySide6.QtWidgets
import rendercanvas.qt
import pygfx as gfx
import importlib.resources
import imageio.v3 as iio
import numpy as np


class GroundtrackRenderer(PySide6.QtWidgets.QWidget):
    """
    Render engine for the groundtrack scene.
    """

    def __init__(self):
        super().__init__()

        self._canvas = rendercanvas.qt.QRenderWidget()
        self._renderer = gfx.WgpuRenderer(self._canvas)
        self._scene = gfx.Scene()
        with importlib.resources.files("hohmannpy.resources").joinpath("gfx/earth_texture_map.jpg").open("rb") as f:
            earth_img = iio.imread(f)
            earth_img = np.ascontiguousarray(np.flipud(earth_img))  # Need to flip array.
            img_height, img_width = earth_img.shape[:2]
        self.pixel_wrap = img_width
        earth_img = gfx.Image(
            gfx.Geometry(grid=gfx.Texture(earth_img, dim=2)),
            gfx.ImageBasicMaterial(clim=(0, 255)),
        )
        self._scene.add(earth_img)
        earth_img.local.position = (-img_width / 2, -img_height / 2, 0)

        self._canvas.request_draw(self.animate)

        layout = PySide6.QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._canvas)
        self._camera = gfx.OrthographicCamera(img_width, img_height)
        self._camera.local.position = (0, 0, 10)

    def animate(self):
        """
        One frame of the animation loop.
        """

        self._renderer.render(self._scene, self._camera)
        self._canvas.request_draw(self.animate)  # Recursive call to start rendering loop.
import PySide6.QtWidgets
import rendercanvas.qt
import pygfx as gfx


class GroundtrackRenderer(PySide6.QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self._canvas = rendercanvas.qt.QRenderWidget()
        self._renderer = gfx.WgpuRenderer(self._canvas)
        self._scene = gfx.Scene()
        self._camera = gfx.OrthographicCamera(110, 110)

        self._canvas.request_draw(self.animate)

        layout = PySide6.QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._canvas)

    def animate(self):
        self._renderer.render(self._scene, self._camera)
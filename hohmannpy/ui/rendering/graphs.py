import PySide6.QtWidgets


class GroundtrackRenderer(PySide6.QtWidgets.QWidget):
    """
    Render engine for the groundtrack scene.
    """

    def __init__(self, sim):
        super().__init__()

        self.sim = sim
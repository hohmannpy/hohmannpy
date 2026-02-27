import PySide6.QtWidgets
import PySide6.QtCore
import PySide6.QtGui


class PropertiesDocker(PySide6.QtWidgets.QDockWidget):
    def __init__(self, sim):
        super().__init__()

        self.setWindowTitle("RSO Properties")
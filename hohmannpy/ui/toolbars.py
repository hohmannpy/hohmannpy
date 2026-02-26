import PySide6.QtWidgets
import PySide6.QtCore
import PySide6.QtGui


class ToolBar(PySide6.QtWidgets.QToolBar):
    def __init__(self):
        super().__init__()

        self.horizon_display = HorizonDisplayModeButton()
        self.addWidget(self.horizon_display)

class HorizonDisplayModeButton(PySide6.QtWidgets.QToolButton):
    mode_changed = PySide6.QtCore.Signal(str)

    def __init__(self):
        super().__init__()

        self.setText("Horizon")
        menu = PySide6.QtWidgets.QMenu(self)

        full_option = PySide6.QtGui.QAction("Full", self, checkable=True)
        past_option = PySide6.QtGui.QAction("Past", self, checkable=True)
        hp_option = PySide6.QtGui.QAction("Half-Period", self, checkable=True)

        full_option.triggered.connect(lambda: self.mode_changed.emit("full"))
        past_option.triggered.connect(lambda: self.mode_changed.emit("past"))
        hp_option.triggered.connect(lambda: self.mode_changed.emit("half_period"))

        options = PySide6.QtGui.QActionGroup(self)
        options.setExclusive(True)
        options.addAction(full_option)
        options.addAction(past_option)
        options.addAction(hp_option)

        menu.addAction(full_option)
        menu.addAction(past_option)
        menu.addAction(hp_option)
        full_option.setChecked(True)

        self.setMenu(menu)
        self.setPopupMode(PySide6.QtWidgets.QToolButton.InstantPopup)
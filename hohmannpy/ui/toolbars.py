import PySide6.QtWidgets
import PySide6.QtCore
import PySide6.QtGui


class ToolBar(PySide6.QtWidgets.QToolBar):
    def __init__(self):
        super().__init__()

        self.horizon_display = HorizonDisplayModeButton()
        self.rso_table = RSOTableButton()

        self.addWidget(self.horizon_display)
        self.addWidget(self.rso_table)

class RSOTableButton(PySide6.QtWidgets.QToolButton):
    rso_table = PySide6.QtCore.Signal()
    def __init__(self):
        super().__init__()

        self.setText("RSO")
        self.clicked.connect(self.rso_table.emit)

class HorizonDisplayModeButton(PySide6.QtWidgets.QToolButton):
    mode_changed = PySide6.QtCore.Signal(str)
    custom_horizon = PySide6.QtCore.Signal(float)

    def __init__(self):
        super().__init__()

        self.setText("Horizon")
        menu = PySide6.QtWidgets.QMenu(self)

        none_option = PySide6.QtGui.QAction("None", self, checkable=True)
        full_option = PySide6.QtGui.QAction("Full", self, checkable=True)
        past_option = PySide6.QtGui.QAction("Past", self, checkable=True)
        hour_option = PySide6.QtGui.QAction("1-Hour", self, checkable=True)
        half_day_option = PySide6.QtGui.QAction("12-Hour", self, checkable=True)
        day_option = PySide6.QtGui.QAction("24-Hour", self, checkable=True)
        custom_option = PySide6.QtGui.QAction("Custom...", self, checkable=True)

        none_option.triggered.connect(lambda: self.mode_changed.emit("none"))
        full_option.triggered.connect(lambda: self.mode_changed.emit("full"))
        past_option.triggered.connect(lambda: self.mode_changed.emit("past"))
        hour_option.triggered.connect(lambda: self.mode_changed.emit("hour"))
        half_day_option.triggered.connect(lambda: self.mode_changed.emit("half_day"))
        day_option.triggered.connect(lambda: self.mode_changed.emit("day"))
        custom_option.triggered.connect(self.custom_dialog)

        options = PySide6.QtGui.QActionGroup(self)
        options.setExclusive(True)
        options.addAction(none_option)
        options.addAction(full_option)
        options.addAction(past_option)
        options.addAction(hour_option)
        options.addAction(half_day_option)
        options.addAction(day_option)
        options.addAction(custom_option)

        menu.addAction(none_option)
        menu.addAction(full_option)
        menu.addAction(past_option)
        menu.addAction(hour_option)
        menu.addAction(half_day_option)
        menu.addAction(day_option)
        menu.addSeparator()
        menu.addAction(custom_option)
        full_option.setChecked(True)

        self.setMenu(menu)
        self.setPopupMode(PySide6.QtWidgets.QToolButton.InstantPopup)

    def custom_dialog(self):
        valid_input = False
        while not valid_input:
            horizon, valid_input = PySide6.QtWidgets.QInputDialog.getDouble(
                self,
                "Custom Horizon",
                "Seconds (>100 for high RSO counts):",
                3600,
                1,
                3600 * 24 * 365.25,
                2
            )
        self.mode_changed.emit("custom")
        self.custom_horizon.emit(horizon)

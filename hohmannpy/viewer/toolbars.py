import importlib.resources
import PySide6.QtWidgets
import PySide6.QtCore
import PySide6.QtGui


class ToolBar(PySide6.QtWidgets.QToolBar):
    """
    A set of buttons/shortcuts always available to the user.
    """

    def __init__(self):
        super().__init__()

        self.horizon_display = HorizonDisplayModeButton()
        self.rso_table = RSOTableButton()
        self.orbit_display = OrbitDisplayModeButton()
        self.sim_speed = SimSpeedButton()
        self.play_pause = PlayPauseButton()
        self.reset = ResetButton()
        self.focus_previous = FocusPreviousButton()
        self.focus_next = FocusNextButton()
        self.focus_earth = FocusEarthButton()

        spacer = PySide6.QtWidgets.QWidget()
        spacer.setSizePolicy(
            PySide6.QtWidgets.QSizePolicy.Expanding,
            PySide6.QtWidgets.QSizePolicy.Preferred
        )

        # LHS, buttons are added left to right in the order in which they appear.
        self.addWidget(self.rso_table)
        self.addWidget(self.sim_speed)
        self.addWidget(self.horizon_display)
        self.addWidget(self.orbit_display)

        # RHS.
        self.addWidget(spacer)  # Spacer auto separates LHS and RHS.

        self.addWidget(self.focus_previous)
        self.addWidget(self.focus_earth)
        self.addWidget(self.focus_next)
        self.addWidget(self.play_pause)
        self.addWidget(self.reset)


# All the toolbar buttons are defined below. Apologies for the limited documentation, they are all just QButton's which
# emitt signals picked up by slots in the Viewer.
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

        period_option = PySide6.QtGui.QAction("Period", self, checkable=True)
        full_option = PySide6.QtGui.QAction("Full", self, checkable=True)
        past_option = PySide6.QtGui.QAction("Past", self, checkable=True)
        hour_option = PySide6.QtGui.QAction("1-Hour", self, checkable=True)
        half_day_option = PySide6.QtGui.QAction("12-Hour", self, checkable=True)
        day_option = PySide6.QtGui.QAction("24-Hour", self, checkable=True)
        custom_option = PySide6.QtGui.QAction("Custom...", self, checkable=True)

        period_option.triggered.connect(lambda: self.mode_changed.emit("period"))
        full_option.triggered.connect(lambda: self.mode_changed.emit("full"))
        past_option.triggered.connect(lambda: self.mode_changed.emit("past"))
        hour_option.triggered.connect(lambda: self.mode_changed.emit("hour"))
        half_day_option.triggered.connect(lambda: self.mode_changed.emit("half_day"))
        day_option.triggered.connect(lambda: self.mode_changed.emit("day"))
        custom_option.triggered.connect(self.custom_dialog)

        options = PySide6.QtGui.QActionGroup(self)
        options.setExclusive(True)
        options.addAction(period_option)
        options.addAction(past_option)
        options.addAction(full_option)
        options.addAction(hour_option)
        options.addAction(half_day_option)
        options.addAction(day_option)
        options.addAction(custom_option)

        menu = PySide6.QtWidgets.QMenu(self)
        menu.addAction(period_option)
        menu.addAction(past_option)
        menu.addAction(full_option)
        menu.addAction(hour_option)
        menu.addAction(half_day_option)
        menu.addAction(day_option)
        menu.addSeparator()
        menu.addAction(custom_option)
        period_option.setChecked(True)

        self.setMenu(menu)
        self.setPopupMode(PySide6.QtWidgets.QToolButton.InstantPopup)

    def custom_dialog(self):
        """
        If the user wants to input a custom horizon this opens a popup dialog where they can enter it.
        """

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


class OrbitDisplayModeButton(PySide6.QtWidgets.QToolButton):
    mode_changed = PySide6.QtCore.Signal(str)

    def __init__(self):
        super().__init__()

        self.setText("View")

        rso_option = PySide6.QtGui.QAction("RSO Only", self, checkable=True)
        traj_option = PySide6.QtGui.QAction("Trajectory Only", self, checkable=True)
        both_option = PySide6.QtGui.QAction("Both", self, checkable=True)

        rso_option.triggered.connect(lambda: self.mode_changed.emit("rso"))
        traj_option.triggered.connect(lambda: self.mode_changed.emit("traj"))
        both_option.triggered.connect(lambda: self.mode_changed.emit("both"))

        options = PySide6.QtGui.QActionGroup(self)
        options.setExclusive(True)
        options.addAction(rso_option)
        options.addAction(traj_option)
        options.addAction(both_option)

        menu = PySide6.QtWidgets.QMenu(self)
        menu.addAction(rso_option)
        menu.addAction(traj_option)
        menu.addAction(both_option)
        both_option.setChecked(True)

        self.setMenu(menu)
        self.setPopupMode(PySide6.QtWidgets.QToolButton.InstantPopup)


class SimSpeedButton(PySide6.QtWidgets.QToolButton):
    mode_changed = PySide6.QtCore.Signal(str)

    def __init__(self):
        super().__init__()

        self.setText("Speed")

        speed1_option = PySide6.QtGui.QAction("1x", self, checkable=True)
        speed2_option = PySide6.QtGui.QAction("10x", self, checkable=True)
        speed3_option = PySide6.QtGui.QAction("100x", self, checkable=True)
        speed4_option = PySide6.QtGui.QAction("1000x", self, checkable=True)
        speed5_option = PySide6.QtGui.QAction("10000x", self, checkable=True)
        speed6_option = PySide6.QtGui.QAction("100000x", self, checkable=True)

        speed1_option.triggered.connect(lambda: self.mode_changed.emit("1x"))
        speed2_option.triggered.connect(lambda: self.mode_changed.emit("10x"))
        speed3_option.triggered.connect(lambda: self.mode_changed.emit("100x"))
        speed4_option.triggered.connect(lambda: self.mode_changed.emit("1000x"))
        speed5_option.triggered.connect(lambda: self.mode_changed.emit("10000x"))
        speed6_option.triggered.connect(lambda: self.mode_changed.emit("100000x"))

        options = PySide6.QtGui.QActionGroup(self)
        options.setExclusive(True)
        options.addAction(speed1_option)
        options.addAction(speed2_option)
        options.addAction(speed3_option)
        options.addAction(speed4_option)
        options.addAction(speed5_option)
        options.addAction(speed6_option)

        menu = PySide6.QtWidgets.QMenu(self)
        menu.addAction(speed1_option)
        menu.addAction(speed2_option)
        menu.addAction(speed3_option)
        menu.addAction(speed4_option)
        menu.addAction(speed5_option)
        menu.addAction(speed6_option)
        speed3_option.setChecked(True)

        self.setMenu(menu)
        self.setPopupMode(PySide6.QtWidgets.QToolButton.InstantPopup)


class PlayPauseButton(PySide6.QtWidgets.QPushButton):
    mode_changed = PySide6.QtCore.Signal(bool)

    def __init__(self):
        super().__init__()

        self.play_icon = self.style().standardIcon(PySide6.QtWidgets.QStyle.SP_MediaPlay)
        self.pause_icon = self.style().standardIcon(PySide6.QtWidgets.QStyle.SP_MediaPause)
        self.setIcon(self.pause_icon)
        self.setToolTip("Pause <i>'SPACE'</i>")
        self.setCheckable(True)
        self.setChecked(False)

        self.clicked.connect(self.on_click)

    def on_click(self):
        if self.isChecked():
            self.setIcon(self.play_icon)
            self.setToolTip("Play <i>'SPACE'</i>")
            self.mode_changed.emit(True)
        else:
            self.setIcon(self.pause_icon)
            self.setToolTip("Pause <i>'SPACE'</i>")
            self.mode_changed.emit(False)


class ResetButton(PySide6.QtWidgets.QPushButton):
    reset = PySide6.QtCore.Signal()

    def __init__(self):
        super().__init__()

        icon_path = importlib.resources.files("hohmannpy.resources").joinpath("gfx/reset_icon.png")
        self.setIcon(PySide6.QtGui.QIcon(str(icon_path)))
        self.setToolTip("Reset <i>'SHIFT + SPACE'</i>")

        self.clicked.connect(self.reset.emit)


class FocusEarthButton(PySide6.QtWidgets.QPushButton):
    focus = PySide6.QtCore.Signal()

    def __init__(self):
        super().__init__()

        icon_path = importlib.resources.files("hohmannpy.resources").joinpath("gfx/earth_icon.png")
        self.setIcon(PySide6.QtGui.QIcon(str(icon_path)))
        self.setToolTip("Focus Earth <i>'F1'</i>")

        self.clicked.connect(self.focus.emit)


class FocusPreviousButton(PySide6.QtWidgets.QPushButton):
    focus = PySide6.QtCore.Signal()

    def __init__(self):
        super().__init__()

        icon = self.style().standardIcon(PySide6.QtWidgets.QStyle.SP_MediaSeekBackward)
        self.setIcon(icon)
        self.setToolTip("Focus Previous <i>'['</i>")

        self.clicked.connect(self.focus.emit)


class FocusNextButton(PySide6.QtWidgets.QPushButton):
    focus = PySide6.QtCore.Signal()

    def __init__(self):
        super().__init__()

        icon = self.style().standardIcon(PySide6.QtWidgets.QStyle.SP_MediaSeekForward)
        self.setIcon(icon)
        self.setToolTip("Focus Next <i>']'</i>")

        self.clicked.connect(self.focus.emit)

import time
import sys
import importlib.resources

import PySide6.QtWidgets
import PySide6.QtCore
import PySide6.QtGui
import scipy as sp
import numpy as np

from . import rendering, toolbars, dockers, tables


# TODO:
#   - Hovering over satellite displays name, clicking focuses it.
#   - Document these classes.
#   - Option to enable docker after closing it.
#   - Pause rendering of non-visible tabs.
#   - FPS tracker.
#   - Favicon not showing.
class MainWindow(PySide6.QtWidgets.QMainWindow):
    def __init__(self, sim):
        super().__init__()
        self.sim = sim
        self.setWindowTitle("HohmannPy Viewer")
        self.resize(1280, 720)

        tabs = PySide6.QtWidgets.QTabWidget()
        self.setCentralWidget(tabs)

        toolbar = toolbars.ToolBar()
        self.addToolBar(toolbar)
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.horizon_display.mode_changed.connect(self.set_horizon)
        toolbar.horizon_display.custom_horizon.connect(self.set_custom_horizon)
        toolbar.rso_table.rso_table.connect(self.open_rso_table)
        toolbar.orbit_display.mode_changed.connect(self.set_orbit)
        toolbar.sim_speed.mode_changed.connect(self.set_sim_speed)
        toolbar.play_pause.mode_changed.connect(self.set_play_pause)
        toolbar.reset.reset.connect(self.reset_sim)
        toolbar.focus_previous.focus.connect(self.set_focus_previous)
        toolbar.focus_earth.focus.connect(self.set_focus_earth)
        toolbar.focus_next.focus.connect(self.set_focus_next)

        orbit_viewer = rendering.orbits.OrbitRenderer(self.sim, tabs)
        gt_viewer = rendering.GroundtrackRenderer(self.sim, tabs)
        tabs.addTab(orbit_viewer, "Orbit")
        tabs.addTab(gt_viewer, "Groundtrack")
        tabs.addTab(PySide6.QtWidgets.QWidget(), "Data Visualizer")
        orbit_viewer.space_pressed.connect(self.on_space_press)

        dock = dockers.PropertiesDocker(self.sim)
        self.addDockWidget(PySide6.QtCore.Qt.RightDockWidgetArea, dock)
        self.resizeDocks(
            [dock],
            [200],
            PySide6.QtCore.Qt.Horizontal
        )  # This call prevents the dock's size from jittering.

        status = self.statusBar()
        self.status_labels = {"sim_time": PySide6.QtWidgets.QLabel("T+00Y:000D:00:H:00M:00S")}
        status.addPermanentWidget(self.status_labels["sim_time"])

        self.elements = {
            "orbit" : orbit_viewer,
            "groundtrack" : gt_viewer,
            "toolbar" : toolbar,
            "tabs" : tabs,
            "dock" : dock,
            "status" : status,
        }

        # Some additional qt stuff for key press handling.
        self.shortcuts = {}
        self.shortcuts["space"] = PySide6.QtGui.QShortcut(PySide6.QtGui.QKeySequence(PySide6.QtCore.Qt.Key_Space), self)
        self.shortcuts["space"].setContext(PySide6.QtCore.Qt.WidgetWithChildrenShortcut)
        self.shortcuts["space"].activated.connect(self.on_space_press)

        self.shortcuts["lb"] = PySide6.QtGui.QShortcut(PySide6.QtGui.QKeySequence(PySide6.QtCore.Qt.Key_BracketLeft), self)
        self.shortcuts["lb"].setContext(PySide6.QtCore.Qt.WidgetWithChildrenShortcut)
        self.shortcuts["lb"].activated.connect(self.on_lb_press)

        self.shortcuts["rb"] = PySide6.QtGui.QShortcut(PySide6.QtGui.QKeySequence(PySide6.QtCore.Qt.Key_BracketRight), self)
        self.shortcuts["rb"].setContext(PySide6.QtCore.Qt.WidgetWithChildrenShortcut)
        self.shortcuts["rb"].activated.connect(self.on_rb_press)

        self.shortcuts["f1"] = PySide6.QtGui.QShortcut(PySide6.QtGui.QKeySequence(PySide6.QtCore.Qt.Key_F1), self)
        self.shortcuts["f1"].setContext(PySide6.QtCore.Qt.WidgetWithChildrenShortcut)
        self.shortcuts["f1"].activated.connect(self.on_f1_press)

        self.shortcuts["shift+space"] = PySide6.QtGui.QShortcut(
            PySide6.QtGui.QKeySequence(
                PySide6.QtCore.Qt.SHIFT | PySide6.QtCore.Qt.Key_Space
            ),
            self
        )
        self.shortcuts["shift+space"].setContext(PySide6.QtCore.Qt.WidgetWithChildrenShortcut)
        self.shortcuts["shift+space"].activated.connect(self.on_shift_space_press)


    @PySide6.QtCore.Slot(str)
    def set_horizon(self, signal):
        self.sim.horizon_display_mode = signal

    @PySide6.QtCore.Slot(str)
    def set_orbit(self, signal):
        self.sim.orbit_display_mode = signal

    @PySide6.QtCore.Slot(float)
    def set_custom_horizon(self, signal):
        self.sim.custom_horizon = signal

    @PySide6.QtCore.Slot(float)
    def set_sim_speed(self, signal):
        match signal:
            case "1x":
                if self.sim.speed_factor != 0:
                    self.sim.speed_factor = 1
                else:
                    self.sim.old_speed_factor = 1
            case "10x":
                if self.sim.speed_factor != 0:
                    self.sim.speed_factor = 10
                else:
                    self.sim.old_speed_factor = 10
            case "100x":
                if self.sim.speed_factor != 0:
                    self.sim.speed_factor = 100
                else:
                    self.sim.old_speed_factor = 100
            case "1000x":
                if self.sim.speed_factor != 0:
                    self.sim.speed_factor = 1000
                else:
                    self.sim.old_speed_factor = 1000
            case "10000x":
                if self.sim.speed_factor != 0:
                    self.sim.speed_factor = 10000
                else:
                    self.sim.old_speed_factor = 10000
            case "100000x":
                if self.sim.speed_factor != 0:
                    self.sim.speed_factor = 100000
                else:
                    self.sim.old_speed_factor = 100000

    @PySide6.QtCore.Slot(bool)
    def set_play_pause(self, signal):
        if signal:
            self.sim.old_speed_factor = self.sim.speed_factor
            self.sim.speed_factor = 0
        else:
            self.sim.speed_factor = self.sim.old_speed_factor

    def open_rso_table(self):
        table = tables.RSOTable(self.sim)
        table.show()

    def reset_sim(self):
        self.sim.initial_local_time = time.perf_counter()
        self.sim.sim_time = 0
        self.statusBar().showMessage("Resetting mission...", 3000)

    def set_focus_previous(self):
        names = list(self.sim.satellites.keys())

        if self.sim.focus is None:
            index = len(names) - 1
            while True:
                name = names[index]
                if self.sim.satellite_display_flags[name]:
                    self.sim.focus = name
                    break
                else:
                    index -= 1
                if index == -1:
                    self.sim.focus = None
                    break
        else:
            index = names.index(self.sim.focus) - 1
            while True:
                if index == -1:
                    self.sim.focus = None
                    break
                name = names[index]

                if self.sim.satellite_display_flags[name]:
                    self.sim.focus = name
                    break
                else:
                    index -= 1

    def set_focus_earth(self):
        self.sim.focus = None

    def set_focus_next(self):
        names = list(self.sim.satellites.keys())

        if self.sim.focus is None:
            index = 0
            while True:
                name = names[index]
                if self.sim.satellite_display_flags[name]:
                    self.sim.focus = name
                    break
                else:
                    index += 1
                if index == len(names):
                    self.sim.focus = None
                    break
        else:
            index = names.index(self.sim.focus) + 1
            while True:
                if index == len(names):
                    self.sim.focus = None
                    break
                name = names[index]

                if self.sim.satellite_display_flags[name]:
                    self.sim.focus = name
                    break
                else:
                    index += 1

    def on_space_press(self):
        self.focus_check()
        self.elements["toolbar"].play_pause.setChecked(not self.elements["toolbar"].play_pause.isChecked())
        self.elements["toolbar"].play_pause.on_click()

    def on_lb_press(self):
        self.focus_check()
        self.set_focus_previous()

    def on_f1_press(self):
        self.focus_check()
        self.set_focus_earth()

    def on_rb_press(self):
        self.focus_check()
        self.set_focus_next()

    def on_shift_space_press(self):
        self.focus_check()
        self.reset_sim()

    def focus_check(self):
        """
        Quick check called by all shortcut methods which prevents a keypress from triggering a shortcut if a text entry
        window is focused.
        """

        focus = PySide6.QtWidgets.QApplication.focusWidget()
        if isinstance(focus, (
                PySide6.QtWidgets.QLineEdit,
                PySide6.QtWidgets.QTextEdit,
                PySide6.QtWidgets.QPlainTextEdit,
                PySide6.QtWidgets.QSpinBox,
                PySide6.QtWidgets.QDoubleSpinBox,
        )):
            return

class SimManager:
    def __init__(
            self,
            satellites,
            initial_global_time,
            final_global_time,
    ):
        self.satellites = satellites

        self.gui = None
        self.local_time = time.perf_counter()
        self.initial_local_time = self.local_time
        self.sim_time = 0
        self.initial_global_time = initial_global_time
        self.final_sim_time = (final_global_time.julian_date - initial_global_time.julian_date) * 86400
        self.speed_factor = 100
        self.old_speed_factor = self.speed_factor
        self.satellite_display_flags = {name: True for name in self.satellites.keys()}
        self.focus = None
        self.orbit_display_mode = "both"
        self.horizon_display_mode = "period"
        self.custom_horizon: int = 24 * 3600  # Defaults to one day.

        self.splines = {"positions" : {}, "velocities" : {}}

        for name, satellite in self.satellites.items():
            times = satellite.time_history
            for i in range(1, times.shape[1]):
                if times[0, i] <= times[0, i - 1]:
                    times[0, i] = times[0, i - 1] + 1e-9
            positions = satellite.position_history.T / 1000
            velocities = satellite.velocity_history.T / 1000

            self.splines["positions"][name] = (
                sp.interpolate.make_interp_spline(
                    times.squeeze(),
                    positions,
                    k=3
                )
            )
            self.splines["velocities"][name] = (
                sp.interpolate.make_interp_spline(
                    times.squeeze(),
                    velocities,
                    k=3
                )
            )

        self.timer = PySide6.QtCore.QTimer()
        self.timer.timeout.connect(self.frame_update)

    def sim_clock(self):
        old_local_time = self.local_time
        self.local_time = time.perf_counter()
        self.sim_time += (self.local_time - old_local_time) * self.speed_factor

        # If the end of the simulation has been reached, reset.
        if self.sim_time > self.final_sim_time:
            self.initial_local_time = time.perf_counter()
            self.sim_time = 0
            self.gui.statusBar().showMessage("End of mission reached, resetting...", 3000)

    def frame_update(self):
        self.sim_clock()

        years = np.floor(self.sim_time / (365.25 * 24 * 60 * 60))
        remainder = self.sim_time % (365.25 * 24 * 60 * 60)
        days = np.floor(remainder / (24 * 60 * 60))
        remainder = remainder % (24 * 60 * 60)
        hours = np.floor(remainder / (60 * 60))
        remainder = remainder % (60 * 60)
        minutes = np.floor(remainder / 60)
        seconds = remainder % 60

        self.gui.status_labels["sim_time"].setText(
            f"T+{years:02.0f}Y:{days:03.0f}D:{hours:02.0f}H:{minutes:02.0f}M:{seconds:05.2f}S"
        )

    def run(self):
        app = PySide6.QtWidgets.QApplication(sys.argv)

        icon_path = importlib.resources.files("hohmannpy.resources").joinpath("gfx/app_icon.png")
        app.setWindowIcon(PySide6.QtGui.QIcon(str(icon_path)))

        self.gui = MainWindow(self)
        self.gui.show()

        self.timer.start(17)
        app.exec()

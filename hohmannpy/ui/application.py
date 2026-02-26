import time
import sys

import PySide6.QtWidgets
import PySide6.QtCore
import scipy as sp
import numpy as np

from . import rendering, toolbars


class MainWindow(PySide6.QtWidgets.QMainWindow):
    def __init__(self, sim):
        super().__init__()
        self.sim = sim
        self.setWindowTitle("HohmannPy")
        self.resize(1280, 720)

        tabs = PySide6.QtWidgets.QTabWidget()
        self.setCentralWidget(tabs)

        toolbar = toolbars.ToolBar()
        self.addToolBar(toolbar)
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.horizon_display.mode_changed.connect(self.set_horizon)

        orbit_viewer = rendering.orbits.OrbitRenderer(self.sim)
        gt_viewer = rendering.GroundtrackRenderer()
        tabs.addTab(orbit_viewer, "Orbit")
        tabs.addTab(gt_viewer, "Groundtrack")

        dock = PySide6.QtWidgets.QDockWidget("Inspector", self)
        dock.setWidget(PySide6.QtWidgets.QLabel("Satellite info here"))
        self.addDockWidget(PySide6.QtCore.Qt.RightDockWidgetArea, dock)

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

    @PySide6.QtCore.Slot(str)
    def set_horizon(self, signal):
        self.elements["orbit"].display_mode = signal

class SimManager:
    def __init__(
            self,
            satellites,
            initial_global_time,
            final_global_time,
    ):
        self.satellites = satellites

        self.orbit_splines = {}
        self.gui = None
        self.local_time = time.perf_counter()
        self.initial_local_time = self.local_time
        self.sim_time = 0
        self.final_sim_time = (final_global_time.julian_date - initial_global_time.julian_date) * 86400
        self.speed_factor = 100

        for name, satellite in self.satellites.items():
            sparse_times = satellite.time_history.squeeze()
            for i in range(1, len(sparse_times)):
                if sparse_times[i] <= sparse_times[i - 1]:
                    sparse_times[i] = sparse_times[i - 1] + 1e-9
            sparse_positions = satellite.position_history.T / 1000

            self.orbit_splines[name] = (
                sp.interpolate.make_interp_spline(
                    times.squeeze(),
                    ,
                    k=1
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
        self.gui = MainWindow(self)
        self.gui.show()

        self.timer.start(16)  # ~60 FPS (16 ms)
        app.exec()
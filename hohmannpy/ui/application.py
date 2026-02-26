import time
import sys
import importlib.resources

import PySide6.QtWidgets
import PySide6.QtCore
import PySide6.QtGui
import scipy as sp
import numpy as np

from . import rendering, toolbars
from ..astro import logging, conversions


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

        orbit_viewer = rendering.orbits.OrbitRenderer(self.sim)
        gt_viewer = rendering.GroundtrackRenderer()
        tabs.addTab(orbit_viewer, "Orbit")
        tabs.addTab(gt_viewer, "Groundtrack")

        dock = PySide6.QtWidgets.QDockWidget("RSO Properties", self)
        dock.setWidget(PySide6.QtWidgets.QLabel("RSO info here"))
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
        self.elements["orbit"].horizon_display_mode = signal

    @PySide6.QtCore.Slot(str)
    def set_orbit(self, signal):
        self.elements["orbit"].orbit_display_mode = signal

    @PySide6.QtCore.Slot(float)
    def set_custom_horizon(self, signal):
        self.elements["orbit"].custom_horizon = signal

    @PySide6.QtCore.Slot(float)
    def set_sim_speed(self, signal):
        match signal:
            case "1x":
                self.sim.speed_factor = 1
            case "10x":
                self.sim.speed_factor = 10
            case "100x":
                self.sim.speed_factor = 100
            case "1000x":
                self.sim.speed_factor = 1000
            case "10000x":
                self.sim.speed_factor = 10000

    def open_rso_table(self):
        table = RSOTable(self.sim)
        table.show()


class RSOTable(PySide6.QtWidgets.QDialog):
    """
    Table which holds information on all the RSOs in the sim.
    """

    def __init__(self, sim):
        super().__init__()
        self.sim = sim

        self.setWindowTitle("RSO Information")
        self.resize(800, 500)

        self.layout = PySide6.QtWidgets.QVBoxLayout(self)  # Wrapper around the QDialog needed to actually display.

        # Create the table. Each row represents a satellite and each column is a different set of information of the
        # satellite.
        self.table = PySide6.QtWidgets.QTableWidget()
        self.table.setRowCount(len(self.sim.satellites.keys()))

        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels(
            [
                "Name", "Display",
                "Semijax (km)", "Eccentricity", "RAAN (rad)", "ArgP (rad)", "Inclination (rad)", "True Anomaly (rad)"
            ]
        )

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, PySide6.QtWidgets.QHeaderView.Stretch)  # Set column to take up any extra space.
        header.setSectionResizeMode(1, PySide6.QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(7, PySide6.QtWidgets.QHeaderView.ResizeToContents)
        self.table.setEditTriggers(PySide6.QtWidgets.QAbstractItemView.NoEditTriggers)  # Need or user can overwrite cells.

        self.checkboxes = []
        for i, (name, satellite) in enumerate(self.sim.satellites.items()):
            self.table.setItem(i, 0, PySide6.QtWidgets.QTableWidgetItem(name))

            # Column with a toggle indicating whether the RSO should be displayed in the sim. This involves creating a
            # QCheckBox widget and then placing it inside another widget which itself is placed inside a table cell.
            # This is the easiest way to ensure the checkbox is center-aligned.
            checkbox = PySide6.QtWidgets.QCheckBox()
            checkbox.setChecked(self.sim.satellite_display_flags[name])
            checkbox.toggled.connect(lambda checked, name=name: self.satellite_toggled(name, checked))

            cell = PySide6.QtWidgets.QWidget()
            cell_layout = PySide6.QtWidgets.QHBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setAlignment(PySide6.QtCore.Qt.AlignCenter)
            cell_layout.addWidget(checkbox)

            self.table.setCellWidget(i, 1, cell)
            self.checkboxes.append(checkbox)

            position = self.sim.splines["positions"][name](self.sim.sim_time) * 1000
            velocity = self.sim.splines["velocities"][name](self.sim.sim_time) * 1000
            grav_param = satellite.orbit.grav_param

            sm_axis, eccentricity, raan, inclination, argp, true_anomaly = (
                conversions.state_2_classical(position, velocity, grav_param)
            )

            sm_axis_col = PySide6.QtWidgets.QTableWidgetItem(f"{sm_axis:.3f}")
            eccentricity_col = PySide6.QtWidgets.QTableWidgetItem(f"{eccentricity:.6f}")
            raan_col = PySide6.QtWidgets.QTableWidgetItem(f"{raan:.6f}")
            inclination_col = PySide6.QtWidgets.QTableWidgetItem(f"{inclination:.6f}")
            argp_col = PySide6.QtWidgets.QTableWidgetItem(f"{argp:.6f}")
            true_anomaly_col = PySide6.QtWidgets.QTableWidgetItem(f"{true_anomaly:.6f}")

            sm_axis_col.setTextAlignment(PySide6.QtCore.Qt.AlignCenter)
            eccentricity_col.setTextAlignment(PySide6.QtCore.Qt.AlignCenter)
            raan_col.setTextAlignment(PySide6.QtCore.Qt.AlignCenter)
            inclination_col.setTextAlignment(PySide6.QtCore.Qt.AlignCenter)
            argp_col.setTextAlignment(PySide6.QtCore.Qt.AlignCenter)
            true_anomaly_col.setTextAlignment(PySide6.QtCore.Qt.AlignCenter)

            self.table.setItem(i, 2, sm_axis_col)
            self.table.setItem(i, 3, eccentricity_col)
            self.table.setItem(i, 4, raan_col)
            self.table.setItem(i, 5, inclination_col)
            self.table.setItem(i, 6, argp_col)
            self.table.setItem(i, 7, true_anomaly_col)

        # Add buttons to enable or disable all satellites.
        display_all_button = PySide6.QtWidgets.QPushButton("Display All")
        hide_all_button = PySide6.QtWidgets.QPushButton("Hide All")

        display_all_button.clicked.connect(self.display_all_toggle)
        hide_all_button.clicked.connect(self.hide_all_toggle)

        self.layout.addWidget(self.table)
        self.layout.addWidget(display_all_button)
        self.layout.addWidget(hide_all_button)

        self.timer = PySide6.QtCore.QTimer(self)
        self.timer.timeout.connect(self.frame_update)
        self.timer.start(250)

    def satellite_toggled(self, name, checked):
        self.sim.satellite_display_flags[name] = checked

    def display_all_toggle(self):
        for checkbox in self.checkboxes:
            checkbox.setChecked(True)

    def hide_all_toggle(self):
        for checkbox in self.checkboxes:
            checkbox.setChecked(False)

    def frame_update(self):
        viewport = self.table.viewport()

        top_index = self.table.rowAt(0)
        bottom_index = self.table.rowAt(viewport.height())

        if top_index == -1:
            return
        if bottom_index == -1:
            bottom_index = self.table.rowCount() - 1

        items = list(self.sim.satellites.items())
        for i in range(top_index, bottom_index):
            name = items[i][0]
            satellite = items[i][1]

            position = self.sim.splines["positions"][name](self.sim.sim_time) * 1000
            velocity = self.sim.splines["velocities"][name](self.sim.sim_time) * 1000
            grav_param = satellite.orbit.grav_param

            sm_axis, eccentricity, raan, inclination, argp, true_anomaly = (
                conversions.state_2_classical(position, velocity, grav_param)
            )

            self.table.item(i, 2).setText(f"{sm_axis:.3f}")
            self.table.item(i, 3).setText(f"{eccentricity:.6f}")
            self.table.item(i, 4).setText(f"{raan:.6f}")
            self.table.item(i, 5).setText(f"{inclination:.6f}")
            self.table.item(i, 6).setText(f"{argp:.6f}")
            self.table.item(i, 7).setText(f"{true_anomaly:.6f}")


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
        self.final_sim_time = (final_global_time.julian_date - initial_global_time.julian_date) * 86400
        self.speed_factor = 100
        self.satellite_display_flags = {name: True for name in self.satellites.keys()}

        self.splines = {"positions" : {}, "velocities" : {}}

        for name, satellite in self.satellites.items():
            times = satellite.time_history
            for i in range(1, len(times)):
                if times[i] <= times[i - 1]:
                    times[i] = times[i - 1] + 1e-9
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
import numpy as np
import PySide6.QtWidgets
import PySide6.QtCore
import PySide6.QtGui

from ..astro import conversions

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

            # Compute and add all the orbital elements.
            position = self.sim.splines["positions"][name](self.sim.sim_time) * 1000
            velocity = self.sim.splines["velocities"][name](self.sim.sim_time) * 1000
            grav_param = satellite.orbit.grav_param

            sm_axis, eccentricity, raan, inclination, argp, true_anomaly = (
                conversions.state_2_classical(position, velocity, grav_param)
            )

            sm_axis_col = PySide6.QtWidgets.QTableWidgetItem(f"{sm_axis / 1000 :.3f}")
            eccentricity_col = PySide6.QtWidgets.QTableWidgetItem(f"{eccentricity:.3f}")
            raan_col = PySide6.QtWidgets.QTableWidgetItem(f"{np.rad2deg(raan):.3f}")
            inclination_col = PySide6.QtWidgets.QTableWidgetItem(f"{np.rad2deg(inclination):.3f}")
            argp_col = PySide6.QtWidgets.QTableWidgetItem(f"{np.rad2deg(argp):.3f}")
            true_anomaly_col = PySide6.QtWidgets.QTableWidgetItem(f"{np.rad2deg(true_anomaly):.3f}")

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

        # Timer used to update table values each frame.
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
        """
        Update the logged orbital elements of only the currently visible table entries to prevent lag.
        """

        # Fetch the indices of currently on-screen table rows.
        viewport = self.table.viewport()

        top_index = self.table.rowAt(0)
        bottom_index = self.table.rowAt(viewport.height())

        if top_index == -1:
            return
        if bottom_index == -1:
            bottom_index = self.table.rowCount() - 1

        # Update the table values for only those rows.
        items = list(self.sim.satellites.items())
        for i in range(top_index, bottom_index + 1):
            name = items[i][0]
            satellite = items[i][1]

            position = self.sim.splines["positions"][name](self.sim.sim_time) * 1000
            velocity = self.sim.splines["velocities"][name](self.sim.sim_time) * 1000
            grav_param = satellite.orbit.grav_param

            sm_axis, eccentricity, raan, inclination, argp, true_anomaly = (
                conversions.state_2_classical(position, velocity, grav_param)
            )

            self.table.item(i, 2).setText(f"{sm_axis / 1000 :.3f}")
            self.table.item(i, 3).setText(f"{eccentricity:.6f}")
            self.table.item(i, 4).setText(f"{np.rad2deg(raan):.3f}")
            self.table.item(i, 5).setText(f"{np.rad2deg(inclination):.3f}")
            self.table.item(i, 6).setText(f"{np.rad2deg(argp):.3f}")
            self.table.item(i, 7).setText(f"{np.rad2deg(true_anomaly):.3f}")
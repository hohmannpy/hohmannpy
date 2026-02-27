import numpy as np
import PySide6.QtWidgets
import PySide6.QtCore
import PySide6.QtGui

from ..astro import conversions


class PropertiesDocker(PySide6.QtWidgets.QDockWidget):
    """
    Docker which displays the state and classical elements of the selected RSO.
    """

    def __init__(self, sim):
        super().__init__()

        self.sim = sim

        self.setWindowTitle("RSO Properties")

        # Docket consists of a QWidget holding a QVBoxLayout which holds a series of headers and QFormLayouts to display
        # data.
        container = PySide6.QtWidgets.QWidget()
        self.setWidget(container)

        layout = PySide6.QtWidgets.QVBoxLayout(container)
        layout.setAlignment(PySide6.QtCore.Qt.AlignTop)

        self.labels = {}  # All data labels.
        self.labels["name"] = PySide6.QtWidgets.QLabel("No RSO Selected")
        self.labels["name"].setAlignment(PySide6.QtCore.Qt.AlignCenter)
        self.labels["name"].setStyleSheet("font-weight: bold;")
        self.labels["state"] = PySide6.QtWidgets.QLabel("State (ECI)")
        self.labels["state"].setAlignment(PySide6.QtCore.Qt.AlignCenter)
        self.labels["state"].setStyleSheet("font-weight: 300;")
        self.labels["classical_elements"] = PySide6.QtWidgets.QLabel("Classical Elements")
        self.labels["classical_elements"].setAlignment(PySide6.QtCore.Qt.AlignCenter)
        self.labels["classical_elements"].setStyleSheet("font-weight: 300;")

        # Setup ECI state display.
        state_form = PySide6.QtWidgets.QFormLayout(container)
        self.state_values = {
            "x_position": ["Position (km) :", PySide6.QtWidgets.QLabel("—")],
            "y_position": ["", PySide6.QtWidgets.QLabel("—")],
            "z_position": ["", PySide6.QtWidgets.QLabel("—")],
            "x_velocity": ["Velocity (km/s) :", PySide6.QtWidgets.QLabel("—")],
            "y_velocity": ["", PySide6.QtWidgets.QLabel("—")],
            "z_velocity": ["", PySide6.QtWidgets.QLabel("—")],
        }
        for value in self.state_values.values():
            value[1].setAlignment(PySide6.QtCore.Qt.AlignRight)
            state_form.addRow(value[0], value[1])

        # Setup classical orbital elements display.
        ce_form = PySide6.QtWidgets.QFormLayout(container)
        self.ce_values = {
            "sm_axis" : ["Semijax (km)", PySide6.QtWidgets.QLabel("—")],
            "eccentricity": ["Eccentricity", PySide6.QtWidgets.QLabel("—")],
            "raan": ["RAAN (deg)", PySide6.QtWidgets.QLabel("—")],
            "argp": ["ArgP (deg)", PySide6.QtWidgets.QLabel("—")],
            "inclination": ["Inclination (deg)", PySide6.QtWidgets.QLabel("—")],
            "true_anomaly": ["True Anomaly", PySide6.QtWidgets.QLabel("—")],
        }
        for value in self.ce_values.values():
            value[1].setAlignment(PySide6.QtCore.Qt.AlignRight)
            ce_form.addRow(value[0] + ":", value[1])

        # Order widgets are added determines how they appear top to bottom.
        layout.addWidget(self.labels["name"])
        layout.addWidget(self.labels["state"])
        layout.addLayout(state_form)
        layout.addWidget(self.labels["classical_elements"])
        layout.addLayout(ce_form)

        # Timer used to update table values each frame.
        self.timer = PySide6.QtCore.QTimer(self)
        self.timer.timeout.connect(self.frame_update)
        self.timer.start(250)

    def frame_update(self):
        # If no RSO is focus leave all values blank, otherwise compute them from the focused RSO's trajectory splines.
        if self.sim.focus is not None:
            position = self.sim.splines["positions"][self.sim.focus](self.sim.sim_time) * 1000
            velocity = self.sim.splines["velocities"][self.sim.focus](self.sim.sim_time) * 1000
            grav_param = self.sim.satellites[self.sim.focus].orbit.grav_param

            self.state_values["x_position"][1].setText(f"{position[0]:.3f}")
            self.state_values["y_position"][1].setText(f"{position[1]:.3f}")
            self.state_values["z_position"][1].setText(f"{position[2]:.3f}")
            self.state_values["x_velocity"][1].setText(f"{velocity[0]:.3f}")
            self.state_values["y_velocity"][1].setText(f"{velocity[1]:.3f}")
            self.state_values["z_velocity"][1].setText(f"{velocity[2]:.3f}")

            sm_axis, eccentricity, raan, inclination, argp, true_anomaly = (
                conversions.state_2_classical(position, velocity, grav_param)
            )

            self.ce_values["sm_axis"][1].setText(f"{sm_axis / 1000 :.3f}")
            self.ce_values["eccentricity"][1].setText(f"{eccentricity:.6f}")
            self.ce_values["raan"][1].setText(f"{np.rad2deg(raan):.3f}")
            self.ce_values["inclination"][1].setText(f"{np.rad2deg(inclination):.3f}")
            self.ce_values["argp"][1].setText(f"{np.rad2deg(argp):.3f}")
            self.ce_values["true_anomaly"][1].setText(f"{np.rad2deg(true_anomaly):.3f}")

            self.labels["name"].setText(self.sim.satellites[self.sim.focus].name)
        else:
            for value in self.state_values.values():
                value[1].setText("—")
            for value in self.ce_values.values():
                value[1].setText("—")
            self.labels["name"].setText("No RSO Selected")
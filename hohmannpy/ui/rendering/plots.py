import PySide6.QtWidgets
import PySide6.QtCharts
import PySide6.QtCore
import PySide6.QtGui
import scipy as sp
import numpy as np

from ...astro import logging


# TODO: (Post-Alpha) Add feature to close an individual plot.
class PlotsRenderer(PySide6.QtWidgets.QWidget):
    """
    Render engine for plots.
    """

    def __init__(self, sim):
        super().__init__()

        self.sim = sim
        self.dense_times = np.arange(
            0,
            self.sim.final_sim_time,
            10
        )
        self.num_plots = 0
        self.dialog = None

        scroll = PySide6.QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)

        grid_container = PySide6.QtWidgets.QWidget()
        self.grid = PySide6.QtWidgets.QGridLayout(grid_container)

        scroll.setWidget(grid_container)

        layout = PySide6.QtWidgets.QVBoxLayout(self)
        layout.addWidget(scroll)

        new_plot_button = PySide6.QtWidgets.QPushButton("Add Plot")
        close_all_button = PySide6.QtWidgets.QPushButton("Close All")
        button_layout = PySide6.QtWidgets.QHBoxLayout()
        new_plot_button.clicked.connect(self.open_new_plot)
        close_all_button.clicked.connect(self.close_all)

        button_layout.addWidget(new_plot_button)
        button_layout.addWidget(close_all_button)
        layout.addLayout(button_layout)

    def create_plot(self, satellite_id: str, var: np.ndarray, var_label: str):
        times = self.sim.satellites[satellite_id].time_history

        for i in range(1, times.shape[1]):
            if times[0, i] <= times[0, i - 1]:
                times[0, i] = times[0, i - 1] + 1e-9

        spline = PySide6.QtCharts.QSplineSeries()
        spline.append([
            PySide6.QtCore.QPointF(x, y)
            for x, y in zip(
                times.squeeze(),
                var.squeeze()
            )
        ])

        # --- Chart ---
        chart = PySide6.QtCharts.QChart()
        chart.addSeries(spline)
        chart.createDefaultAxes()
        chart.legend().hide()
        y_axis = chart.axes(PySide6.QtCore.Qt.Vertical)[0]
        x_axis = chart.axes(PySide6.QtCore.Qt.Horizontal)[0]
        y_axis.setLabelFormat("%.5g")
        x_axis.setTitleText("Time [s]")
        y_axis.setTitleText(var_label)

        # --- View ---
        plot = PySide6.QtCharts.QChartView(chart)
        plot.setRenderHint(PySide6.QtGui.QPainter.Antialiasing)
        plot.setMinimumSize(500, 250)

        row_index = self.num_plots % 2
        col_index = self.num_plots // 2
        self.grid.addWidget(plot, row_index, col_index)
        self.grid.setRowStretch(row_index, 1)
        self.grid.setColumnStretch(col_index, 1)

        self.num_plots += 1

    def open_new_plot(self):
        self.dialog = NewPlotDialog(self, self.sim)
        self.dialog.setAttribute(PySide6.QtCore.Qt.WA_DeleteOnClose)
        self.dialog.show()

    def close_all(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.num_plots = 0


class NewPlotDialog(PySide6.QtWidgets.QDialog):
    """
    Dialog which pops up when you create a new graph.
    """

    def __init__(self, parent_window, sim):
        super().__init__()
        self.sim = sim
        self.parent_window = parent_window

        self.setWindowTitle("New Plot")
        self.resize(300, 100)

        layout = PySide6.QtWidgets.QVBoxLayout(self)  # Wrapper around the QDialog needed to actually display.

        # Create satellite and plot data selectors.
        combo_layout = PySide6.QtWidgets.QHBoxLayout()

        dd1 = PySide6.QtWidgets.QComboBox()
        dd2 = PySide6.QtWidgets.QComboBox()
        vars = {}
        for name, satellite in self.sim.satellites.items():
            dd1.addItem(name)
            for logger in satellite.loggers:
                for label in logger.labels:
                    index = logger.labels.index(label)

                    if isinstance(logger, logging.StateLogger):
                        if index == 0:
                            continue
                        elif 1 <= index < 4:
                            row = index - 1
                            index = 1
                        elif 4 <= index < 7:
                            row = index - 4
                            index = 2
                    else:
                        row = 0

                    var = getattr(logger, logger.attributes[index])
                    var = var[row, :]

                    if label[-5:] == "[rad]":
                        label = label[:-5] + "[deg]"
                        var = np.rad2deg(var)
                    if label not in (vars.keys()):
                        vars[label] = {}
                    vars[label][name] = var

                    dd2.addItem(label, vars[label])

        combo_layout.addWidget(dd1)
        combo_layout.addWidget(dd2)
        self.dropdowns = {
            "dd1": dd1,
            "dd2": dd2,
        }

        # Add confirm button and finalize layout.
        confirm_button = PySide6.QtWidgets.QPushButton("Create")
        confirm_layout = PySide6.QtWidgets.QHBoxLayout()
        confirm_layout.addStretch()
        confirm_layout.addWidget(confirm_button)

        layout.addLayout(combo_layout)
        layout.addLayout(confirm_layout)

        confirm_button.clicked.connect(self.create_plot)

    def create_plot(self):
        satellite_id = self.dropdowns["dd1"].currentText()
        var_label = self.dropdowns["dd2"].currentText()
        var = self.dropdowns["dd2"].currentData()[satellite_id]
        self.parent_window.create_plot(satellite_id, var, var_label)
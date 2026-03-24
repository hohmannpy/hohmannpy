import PySide6.QtWidgets
import PySide6.QtCharts
import PySide6.QtCore
import PySide6.QtGui
import numpy as np

from ...astro import logging, conversions


class PlotsRenderer(PySide6.QtWidgets.QWidget):
    """
    Render engine for plots.
    """

    def __init__(self, sim, tabs):
        super().__init__()

        self.sim = sim
        self.tabs = tabs
        self.num_plots = 0
        self.dialog = None  # Dialog widget that pops up when creating a new plot.

        # The goal of this renderer is to render as many plots as the user wants in a 2xN grid. This necessitates some
        # finagling with the Qt stack. The resulting layout looks like this:
        #   Plot Renderer -> QVBoxLayout -> QScrollArea -> QGridLayout -> QWidget -> QCharts
        #                                -> QHBoxLayout -> QButton(s)
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

        # QTimer render loop which updates all plots based on the current horizon.
        self.timer = PySide6.QtCore.QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(33)

    def create_plot(self, satellite_id: str, var: np.ndarray, var_label: str):
        """
        Add a new plot.
        """

        times = self.sim.satellites[satellite_id].time_history  # All plots have time ax the x-axis.

        # Generate the data to plot. Note that as a result of impulsive burns there may be multiple data entries at the
        # same time point. Detect this and slightly increment them or splining will throw an error. Splining is handled
        # automatically via passing the data points into a QSplineSeries.
        for i in range(1, times.shape[1]):
            if times[0, i] <= times[0, i - 1]:
                times[0, i] = times[0, i - 1] + 1e-9

        spline = PySide6.QtCharts.QLineSeries()
        spline.append([
            PySide6.QtCore.QPointF(x, y)
            for x, y in zip(
                times.squeeze(),
                var.squeeze()
            )
        ])

        # Create the QChart and then inset it into a QWidget.
        chart = PySide6.QtCharts.QChart()
        chart.addSeries(spline)
        chart.createDefaultAxes()
        chart.legend().hide()
        y_axis = chart.axes(PySide6.QtCore.Qt.Vertical)[0]
        x_axis = chart.axes(PySide6.QtCore.Qt.Horizontal)[0]
        y_axis.setLabelFormat("%.5g")
        x_axis.setTitleText("Time [s]")
        y_axis.setTitleText(f"{satellite_id}: {var_label}")
        chart.setProperty("satellite_id", satellite_id)

        plot = PySide6.QtCharts.QChartView(chart)
        plot.setRenderHint(PySide6.QtGui.QPainter.Antialiasing)  # Need or lines are super jagged.
        plot.setMinimumSize(500, 250)

        # Plots a rendered in a 2xN grid, moving first down columns. This logic ensures plots are placed correctly.
        row_index = self.num_plots % 2
        col_index = self.num_plots // 2
        self.grid.addWidget(plot, row_index, col_index)
        self.grid.setRowStretch(row_index, 1)
        self.grid.setColumnStretch(col_index, 1)

        self.num_plots += 1
        self.animate()  # Force a frame update so plot renders automatically.

    def open_new_plot(self):
        """
        Opens the dialog to create a new plot.
        """

        self.dialog = NewPlotDialog(self, self.sim)
        self.dialog.setAttribute(PySide6.QtCore.Qt.WA_DeleteOnClose)  # Need or dialog will remain on heap upon close.
        self.dialog.show()

    def close_all(self):
        """
        Close all open plots
        """

        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.num_plots = 0

    def animate(self):
        """
        One frame of the animation loop.
        """

        # Don't render if this tab isn't visible.
        if self.tabs.currentIndex() != 2:
            return

        # For each plot, clamp the time axis to only display data points in the current horizon. No special splining
        # or buffer logic is needed here because QCharts handle this all automatically via the setRange() method owned
        # by their axes.
        for i in range(self.grid.count()):
            chart = self.grid.itemAt(i).widget().chart()

            match self.sim.horizon_display_mode:
                case "period":
                    name = chart.property("satellite_id")
                    position = self.sim.splines["positions"][name](self.sim.sim_time) * 1000
                    velocity = self.sim.splines["velocities"][name](self.sim.sim_time) * 1000
                    grav_param = self.sim.satellites[name].orbit.grav_param

                    sm_axis, _, _, _, _, _ = conversions.state_2_classical(position, velocity, grav_param)
                    period = 2 * np.pi * np.sqrt(sm_axis ** 3 / grav_param)

                    horizon = period
                    lower_time = self.sim.sim_time - horizon
                    upper_time = self.sim.sim_time + horizon
                case "full":
                    lower_time = 0
                    upper_time = self.sim.final_sim_time
                case "past":
                    lower_time = 0
                    upper_time = self.sim.sim_time
                case "hour":
                    horizon = 60 * 60
                    lower_time = self.sim.sim_time - horizon
                    upper_time = self.sim.sim_time + horizon
                case "half_day":
                    horizon = 60 * 60 * 12
                    lower_time = self.sim.sim_time - horizon
                    upper_time = self.sim.sim_time + horizon
                case "day":
                    horizon = 60 * 60 * 24
                    lower_time = self.sim.sim_time - horizon
                    upper_time = self.sim.sim_time + horizon
                case "custom":
                    horizon = self.sim.custom_horizon
                    lower_time = self.sim.sim_time - horizon
                    upper_time = self.sim.sim_time + horizon

            if lower_time < 0:
                lower_time = 0

            axis_x = chart.axes(PySide6.QtCore.Qt.Horizontal)[0]
            axis_x.setRange(lower_time, upper_time)


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

        # Create satellite and plot data selectors. This consists of two QComboBox-es (drop down menus). The first
        # allows the user to select which satellite to plot data from and the second which data to plot. The satellites
        # consist of all satellites propagated for the Mission. The available data consists of all the data stored by
        # Loggers stored by the mission.
        combo_layout = PySide6.QtWidgets.QHBoxLayout()
        dd1 = PySide6.QtWidgets.QComboBox()
        dd2 = PySide6.QtWidgets.QComboBox()

        vars = {}
        for name, satellite in self.sim.satellites.items():
            dd1.addItem(name)  # Append a name to the first dropdown.

            # The second drop down is more complicated. Each satellite has its own Logger instance, but each of those
            # Loggers stores the same type of data. The way a QComboBox works, it has a display name as well as stored
            # reference to the data to return when a given option is selected. We only want to display each data type
            # option, for example "Eccentricity", but when the user selects that option it needs to then store a
            # reference to the logged "Eccentricity" of whatever satellite is currently selected in the first dropdown.
            # To handle this, we store all data in a dict vars. Each item in vars has a key corresponding to the name
            # of the data type and a value corresponding to another dict. This dict has as keys the name of each
            # satellite and as values the data corresponding to the other dict's key.
            for logger in satellite.loggers:
                for label in logger.labels:
                    index = logger.labels.index(label)

                    # This is a special case as all other Loggers only store data as (1, N) vectors. StateLogger stores
                    # position and velocity specially as (3, N) vectors so need to split that up into (1, N) arrays so
                    # that the user can plot "x-Position", "y-Position", etc;
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

                    var = getattr(logger, logger.attributes[index])  # See Logger class for details on why this works.
                    var = var[row, :]

                    # If the data was logged in radians, convert to degrees.
                    if label[-5:] == "[rad]":
                        label = label[:-5] + "[deg]"
                        var = np.rad2deg(var)

                    if label not in (vars.keys()):
                        vars[label] = {}
                    vars[label][name] = var

        for label, var in vars.items():
            dd2.addItem(label, var)

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
        """
        Create a plot using the currently selected dropdown options.
        """

        satellite_id = self.dropdowns["dd1"].currentText()
        var_label = self.dropdowns["dd2"].currentText()
        var = self.dropdowns["dd2"].currentData()[satellite_id]
        self.parent_window.create_plot(satellite_id, var, var_label)

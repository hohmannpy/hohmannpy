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
#   - (Post-alpha) Hovering over satellite displays name, clicking focuses it.
#   - (Post-alpha) FPS tracking on all tabs.
#   - (Post-alpha) Add an option to display the ECI basis in the orbit viewer and latitude and longitude lines in the
#       groundtrack viewer. Also add a visible Sun on screen.
#   - (Post-alpha) The groundtrack renderer is still a little laggy. Work on ways to improve this.
#   - (Post-alpha) Laggy when a long trajectory is rendered, need to make step size more dynamic.
#   - (Post-alpha) Redo variables and docs to use privatization.
#   - (Post-beta) Fix spacecraft not being shadowed when in eclipse by turning off lighting.
class Viewer(PySide6.QtWidgets.QMainWindow):
    """
    This is the main QtWidget of the HohmannPy Viewer application. This can be used to visually display a 3D rendering
    of a :class:`~hohmannpy.astro.Mission` once it has run. Additional data visualization functionality is included.

    It holds all the additional Qt functionality needed for the application to function.

    Parameters
    ----------
    sim : :class:`hohmannpy.viewer.ViewerManager`
        Handles all the time keeping and data handling for the HohmannPy simulation displayed by this HohmannPy Viewer.

    Attributes
    ----------
    sim : :class:`hohmannpy.viewer.ViewerManager`
        Handles all the time keeping and data handling for the HohmannPy simulation displayed by this HohmannPy Viewer.
    status_labels : dict[str, :class:`~PySide6.QtWidgets.QLabel`
        Dictionary containing all the information to display in the status bar at the bottom of the application.
    elements : dict[str, :class:`~PySide6.QtWidgets.QWidget`]
        Reference to all Qt widgets displayed by this application.
    """

    def __init__(self, sim):
        super().__init__()
        self.sim = sim

        # Set up the main window.
        self.setWindowTitle("HohmannPy Viewer")
        self.resize(1280, 720)

        # Tabs are the primary view ports and each one provides a different way to view the Mission. Currently, three
        # tabs are included:
        #   1) orbit_viewer: 3D scene of the spacecraft in orbit.
        #   2) gt_viewer: 2D scene of the spacecraft's groundtracks
        #   3) plot_viewer: Plotting functionality for all the different data logged during the Mission.
        # Since each tab takes a lot of CPU and GPU power, all but the currently displayed tab pause their rendering
        # loops when they aren't active.
        tabs = PySide6.QtWidgets.QTabWidget()
        self.setCentralWidget(tabs)

        orbit_viewer = rendering.OrbitRenderer(self.sim, tabs)
        tabs.addTab(orbit_viewer, "Orbit")
        if sim.include_rotation:
            attitude_viewer = rendering.ProximityRenderer(self.sim, tabs)
            tabs.addTab(attitude_viewer, "Proximity")
        else:
            attitude_viewer = None
        gt_viewer = rendering.GroundtrackRenderer(self.sim, tabs)
        tabs.addTab(gt_viewer, "Groundtrack")
        plot_viewer = rendering.PlotsRenderer(self.sim, tabs)
        tabs.addTab(plot_viewer, "Data Visualizer")

        # SPACE pauses the sim using Qt signal/slot functionality. Linkage to orbit_viewer is arbitrary, it could have
        # been linked to any widget tied to this class.
        orbit_viewer.space_pressed.connect(self.on_space_press)

        # The toolbar sits above the tabs and consists of a series of buttons and hotkeys that provided functionality
        # needed in all tabs. Since clicking anything in the toolbar should run some logic all items are linked to some
        # Qt signal/slot functionality.
        toolbar = toolbars.ToolBar()
        self.addToolBar(toolbar)
        toolbar.setMovable(False)
        toolbar.setFloatable(False)

        toolbar.horizon_display.mode_changed.connect(self.set_horizon)
        toolbar.horizon_display.custom_horizon.connect(self.set_custom_horizon)  # Submenu action inside above button.
        toolbar.rso_table.rso_table.connect(self.open_rso_table)
        toolbar.orbit_display.mode_changed.connect(self.set_orbit)
        toolbar.sim_speed.mode_changed.connect(self.set_sim_speed)
        toolbar.play_pause.mode_changed.connect(self.set_play_pause)
        toolbar.reset.reset.connect(self.reset_sim)
        toolbar.focus_previous.focus.connect(self.set_focus_previous)
        toolbar.focus_earth.focus.connect(self.set_focus_earth)
        toolbar.focus_next.focus.connect(self.set_focus_next)

        # Docks are additional information GUIs which can be moved around the application. For now, this just consists
        # of a real time data display table for whatever RSO the user is currently looking at.
        dock = dockers.PropertiesDocker(self.sim)
        self.addDockWidget(PySide6.QtCore.Qt.RightDockWidgetArea, dock)
        self.resizeDocks(
            [dock],
            [200],
            PySide6.QtCore.Qt.Horizontal
        )  # This call prevents the dock's size from jittering.

        # The status bar is at the bottom of the application and displays persistent information and dynamic messages
        # which pop up whenever something regarding the sim changes (think of it like a message log).
        status = self.statusBar()
        self.status_labels = {"sim_time": PySide6.QtWidgets.QLabel("T+00Y:000D:00:H:00M:00S")}
        status.addPermanentWidget(self.status_labels["sim_time"])

        # Store references to all the child widgets of the application in a dict.
        self.elements = {
            "orbit" : orbit_viewer,
            "attitude" : attitude_viewer,
            "groundtrack" : gt_viewer,
            "plots" : plot_viewer,
            "toolbar" : toolbar,
            "tabs" : tabs,
            "dock" : dock,
            "status" : status,
            "rso_table" : None,  # Only created when the "RSO" toolbar button is pressed.
        }

        # Keyboard shortcuts which mirror functionality provided by toolbar buttons.
        self.shortcuts = {}
        self.shortcuts["space"] = PySide6.QtGui.QShortcut(PySide6.QtGui.QKeySequence(PySide6.QtCore.Qt.Key_Space), self)
        self.shortcuts["space"].setContext(PySide6.QtCore.Qt.WidgetWithChildrenShortcut)
        self.shortcuts["space"].activated.connect(self.on_space_press)

        self.shortcuts["lb"] = PySide6.QtGui.QShortcut(
            PySide6.QtGui.QKeySequence(PySide6.QtCore.Qt.Key_BracketLeft), self
        )
        self.shortcuts["lb"].setContext(PySide6.QtCore.Qt.WidgetWithChildrenShortcut)
        self.shortcuts["lb"].activated.connect(self.on_lb_press)

        self.shortcuts["rb"] = PySide6.QtGui.QShortcut(
            PySide6.QtGui.QKeySequence(PySide6.QtCore.Qt.Key_BracketRight), self
        )
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

        # Add a menu which sits above the toolbar. Currently, it just allows you to reopen dockers if you accidentally
        # close them.
        menu = self.menuBar().addMenu("Menu")
        docker_menu = menu.addMenu("Dockers")
        docker_menu.addAction(self.elements["dock"].toggleViewAction())  # This is what lets you reopen dockers.

    # These are the slot component of all the signals linked in the __init__().
    @PySide6.QtCore.Slot(str)
    def set_horizon(self, signal):
        """
        Sets how far into the past/future to display data.
        """

        self.sim.horizon_display_mode = signal

    @PySide6.QtCore.Slot(str)
    def set_orbit(self, signal):
        """
        Sets whether to display just the RSO, the orbit and the RSO, or just the orbit.
        """

        self.sim.orbit_display_mode = signal

    @PySide6.QtCore.Slot(float)
    def set_custom_horizon(self, signal):
        """
        The "horizon" toolbar button allows you to pick from several preset horizons or create your own.
        """

        self.sim.custom_horizon = signal

    @PySide6.QtCore.Slot(float)
    def set_sim_speed(self, signal):
        """
        Adjust how much faster than real time the sim runs at.
        """

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
        """
        Play/pause the sim.
        """

        if signal:
            self.sim.old_speed_factor = self.sim.speed_factor
            self.sim.speed_factor = 0
        else:
            self.sim.speed_factor = self.sim.old_speed_factor

    def open_rso_table(self):
        """
        Opens the RSO table. This is an additional application window application via the "RSO" button on the toolbar
        which allows you to view information on all currently active RSOs in the application in a single list.
        """

        # Some additional logic is included here because Qt puts any widgets instantiated from a MainWindow object
        # permanently onto the heap. As a result, they aren't flushed during Python memory collection and instead
        # persist in the background. The table has to update a ton of data on each satellite every frame and so having
        # it run in the background is laggy. To account for this, "WA_DeleteOnClose", is enabled which force deletes it
        # when closed.
        if self.elements["rso_table"] is None:
            self.elements["rso_table"] = tables.RSOTable(self.sim)  # Give object a reference to this widget.
            self.elements["rso_table"].setAttribute(PySide6.QtCore.Qt.WA_DeleteOnClose)
            self.elements["rso_table"].destroyed.connect(self.on_rso_table_closed)  # Delete reference when closed.
            self.elements["rso_table"].show()

        # This prevents two tables from being open at once. When the table opens it is stored in the elements attribute
        # of this object. If there is data in that attribute (i.e. it isn't set to None), when the user attempts to open
        # the table from the toolbar it instead just brings the prexisting table back into focus.
        else:
            self.elements["rso_table"].raise_()
            self.elements["rso_table"].activateWindow()

    def on_rso_table_closed(self):
        """
        Remove reference to the table once it is closed.
        """

        self.elements["rso_table"] = None

    def reset_sim(self):
        """
        Reset the sim timer back to the start of the Mission.
        """

        self.sim.initial_local_time = time.perf_counter()
        self.sim.sim_time = 0
        self.statusBar().showMessage("Resetting mission...", 3000)

    def set_focus_previous(self):
        """
        Display data in the docker (and focus the camera on if the orbit tab is open), the previous satellite in the
        sim's satellite list (with respect to the currently selected satellite).

        Satellites are stored in the order in which they were added to the Mission.
        """

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
        """
        Focus the Earth, this is equivalent to displaying no data in the docker and focusing on the Earth in the orbit
        tab.
        """

        self.sim.focus = None

    def set_focus_next(self):
        """
        Display data in the docker (and focus the camera on if the orbit tab is open), the next satellite in the
        sim's satellite list (with respect to the currently selected satellite).

        Satellites are stored in the order in which they were added to the Mission.
        """

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

    # These next methods just connect hotkeys to their respective toolbar buttons.
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


class ViewerManager:
    """
    Companion class to :class:`~hohmannpy.viewer.Viewer`. While the ``Viewer`` is responsible for data visualization and
    rendering, this class is the one which actually stores and processes data from a :class:`~hohmannpy.astro.Mission`
    and keeps an internal simulation clock.

    Parameters
    ----------
    satellites : dict[str, :class:`~hohamnpy.astro.Satellite`]
        All RSOs simulated by the ``Mission``. All data needed by the ``Viewer`` is stored in these objects.
    initial_global_time : :class:`~hohmannpy.astro.Time`
        Real-world date and time at which the simulation started.
    final_global_time : :class:`~hohmannpy.astro.Time`
        Real-world date and time at which the simulation ended.

    Attributes
    ----------
    satellites : dict[str, :class:`~hohamnpy.astro.Satellite`]
        All RSOs simulated by the ``Mission``. All data needed by the ``Viewer`` is stored in these objects.
    gui : :class:`~hohmannpy.viewer.Viewer`
        Reference to the companion ``Viewer``.
    local_time : float
        The current time as reported by ``time.perf_counter()``.
    initial_local_time : ``time.perf_counter()``
        The ``local_time`` at which the ``sim_time`` was last set to zero.
    sim_time : float
        How far into simulating the ``Mission`` the ``Viewer`` is. This may diverge from the ``local_time`` because it
        may be accelerated  (ex. 1 simulation hour/1 real second).
    final_sim_time : float
        How many seconds elapse between the initial and final global times the simulation ran for.
    initial_global_time : :class:`~hohmannpy.astro.Time`
        Real-world date and time at which the simulation started. Referenced to properly orient the Earth's GMST when it
        is rendered.
    speed_factor : float
        How many simulation seconds should pass for each real second.
    old_speed_factor : float
        When the simulation is paused, the ``speed_factor`` is set to zero. However, its value before pause is stored
        using this variable so that it can be set back to that value when the user unpauses.
    satellite_display_flags : dict[str, bool]
        Corresponds to the ``satellites`` viewable in the ``Viewer``. ``True`` means include the satellite in the scene
        and ``False`` means disable it.
    focus : str
        Which RSO the camera should center on in the orbit view. If set to ``None`` the camera instead centers on the
        Earth.
    orbit_display_mode : str
        Sets whether to display just the RSO, the orbit and the RSO, or just the orbit.
    horizon_display_mode : str
        Sets how far into the past and future to display data (in seconds). If set to "custom" than it uses the
        user-entered time stored in ``custom_horizon``.
    splines : dict[str, dict[str, name]]
        Position and velocity data for each satellite after propagation is not smooth, so to account for this the data
        is splined. Whenever a trajectory is needed for rendering purposes a densified version of this data is used.
    timer : :class:`~PySide6.QtCore.QTimer`
        Qt wrapper that calls the ``frame_update`` method on each frame automatically.
    """

    def __init__(
            self,
            satellites,
            initial_global_time,
            final_global_time,
            step_size,
            include_rotation
    ):
        self.satellites = satellites
        self.include_rotation = include_rotation

        self.gui = None

        self.initial_global_time = initial_global_time
        self.local_time = time.perf_counter()
        self.initial_local_time = self.local_time
        self.sim_time = 0
        self.final_sim_time = (final_global_time.julian_date - initial_global_time.julian_date) * 86400
        self.speed_factor = 100
        self.old_speed_factor = self.speed_factor
        self.step_size = step_size

        self.satellite_display_flags = {name: True for name in self.satellites.keys()}

        self.focus = None
        self.orbit_display_mode = "both"
        self.horizon_display_mode = "period"
        self.custom_horizon: int = 24 * 3600  # Defaults to one day.

        self.splines = {"positions" : {}, "velocities" : {}, "attitudes" : {}}

        # Generate the data splines. Note that as a result of impulsive burns there may be multiple data entries at the
        # same time point. Detect this and slightly increment them or splining will throw an error.
        for name, satellite in self.satellites.items():
            times = satellite.time_history
            positions = satellite.position_history.T / 1000
            velocities = satellite.velocity_history.T / 1000

            for i in range(1, times.shape[1]):
                if times[0, i] <= times[0, i - 1]:
                    times[0, i] = times[0, i - 1] + 1e-9
            times = times.squeeze()

            self.splines["positions"][name] = (
                sp.interpolate.make_interp_spline(
                    times,
                    positions,
                    k=3
                )
            )

            if self.include_rotation:
                attitudes = satellite.quaternion_history.T
                self.splines["attitudes"][name] = (
                    sp.interpolate.make_interp_spline(
                        times,
                        attitudes,
                        k=3
                    )
                )

            if len([x for x in satellite._events if x[1] == "impulsive"]) > 0:
                jump_times = [x[0] for x in satellite._events if x[1] == "impulsive"]
                start_time = times[0]
                splines = []

                for jump_time in jump_times:
                    time_chunk = times[(times >= start_time) & (times < jump_time)]
                    vel_chunk = velocities[(times >= start_time) & (times < jump_time), :]
                    splines.append(
                        sp.interpolate.make_interp_spline(
                            time_chunk,
                            vel_chunk,
                            k=1
                        )
                    )
                    start_time = jump_time

                time_chunk = times[(times >= start_time)]
                vel_chunk = velocities[(times >= start_time), :]
                splines.append(
                    sp.interpolate.make_interp_spline(
                        time_chunk,
                        vel_chunk,
                        k=1
                    )
                )

                def full_spline(time):
                    for j in range(len(jump_times)):
                        if time < jump_times[j]:
                            return splines[j](time)
                    else:
                        return splines[-1](time)

                self.splines["velocities"][name] = full_spline

            else:
                self.splines["velocities"][name] = (
                    sp.interpolate.make_interp_spline(
                        times,
                        velocities,
                        k=3
                    )
                )

        # Set up the timer.
        self.timer = PySide6.QtCore.QTimer()
        self.timer.timeout.connect(self.frame_update)

    def sim_clock(self):
        """
        Increment the sim clock based on how much real time has passed since it was last called.
        """

        old_local_time = self.local_time
        self.local_time = time.perf_counter()
        self.sim_time += (self.local_time - old_local_time) * self.speed_factor

        # If the end of the simulation has been reached, reset.
        if self.sim_time > self.final_sim_time:
            self.initial_local_time = time.perf_counter()
            self.sim_time = 0
            self.gui.statusBar().showMessage("End of mission reached, resetting...", 3000)

    def frame_update(self):
        """
        Functionality to run each frame.

        NOTE: Each tab in the HohmannPy Viewer has an internal rendering loop which is called separate from this.
        """

        # Increment sim time based on how much should have passeed since the last frame.
        self.sim_clock()

        # Display the current in-sim UT1 time and Gregorian date in the Viewer's status bar.
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
        """
        Only externally facing method in all of :mod:`~hohmannpy.viewer` package. Call this to launch the HohmannPy
        Viewer.
        """

        app = PySide6.QtWidgets.QApplication(sys.argv)

        icon_path = importlib.resources.files("hohmannpy.resources").joinpath("gfx/app_icon.png")
        app.setWindowIcon(PySide6.QtGui.QIcon(str(icon_path)))  # Add an app icon.

        self.gui = Viewer(self)
        self.gui.show()

        self.timer.start(17)  # Update the Viewer at approximately 60 FPS.
        app.exec()

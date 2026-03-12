from __future__ import annotations

import numpy as np


class Time:
    r"""
    Stores a date and time in UT1 and allows for rapid conversion to other time schemes (including Julian dates and
    Greenwich mean-sidereal time).

    Takes in a date (MM/DD/YYYY) and time (HH:MM:SS.S) and via the use of @property automatically converts to the
    corresponding Julian date and Greenwich mean-sidereal time (GMST). The input time should be in UT1 but technically
    UTC+0 may also be used with approximately 1 s loss in accuracy.

    Parameters
    ----------
    date: str
        Current Gregorian date (MM/DD/YYYY).
    time: str
        Current UT1 time (HH:MM:SS.S).

    Notes
    -----
    The Julian date and GMST are calculated from Algorithms 14 and 15 respectively in Vallado [1]_.

    .. [1] Vallado, D. A., Fundamentals of Astrodynamics and Applications, 3rd ed., Microcosm Press/Springer, 2007.
    """

    def __init__(self, date: str, time: str):
        # Error checking to make sure date and time are entered correctly.
        if (not date[2] == '/') or (not date[5] == '/'):
            raise ValueError('Invalid time, please enter the date in MM/DD/YYYY format.')
        if (not time[2] == ':') or (not time[5] == ':'):
            raise ValueError('Invalid time, please enter the date in HH:MM:SS.S format.')

        self._date = date
        self._time = time
        self._julian_date = None
        self._gmst = None

    @property
    def julian_date(self):
        """
        Julian date computed from the current Gregorian date and UTD1 time. Only valid for dates between March 1, 1900
        and February 28, 2100.
        """

        month = int(self._date[:2])
        day = int(self._date[3:5])
        year = int(self._date[6:])

        hours = int(self._time[:2])
        minutes = int(self._time[3:5])
        seconds = float(self._time[6:])

        self._julian_date = (
                367 * year - int(7 * (year + int((month + 9) / 12)) / 4) + int(275 * month / 9) + day + 1721013.5
                    + ((seconds / 60 + minutes) / 60 + hours) / 24
        )

        return self._julian_date

    @property
    def gmst(self):
        """
        GMST computed from the current Julian date.
        """

        centuries = (self.julian_date - 2451545) / 36525  # Julian centuries elapsed since J2000.0.

        self._gmst = (
                67310.54841 + (876600 * 3600 + 8640184.812866) * centuries
                    + 0.0093104 * centuries ** 2
                    - 6.2e-6 * centuries ** 3
        ) % 86400 # GMST in seconds wrapped to [0, 86400).
        self._gmst = np.deg2rad(self._gmst / 240) # Convert to degrees and then radians.

        return self._gmst

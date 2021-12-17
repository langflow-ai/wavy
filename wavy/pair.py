import numpy as np
import pandas as pd


class TimePair:

    DIMS = ("assets", "timesteps", "channels")

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        x_equals = self.x.equals(other.x)
        y_equals = self.y.equals(other.y)
        return x_equals and y_equals

    def __repr__(self):
        return f"<TimePair, lookback {self.lookback}, horizon {self.horizon}>"

    @property
    def lookback(self):
        """
        TimePair lookback.
        """
        return len(self.x)

    @property
    def horizon(self):
        """
        TimePair horizon.
        """
        return len(self.y)

    @property
    def start(self):
        """
        TimePair start.

        Example:

        >>> panelside.start
        Timestamp('2005-12-27 00:00:00')
        """
        return self.x.start

    @property
    def end(self):
        """
        TimePair end.

        Example:

        >>> panelside.end
        Timestamp('2005-12-30 00:00:00')
        """
        return self.y.end

    @property
    def shape(self):
        """
        TimePair shape.

        Example:

        >>> panelside.shape
           assets  timesteps  channels
        x       2          2         2
        y       2          2         2
        """
        return pd.DataFrame([self.x.tensor.shape, self.y.tensor.shape], index=["x", "y"], columns=self.DIMS)

    def filter(self, assets=None, channels=None):
        # ? Will assets and channels be the same on x and y
        x = self.x.filter(assets=assets, channels=channels)
        y = self.y.filter(assets=assets, channels=channels)
        return TimePair(x, y)

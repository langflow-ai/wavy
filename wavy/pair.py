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
        return len(self.x)

    @property
    def horizon(self):
        return len(self.y)

    @property
    def start(self):
        return self.x.start

    @property
    def end(self):
        return self.y.end

    @property
    def shape(self):
        return pd.DataFrame([self.x.numpy().shape, self.y.numpy().shape], index=["x", "y"], columns=self.DIMS)

    def filter(self, assets=None, channels=None):
        x = self.x.filter(assets=assets, channels=channels)
        y = self.y.filter(assets=assets, channels=channels)
        return TimePair(x, y)

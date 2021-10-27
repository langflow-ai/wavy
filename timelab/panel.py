from itertools import compress
from typing import Iterable

import numpy as np
import pandas as pd

from .block import TimeBlock
from .pair import TimePair
from .side import PanelSide


def from_pairs(pairs):
    if len(pairs) == 0:
        raise ValueError("Cannot build TimePanel from empty list")
    blocks = [(pair.x, pair.y) for pair in pairs]
    x = PanelSide([block[0] for block in blocks])
    y = PanelSide([block[1] for block in blocks])
    return TimePanel(x, y)


def from_xy_data(x, y, lookback, horizon, gap=0):

    x_timesteps = len(x.index)

    if x_timesteps - lookback - horizon - gap <= -1:
        raise ValueError("Not enough timesteps to build")

    end = x_timesteps - horizon - gap + 1

    # Convert to blocks
    x = TimeBlock(x)
    y = TimeBlock(y)

    indexes = np.arange(lookback, end)
    xblocks, yblocks = [], []

    for i in indexes:
        xblocks.append(x.iloc[i - lookback : i])
        yblocks.append(y.iloc[i + gap : i + gap + horizon])
    return TimePanel(PanelSide(xblocks), PanelSide(yblocks))


def from_data(df, lookback, horizon, gap=0, x_assets=None, y_assets=None, x_channels=None, y_channels=None, assets=None, channels=None):

    if assets:
        x_assets, y_assets = assets, assets
    if channels:
        x_channels, y_channels = channels, channels


    df = TimeBlock(df)

    if df.T.index.nlevels == 1:
        df = df.add_level('asset')

    xdata = df.filter(x_assets, x_channels)
    ydata = df.filter(y_assets, y_channels)
    return from_xy_data(xdata, ydata, lookback, horizon, gap)


class TimePanel:

    DIMS = ("size", "assets", "timesteps", "channels")

    def __init__(self, x, y):
        self._x, self._y = x, y
        self.train_size, self.val_size, self.test_size = None, None, None

        # freq checks
        # if x.isnull().values.any() or y.isnull().values.any():
        #     warnings.warn("Data contains null values.")

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @x.setter
    def x(self, value):
        if not isinstance(value, PanelSide):
            print(type(value))
            raise ValueError(f"'x' must be of type PanelSide, it is {type(value)}")
        if len(value) != len(self.x):
            raise ValueError("'x' must keep the same length")
        if len({len(block) for block in value.blocks}) != 1:
            raise ValueError("'x' blocks must have the same length")
        self._x = value

    @y.setter
    def y(self, value):
        if not isinstance(value, PanelSide):
            raise ValueError("'y' must be of type PanelSide")
        if len(value) != len(self.y):
            raise ValueError("'y' must keep the same length")
        if len({len(block) for block in value.blocks}) != 1:
            raise ValueError("'y' blocks must have the same length")
        self._y = value

    @property
    def pairs(self):
        return [TimePair(x, y) for x, y in zip(self.x.blocks, self.y.blocks)]

    @property
    def lookback(self):
        return len(self.x.first)

    @property
    def horizon(self):
        return len(self.y.first)

    @property
    def shape(self):
        return pd.DataFrame([self.x.shape, self.y.shape], index=["x", "y"], columns=self.DIMS)

    @property
    def start(self):
        return self.x.start

    @property
    def end(self):
        return self.y.end

    @property
    def index(self):
        return sorted(list(set(self.x.index + self.y.index)))

    def apply(self, func, axis):
        x = self.x.apply(func=func, axis=axis)
        y = self.y.apply(func=func, axis=axis)
        return TimePanel(x, y)

    def dropna(self, x=True, y=True):
        x_nan = self.x.findna() if x else []
        y_nan = self.y.findna() if y else []
        nan_values = set(x_nan + y_nan)
        idx = {i for i in range(len(self)) if i not in nan_values}
        if not idx:
            raise ValueError("'dropna' would create empty TimePanel")
        return self[idx]

    def __repr__(self):
        summary = pd.Series(
            {
                "size": self.__len__(),
                "lookback": self.lookback,
                "horizon": self.horizon,
                # "gap": self.gap,
                "num_xassets": len(self.x.assets),
                "num_yassets": len(self.y.assets),
                "num_xchannels": len(self.x.channels),
                "num_ychannels": len(self.y.channels),
                "start": self.x.start,
                "end": self.y.end,
            },
            name="TimePanel",
        )

        print(summary)
        return f"<TimePanel, size {self.__len__()}>"

    def set_training_split(self, val_size=0.2, test_size=0.1):
        """
        Time series split into training, validation, and test sets, avoiding data leakage.
        Splits the panel in training, validation, and test panels, accessed with the properties
        .train, .val and .test. The sum of the three sizes inserted must equals one.

        Parameters
        ----------
        val_size : float
            Percentage of data used for the validation set.
        test_size : float
            Percentage of data used for the test set.

        Returns
        -------
        panel : TimePanel
            New panel with the pairs split into training, validation, and test sets.
            To use each set, one must access the properties .train, .val and .test.


        Examples
        -------
        >>> panel.set_training_split(0.2, 0.1)
        >>> train = panel.train
        >>> val = panel.val
        >>> test = panel.test
        """

        train_size = len(self) - int(len(self) * test_size)

        self.test_size = int(len(self) * test_size)
        self.val_size = int(train_size * val_size)
        self.train_size = train_size - self.val_size
        assert self.train_size + self.val_size + self.test_size == len(self)

    @property
    def train(self):
        """
        Returns the TimePanel with the pairs of the training set, according to
        the parameters given in the 'set_train_val_test_sets' function.

        Returns
        -------
        TimePanel
            TimePanel with the pairs of the training set.

        """
        if self.train_size:
            return self[: self.train_size]

    @property
    def val(self):
        """
        Returns the TimePanel with the pairs of the validation set, according to
        the parameters given in the 'set_train_val_test_sets' function.

        Returns
        -------
        TimePanel
            TimePanel with the pairs of the validation set.

        """
        if self.val_size and self.train_size:
            return self[self.train_size : int(self.train_size + self.val_size)]

    @property
    def test(self):
        """
        Returns the TimePanel with the pairs of the testing set, according to
        the parameters given in the 'set_train_val_test_sets' function.

        Returns
        -------
        TimePanel
            TimePanel with the pairs of the testing set.

        """
        if self.val_size and self.train_size:
            return self[self.train_size + self.val_size :]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, key):
        if isinstance(key, Iterable):
            key_set = set(key)
            if key_set == {False, True}:
                pairs = list(compress(self.pairs, key))
            else:
                pairs = [pair for i, pair in enumerate(self.pairs) if i in key_set]
        else:
            pairs = self.pairs[key]

        return from_pairs(pairs)

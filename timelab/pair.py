from copy import copy
from typing import List
import numpy as np
import pandas as pd

from .multicol import MultiColumn, rebuild_from_index


def add_pair(func, pair):
    args = func.__code__.co_varnames
    if "pair" in args:

        def func_(x):
            return func(x, pair=pair)

    else:
        func_ = func
    return func_


def add_axis(func):
    args = func.__code__.co_varnames
    if not ("axis" in args or "args" in args):

        def func_(x, axis):
            return func(x)

    else:
        func_ = func
    return func_


def find_gap(x_indexes, y_indexes):
    diff = None
    if len(x_indexes) > 1:
        diff = pd.DatetimeIndex([x_indexes[1]]) - pd.DatetimeIndex([x_indexes[0]])
    elif len(y_indexes) > 1:
        diff = pd.DatetimeIndex([y_indexes[1]]) - pd.DatetimeIndex([y_indexes[0]])
    if diff is not None:
        x_y_diff = pd.DatetimeIndex([y_indexes[0]]) - pd.DatetimeIndex([x_indexes[-1]])

        if x_y_diff.total_seconds() % diff.total_seconds() == 0:
            gap = x_y_diff / diff
            return int(gap[0] - 1)
        else:
            print("Gap not found")
            raise ValueError(f"{x_y_diff.total_seconds()} % {diff.total_seconds()} != 0")


def from_dataframe(
    dataframe, lookback, horizon, gap=0, xunits=None, xchannels=None, yunits=None, ychannels=None,
):
    # Add start index as parameter
    xframe = dataframe.iloc[:lookback, :]
    yframe = dataframe.iloc[lookback + gap : lookback + gap + horizon, :]
    xframe = MultiColumn(xframe).filter(xunits, xchannels)
    yframe = MultiColumn(yframe).filter(yunits, ychannels)

    return from_frames(xframe, yframe, gap)


def from_frames(xframe, yframe, gap=None):
    """
    Pair of two time series dataframes.
    xframe: multi-level columns dataframe
    yframe: multi-level columns dataframe
    """

    xframe = MultiColumn(xframe)
    yframe = MultiColumn(yframe)

    xindex = list(map(str, xframe.index))
    yindex = list(map(str, yframe.index))

    xunits = xframe.units
    yunits = yframe.units
    xchannels = xframe.channels
    ychannels = yframe.channels
    lookback = len(xframe)
    horizon = len(yframe)

    # TODO: Needs tests
    X = np.array([xframe[i].values for i in xunits])
    y = np.array([yframe[i].values for i in yunits])
    # X = xframe.values
    # y = yframe.values

    return TimePair(
        xvalues=X,
        yvalues=y,
        xindex=xindex,
        xunits=xunits,
        xchannels=xchannels,
        yunits=yunits,
        yindex=yindex,
        ychannels=ychannels,
        lookback=lookback,
        horizon=horizon,
        gap=gap,
    )


class PairBlock:
    def __init__(self, pair, values, units, channels, index, name):
        self.name = name
        self._pair = pair
        self.units = units
        self.channels = channels
        self.index = index
        self.start = self.index[0]
        self.end = self.index[-1]
        self.values = values

    @property
    def frame(self):
        """
        Dataframe with the X values of this pair.

        """
        return rebuild_from_index(self.values, self.index, self.units, self.channels, smash_dims=True)


class TimePair:
    # Constant Frequency => X_freq == y_freq

    def __init__(
        self, xvalues, yvalues, xunits, xchannels, xindex, yunits, ychannels, yindex, lookback, horizon, gap=None
    ):

        if not isinstance(xunits, list):
            raise TypeError(f"Attribute 'xunits' must be a list, it is {type(xunits)}, {xunits}")
        if not isinstance(yunits, list):
            raise TypeError(f"Attribute 'yunits' must be a list, it is {type(yunits)}")
        if not isinstance(xchannels, list):
            raise TypeError(f"Attribute 'xchannels' must be a list, it is {type(xchannels)}")
        if not isinstance(ychannels, list):
            raise TypeError(f"Attribute 'ychannels' must be a list, it is {type(ychannels)}")

        # Assert or warn that no unit (column level 0) repeat names
        self._x = PairBlock(pair=self, values=xvalues, units=xunits, channels=xchannels, index=xindex, name='x')
        self._y = PairBlock(pair=self, values=yvalues, units=yunits, channels=ychannels, index=yindex, name='y')
        self.lookback = lookback
        self.horizon = horizon

        if gap is not None:
            self.gap = gap
        else:
            self.gap = find_gap(xindex, yindex)
            if lookback == 1 and horizon == 1:
                raise ValueError("If lookback and horizon equals 1, gap should be provided.")

        # Assert data indexes don't overlap (y must come after x)
        if self._x.end >= self._y.start:
            raise ValueError(
                f"Data indexes can't overlap (y must come after x). xend: {self._x.end} ystart: {self._y.start}"
            )

        self._active_block = None

    def __eq__(self, other):
        x_equals = np.all(np.nan_to_num(self.x.values, nan=0) == np.nan_to_num(other.x.values, nan=0))
        y_equals = np.all(np.all(np.nan_to_num(self.x.values, nan=0) == np.nan_to_num(other.x.values, nan=0)))

        return x_equals and y_equals

    @property
    def x(self):
        pair = copy(self)
        pair._active_block = "x"
        return pair

    @property
    def y(self):
        pair = copy(self)
        pair._active_block = "y"
        return pair

    @property
    def xframe(self):
        return self._x.frame

    @property
    def yframe(self):
        return self._y.frame

    @property
    def units(self):
        return self._get_block_attr("units")

    @property
    def channels(self):
        return self._get_block_attr("channels")

    @property
    def start(self):
        return self._get_block_attr("start")

    @property
    def end(self):
        return self._get_block_attr("end")

    @property
    def index(self):
        return self._get_block_attr("index")

    @property
    def values(self):
        return self._get_block_attr("values")

    # TODO: Repeated function from Panel, merge
    def _get_block_attr(self, name):
        if self._active_block == "x":
            return getattr(self._x, name)
        elif self._active_block == "y":
            return getattr(self._y, name)
        if self._active_block is None:
            return (getattr(self._x, name), getattr(self._y, name))

    # TODO: Implement from block
    def _sel_units(self, xunits=None, yunits=None):

        if isinstance(xunits, str):
            xunits = [xunits]
        if isinstance(yunits, str):
            yunits = [yunits]

        if not xunits:
            xunits = self._x.units
        if not yunits:
            yunits = self._y.units

        # Improve variable naming
        xus = [self._x.units.index(xu) for xu in xunits]
        yus = [self._y.units.index(yu) for yu in yunits]

        X_sel = self._x.values[xus, :, :]
        y_sel = self._y.values[yus, :, :]

        return TimePair(
            xvalues=X_sel,
            yvalues=y_sel,
            xindex=self._x.index,
            xunits=xunits,
            xchannels=self._x.channels,
            yunits=yunits,
            yindex=self._y.index,
            ychannels=self._y.channels,
            lookback=self.lookback,
            horizon=self.horizon,
            gap=self.gap,
        )

    def apply(self, func, on="timestamps", new_channel=None):
        if self._active_block == "x":
            result = self._xapply(func=func, on=on, new_channel=new_channel)
        elif self._active_block == "y":
            result = self._yapply(func=func, on=on, new_channel=new_channel)
        elif self._active_block is None:
            result = self._xapply(func=func, on=on, new_channel=new_channel)
            result = result._yapply(func=func, on=on, new_channel=new_channel)
        return result


    # TODO: Implement from block
    def _xapply(self, func, on="timestamps", new_channel=None):
        """
        Parameters:
        func :  array function with axis argument
        on :    "channels", "timestamps"
        """

        axis_map = {"timestamps": 1, "channels": 2}

        func_ = add_pair(func, self)
        func_ = add_axis(func_)

        X_ = func_(self._x.values, axis=axis_map[on])
        # check if the answer is an scalar or an array
        if hasattr(X_, "__len__"):
            shape = X_.shape
            lookback = shape[1] if X_.ndim == 3 else 1
        else:
            lookback = 1
            shape = (1, 1)

        if on == "channels":
            assert new_channel, "Must set new channel name"
            xchannels = [new_channel]
            X_ = X_.reshape((shape[0], lookback, 1))

        elif on == "timestamps":
            assert not new_channel, "No channel is created if applying on timestamps"
            xchannels = self._x.channels
            X_ = X_.reshape((shape[0], lookback, shape[-1]))

        return TimePair(
            xvalues=X_,
            yvalues=self._y.values,
            xindex=self._x.index[:lookback],
            xunits=self._x.units,
            xchannels=xchannels,
            yunits=self._y.units,
            yindex=self._y.index,
            ychannels=self._y.channels,
            lookback=lookback,
            horizon=self.horizon,
            gap=self.gap,
        )

    # TODO: Implement from block
    def _yapply(self, func, on="timestamps", new_channel=None):
        """
        Parameters:
        func :  array function with axis argument
        on :    "channels", "timestamps"
        """

        axis_map = {"timestamps": 1, "channels": 2}

        func_ = add_pair(func, self)
        func_ = add_axis(func_)

        y_ = func_(self._y.values, axis=axis_map[on])
        # check if the answer is an scalar or an array
        if hasattr(y_, "__len__"):
            shape = y_.shape
            horizon = shape[1] if y_.ndim == 3 else 1
        else:
            horizon = 1
            shape = (1, 1)

        if on == "channels":
            assert new_channel, "Must set new channel name"
            ychannels = [new_channel]
            y_ = y_.reshape((shape[0], horizon, 1))

        elif on == "timestamps":
            assert not new_channel, "No channel is created if applying on timestamps"
            ychannels = self._y.channels
            y_ = y_.reshape((shape[0], horizon, shape[-1]))

        return TimePair(
            xvalues=self._x.values,
            yvalues=y_,
            xindex=self._x.index,
            xunits=self._x.units,
            xchannels=self._x.channels,
            yunits=self._y.units,
            yindex=self._y.index[:horizon],
            ychannels=ychannels,
            lookback=self.lookback,
            horizon=horizon,
            gap=self.gap,
        )

    # TODO: Implement from block
    def _sel_channels(self, xchannels=None, ychannels=None):

        if isinstance(xchannels, str):
            xchannels = [xchannels]
        if isinstance(ychannels, str):
            ychannels = [ychannels]

        if not xchannels:
            xchannels = self._x.channels
        if not ychannels:
            ychannels = self._y.channels

        xcs = [self._x.channels.index(xc) for xc in xchannels]
        ycs = [self._y.channels.index(yc) for yc in ychannels]

        X_sel = self._x.values[:, :, xcs]
        y_sel = self._y.values[:, :, ycs]

        return TimePair(
            xvalues=X_sel,
            yvalues=y_sel,
            xindex=self._x.index,
            xunits=self._x.units,
            xchannels=xchannels,
            yunits=self._y.units,
            yindex=self._y.index,
            ychannels=ychannels,
            lookback=self.lookback,
            horizon=self.horizon,
            gap=self.gap,
        )

    # TODO: Implement from block
    def filter(self, units=None, channels=None):
        """
        Returns the pair with only the select units and channels

        Parameters
        ----------
        xunits : str, optional
            Selected xunits.
        xchannels : str, optional
            Selected xchannels.
        yunits : str, optional
            Selected yunits.
        ychannels : str, optional
            Selected ychannels.


        Returns
        -------
        selected : TimePair
            New TimePair with only the selected channels and units.

        """
        if self._active_block == "x":
            selected = self._sel_units(xunits=units, yunits=None)
            selected = selected._sel_channels(xchannels=channels, ychannels=None)
        elif self._active_block == "y":
            selected = self._sel_units(xunits=None, yunits=units)
            selected = selected._sel_channels(xchannels=None, ychannels=channels)
        elif self._active_block is None:
            selected = self._sel_units(xunits=units, yunits=units)
            selected = selected._sel_channels(xchannels=channels, ychannels=channels)

        return selected

    # TODO: Implement from block
    def add_channel(self, new_pair):
        """
        Returns the pair with the new channel

        Parameters
        ----------
        new_pair : TimePair
            TimePair with the channel to be added.
        mode : str
            Select the channel from the xdata or ydata. Options:
            'X', 'y'. The default is "X".

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        if self._active_block == "x":
            if new_pair._x.values.shape[1] != self._x.values.shape[1]:
                new_pair._x.values = (
                    np.ones((new_pair._x.values.shape[0], self._x.values.shape[1], new_pair._x.values.shape[2]))
                    * new_pair._x.values
                )
            X = np.concatenate((self._x.values, new_pair._x.values), axis=2)
            xchannels = self._x.channels + new_pair._x.channels
            y = self._y.values
            ychannels = self._y.channels

        elif self._active_block == "y":
            if new_pair._y.values.shape[1] != self._y.values.shape[1]:
                new_pair._y.values = (
                    np.ones((new_pair._y.values.shape[0], self._y.values.shape[1], new_pair._y.values.shape[2]))
                    * new_pair._y.values
                )
            X = self._x.values
            xchannels = self._x.channels
            y = np.concatenate((self._y.values, new_pair._y.values), axis=2)
            ychannels = self._y.channels + new_pair._y.channels

        return TimePair(
            xvalues=X,
            yvalues=y,
            xindex=self._x.index,
            xunits=self._x.units,
            xchannels=xchannels,
            yunits=self._y.units,
            yindex=self._y.index,
            ychannels=ychannels,
            lookback=self.lookback,
            horizon=self.horizon,
            gap=self.gap,
        )


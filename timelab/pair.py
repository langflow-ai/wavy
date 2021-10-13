from copy import copy
from typing import List
import numpy as np
import pandas as pd

from .utils import _get_active, _get_block_attr

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
    xvalues = np.array([xframe[i].values for i in xunits])
    yvalues = np.array([yframe[i].values for i in yunits])

    return TimePair(
        xvalues=xvalues,
        yvalues=yvalues,
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


def from_blocks(x_block, y_block):
    return TimePair(
        xvalues=x_block.values,
        yvalues=y_block.values,
        xindex=x_block.index,
        xunits=x_block.units,
        xchannels=x_block.channels,
        yunits=y_block.units,
        yindex=y_block.index,
        ychannels=y_block.channels,
        lookback=y_block.lookback,
        horizon=y_block.horizon,
        gap=y_block.gap,
    )


class PairBlock:
    def __init__(self, values, units, channels, index, name, lookback, horizon, gap):
        self.name = name
        self.units = units
        self.channels = channels
        self.index = index
        self.start = self.index[0]
        self.end = self.index[-1]
        self.values = values
        self.lookback = lookback
        self.horizon = horizon
        self.gap = gap

    @property
    def frame(self):
        """
        Dataframe with the X values of this pair.

        """
        return rebuild_from_index(self.values, self.index, self.units, self.channels, smash_dims=True)

    def _sel_units(self, units=None):

        if isinstance(units, str):
            units = [units]

        if not units:
            units = self.units

        # Improve variable naming
        return [self.units.index(unit) for unit in units]

    def _sel_channels(self, channels=None):

        if isinstance(channels, str):
            channels = [channels]

        if not channels:
            channels = self.channels

        return [self.channels.index(channel) for channel in channels]

    def filter(self, units=None, channels=None):
        units = self._sel_units(units=units)
        channels = self._sel_channels(channels=channels)
        return PairBlock(
            self.values[units or slice(None), :, channels or slice(None)],
            units,
            channels or self.channels,
            self.index,
            self.name,
            self.lookback,
            self.horizon,
            self.gap,
        )


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
        self.lookback = lookback
        self.horizon = horizon

        if gap is None:
            self.gap = find_gap(xindex, yindex)
            if lookback == 1 and horizon == 1:
                raise ValueError("If lookback and horizon equals 1, gap should be provided.")
        else:
            self.gap = gap

        self._x = PairBlock(
            values=xvalues,
            units=xunits,
            channels=xchannels,
            index=xindex,
            name="x",
            lookback=self.lookback,
            horizon=self.horizon,
            gap=self.gap,
        )
        self._y = PairBlock(
            values=yvalues,
            units=yunits,
            channels=ychannels,
            index=yindex,
            name="y",
            lookback=self.lookback,
            horizon=self.horizon,
            gap=self.gap,
        )

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

    def __repr__(self):
        if self._active_block:
            return "<TimePair Active Block>"
        return f"<TimePair, lookback {self.lookback}, horizon {self.horizon}>"

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
        return _get_block_attr(self, "units")

    @property
    def channels(self):
        return _get_block_attr(self, "channels")

    @property
    def start(self):
        return _get_block_attr(self, "start")

    @property
    def end(self):
        return _get_block_attr(self, "end")

    @property
    def index(self):
        return _get_block_attr(self, "index")

    @property
    def values(self):
        if self._active_block is None:
            raise AttributeError("'TimePair' object has no attribute 'values'")
        return _get_block_attr(self, "values")

    @values.setter
    def values(self, values):
        if self._active_block is None:
            self._values = values
        _get_active(self).values = values

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
        if self._active_block is None:

            x_block = self._x.filter(units=units, channels=channels)
            y_block = self._y.filter(units=units, channels=channels)
            return from_blocks(x_block, y_block)
        else:
            pair_block = _get_active(self).filter(units=units, channels=channels)
            if self._active_block == "x":
                return from_blocks(pair_block, self._y)
            else:
                return from_blocks(self._x, pair_block)

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

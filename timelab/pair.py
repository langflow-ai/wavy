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
            raise ValueError(
                f"{x_y_diff.total_seconds()} % {diff.total_seconds()} != 0"
            )


def from_dataframe(
    dataframe,
    lookback,
    horizon,
    gap=0,
    xunits=None,
    xchannels=None,
    yunits=None,
    ychannels=None,
):
    # Add start index as parameter
    xframe = dataframe.iloc[:lookback, :]
    yframe = dataframe.iloc[lookback + gap : lookback + gap + horizon, :]
    xframe = MultiColumn(xframe).filter(xunits, xchannels)
    yframe = MultiColumn(yframe).filter(xunits, xchannels)

    return from_frames(xframe, yframe, gap)


def from_frames(xframe, yframe, gap=None):
    """
        Pair of two time series dataframes.
        xframe: multi-level columns dataframe
        yframe: multi-level columns dataframe
    """

    xframe = MultiColumn(xframe)
    yframe = MultiColumn(yframe)

    x_idx = list(map(str, xframe.index))
    y_idx = list(map(str, yframe.index))
    indexes = (x_idx, y_idx)

    xunits = xframe.units
    yunits = yframe.units
    xchannels = xframe.channels
    ychannels = yframe.channels
    lookback = len(xframe)
    horizon = len(yframe)

    # Assert frequency is constant
    # ! Must find a way to check when size is 2
    #     if lookback > 2:
    #         assert check_frequency(xframe)
    #     if horizon > 2:
    #         assert check_frequency(yframe)

    # TODO: Needs tests
    X = np.array([xframe[i].values for i in xunits])
    y = np.array([yframe[i].values for i in yunits])
    # X = xframe.values
    # y = yframe.values

    return TimePair(
        X=X,
        y=y,
        indexes=indexes,
        xunits=xunits,
        xchannels=xchannels,
        yunits=yunits,
        ychannels=ychannels,
        lookback=lookback,
        horizon=horizon,
        gap=gap,
    )


class TimePair:
    # Constant Frequency => X_freq == y_freq

    def __init__(
        self,
        X,
        y,
        indexes,
        xunits,
        xchannels,
        yunits,
        ychannels,
        lookback,
        horizon,
        gap=None,
    ):

        if not isinstance(xunits, list):
            raise TypeError(f"Attribute 'xunits' must be a list, it is {type(xunits)}")
        if not isinstance(yunits, list):
            raise TypeError(f"Attribute 'yunits' must be a list, it is {type(yunits)}")
        if not isinstance(xchannels, list):
            raise TypeError(
                f"Attribute 'xchannels' must be a list, it is {type(xchannels)}"
            )
        if not isinstance(ychannels, list):
            raise TypeError(
                f"Attribute 'ychannels' must be a list, it is {type(ychannels)}"
            )

        # Assert or warn that no unit (column level 0) repeat names

        self.indexes = indexes
        self.xunits = xunits
        self.yunits = yunits
        self.xchannels = xchannels
        self.ychannels = ychannels
        self.lookback = lookback
        self.horizon = horizon

        if gap is not None:
            self.gap = gap
        else:
            self.gap = find_gap(*indexes)
            if lookback == 1 and horizon == 1:
                raise ValueError(
                    "If lookback and horizon equals 1, gap should be provided."
                )

        self.xstart = self.indexes[0][0]
        self.ystart = self.indexes[1][0]
        self.xend = self.indexes[0][-1]
        self.yend = self.indexes[1][-1]

        # Assert data indexes don't overlap (y must come after x)
        assert (
            self.xend < self.ystart
        ), f"Data indexes can't overlap (y must come after x). xend: {self.xend} ystart: {self.ystart}"

        self.X = X
        self.y = y

    @property
    def xframe(self):
        """
        Dataframe with the X values of this pair.

        """
        return rebuild_from_index(
            self.X, self.indexes[0], self.xunits, self.xchannels, smash_dims=True
        )

    @property
    def yframe(self):
        """
        Dataframe with the y values of this pair.

        """
        return rebuild_from_index(
            self.y, self.indexes[1], self.yunits, self.ychannels, smash_dims=True
        )

    def _sel_units(self, xunits=None, yunits=None):

        if isinstance(xunits, str):
            xunits = [xunits]
        if isinstance(yunits, str):
            yunits = [yunits]

        if not xunits:
            xunits = self.xunits
        if not yunits:
            yunits = self.yunits

        # Improve variable naming
        xus = (self.xunits.index(xu) for xu in xunits)
        yus = (self.yunits.index(yu) for yu in yunits)

        X_sel = self.X[xus, :, :]
        y_sel = self.y[yus, :, :]

        return TimePair(
            X_sel,
            y_sel,
            self.indexes,
            xunits,
            self.xchannels,
            yunits,
            self.ychannels,
            self.lookback,
            self.horizon,
            self.gap,
        )

    def xapply(self, func, on="timestamps", new_channel=None):
        """
        Parameters:
        func :  array function with axis argument
        on :    "channels", "timestamps"
        """

        axis_map = {"timestamps": 1, "channels": 2}

        func_ = add_pair(func, self)
        func_ = add_axis(func_)

        X_ = func_(self.X, axis=axis_map[on])
        # check if the answer is an scalar or an array
        if hasattr(X_, "__len__"):
            shape = X_.shape
            if X_.ndim == 3:
                lookback = shape[1]
            else:
                lookback = 1
        else:
            lookback = 1
            shape = (1,1)

        if on == "channels":
            assert new_channel, "Must set new channel name"
            xchannels = [new_channel]
            indexes = (self.indexes[0][:lookback], self.indexes[1])
            X_ = X_.reshape((shape[0], lookback, 1))

        elif on == "timestamps":
            assert not new_channel, "No channel is created if applying on timestamps"
            xchannels = self.xchannels
            indexes = (self.indexes[0][:lookback], self.indexes[1])
            X_ = X_.reshape((shape[0], lookback, shape[-1]))

        return TimePair(
            X_,
            self.y,
            indexes,
            self.xunits,
            xchannels,
            self.yunits,
            self.ychannels,
            lookback,
            self.horizon,
            self.gap,
        )

    def yapply(self, func, on="timestamps", new_channel=None):
        """
        Parameters:
        func :  array function with axis argument
        on :    "channels", "timestamps"
        """

        axis_map = {"timestamps": 1, "channels": 2}

        func_ = add_pair(func, self)
        func_ = add_axis(func_)

        y_ = func_(self.y, axis=axis_map[on])
        # check if the answer is an scalar or an array
        if hasattr(y_, "__len__"):
            shape = y_.shape
            if y_.ndim == 3:
                horizon = shape[1]
            else:
                horizon = 1
        else:
            horizon = 1
            shape = (1,1)

        if on == "channels":
            assert new_channel, "Must set new channel name"
            ychannels = [new_channel]
            indexes = (self.indexes[0], self.indexes[1][:horizon])
            y_ = y_.reshape((shape[0], horizon, 1))

        elif on == "timestamps":
            assert not new_channel, "No channel is created if applying on timestamps"
            ychannels = self.ychannels
            indexes = (self.indexes[0], self.indexes[1][:horizon])
            y_ = y_.reshape((shape[0], horizon, shape[-1]))

        pair = TimePair(
            self.X,
            y_,
            indexes,
            self.xunits,
            self.xchannels,
            self.yunits,
            ychannels,
            self.lookback,
            horizon,
            self.gap,
        )

        return pair

    def _sel_channels(self, xchannels=None, ychannels=None):

        if isinstance(xchannels, str):
            xchannels = [xchannels]
        if isinstance(ychannels, str):
            ychannels = [ychannels]

        if not xchannels:
            xchannels = self.xchannels
        if not ychannels:
            ychannels = self.ychannels

        xcs = (self.xchannels.index(xc) for xc in xchannels)
        ycs = (self.ychannels.index(yc) for yc in ychannels)

        X_sel = self.X[:, :, xcs]
        y_sel = self.y[:, :, ycs]

        return TimePair(
            X_sel,
            y_sel,
            self.indexes,
            self.xunits,
            xchannels,
            self.yunits,
            ychannels,
            self.lookback,
            self.horizon,
            self.gap,
        )

    def filter(self, xunits=None, xchannels=None, yunits=None, ychannels=None):
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

        selected = self._sel_units(xunits=xunits, yunits=yunits)
        selected = selected._sel_channels(xchannels=xchannels, ychannels=ychannels)
        return selected

    def add(self, new_pair, mode):
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

        if mode == "X":
            if new_pair.X.shape[1] != self.X.shape[1]:
                new_pair.X = (
                    np.ones((new_pair.X.shape[0], self.X.shape[1], new_pair.X.shape[2]))
                    * new_pair.X
                )
            X = np.concatenate((self.X, new_pair.X), axis=2)
            xchannels = self.xchannels + new_pair.xchannels
            y = self.y
            ychannels = self.ychannels

        elif mode == "y":
            if new_pair.y.shape[1] != self.y.shape[1]:
                new_pair.y = (
                    np.ones((new_pair.y.shape[0], self.y.shape[1], new_pair.y.shape[2]))
                    * new_pair.y
                )
            X = self.X
            xchannels = self.xchannels
            y = np.concatenate((self.y, new_pair.y), axis=2)
            ychannels = self.ychannels + new_pair.ychannels

        return TimePair(
            X,
            y,
            self.indexes,
            self.xunits,
            xchannels,
            self.yunits,
            ychannels,
            self.lookback,
            self.horizon,
            self.gap,
        )

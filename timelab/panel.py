from re import X
import warnings
from copy import copy

import numpy as np
import pandas as pd
import plotly.express as px
from tqdm import tqdm

from . import frequency as freq
from .multicol import MultiColumn, rebuild_from_index
from .pair import TimePair
from .utils import bfill, ffill, get_null_indexes, smash_array, get_all_unique, _get_active, _get_block_attr


def from_xy_data(xdata, ydata, lookback, horizon, gap=0):
    """
     Gets TimePanels from x,y DataFrames

     Parameters
     ----------
     xdata : DataFrame
         Inputs to the model.
     ydata : DataFrame
         Outputs to the model.
     lookback : int
         How many time steps from the past are considered for the xdata.
     horizon : int
         How many time steps to the future are considered for the ydata.
     gap : int, optional
         How many time steps are ignored between the lookback and the horizon. The default is 0.

     Raises
     ------
    ValueError
         If the number of total time steps in the given dataset is not enough to create a TimePair.
         I.e. the number of time steps has to be greater than or equal to 'lookback + horizon + gap'.

     Returns
     -------
     TimePanel
         TimePanel object containing all (input,output) TimePairs of te given dataset.

    """

    try:
        if not freq.infer(xdata) or not freq.infer(ydata):
            warnings.warn("Frequency is either not constant or negative.")
    except Exception:
        pass

    if xdata.isnull().values.any() or ydata.isnull().values.any():
        warnings.warn("Data contains null values.")

    x_timesteps = len(xdata.index)

    # assert y_index match gap, horizon, lookback with x_index
    total = x_timesteps - lookback - horizon - gap + 1

    if total <= 0:
        raise ValueError("Not enough timesteps to extract X and y.")

    # pairs = list()
    end = x_timesteps - horizon - gap + 1

    xdata = pd.DataFrame(xdata)
    ydata = pd.DataFrame(ydata)

    # Get Indexes from the dataframes
    x_data_indexes = list(map(str, xdata.index))
    y_data_indexes = list(map(str, ydata.index))

    # Get units and channels
    xdata = MultiColumn(xdata)
    xunits = xdata.units
    ydata = MultiColumn(ydata)
    yunits = ydata.units
    xchannels = xdata.channels
    ychannels = ydata.channels

    # Convert from datafram to numpy array
    X = np.array([xdata[i].values for i in xunits])
    y = np.array([ydata[i].values for i in yunits])

    indexes = np.arange(lookback, end)
    indexes = tqdm(indexes)
    pairs = [
        TimePair(
            xvalues=X[:, i - lookback : i, :],
            yvalues=y[:, i + gap : i + gap + horizon, :],
            xindex=x_data_indexes[i - lookback : i],
            yindex=y_data_indexes[i + gap : i + gap + horizon],
            xunits=xunits,
            yunits=yunits,
            xchannels=xchannels,
            ychannels=ychannels,
            lookback=lookback,
            horizon=horizon,
            gap=gap,
        )
        for i in indexes
    ]

    return TimePanel(pairs)


def from_ypred(panel, ypred):
    panel_ = copy(panel)
    for i, pair in enumerate(panel_.pairs):
        # TODO: Replace with setitem
        pair.y = ypred[i]
    return panel_


def from_data(
    data, lookback, horizon, gap=0, freq=None, xunits=None, yunits=None, xchannels=None, ychannels=None,
):
    """
    Gets TimePanels from a DataFrame.

    Parameters
    ----------
    data : DataFrame
        DataFrame.
    lookback : int
        How many time steps from the past are considered for the xdata.
    horizon : int
        How many time steps to the future are considered for the ydata.
    gap : int, optional
        How many time steps are ignored between the lookback and the horizon. The default is 0.
    freq : int, optional
        Data frequency. The default is None.
    xunits : str or list of str
        Units of the X data.
    yunits : str or list of str
        Units of the y data.
    xchannels : str or list of str
        Channels of the X data.
    ychannels : str or list of str
        Channels of the y data.

    Returns
    -------
    TimePanel
        TimePanel object containing all (input,output) TimePairs of te given dataset.

    """

    # TODO: Need tests and change how to get parameters

    # Assert that all units have the same channels
    # freq = None // if defined, reshape the input data

    if freq:
        # Raise warning that is dropping nans
        data = data.resample(freq).ffill().dropna()

    data = MultiColumn(data, sep=None)
    xdata = data.filter(xunits, xchannels)
    ydata = data.filter(yunits, ychannels)
    return from_xy_data(xdata, ydata, lookback, horizon, gap)


def from_arrays(
    X, y, index, xindex, yindex, lookback, horizon, gap, xunits, yunits, xchannels, ychannels,
):
    """
    Gets TimePanels from arrays.

    Parameters
    ----------
    X : numpy array
        Array with the X values.
    y : numpy array
        Array with the y values.
    index: Datetime index or str
        Index of each row.
    xindex: Datetime index or str
        Index of each row of the X array.
    yindex: Datetime index or str
        Index of each row of the y array.
    lookback : int
        How many time steps from the past are considered for the xdata.
    horizon : int
        How many time steps to the future are considered for the ydata.
    gap : int, optional
        How many time steps are ignored between the lookback and the horizon. The default is 0.
    freq : int, optional
        Data frequency. The default is None.
    xunits : str or list of str
        Units of the X data.
    yunits : str or list of str
        Units of the y data.
    xchannels : str or list of str
        Channels of the X data.
    ychannels : str or list of str
        Channels of the y data.

    Returns
    -------
    TimePanel
        TimePanel object containing all (input,output) TimePairs of te given dataset.

    """

    xindex = index[: -horizon - gap]
    yindex = index[lookback + gap :]
    xdata = TimePanel.make_xdata(X, index, xindex, xunits, xchannels)
    ydata = TimePanel.make_ydata(y, index, yindex, yunits, ychannels)
    return from_xy_data(xdata, ydata, lookback, horizon, gap)


class PanelBlock:
    def __init__(self, pairs, name):
        self.name = name
        self.pairs = [getattr(pair, self.name) for pair in pairs]
        self.values = np.array([pair.values for pair in self.pairs])
        self.first = self.pairs[0]
        self.last = self.pairs[1]

        self.units = self.first.units
        self.channels = self.first.channels
        self.start = self.first.start
        self.end = self.last.end

        self.lookback = self.first.lookback
        self.horizon = self.first.horizon
        self.gap = self.first.gap

    """
    All functions are properties for calling on timepanel
    """

    @property
    def shape(self):
        return pd.Series(self.values.shape, index=self.dims)

    def findna(self):
        return set(get_null_indexes(self.values))

    def dropna(self):
        """ Drops the pair containing NaN values"""
        null_indexes = self.findna
        return [pair for idx, pair in enumerate(self.pairs) if idx not in null_indexes]

    def apply(self, func, on="timestamps", new_channel=None):
        return [pair.apply(func=func, on=on, new_channel=new_channel) for pair in tqdm(self.pairs)]

    def filter(self, units=None, channels=None):
        """
        Selects only the given channels and units and returns another TimePanel.

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
        TimePanel
            New TimePanel with only the selected channels and units.

        """
        return [pair.filter(units, channels) for pair in tqdm(self.pairs)]


class TimePanel:
    """
    This class creates a TimePanel object that contains pairs with
    inputs (xdata) and outputs (ydata).

    """

    # TODO:
    # Check if all forms of initialization match with each other

    def __init__(self, pairs):
        self._active_block = None
        if not pairs:
            raise ValueError("Cannot instantiate TimePanel with empty pairs")

        self.pairs = pairs
        self._x = PanelBlock(self.pairs, "x")
        self._y = PanelBlock(self.pairs, "y")

        # TODO: either remove or improve infer_freq
        # self.freq = self.infer_freq()

        self.train_size, self.val_size, self.test_size = None, None, None

    @property
    def x(self):
        panel = copy(self)
        panel._active_block = "x"
        return panel

    @property
    def y(self):
        panel = copy(self)
        panel._active_block = "y"
        return panel

    @property
    def dims(self):
        return ["size", "units", "timesteps", "channels"]

    @property
    def shape(self):
        if self._active_block is None:
            return pd.DataFrame([self.x.shape, self.y.shape], index=["X", "y"])
        return _get_block_attr(self, "shape")

    @property
    def pairs(self):
        if self._active_block is None:
            return self._pairs
        return _get_block_attr(self, "pairs")

    @pairs.setter
    def pairs(self, pairs):
        self._pairs = pairs

    @property
    def first(self):
        return _get_block_attr(self, "first")

    @property
    def last(self):
        return _get_block_attr(self, "last")

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
    def values(self):
        return _get_block_attr(self, "values")

    @property
    def gap(self):
        return _get_block_attr(self, "gap")

    @property
    def lookback(self):
        return _get_block_attr(self, "lookback")

    @property
    def horizon(self):
        return _get_block_attr(self, "horizon")

    @property
    def X_(self):
        return self.x.values

    @property
    def y_(self):
        return self.y.values

    @property
    def index(self):
        """
        Returns the indexes of the dataframe.

        Returns
        -------
        list
            List of indexes of the dataframe.

        """
        # TODO: Improve code, should be partially on block

        if self._active_block is None:
            return sorted(list(set(self.x.index + self.y.index)))
        indexes = np.array([pair.index for pair in self.pairs])
        return [str(idx) for idx in get_all_unique(indexes)]

    def findna(self):
        if self._active_block is None:
            return {pair for pair in self.x.findna() if pair in self.y.findna()}
        return _get_block_attr(self, "findna")()

    def dropna(self):
        """Remove pairs with nan values

        Args:
            panel ([type]): [description]

        Returns:
            [type]: [description]
        """
        if self._active_block is None:
            pairs = [pair for pair in self.x.dropna() if pair in self.y.dropna()]
        else:
            pairs = _get_block_attr(self, "dropna")()
        if not pairs:
            # TODO: Return empty TimePanel
            raise RuntimeError("Cannot dropna if all TimePairs contain NaN values.")
        return TimePanel(pairs)

        # def infer_freq(self):
        # TODO: Fix
        """
        Infer panel data frequency.

        Returns
        -------
        string
            Frequency guess representing the seasonality of the data.

        """
        # return freq.infer(self.x.data)

    def apply(self, func, on="timestamps", new_channel=None):
        if self._active_block is None:
            pairs = self.x.apply(func=func, on=on, new_channel=new_channel)
            panel = TimePanel(pairs)
            pairs = panel.y.apply(func=func, on=on, new_channel=new_channel)
            return TimePanel(pairs)
        pairs = self.apply(func=func, on=on, new_channel=new_channel)
        return TimePanel(pairs)

    def xframe(self, idx):
        """
        Dataframe with the xdata from the selected pair.

        Parameters
        ----------
        idx : int
            index.

        Returns
        -------
        Dataframe
            Dataframe with the xdata from the selected pair.

        """
        return self.pairs[idx].xframe

    def yframe(self, idx):
        """
        Dataframe with the ydata from the selected pair.

        Parameters
        ----------
        idx : int
            index.

        Returns
        -------
        Dataframe
            Dataframe with the ydata from the selected pair.

        """
        return self.pairs[idx].yframe

    def add_channel(self, new_panel):
        """
        Adds a new channel from another panel to the current panel.

        Parameters
        ----------
        new_panel : TimePanel
            TimePanel with the channel that will be added.

        Raises
        ------

        Returns
        -------
        TimePanel
            Current TimePanel with the new channel added.

        Examples
        -------
        >>> new_panel = panel.xapply(np.max, on='timestamps')
        >>> panel = panel.add_channel(new_panel, mode='X')

        """
        if self._active_block is None:
            panel = self.x.add_channel(new_panel)
            panel = panel.y.add_channel(new_panel)
            return panel
        pairs = [pair.add_channel(new_panel[index]) for index, pair in tqdm(enumerate(self.pairs))]
        return TimePanel(pairs)

    def fillna(self, value=None, method=None):
        # TODO: Make explicit for y and X
        """Fills the numpy array with parameter value
        or using one of the methods 'ffill' or 'bfill'.

        Parameters
        ----------
        value : int
            Value to replace NaN values
        method : str
            One of 'ffill' or 'bfill'.

        Raises
        ------
        ValueError
            Parameter method must be 'ffill' or 'bfill' but you passed '{method}'.

        ValueError
            Parameter value must be  int or float.

        Returns
        -------
        TimePanel
            New TimePanel after filling NaN values.
        """
        if method not in ["ffill", "bfill", None]:
            raise ValueError(f"Parameter method must be 'ffill' or 'bfill' but you passed '{method}'.")

        if value is not None:
            if isinstance(value, (int, float)):

                def func(x, axis=None):
                    return np.nan_to_num(x.astype(float), nan=value)

            else:
                raise ValueError(f"Parameter value must be int or float. It is {type(value)}.")

        elif method == "ffill":
            func = ffill
        elif method == "bfill":
            func = bfill

        if self._active_block is None:
            pairs = self.x.apply(func)
            pairs = TimePanel(pairs).y.apply(func)
            return TimePanel(pairs)
        else:
            pairs = _get_active(self).apply(func)
        return TimePanel(pairs)

    @staticmethod
    def make_dataframe(data, index, block_index, units, channels):
        data = smash_array(data)
        all_ = get_all_unique(data)
        data_ = rebuild_from_index(all_, block_index, units, channels, to_datetime=True)
        data = pd.DataFrame(index=index, columns=pd.MultiIndex.from_product([units, channels]))
        data.loc[block_index, (units, channels)] = data_.values
        return MultiColumn(data)

    @property
    def data(self):
        """
        Returns a dataframe with all the X values.

        Returns
        -------
        Dataframe
            Dataframe with all the X values.

        """
        if self._active_block is None:
            xdata = self.make_dataframe(self.X_, self.index, self.x.index, self.x.units, self.x.channels)
            ydata = self.make_dataframe(self.y_, self.index, self.y.index, self.y.units, self.y.channels)
            return xdata, ydata
        attr = getattr(self, self._active_block)
        return self.make_dataframe(attr.values, self.index, attr.index, attr.units, attr.channels)

    def train_test_split(self, test_size=0.2):
        """
        Time series split into training and testing sets avoiding data leakage. Splits
        the panel in train and test panels.

        Parameters
        ----------
        split_size : float
            Percentage of data used for the test set.

        Returns
        -------
        train : TimePanel
            Part of the total number of TimePanels according to the percentage provided.
        test : TimePanel
            Part of the total number of TimePanels according to the percentage provided.
        """
        if 0 < test_size < 1:
            train_size = int(len(self) * (1 - test_size))
        else:
            train_size = len(len(self) - test_size)

        train = self[:train_size]
        test = self[train_size:]

        return train, test

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

    def view(self):
        """
        Displays the main infos of the TimePanel.

        Returns
        -------
        None.

        """

        print("TimePanel")

        if self.x.units == self.y.units and self.x.channels == self.y.channels:
            summary = pd.Series(
                {
                    "size": self.__len__(),
                    "lookback": self.lookback,
                    "horizon": self.horizon,
                    "gap": self.gap,
                    "units": self.x.units,
                    "channels": self.x.channels,
                    "start": self.x.start,
                    "end": self.y.end,
                },
                name="TimePanel",
            )
            print(summary)
            return

        summary = pd.Series(
            {
                "size": self.__len__(),
                "lookback": self.lookback,
                "horizon": self.horizon,
                "gap": self.gap,
                "num_xunits": len(self.x.units),
                "num_yunits": len(self.y.units),
                "num_xchannels": len(self.x.channels),
                "num_ychannels": len(self.y.channels),
                "start": self.x.start,
                "end": self.y.end,
            },
            name="TimePanel",
        )
        print(summary)

    def head(self, n=5):
        """
        Returns the TimePanel with the first n pairs.

        Parameters
        ----------
        n : int, optional
            Number of pairs. The default is 5.

        Returns
        -------
        TimePanel
             TimePanel with the first n pairs..

        """
        return self[:n]

    def tail(self, n=5):
        """
        Returns the TimePanel with the last n pairs.

        Parameters
        ----------
        n : int, optional
            Number of pairs. The default is 5.

        Returns
        -------
        TimePanel
             TimePanel with the first n pairs..

        """
        return self[-n:]

    def filter(self, units=None, channels=None):
        """
        Selects only the given channels and units and returns another TimePanel.

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
        TimePanel
            New TimePanel with only the selected channels and units.

        """
        if self._active_block is None:
            panel = TimePanel([pair.filter(units, channels) for pair in tqdm(self.x.pairs)])
            pairs = [pair.y.filter(units, channels) for pair in tqdm(panel.y.pairs)]
        else:
            # TODO This part should be implemented on block
            pairs = [pair.filter(units, channels) for pair in tqdm(self.pairs)]
        return TimePanel(pairs)

    def split_units(self, yunits=False):
        """
        Return a list of panels, one panel for each unit.

        Parameters
        ----------
        yunits : bool, optional
            If yunits is False, will split only xunits,
            leaving yunits as it is. The default is False.

        Returns
        -------
        panels : List of TimePanels
            List of panels, one panel for each unit.
        """

        index_units = np.arange(0, len(self.xunits))
        indexes = np.arange(0, len(self.pairs))

        return [
            TimePanel(
                [
                    TimePair(
                        xvalues=self.pairs[i].X_[index_unit, :, :].reshape(1, self.lookback, -1),
                        yvalues=self.pairs[i].y_[index_unit, :, :].reshape(1, self.horizon, -1)
                        if yunits
                        else self.pairs[i].y,
                        xindex=self.pairs[i].x.index,
                        xunits=[self.pairs[i].x.units[index_unit]],
                        yunits=[self.pairs[i].y.units[index_unit]] if yunits else self.pairs[i].y.units,
                        yindex=self.pairs[i].y.index,
                        xchannels=self.pairs[i].x.channels,
                        ychannels=self.pairs[i].y.channels,
                        lookback=self.pairs[i].lookback,
                        horizon=self.pairs[i].horizon,
                        gap=self.pairs[i].gap,
                    )
                    for i in indexes
                ]
            )
            for index_unit in tqdm(index_units)
        ]

    def swap_dims(self):
        """
        Swap units with channels.
        """

        xdata = self.xdata.swaplevel(i=-2, j=-1, axis=1)
        ydata = self.ydata.swaplevel(i=-2, j=-1, axis=1)
        return from_xy_data(xdata, ydata, horizon=self.horizon, lookback=self.lookback, gap=self.gap)

    # TODO: Stopped here (08/10)
    def replace(self, data):
        if self._active_block is None:
            raise NotImplementedError("Replace must be applied to attributes TimePanel.x or TimePanel.y")
        pairs = []
        for i, pair in enumerate(self.pairs):
            pair_ = copy(pair)

            setattr(pair_, self._active_block, data[i])
            pairs.append(pair_)
        return TimePanel(pairs)

    def xflat(self):
        """
        Flattens X output for shallow ML models.

        Args:
            use_index (bool, optional): If True, uses "yindex" for dates to match with ML y_true.
            Defaults to True.

        Returns:
            DataFrame: DataFrame where each "xframe" is represented in a row.
        """
        # avoid indexing to 0
        index = self.xindex if self.lookback == 1 else self.xindex[: -self.lookback + 1]

        xflat = np.array([i.flatten() for i in self.X])
        xflat = pd.DataFrame(xflat, index=index)
        return xflat

    def yflat(self):
        """
        Flattens y output for shallow ML models.

        Args:
            use_index (bool, optional): If True, uses "yindex" for dates to match with ML y_true.
            Defaults to True.

        Returns:
            DataFrame: DataFrame where each "yframe" is represented in a row.
        """

        index = self.yindex if self.horizon == 1 else self.yindex[: -self.horizon + 1]
        yflat = np.array([i.flatten() for i in self.y])
        yflat = pd.DataFrame(yflat, index=index)
        return yflat

    def __len__(self):
        return len(self.pairs)

    def __repr__(self):
        self.view()
        return f"<TimePanel, size {self.__len__()}>"

    def __getitem__(self, key):
        _, _, selection = None, None, None

        if isinstance(key, int):
            selection = self.pairs[key]
            if selection:
                return selection

        elif isinstance(key, str):
            selection = [pair for pair in self.pairs if pd.Timestamp(pair.xstart) == pd.Timestamp(key)]
            if selection:
                return selection[0]  # No xstart repeat

        elif isinstance(key, slice):
            selection = self.pairs
            if isinstance(key.start, pd.Timestamp) or isinstance(key.stop, pd.Timestamp):
                if key.start:
                    selection = [pair for pair in selection if pd.Timestamp(pair.xstart) >= key.start]
                if key.stop:
                    selection = [pair for pair in selection if pd.Timestamp(pair.xstart) < key.stop]

            elif isinstance(key.start, int) or isinstance(key.stop, int):
                if key.start and key.stop:
                    selection = selection[key.start : key.stop]
                elif key.start:
                    selection = selection[key.start :]
                elif key.stop:
                    selection = selection[: key.stop]

            elif isinstance(key.start, str) or isinstance(key.stop, str):
                if key.start:
                    selection = [pair for pair in selection if pd.Timestamp(pair.xstart) >= pd.Timestamp(key.start)]
                if key.stop:
                    selection = [pair for pair in selection if pd.Timestamp(pair.xstart) < pd.Timestamp(key.stop)]

        if selection:
            return TimePanel(selection)
        return

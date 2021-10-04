from copy import copy
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
from tqdm import tqdm

from . import frequency as freq
from .multicol import MultiColumn, rebuild_from_index
from .pair import TimePair
from .utils import all_equal, bfill, ffill, get_null_indexes, smash_array


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
    except:
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
            X[:, i - lookback: i, :],
            y[:, i + gap: i + gap + horizon, :],
            indexes=(
                x_data_indexes[i - lookback: i],
                y_data_indexes[i + gap: i + gap + horizon],
            ),
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

# def _from_xypred_data(xdata, ydata, y_pred, lookback, horizon, gap=0):
#     """
#     Gets TimePanels from x,y DataFrames and y_pred got from the trained model.
#     This function is only callable by the TimePanel class.

#     Parameters
#     ----------
#     xdata : DataFrame
#         Inputs to the model.
#     ydata : DataFrame
#         Outputs to the model.
#     y_pred: numpy array
#         Data outputted from the trained model.
#     lookback : int
#         How many time steps from the past are considered for the xdata.
#     horizon : int
#         How many time steps to the future are considered for the ydata.
#     gap : int, optional
#         How many time steps are ignored between the lookback and the horizon. The default is 0.

#     Raises
#     ------
#    ValueError
#         If the number of total time steps in the given dataset is not enough to create a TimePair.
#         I.e. the number of time steps has to be greater than or equal to 'lookback + horizon + gap'.

#     Returns
#     -------
#     TimePanel
#         TimePanel object containing all (input,output) TimePairs of te given dataset.

#     """

#     x_timesteps = len(xdata.index)

#     # assert y_index match gap, horizon, lookback with x_index
#     total = x_timesteps - lookback - horizon - gap + 1

#     if total <= 0:
#         raise ValueError("Not enough timesteps to extract X and y.")

#     end = x_timesteps - horizon - gap + 1

#     xdata = pd.DataFrame(xdata)
#     ydata = pd.DataFrame(ydata)

#     # Get Indexes from the dataframes
#     x_data_indexes = list(map(str, xdata.index))
#     y_data_indexes = list(map(str, ydata.index))

#     # Get units and channels
#     xdata = MultiColumn(xdata)
#     ydata = MultiColumn(ydata)
#     xunits = xdata.units
#     yunits = ydata.units
#     xchannels = xdata.channels
#     ychannels = ydata.channels

#     # Convert from dataframe to numpy array
#     X = np.array([xdata[i].values for i in xunits])
#     indexes = np.arange(lookback, end)
#     print(X.shape)
#     print(y_pred.shape)

#     pairs = []

#     for i in indexes:
#         xi = X[:, i - lookback: i, :]
#         print(xi);break
#         # yi = np.array([y_pred[:, unit, :, :] for unit, _ in enumerate(yunits)])
#         # idxi = (x_data_indexes[i - lookback: i],
#         #         y_data_indexes[i + gap: i + gap + horizon],)

#     # pairs = [
#     #     TimePair(
#     #         X[:, i - lookback: i, :],
#     #         np.array([y_pred[:, unit, :, :]
#     #                  for unit, _ in enumerate(yunits)]),
#     #         indexes=(
#     #             x_data_indexes[i - lookback: i],
#     #             y_data_indexes[i + gap: i + gap + horizon],
#     #         ),
#     #         xunits=xunits,
#     #         yunits=yunits,
#     #         xchannels=xchannels,
#     #         ychannels=ychannels,
#     #         lookback=lookback,
#     #         horizon=horizon,
#     #         gap=gap,
#     #     )
#     #     for i in tqdm(indexes)
#     # ]

#     # return TimePanel(pairs)


def from_data(
    data,
    lookback,
    horizon,
    gap=0,
    freq=None,
    xunits=None,
    yunits=None,
    xchannels=None,
    ychannels=None,
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
    X,
    y,
    index,
    xindex,
    yindex,
    lookback,
    horizon,
    gap,
    xunits,
    yunits,
    xchannels,
    ychannels,
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
    yindex = index[lookback + gap:]
    xdata = TimePanel.make_xdata(X, index, xindex, xunits, xchannels)
    ydata = TimePanel.make_ydata(y, index, yindex, yunits, ychannels)
    return from_xy_data(xdata, ydata, lookback, horizon, gap)


class TimePanel:
    """
    This class creates a TimePanel object that contains pairs with
    inputs (xdata) and outputs (ydata).

    """

    # TODO:
    # Check if all forms of initialization match with each other
    # Need tests

    def __init__(self, pairs):

        self.pairs = pairs

        self.first = self.pairs[0]
        self.last = self.pairs[-1]

        self.lookback = self.first.lookback
        self.horizon = self.first.horizon

        self.xunits = self.first.xunits
        self.yunits = self.first.yunits
        self.xchannels = self.first.xchannels
        self.ychannels = self.first.ychannels
        self.gap = self.first.gap

        self.xstart = self.first.xstart
        self.ystart = self.first.ystart
        self.xend = self.last.xend
        self.yend = self.last.yend

        # TODO: either remove or improve find_freq
        self.freq = self.find_freq()

        self.set_arrays()

        self.train_size, self.val_size, self.test_size = None, None, None

    @property
    def dims(self):
        return ["size", "units", "timesteps", "channels"]

    @property
    def xshape(self):
        return pd.Series(self.X.shape, index=self.dims)

    @property
    def yshape(self):
        return pd.Series(self.y.shape, index=self.dims)

    @property
    def shape(self):
        return pd.DataFrame([self.xshape, self.yshape], index=['X', 'y'])

    def _findna(self):
        X_indexes = get_null_indexes(self.X)
        y_indexes = get_null_indexes(self.y)
        return set(X_indexes + y_indexes)

    def dropna(self):
        # TODO: Separate dropna for y and X
        """ Remove pairs with nan values

        Args:
            panel ([type]): [description]

        Returns:
            [type]: [description]
        """ ""

        null_indexes = self._findna()
        new_pairs = [i for idx, i in enumerate(
            self.pairs) if idx not in null_indexes]
        print(len(self.pairs) - len(new_pairs))
        return TimePanel(new_pairs)

    def find_freq(self):
        """
        Finds the xframe frequency.

        Returns
        -------
        string
            Frequency guess representing the seasonality of the data.

        """
        # ! Does not consider y frequency
        df = self.xframe(0)
        if len(df) < 3:
            df = df.append(self.xframe(1)).append(self.xframe(2))
        return freq.infer(df)

    def set_arrays(self):
        """
        Convert all X,y pairs to numpy arrays.

        Returns
        -------
        None.

        """
        # ? Maybe leave as a property?
        self.X = np.array([pair.X for pair in self.pairs])
        self.y = np.array([pair.y for pair in self.pairs])

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
            train_size = len(len(panel) - test_size)

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

    def from_predictions(self, model):
        """
        Gets a new TimePanel with the predicted outputs from the trained model.

        Parameters
        ----------
        model: model
            Trained model.

        Returns
        -------
        TimePanel
            TimePanel object containing all (input,output) TimePairs with the original inputs and predicted outputs.

        """

        ypred = [
            model.predict(self.X[:, unit, :, :]) for unit, _ in enumerate(self.units)
        ]

        return _from_xypred_data(
            self.xdata,
            self.ydata,
            ypred,
            horizon=self.horizon,
            lookback=self.lookback,
            gap=self.gap,
        )

    def xapply(self, func, on="timestamps", new_channel=None):
        """
        Applies a specific function to the TimePanel's xframes.

        Parameters
        ----------
        func : function
            Function to apply in the TimePanel's xframes.
        on :
            Direction of the applied function. If 'timestamps' is inserted,
            the function is applied in the vertical direction. If 'channels'
            is inserted, the function is applied in the horizontal direction.
            If you want to add the new feature to the old panel, please use the
            'add_channel' function.
        new_channel : str
            Name of the new channel after applying the function.
            If 'channels' is selected one must insert a new channel name.

        Returns
        -------
        TimePanel
            New TimePanel after applying the function.

        Examples:
        -------
        Applying on channels

        >>> def min_last10(X,pair):
                return np.min(pair.X[:,-10:,0]).reshape(1,-1)
        >>> new_panel = panel.xapply(min_last10, on='channels', new_channel='min_last10')

        Applying on timestamps

        >>> def last_ones(x):
                return x[:,-1:,:]
        >>> new_panel = panel.xapply(last_ones,on='timestamps')

        """

        pairs = [
            pair.xapply(func=func, on=on, new_channel=new_channel)
            for pair in tqdm(self.pairs)
        ]
        return TimePanel(pairs)

    def yapply(self, func, on="timestamps", new_channel=None):
        """
        Applies a specific function to the TimePanel's yframes.

        Parameters
        ----------
        func : function
            Function to apply in the TimePanel's yframes.
        on :
            Direction of the applied function. If 'timestamps' is inserted,
            the function is applied in the vertical direction. If 'channels'
            is inserted, the function is applied in the horizontal direction.
            If you want to add the new feature to the old panel, please use the
            'add_channel' function.
        new_channel : str
            Name of the new channel after applying the function. If 'channels'
            is selected one must insert a new channel name.

        Returns
        -------
        TimePanel
            New TimePanel after applying the function.

        Examples:
        -------
        Applying on channels

        >>> def mean_last10(y,pair):
                return np.mean(pair.y[:,-10:,0]).reshape(1,-1)
        >>> new_panel = panel.yapply(mean_last10, on='channels', new_channel='mean_last10')

        Applying on timestamps

        >>> new_panel = panel.yapply(np.max, on='timestamps')

        """

        pairs = [
            pair.yapply(func=func, on=on, new_channel=new_channel)
            for pair in tqdm(self.pairs)
        ]
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

    def add_channel(self, new_panel, mode="X"):
        """
        Adds a new channel from another panel to the current panel.

        Parameters
        ----------
        new_panel : TimePanel
            TimePanel with the channel that will be added.
        mode : str
            Select the channel from the xdata or ydata. Options:
            'X', 'y'. The default is "X".

        Raises
        ------
        ValueError
            Mode must be 'X' or 'y'.

        Returns
        -------
        TimePanel
            Current TimePanel with the new channel added.

        Examples
        -------
        >>> new_panel = panel.xapply(np.max, on='timestamps')
        >>> panel = panel.add_channel(new_panel, mode='X')

        """

        if mode not in ["X", "y"]:
            raise ValueError("Mode must be 'X' or 'y'.")
        pairs = [
            pair.add(new_panel[index], mode=mode)
            for index, pair in tqdm(enumerate(self.pairs))
        ]

        return TimePanel(pairs)

    @staticmethod
    def channel_apply_numpy(x, func, ychannels):
        units = np.arange(0, x.shape[0])
        result = [
            func(pd.DataFrame(x[unit, :, :], columns=ychannels)) for unit in units
        ]
        return np.array(result)

    def fillna(self, value=None, method=None):
        # TODO: Make explicit for y and X
        """ Fills the numpy array with parameter value
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
            raise ValueError(
                f"Parameter method must be 'ffill' or 'bfill' but you passed '{method}'."
            )

        if value is not None:
            if isinstance(value, (int, float)):
                def func(x, axis=None): return np.nan_to_num(
                    x.astype(float), nan=value)
            else:
                raise ValueError(
                    f"Parameter value must be int or float. It is {type(value)}."
                )

        elif method == "ffill":
            func = ffill
        elif method == "bfill":
            func = bfill
        return self.xapply(func)

    @staticmethod
    def get_all_unique(array):
        all_ = [i for i in array[0]]
        for i in array[1:]:
            all_.append(i[-1])
        return np.array(all_)

    @staticmethod
    def make_xdata(X, index, xindex, xunits, xchannels):
        X = smash_array(X)
        all_X = TimePanel.get_all_unique(X)
        xdata_ = rebuild_from_index(
            all_X, xindex, xunits, xchannels, to_datetime=True)
        xdata = pd.DataFrame(
            index=index, columns=pd.MultiIndex.from_product(
                [xunits, xchannels])
        )
        xdata.loc[xindex, (xunits, xchannels)] = xdata_.values
        return MultiColumn(xdata)

    @staticmethod
    def make_ydata(y, index, yindex, yunits, ychannels):
        y = smash_array(y)
        all_y = TimePanel.get_all_unique(y)
        ydata_ = rebuild_from_index(
            all_y, yindex, yunits, ychannels, to_datetime=True)
        ydata = pd.DataFrame(
            index=index, columns=pd.MultiIndex.from_product(
                [yunits, ychannels])
        )
        ydata.loc[yindex, (yunits, ychannels)] = ydata_.values
        return MultiColumn(ydata)

    @property
    def xdata(self):
        """
        Returns a dataframe with all the X values.

        Returns
        -------
        Dataframe
            Dataframe with all the X values.

        """
        return self.make_xdata(
            self.X, self.index, self.xindex, self.xunits, self.xchannels
        )

    @property
    def ydata(self):
        """
        Returns a dataframe with all the y values.

        Returns
        -------
        Dataframe
            Dataframe with all the y values.

        """
        return self.make_ydata(
            self.y, self.index, self.yindex, self.yunits, self.ychannels
        )

    @property
    def xindex(self):
        """
        Returns the indexes of the dataframe with the X values.

        Returns
        -------
        list
            List of indexes of the dataframe with the X values.

        """
        x_indexes = np.array([i.indexes[0] for i in self.pairs])
        return [str(i) for i in self.get_all_unique(x_indexes)]

    @property
    def yindex(self):
        """
        Returns the indexes of the dataframe with the y values.

        Returns
        -------
        list
            List of indexes of the dataframe with the y values.

        """
        y_indexes = np.array([i.indexes[1] for i in self.pairs])
        return [str(i) for i in self.get_all_unique(y_indexes)]

    @property
    def index(self):
        """
        Returns the indexes of the dataframe with the X and y values.

        Returns
        -------
        list
            List of indexes of the dataframe with the X and y values.

        """
        return sorted(list(set(self.xindex + self.yindex)))

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
            return self[self.train_size: int(self.train_size + self.val_size)]

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
            return self[self.train_size + self.val_size:]

    def view(self):
        """
        Displays the main infos of the TimePanel.

        Returns
        -------
        None.

        """

        print("TimePanel")

        if self.xunits == self.yunits and self.xchannels == self.ychannels:
            summary = pd.Series(
                {
                    "size": self.__len__(),
                    "lookback": self.lookback,
                    "horizon": self.horizon,
                    "gap": self.gap,
                    "units": self.units,
                    "channels": self.channels,
                    "start": self.xstart,
                    "end": self.yend,
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
                "num_xunits": len(self.xunits),
                "num_yunits": len(self.yunits),
                "num_xchannels": len(self.xchannels),
                "num_ychannels": len(self.ychannels),
                "start": self.xstart,
                "end": self.yend,
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

    def filter(self, xunits=None, xchannels=None, yunits=None, ychannels=None):
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

        pairs = [
            pair.filter(xunits, xchannels, yunits, ychannels)
            for pair in tqdm(self.pairs)
        ]
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
                        X=self.pairs[i]
                        .X[index_unit, :, :]
                        .reshape(1, self.lookback, -1),
                        y=self.pairs[i].y[index_unit, :, :].reshape(
                            1, self.horizon, -1)
                        if yunits
                        else self.pairs[i].y,
                        indexes=self.pairs[i].indexes,
                        xunits=[self.pairs[i].xunits[index_unit]],
                        yunits=[self.pairs[i].yunits[index_unit]]
                        if yunits
                        else self.pairs[i].yunits,
                        xchannels=self.pairs[i].xchannels,
                        ychannels=self.pairs[i].ychannels,
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
        return from_xy_data(
            xdata, ydata, horizon=self.horizon, lookback=self.lookback, gap=self.gap
        )

    def plot_data(self, start=None, end=None, channels=None, units=None, on="xdata"):
        """
        Plots the data according to the given parameters.

        Parameters
        ----------
        start : Timestamp
            First date to be plotted.
        end : Timestamp
            Last date to be plotted.
        channels : str or list of strs
            Channels to be plotted. If none is inserted, all the channels will be plotted.
        units : str or list of strs
            Units to be plotted. If none is inserted, all the units will be plotted.
        on : str
            Data to be plotted, options:
                "xdata",
                "ydata"

        Raises
        ------
        ValueError
            When no start or end date is inserted.

        Returns
        -------
        None.

        """

        if start is None:
            raise ValueError("Must enter the start date!")
        if end is None:
            raise ValueError("Must enter the start date!")

        if channels is None:
            channels = self.channels
        elif isinstance(channels, str):
            channels = [channels]

        if units is None:
            units = self.units
        elif isinstance(units, str):
            units = [units]

        if on == "xdata":
            data = self.xdata[start:end]
        elif on == "ydata":
            data = self.ydata[start:end]
        else:
            raise ValueError("Please select 'xdata' or 'ydata'")

        for unit in units:
            for channel in channels:
                data_aux = data

                data_aux = data_aux[unit][channel]
                indexes = data_aux.index

                fig = px.line(data_aux, x=indexes, y=channel, title=unit)
                fig.show()

    @property
    def channels(self):
        if self.xchannels == self.ychannels:
            return self.xchannels
        else:
            return {"xchannels": self.xchannels, "ychannels": self.ychannels}

    @property
    def units(self):
        if self.xunits == self.yunits:
            return self.xunits
        else:
            return {"xunits": self.xunits, "yunits": self.yunits}

    # TODO: Add better name, not inplace (sub?)
    def xsub(self, X):
        """Replace X changing frames.

        Args:
            X ([type]): [description]

        Returns:
            [type]: [description]
        """""
        pairs = []
        for i, pair in enumerate(self.pairs):
            pair_ = copy(pair)

            # TODO: Replace with setitem
            pair_.X = X[i]
            pairs.append(pair_)
        return TimePanel(pairs)

    # TODO: Add better name, not inplace (sub?)
    def ysub(self, y):
        pairs = []
        for i, pair in enumerate(self.pairs):
            pair_ = copy(pair)

            # TODO: Replace with setitem
            pair_.y = y[i]
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
        index = self.xindex if self.lookback == 1 else self.xindex[: -
                                                                   self.lookback + 1]

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
        start, stop, selection = None, None, None

        if isinstance(key, int):
            selection = self.pairs[key]
            if selection:
                return selection

        elif isinstance(key, str):
            selection = [
                pair
                for pair in self.pairs
                if pd.Timestamp(pair.xstart) == pd.Timestamp(key)
            ]
            if selection:
                return selection[0]  # No xstart repeat

        elif isinstance(key, slice):
            selection = self.pairs
            if isinstance(key.start, pd.Timestamp) or isinstance(
                key.stop, pd.Timestamp
            ):
                if key.start:
                    selection = [
                        pair
                        for pair in selection
                        if pd.Timestamp(pair.xstart) >= key.start
                    ]
                if key.stop:
                    selection = [
                        pair
                        for pair in selection
                        if pd.Timestamp(pair.xstart) < key.stop
                    ]

            elif isinstance(key.start, int) or isinstance(key.stop, int):
                if key.start and key.stop:
                    selection = selection[key.start: key.stop]
                elif key.start:
                    selection = selection[key.start:]
                elif key.stop:
                    selection = selection[: key.stop]

            elif isinstance(key.start, str) or isinstance(key.stop, str):
                if key.start:
                    selection = [
                        pair
                        for pair in selection
                        if pd.Timestamp(pair.xstart) >= pd.Timestamp(key.start)
                    ]
                if key.stop:
                    selection = [
                        pair
                        for pair in selection
                        if pd.Timestamp(pair.xstart) < pd.Timestamp(key.stop)
                    ]

        if selection:
            return TimePanel(selection)
        return

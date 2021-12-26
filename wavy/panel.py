from itertools import compress
from typing import Iterable

import numpy as np
import pandas as pd

from .block import Block
from .pair import TimePair
from .side import Side

from typing import List


def from_pairs(pairs: List):
    """
    Creates a panel from a list of pairs.

    Args:
        pairs (List[TimePair]): List of TimePair

    Returns:
        ``Panel``: Renamed Panel

    Example:

    >>> from_pairs(timepairs)
    size                               1
    lookback                           2
    horizon                            2
    num_xassets                        2
    num_yassets                        2
    num_xchannels                      2
    num_ychannels                      2
    start            2005-12-27 00:00:00
    end              2005-12-30 00:00:00
    Name: Panel, dtype: object
    <Panel, size 1>
    """
    if len(pairs) == 0:
        raise ValueError("Cannot build Panel from empty list")
    blocks = [(pair.x, pair.y) for pair in pairs]
    x = Side([block[0] for block in blocks])
    y = Side([block[1] for block in blocks])
    return Panel(x, y)


def from_xy_data(x, y, lookback:int, horizon:int, gap:int = 0):
    """
    Create a panel from two dataframes.

    Args:
        x (DataFrame): x DataFrame
        y (DataFrame): y DataFrame
        lookback (int): lookback size
        horizont (int): horizont size
        gap (int): gap between x and y

    Returns:
        ``Panel``: Renamed Panel

    Example:

    >>> from_xy_data(x, y, 5, 5, 0)
    size                               1
    lookback                           2
    horizon                            2
    num_xassets                        2
    num_yassets                        2
    num_xchannels                      2
    num_ychannels                      2
    start            2005-12-27 00:00:00
    end              2005-12-30 00:00:00
    Name: Panel, dtype: object
    <Panel, size 1>
    """

    x_timesteps = len(x.index)

    if x_timesteps - lookback - horizon - gap <= -1:
        raise ValueError("Not enough timesteps to build")

    end = x_timesteps - horizon - gap + 1

    # Convert to blocks
    x = Block(x)
    y = Block(y)

    indexes = np.arange(lookback, end)
    xblocks, yblocks = [], []

    for i in indexes:
        xblocks.append(x.iloc[i - lookback : i])
        yblocks.append(y.iloc[i + gap : i + gap + horizon])
    return Panel(Side(xblocks), Side(yblocks), gap=gap)


def from_data(df, lookback:int, horizon:int, gap:int = 0, x_assets: List[str] = None, y_assets: List[str] = None, x_channels: List[str] = None, y_channels: List[str] = None, assets: List[str] = None, channels: List[str] = None):
    """
    Create a panel from a dataframe.

    Args:
        df (DataFrame): Values DataFrame
        lookback (int): lookback size
        horizont (int): horizont size
        gap (int): gap between x and y
        x_assets (list): List of x assets
        y_assets (list): List of y assets
        x_channels (list): List of x channels
        y_channels (list): List of y channels
        assets (list): List of assets
        channels (list): List of channels

    Returns:
        ``Panel``: Renamed Panel

    Example:

    >>> from_data(df, 5, 5, 0)
    size                               1
    lookback                           2
    horizon                            2
    num_xassets                        2
    num_yassets                        2
    num_xchannels                      2
    num_ychannels                      2
    start            2005-12-27 00:00:00
    end              2005-12-30 00:00:00
    Name: Panel, dtype: object
    <Panel, size 1>
    """

    if assets:
        x_assets, y_assets = assets, assets
    if channels:
        x_channels, y_channels = channels, channels

    df = Block(df)

    if df.T.index.nlevels == 1:
        df = df.add_level('asset')

    xdata = df.filter(x_assets, x_channels)
    ydata = df.filter(y_assets, y_channels)
    return from_xy_data(xdata, ydata, lookback, horizon, gap)


class Panel:

    _DIMS = ("size", "assets", "timesteps", "channels")

    def __init__(self, x, y, gap=0):

        class _IXIndexer:
            def __getitem__(self, item):
                return Panel(x.ix[item], y.ix[item])
        class _iLocIndexer:
            def __getitem__(self, item):
                
                return Panel(x.iloc[item], y.iloc[item])
        class _LocIndexer:
            def __getitem__(self, item):
                return Panel(x.loc[item], y.loc[item])
        class _AtIndexer:
            def __getitem__(self, item):
                return Panel(x.at[item], y.at[item])
        class _iAtIndexer:
            def __getitem__(self, item):
                return Panel(x.iat[item], y.iat[item])

        self._x, self._y = x, y
        self.gap = gap
        self.ix = _IXIndexer()
        self.iloc = _iLocIndexer()
        self.loc = _LocIndexer()
        self.at = _AtIndexer()
        self.iat = _iAtIndexer()
        self.set_training_split()

    def __len__(self):
        return len(self.pairs)

    # TODO implement returning a Panel with only the key element
    def __getitem__(self, key):
        if isinstance(key, slice):
            # x = self.x[key]
            # y = self.y[key]

            return Panel(Side(self.x[key]), Side(self.y[key]))
            #     key_set = set(key)
            #     if key_set == {False, True}:
            #         pairs = list(compress(self.pairs, key))
            #     else:
            #         pairs = [pair for i, pair in enumerate(self.pairs) if i in key_set]
            # else:
        elif isinstance(key, int):
            return Panel(Side([self.x[key]]), Side([self.y[key]]))

        # return from_pairs(pairs)
        # return Panel(Side(xblocks), Side(yblocks), x, y)

    # TODO getter and setter for full_x and full_y

    @property
    def x(self):
        """
        Side with x Blocks.

        Returns:
            ``Side``: Side with x Blocks
        """
        return self._x

    @property
    def y(self):
        """
        Side with y Blocks.

        Returns:
            ``Side``: Side with y Blocks
        """
        return self._y

    @x.setter
    def x(self, value):
        """
        Set x with Side.
        """
        if not isinstance(value, Side):
            print(type(value))
            raise ValueError(f"'x' must be of type Side, it is {type(value)}")
        if len(value) != len(self.x):
            raise ValueError("'x' must keep the same length")
        if len({len(block) for block in value.blocks}) != 1:
            raise ValueError("'x' blocks must have the same length")
        self._x = value

    @y.setter
    def y(self, value):
        """
        Set y with Side.
        """
        if not isinstance(value, Side):
            raise ValueError("'y' must be of type Side")
        if len(value) != len(self.y):
            raise ValueError("'y' must keep the same length")
        if len({len(block) for block in value.blocks}) != 1:
            raise ValueError("'y' blocks must have the same length")
        self._y = value

    @property
    def pairs(self):
        """
        List of TimePairs.

        Returns:
            ``List[TimePair]``: List of TimePair
        """
        return [TimePair(x, y) for x, y in zip(self.x.blocks, self.y.blocks)]

    @property
    def lookback(self):
        """
        Lookback size value.

        Returns:
            ``int``: Lookback size value
        """
        return len(self.x.first)

    @property
    def horizon(self):
        """
        Horizon size value.

        Returns:
            ``int``: Horizon size value
        """
        return len(self.y.first)

    # Could return pairs
    # TODO first
    # TODO last

    @property
    def start(self):
        """
        Panel first index.

        Example:

        >>> panel.start
        Timestamp('2005-12-21 00:00:00')
        """
        return self.x.start

    @property
    def end(self):
        """
        Panel last index.

        Example:

        >>> panel.end
        Timestamp('2005-12-21 00:00:00')
        """
        return self.y.end

    @property
    def assets(self):
        """
        Panel assets.

        Example:

        >>> panel.assets
        0    AAPL
        1    MSFT
        dtype: object
        """
        return self.x.first.assets

    @property
    def channels(self):
        """
        Panel channels.

        Example:

        >>> panel.channels
        0    Open
        1    Close
        dtype: object
        """
        return self.x.first.channels

    @property
    def timesteps(self):
        """
        Panel timesteps.

        Example:

        >>> panel.timesteps
        [Timestamp('2005-12-27 00:00:00'),
         Timestamp('2005-12-28 00:00:00'),
         Timestamp('2005-12-29 00:00:00'),
         Timestamp('2005-12-30 00:00:00')]
        """
        # The same as the index
        return self.index

    @property
    def index(self):
        """
        Panel index.

        Example:

        >>> panel.index
        [Timestamp('2005-12-27 00:00:00'),
         Timestamp('2005-12-28 00:00:00'),
         Timestamp('2005-12-29 00:00:00'),
         Timestamp('2005-12-30 00:00:00')]
        """
        return sorted(list(set(list(self.x.index) + list(self.y.index))))

    @property
    def shape(self):
        """
        Panel shape.

        Example:

        >>> panel.shape
           size  assets  timesteps  channels
        x     1       2          2         2
        y     1       2          2         2
        """
        return pd.DataFrame([self.x.shape, self.y.shape], index=["x", "y"], columns=self._DIMS)

    # TODO tensor4d
    # TODO tensor3d

    def filter(self, assets: List[str] = None, channels: List[str] = None):
        """
        Panel subset according to the specified assets and channels.

        Args:
            assets (list): List of assets
            channels (list): List of channels

        Returns:
            ``Panel``: Filtered Panel
        """
        x = self.x.filter(assets=assets, channels=channels)
        y = self.y.filter(assets=assets, channels=channels)
        return Panel(x, y)

    def drop(self, assets=None, channels=None):
        """
        Subset of the Panel columns discarding the specified assets and channels.

        Args:
            assets (list): List of assets
            channels (list): List of channels

        Returns:
            ``Panel``: Filtered Panel
        """
        x = self.x.drop(assets=assets, channels=channels)
        y = self.y.drop(assets=assets, channels=channels)
        return Panel(x, y)

    def rename_assets(self, dict: dict):
        """
        Rename asset labels.

        Args:
            dict (dict): Dictionary with assets to rename

        Returns:
            ``Panel``: Renamed Panel
        """
        x = self.x.rename_assets(dict=dict)
        y = self.y.rename_assets(dict=dict)
        return Panel(x, y)

    def rename_channels(self, dict: dict):
        """
        Rename channel labels.

        Args:
            dict (dict): Dictionary with channels to rename

        Returns:
            ``Panel``: Renamed Panel
        """
        x = self.x.rename_channels(dict=dict)
        y = self.y.rename_channels(dict=dict)
        return Panel(x, y)

    def apply(self, func, axis):
        """
        Apply a function along an axis of the DataBlock.

        Args:
            func (function): Function to apply to each column or row.
            on (str, default 'row'): Axis along which the function is applied:

                * 'timestamps': apply function to each timestamps.
                * 'channels': apply function to each channels.

        Returns:
            ``Panel``: Result of applying `func` along the given axis of the Panel.
        """
        x = self.x.apply(func=func, axis=axis)
        y = self.y.apply(func=func, axis=axis)
        return Panel(x, y)

    def update(self, values=None, index: List = None, assets: List = None, channels: List = None):
        """
        Update function for any of Panel properties.

        Args:
            values (ndarray): New values Dataframe.
            index (list): New list of index.
            assets (list): New list of assets
            channels (list): New list of channels

        Returns:
            ``Panel``: Result of updated Panel.
        """
        x = Side([block.update(values[i][0], index, assets, channels) for i, block in enumerate(self.x)])
        y = Side([block.update(values[i][1], index, assets, channels) for i, block in enumerate(self.y)])
        return Panel(x, y)

    def sort_assets(self, order: List[str] = None):
        """
        Sort assets in alphabetical order.

        Args:
            order (List[str]): Asset order to be sorted.

        Returns:
            ``Panel``: Result of sorting assets.
        """
        x = self.x.sort_assets(order=order)
        y = self.y.sort_assets(order=order)
        return Panel(x, y)

    def sort_channels(self, order: List[str] = None):
        """
        Sort channels in alphabetical order.

        Args:
            order (List[str]): Channel order to be sorted.

        Returns:
            ``Panel``: Result of sorting channels.
        """
        x = self.x.sort_channels(order=order)
        y = self.y.sort_channels(order=order)
        return Panel(x, y)

    def swap_cols(self):
        """
        Swap columns levels, assets becomes channels and channels becomes assets

        Returns:
            ``Panel``: Result of swapping columns.
        """
        x = self.x.swap_cols()
        y = self.y.swap_cols()
        return Panel(x, y)

    def countna(self):
        """
        Count NA/NaN cells for each Panel.

        Returns:
            ``Panel``: NaN count for each Panel.
        """
        values = self.x.countna().values + self.y.countna().values
        return pd.DataFrame(values, index=range(len(self.x.blocks)), columns=['nan'])

    def fillna(self, value=None, method: str = None):
        """
        Fill NA/NaN values using the specified method.

        Returns:
            ``Panel``: Panel with missing values filled.
        """
        x = self.x.fillna(value=value, method=method)
        y = self.y.fillna(value=value, method=method)
        return Panel(x, y)

    def dropna(self, x=True, y=True):
        """
        Drop pairs with missing values from the panel.

        Returns:
            ``Panel``: Panel with missing values dropped.
        """
        nan_values = self.findna()
        idx = {i for i in range(len(self)) if i not in nan_values}
        if not idx:
            raise ValueError("'dropna' would create empty Panel")
        return self[idx]

    def findna(self, x=True, y=True):
        """
        Find NA/NaN values index.

        Returns:
            ``List``: List with index of missing values.
        """
        x_nan = self.x.findna() if x else []
        y_nan = self.y.findna() if y else []
        return list(set(x_nan + y_nan))

    def __repr__(self):
        summary = pd.Series(
            {
                "size": self.__len__(),
                "lookback": self.lookback,
                "horizon": self.horizon,
                "gap": self.gap,
                "num_xassets": len(self.x.assets),
                "num_yassets": len(self.y.assets),
                "num_xchannels": len(self.x.channels),
                "num_ychannels": len(self.y.channels),
                "start": self.x.start,
                "end": self.y.end,
            },
            name="Panel",
        )

        print(summary)
        return f"<Panel, size {self.__len__()}>"

    def set_training_split(self, val_size=0.2, test_size=0.1):
        """
        Time series split into training, validation, and test sets, avoiding data leakage.
        Splits the panel in training, validation, and test panels, accessed with the properties
        .train, .val and .test. The sum of the three sizes inserted must equals one.

        Args:
            val_size (float): Percentage of data used for the validation set.
            test_size (float): Percentage of data used for the test set.

        Returns:
            ``DataBlock``: New panel with the pairs split into training, validation,
            and test sets. To use each set, one must access the properties .train,
            .val and .test.


        Example:

        >>> panel.set_training_split(val_size=0.2, test_size=0.1)
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
        Returns the Panel with the pairs of the training set, according to
        the parameters given in the 'set_train_val_test_sets' function.

        Returns:
            ``Panel``: Panel with the pairs of the training set.
        """
        if self.train_size:
            return self[: self.train_size]

    @property
    def val(self):
        """
        Returns the Panel with the pairs of the validation set, according to
        the parameters given in the 'set_train_val_test_sets' function.

        Returns:
            ``Panel``: Panel with the pairs of the validation set.

        """
        if self.val_size and self.train_size:
            return self[self.train_size : int(self.train_size + self.val_size)]

    @property
    def test(self):
        """
        Returns the Panel with the pairs of the testing set, according to
        the parameters given in the 'set_train_val_test_sets' function.

        Returns:
            ``Panel``: Panel with the pairs of the testing set.

        """
        if self.val_size and self.train_size:
            return self[self.train_size + self.val_size :]


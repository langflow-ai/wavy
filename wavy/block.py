import functools
from collections import OrderedDict
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from typing import List
from .utils import add_dim, add_level


def from_dataframe(df: DataFrame, asset: str = 'asset'):
    """
    Generate TimeBlock from DataFrame

    Args:
        df (DataFrame): Pandas DataFrame
        asset (string): Asset name

    Returns:
        ``TimeBlock``: Constructed TimeBlock

    Example:

    >>> data
                    Open     Close
    Date
    2005-12-21  2.218566  2.246069
    2005-12-22  2.258598  2.261960

    >>> timeblock = from_dataframe(data, 'AAPL')
                    AAPL
                    Open     Close
    Date
    2005-12-21  2.218566  2.246069
    2005-12-22  2.258598  2.261960

    >>> type(timeblock)
    wavy.block.TimeBlock

    """

    # Add level if level equals to 1
    if df.T.index.nlevels == 1:
        df = add_level(df, asset)

    # Create TimeBlock
    tb = TimeBlock(
            pd.DataFrame(
                df.values,
                index=df.index,
                columns=df.columns,
                )
            )

    return tb


def from_dict(data: dict):
    """
    Generate TimeBlock from dictionary

    Args:
        data ({str: DataFrame}): Dictionary containing asset name and DataFrame

    Returns:
        ``TimeBlock``: Constructed TimeBlock

    Example:

    >>> dict
    {'AAPL':                 Open     Close
    Date
    2005-12-21  2.218566  2.246069
    2005-12-22  2.258598  2.261960,
    'MSFT':                  Open      Close
    Date
    2005-12-21  19.577126  19.475122
    2005-12-22  19.460543  19.373114}

    >>> from_dict(dict)
                    AAPL                 MSFT
                    Open     Close       Open      Close
    Date
    2005-12-21  2.218566  2.246069  19.577126  19.475122
    2005-12-22  2.258598  2.261960  19.460543  19.373114
    """

    previous_channels = None
    for _, value in data.items():
        assert isinstance(value, DataFrame), 'Data must be a DataFrame'

        # ? Remove upper level if data is multilevel
        assert value.T.index.nlevels == 1, 'Data cannot be multilevel'

        channels_flag = value.shape[1] == previous_channels if previous_channels else True
        assert channels_flag, 'Data with different number of channels'
        previous_channels = value.shape[1]

    return TimeBlock(pd.concat(data.values(), axis=1, keys=data.keys()))


def from_dataframes(data: List[DataFrame], assets: List[str] = None):
    """
    Generate a TimeBlock from a list of dataframes. Each dataframe becomes one asset.

    Args:
        data (list): List of dataframes
        assets (list): List of assets

    Returns:
        ``TimeBlock``: Constructed TimeBlock

    Example:

    >>> aapl
                    Open     Close
    Date
    2005-12-21  2.218566  2.246069
    2005-12-22  2.258598  2.261960

    >>> msft
                     Open      Close
    Date
    2005-12-21  19.577126  19.475122
    2005-12-22  19.460543  19.373114

    Generating TimeBlock with list of dataframes

    >>> from_dataframes([aapl, msft])
             asset_0              asset_1
                    Open     Close       Open      Close
    Date
    2005-12-21  2.218566  2.246069  19.577126  19.475122
    2005-12-22  2.258598  2.261960  19.460543  19.373114


    Generating TimeBlock with list of dataframes and assets

    >>> from_dataframes([aapl, msft], ['AAPL', 'MSFT'])
                AAPL                 MSFT
                    Open     Close       Open      Close
    Date
    2005-12-21  2.218566  2.246069  19.577126  19.475122
    2005-12-22  2.258598  2.261960  19.460543  19.373114
    """
    if assets:
        dict = {assets[k]: v for k, v in enumerate(data)}
    else:
        dict = {"asset_" + str(k): v for k, v in enumerate(data)}

    return from_dict(dict)


def from_tensor(values, index: List = None, assets: List[str] = None, channels: List[str] = None):
    """
    Generate a TimeBlock from list of attributes.

    Args:
        values (ndarray): Dataframes of size (assets x index x channels)
        index (list): List of index
        assets (list): List of assets
        channels (list): List of channels

    Returns:
        ``TimeBlock``: Constructed TimeBlock

    Example:

    >>> values
    array([[[ 2.21856582,  2.24606872],
            [ 2.25859845,  2.26195979]],
           [[19.57712554, 19.47512245],
            [19.46054323, 19.37311363]]])

    >>> index
    DatetimeIndex(['2005-12-21', '2005-12-22'], dtype='datetime64[ns]', name='Date', freq=None)

    >>> assets
    Index(['AAPL', 'MSFT'], dtype='object')

    >>> from_tensor(values, index=index, assets=assets, channels=channels)
                    AAPL                 MSFT
                    Open     Close       Open      Close
    Date
    2005-12-21  2.218566  2.246069  19.577126  19.475122
    2005-12-22  2.258598  2.261960  19.460543  19.373114
    """

    values = np.concatenate(values, axis=1)

    return from_matrix(values, index=index, assets=assets, channels=channels)


def from_matrix(values, index: List = None, assets: List[str] = None, channels: List[str] = None):
    """
    Generate a TimeBlock from list of attributes.

    Args:
        values (ndarray): Dataframes of size (index x [assets * channels])
        index (list): List of index
        assets (list): List of assets
        channels (list): List of channels

    Returns:
        ``TimeBlock``: Constructed TimeBlock

    Example:

    >>> values
    array([[ 2.21856582,  2.24606872, 19.57712554, 19.47512245],
           [ 2.25859845,  2.26195979, 19.46054323, 19.37311363]])

    >>> index
    DatetimeIndex(['2005-12-21', '2005-12-22'], dtype='datetime64[ns]', name='Date', freq=None)

    >>> assets
    Index(['AAPL', 'MSFT'], dtype='object')

    >>> from_matrix(values, index=index, assets=assets, channels=channels)
                    AAPL                 MSFT
                    Open     Close       Open      Close
    Date
    2005-12-21  2.218566  2.246069  19.577126  19.475122
    2005-12-22  2.258598  2.261960  19.460543  19.373114
    """

    values = add_dim(values, n = 3 - len(values.shape))
    if assets is None:
        assets = range(values.shape[0])
    if index is None:
        index = range(values.shape[1])
    if channels is None:
        channels = range(values.shape[2])

    columns = pd.MultiIndex.from_product([assets, channels])
    df = pd.DataFrame(index=index, columns=columns)
    df.loc[:, (slice(None), slice(None))] = values.reshape(df.shape)
    return TimeBlock(df)


def _rebuild(func):
    # Avoid problem with frozen list from pandas
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        df = func(*args, **kwargs)
        return from_matrix(df.values, df.index, df.assets, df.channels)

    return wrapper


class _AssetSeries(pd.Series):
    def __init__(self, df, *args, **kwargs):
        super().__init__(df, *args, **kwargs)

    @property
    def _constructor_expanddim(self):
        return TimeBlock

    @property
    def _constructor(self):
        return _AssetSeries


class TimeBlock(pd.DataFrame):

    def __init__(self, df, *args, **kwargs):
        super().__init__(df, *args, **kwargs)

    @property
    def _constructor(self):
        return TimeBlock

    @property
    def _constructor_sliced(self):
        return _AssetSeries

    @property
    def start(self):
        """
        TimeBlock first index.

        Example:

        >>> timeblock
                        AAPL                 MSFT
                        Open     Close       Open      Close
        Date
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> timeblock.start
        Timestamp('2005-12-21 00:00:00')
        """
        return self.index[0]

    @property
    def end(self):
        """
        TimeBlock last index.

        Example:

        >>> timeblock
                        AAPL                 MSFT
                        Open     Close       Open      Close
        Date
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> timeblock.end
        Timestamp('2005-12-22 00:00:00')
        """
        return self.index[-1]

    @property
    def assets(self):
        """
        TimeBlock assets.

        Example:

        >>> timeblock
                        AAPL                 MSFT
                        Open     Close       Open      Close
        Date
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> timeblock.assets
        0    AAPL
        1    MSFT
        dtype: object
        """
        assets = [col[0] for col in self.columns]
        # OrderedDict to keep order
        # ? Is it correct to  order, what happens in case the user wants to _rebuild the block?
        return pd.Series(tuple(OrderedDict.fromkeys(assets)))

    @property
    def channels(self):
        """
        TimeBlock channels.

        Example:

        >>> timeblock
                        AAPL                 MSFT
                        Open     Close       Open      Close
        Date
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> timeblock.channels
        0    Open
        1    Close
        dtype: object
        """
        channels = [col[1] for col in self.columns]
        # OrderedDict to keep order
        # ? Is it correct to  order, what happens in case the user wants to _rebuild the block?
        return pd.Series(list(OrderedDict.fromkeys(channels)))

    @property
    def timesteps(self):
        """
        TimeBlock timesteps.

        Example:

        >>> timeblock.timesteps
        DatetimeIndex(['2005-12-21', '2005-12-22', '2005-12-23'], dtype='datetime64[ns]', name='Date', freq=None)
        """
        # The same as the index
        return self.index

    # Causing error, overwriting shape function
    # @property
    # def shape(self):
    #     """
    #     TimeBlock shape.

    #     Example:

    #     >>> timeblock.shape
    #     (2, 2, 2)
    #     """
    #     return self.tensor.shape

    @property
    def tensor(self):
        """
        3D matrix with DataBlock value.

        Example:

        >>> timeblock
                        AAPL                 MSFT
                        Open     Close       Open      Close
        Date
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> timeblock.tensor
        array([[[ 2.21856582,  2.24606872],
                [ 2.25859845,  2.26195979]],
               [[19.57712554, 19.47512245],
                [19.46054323, 19.37311363]]])
        """
        new_shape = (len(self), len(self.assets), len(self.channels))
        values = self.values.reshape(*new_shape)
        return values.transpose(1, 0, 2)

    @property
    def matrix(self):
        """
        2D matrix with DataBlock value.

        Example:

        >>> timeblock
                        AAPL                 MSFT
                        Open     Close       Open      Close
        Date
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> timeblock.matrix
        array([[ 2.21856582,  2.24606872, 19.57712554, 19.47512245],
            [ 2.25859845,  2.26195979, 19.46054323, 19.37311363]])
        """
        return self.values

    def filter(self, assets: List[str] = None, channels: List[str] = None):
        """
        TimeBlock subset according to the specified assets and channels.

        Args:
            assets (list): List of assets
            channels (list): List of channels

        Returns:
            ``TimeBlock``: Filtered TimeBlock

        Example:

        >>> timeblock
                        AAPL                 MSFT
                        Open     Close       Open      Close
        Date
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> timeblock.filter(assets=['AAPL'], channels=['Open'])
                        AAPL
                        Open
        Date
        2005-12-21  2.218566
        2005-12-22  2.258598
        """
        filtered = self._filter_assets(assets)
        filtered = filtered._filter_channels(channels)
        return filtered

    @_rebuild
    def _filter_assets(self, assets):
        if type(assets) == str:
            assets = [assets]

        # TODO improve speed

        if assets is not None and any(asset not in self.assets for asset in assets):
            raise ValueError(f"{assets} not found in columns. Columns: {list(self.assets)}")

        return self.loc[:, (assets, slice(None))] if assets else self

    @_rebuild
    def _filter_channels(self, channels):
        if type(channels) == str:
            channels = [channels]

        # TODO improve speed

        if channels is not None and any(channel not in self.channels for channel in channels):
            raise ValueError(f"{channels} not found in columns. Columns: {list(self.channels)}")

        return self.loc[:, (slice(None), channels)][self.assets] if channels else self

    def drop(self, assets: List[str] = None, channels: List[str] = None):
        """
        Subset of the dataframe columns discarding the specified assets and channels.

        Args:
            assets (list): List of assets
            channels (list): List of channels

        Returns:
            ``TimeBlock``: Filtered TimeBlock

        Example:

        >>> timeblock
                        AAPL                 MSFT
                        Open     Close       Open      Close
        Date
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> timeblock.drop(assets=['AAPL'], channels=['Open'])
                         MSFT
                        Close
        Date
        2005-12-21  19.475122
        2005-12-22  19.373114
        """
        filtered = self._drop_assets(assets)
        filtered = filtered._drop_channels(channels)
        return filtered

    @_rebuild
    def _drop_assets(self, assets):
        if isinstance(assets, str):
            assets = [assets]
        new_assets = [u for u in self.assets if u not in assets]
        return self._filter_assets(new_assets)

    @_rebuild
    def _drop_channels(self, channels):
        if isinstance(channels, str):
            channels = [channels]
        new_channels = [c for c in self.channels if c not in channels]
        return self._filter_channels(new_channels)

    def rename_assets(self, dict: dict):
        """
        Rename asset labels.

        Args:
            dict (dict): Dictionary with assets to rename

        Returns:
            ``TimeBlock``: Renamed TimeBlock

        Example:

        >>> timeblock
                        AAPL                 MSFT
                        Open     Close       Open      Close
        Date
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> timeblock.rename_assets({'AAPL': 'Apple', 'MSFT': 'Microsoft'})
                       Apple            Microsoft
                        Open     Close       Open      Close
        Date
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114
        """

        values = dict.keys()
        new_values = dict.values()

        assets = self.assets.replace(to_replace=values, value=new_values)
        return self.update(assets=assets.values)

    def rename_channels(self, dict: dict):
        """
        Rename channel labels.

        Args:
            dict (dict): Dictionary with channels to rename

        Returns:
            ``TimeBlock``: Renamed TimeBlock

        Example:

        >>> timeblock
                        AAPL                 MSFT
                        Open     Close       Open      Close
        Date
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> timeblock.rename_channels({'Open': 'Op', 'Close': 'Cl'})
                        AAPL                 MSFT
                        Op        Cl         Op         Cl
        Date
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114
        """
        values = dict.keys()
        new_values = dict.values()

        channels = self.channels.replace(to_replace=values, value=new_values)
        return self.update(channels=channels.values)

    def apply(self, func, on: str = 'timestamps'):
        """
        Apply a function along an axis of the DataBlock.

        Args:
            func (function): Function to apply to each column or row.
            on (str, default 'row'): Axis along which the function is applied:

                * 'timestamps': apply function to each timestamps.
                * 'channels': apply function to each channels.

        Returns:
            ``TimeBlock``: Result of applying `func` along the given axis of the TimeBlock.

        Example:

        >>> timeblock
                        AAPL                 MSFT
                        Open     Close       Open      Close
        Date
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> timeblock.apply(np.max, on='rows')
            AAPL                MSFT
            Open    Close       Open      Close
        0  2.258598  2.26196  19.577126  19.475122

        >>> timeblock.apply(np.max, on='columns')
                        AAPL       MSFT
                        amax       amax
        Date
        2005-12-21  2.246069  19.577126
        2005-12-22  2.261960  19.460543
        """

        if on == 'timestamps ':
            return self._timestamp_apply(func)
        elif on == 'channels':
            return self._channel_apply(func)

        raise ValueError(f"{axis} not acceptable for 'axis'. Available values are [0, 1]")

    def _timestamp_apply(self, func):
        df = self.as_dataframe().apply(func, axis=0)
        if isinstance(df, pd.Series):
            return df.to_frame().T
        return df.T

    def _channel_apply(self, func):
        splits = self.split_assets()
        new = from_matrix(np.swapaxes(np.array([asset.as_dataframe().apply(func, axis=1).values for asset in splits]), 0,1), index=self.index, assets=self.assets, channels=[func.__name__])
        return new

    def update(self, values=None, index: List = None, assets: List = None, channels: List = None):
        """
        Update function for any of DataBlock properties.

        Args:
            values (ndarray): New values Dataframe.
            index (list): New list of index.
            assets (list): New list of assets
            channels (list): New list of channels

        Returns:
            ``TimeBlock``: Result of updated TimeBlock.

        Example:

        >>> timeblock
                        AAPL                 MSFT
                        Open     Close       Open      Close
        Date
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> timeblock.update(assets=['Microsoft', 'Apple'], channels=['Op', 'Cl'])
                 Microsoft                 Apple           
                        Op         Cl         Op         Cl
        Date                                                  
        2005-12-21  19.577126  19.475122  19.460543  19.373114
        2005-12-22   2.218566   2.246069   2.258598   2.261960
        """
        assets = assets if assets is not None else self.assets
        index = index if index is not None else self.index
        channels = channels if channels is not None else self.channels
        values = values if values is not None else self.matrix

        if values is not None:
            if len(values.shape) == 3:
                db = from_tensor(values, index, assets, channels)
            elif len(values.shape) == 2:
                db = from_matrix(values, index, assets, channels)
        return db

    def split_assets(self):
        """
        Split DataBlock into assets.

        Returns:
            ``List``: List of DataBlock, each one being one asset.

        Example:

        >>> timeblock
                        AAPL                 MSFT
                        Open     Close       Open      Close
        Date
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> timeblock.split_assets()
        [               AAPL
                        Open     Close
        Date
        2005-12-21  2.218566  2.246069
        2005-12-22  2.258598  2.261960,
                        MSFT
                        Open      Close
        Date
        2005-12-21  19.577126  19.475122
        2005-12-22  19.460543  19.373114]
        """
        return [self.filter(asset) for asset in self.assets]

    def sort_assets(self, order: List[str] = None):
        """
        Sort assets in alphabetical order.

        Args:
            order (List[str]): Asset order to be sorted.

        Returns:
            ``DataBlock``: Result of sorting assets.

        Example:

        >>> timeblock
                        MSFT                 AAPL
                        Open      Close      Open     Close
        Date
        2005-12-21  19.577126  19.475122  2.218566  2.246069
        2005-12-22  19.460543  19.373114  2.258598  2.261960

        >>> timeblock.sort_assets()
                        AAPL                 MSFT
                        Open     Close       Open      Close
        Date
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114
        """
        assets = sorted(self.assets) if order is None else order
        channels = self.channels
        pair = [(asset, channel) for asset in assets for channel in channels]
        return self.reindex(pair, axis=1)

    def sort_channels(self, order: List[str] = None):
        """
        Sort channels in alphabetical order.

        Args:
            order (List[str]): Channel order to be sorted.

        Returns:
            ``DataBlock``: Result of sorting channels.

        Example:

        >>> timeblock
                        MSFT                 AAPL
                        Open      Close      Open     Close
        Date
        2005-12-21  19.577126  19.475122  2.218566  2.246069
        2005-12-22  19.460543  19.373114  2.258598  2.261960

        >>> timeblock.sort_channels()
                         MSFT                 AAPL
                        Close       Open     Close      Open
        Date
        2005-12-21  19.475122  19.577126  2.246069  2.218566
        2005-12-22  19.373114  19.460543  2.261960  2.258598
        """
        assets = self.assets
        channels = sorted(self.channels) if order is None else order
        pair = [(asset, channel) for asset in assets for channel in channels]
        return self.reindex(pair, axis=1)

    def swap_cols(self):
        """
        Swap columns levels, assets becomes channels and channels becomes assets

        Returns:
            ``DataBlock``: Result of swapping columns.

        Example:

        >>> timeblock
                        AAPL                 MSFT
                        Open     Close       Open      Close
        Date
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> timeblock.swap_cols()
                       Close                 Open
                        AAPL       MSFT      AAPL       MSFT
        Date
        2005-12-21  2.246069  19.475122  2.218566  19.577126
        2005-12-22  2.261960  19.373114  2.258598  19.460543
        """
        channels = self.channels
        return self.T.swaplevel(i=- 2, j=- 1, axis=0).T.sort_assets(channels)

    def countna(self, type: str = 'asset'):
        """
        Count NA/NaN cells for each asset or channel.

        Returns:
            ``List``: List of NaN count.

        Example:

        >>> timeblock
                        AAPL                 MSFT
                        Open     Close       Open      Close
        Date
        2005-12-21  2.218566  2.246069        NaN  19.475122
        2005-12-22  2.258598       NaN  19.460543  19.373114

        >>> timeblock.countna('asset')
        AAPL    1
        MSFT    1
        dtype: int64

        >>> timeblock.countna('channel')
        AAPL  Open     0
              Close    1
        MSFT  Open     1
              Close    0
        dtype: int64
        """
        if type == 'asset':
            s = pd.Series(dtype=int)
            for asset in self.assets:
                s[asset] = len(self[asset]) - len(self[asset].dropna())
        elif type == 'channel':
            s = self.isnull().sum(axis = 0)
        return s

    def as_dataframe(self):
        """
        Generate DataFrame from TimeBlock

        Returns:
            ``DataFrame``: Constructed DataFrame

        Example:

        >>> type(timeblock)
        wavy.block.TimeBlock

        >>> type(timeblock.as_dataframe())
        pandas.core.frame.DataFrame
        """
        return pd.DataFrame(self.values, index=self.index, columns=self.columns)

    def fillna(self, value=None, method=None, axis=None, inplace=False, limit=None, downcast=None):
        # TODO link pandas documentation (Similar/Inherited to pandas.dataframe.fillna)
        """
        Fill NA/NaN values using the specified method.

        Returns:
            ``DataBlock``: DataBlock with missing values filled.

        Example:

        >>> timeblock
                        AAPL                 MSFT
                        Open     Close       Open      Close
        Date
        2005-12-21  2.218566  2.246069        NaN  19.475122
        2005-12-22  2.258598       NaN  19.460543  19.373114

        >>> timeblock.fillna(0)
                        MSFT                 AAPL          
                        Open      Close      Open     Close
        Date                                                
        2005-12-21  19.577126  19.475122  0.000000  2.246069
        2005-12-22  19.460543   0.000000  2.258598  2.261960
        """
        return super().fillna(value, method, axis, inplace, limit, downcast)

    # TODO Not implemented error for all pandas functions not used in wavy

    # TODO dropna

    # TODO add findna???
    # TODO add_channel???
    # TODO flat???
    # TODO flatten???
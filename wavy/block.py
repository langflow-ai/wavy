import functools
from collections import OrderedDict
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from typing import List, overload
from .utils import add_dim, add_level
from multipledispatch import dispatch


def from_dataframe(df: DataFrame, asset: str ='asset'):
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

    >>> datablock = from_dataframe(data, 'AAPL')
                    AAPL          
                    Open     Close
    Date                          
    2005-12-21  2.218566  2.246069
    2005-12-22  2.258598  2.261960

    >>> type(datablock)
    wavy.block.TimeBlock

    """
    # Recreate columns to avoid pandas issue
    # avoid problem
    # TODO: [RN] check if problem still exists on newer versions

    # Add level if level equals to 1
    # TODO: [RN] confirm add_level functions
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


# @dispatch(dict)
# # def from_dataframes(data: dict):
def from_dictionary(data: dict):
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

    >>> from_dictionary(dict)
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
    
    return from_dictionary(dict)


def from_tensor(values, index=None, assets=None, channels=None):
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


def from_matrix(values, index=None, assets=None, channels=None):
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


def rebuild(func):
    # TODO: [RN] tests to check if can be removed
    # Avoid problem with frozen list from pandas
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        df = func(*args, **kwargs)
        return from_dataframe(df)

    return wrapper


class _AssetSeries(pd.Series):
    # ? I made it internal
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
        Datablock first index.

        Example:

        >>> datablock
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> datablock.assets
        Timestamp('2005-12-21 00:00:00')
        """
        return self.index[0]

    @property
    def end(self):
        """
        Datablock last index.

        Example:

        >>> datablock
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> datablock.assets
        Timestamp('2005-12-22 00:00:00')
        """
        return self.index[-1]

    @property
    def assets(self):
        """
        Datablock assets.

        Example:

        >>> datablock
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> datablock.assets
        0    AAPL
        1    MSFT
        dtype: object
        """
        assets = [col[0] for col in self.columns]
        # OrderedDict to keep order
        # ? Is it correct to  order, what happens in case the user wants to rebuild the block?
        return pd.Series(tuple(OrderedDict.fromkeys(assets)))

    @property
    def channels(self):
        """
        Datablock channels.

        Example:

        >>> datablock
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> datablock.channels
        0    Open
        1    Close
        dtype: object
        """
        channels = [col[1] for col in self.columns]
        # OrderedDict to keep order
        # ? Is it correct to  order, what happens in case the user wants to rebuild the block?
        return pd.Series(list(OrderedDict.fromkeys(channels)))

    @property
    def tensor(self):
        """
        3D matrix with DataBlock value.

        Example:

        >>> datablock
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> datablock.tensor
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

        >>> datablock
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> datablock.matrix
        array([[ 2.21856582,  2.24606872, 19.57712554, 19.47512245],
            [ 2.25859845,  2.26195979, 19.46054323, 19.37311363]])
        """
        return self.values

    @rebuild
    def filter(self, assets: List[str] = None, channels: List[str] = None):
        """
        Subset the dataframe columns according to the specified assets and channels.

        Args:
            assets (list): List of assets
            channels (list): List of channels

        Returns:
            ``TimeBlock``: Filtered TimeBlock

        Example:

        >>> datablock
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> datablock.filter(assets=['AAPL'], channels=['Open'])
                        AAPL
                        Open
        Date                
        2005-12-21  2.218566
        2005-12-22  2.258598
        """
        filtered = self._filter_assets(assets)
        filtered = filtered._filter_channels(channels)
        return filtered

    @rebuild
    def _filter_assets(self, assets):
        if type(assets) == str:
            assets = [assets]

        # if assets is not None and any(asset not in self.columns.levels[0] for asset in assets):
        #     raise ValueError(f"{assets} not found in columns. Columns: {list(self.columns.levels[0])}")

        return self.loc[:, (assets, slice(None))] if assets else self

    @rebuild
    def _filter_channels(self, channels):
        if type(channels) == str:
            channels = [channels]

        # if channels is not None and any(channel not in self.columns.levels[1] for channel in channels):
        #     raise ValueError(f"{channels} not found in columns. Columns:{list(self.columns.levels[1])}")

        return self.loc[:, (slice(None), channels)][self.assets] if channels else self

    def drop(self, assets=None, channels=None):
        """"""
        # TODO: check the necessity of drop function once we have filter
        filtered = self._drop_assets(assets)
        filtered = filtered._drop_channels(channels)
        return filtered

    def _drop_assets(self, assets):
        if isinstance(assets, str):
            assets = [assets]
        new_assets = [u for u in self.assets if u not in assets]
        return self.filter_assets(new_assets)

    def _drop_channels(self, channels):
        if isinstance(channels, str):
            channels = [channels]
        new_channels = [c for c in self.channels if c not in channels]
        return self.filter_channels(new_channels)

    def rename_assets(self, dict: dict):
        """
        Alter asset labels.

        Args:
            dict (list): Dictionary with assets to rename

        Returns:
            ``TimeBlock``: Renamed TimeBlock

        Example:

        >>> datablock
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> datablock.rename_assets({'AAPL': 'Apple', 'MSFT': 'Microsoft'})
                       Apple            Microsoft           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114
        """

        values = dict.keys()
        new_values = dict.values()

        assets = self.assets.replace(to_replace=values, value=new_values)
        return self._update(assets=assets.values)

    def rename_channels(self, dict: dict):
        """
        Alter channel labels.

        Args:
            dict (list): Dictionary with channels to rename

        Returns:
            ``TimeBlock``: Renamed TimeBlock

        Example:

        >>> datablock
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> datablock.rename_channels({'Open': 'Op', 'Close': 'Cl'})
                        AAPL                 MSFT           
                        Op        Cl         Op         Cl
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114
        """
        values = dict.keys()
        new_values = dict.values()

        channels = self.channels.replace(to_replace=values, value=new_values)
        return self._update(channels=channels.values)

    def apply(self, func, axis=0):
        """
        Apply a function along an axis of the DataBlock.

        Args:
            func (function): Function to apply to each column or row.
            axis ({0 or 'index', 1 or 'columns'}, default 0): Axis along which the function is applied:
                
                * 0 or 'index': apply function to each column.
                * 1 or 'columns': apply function to each row.

        Returns:
            ``TimeBlock``: Result of applying `func` along the given axis of the DataFrame.

        Example:

        >>> datablock
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> datablock.apply(np.max, axis=0)
            AAPL                MSFT           
            Open    Close       Open      Close
        0  2.258598  2.26196  19.577126  19.475122

        >>> datablock.apply(np.max, axis=1)
                        AAPL       MSFT
                        amax       amax
        Date                           
        2005-12-21  2.246069  19.577126
        2005-12-22  2.261960  19.460543
        """
        # ? think about overwriting pandas apply function
        if axis == 0 or axis == 'index':
            return self._timestamp_apply(func)
        elif axis == 1 or axis == 'columns':
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

    def _update(self, values=None, index=None, assets=None, channels=None):
        # TODO check which functions need to use _update
        values = values if values is not None else self.values
        assets = assets if assets is not None else self.assets
        index = index if index is not None else self.index
        channels = channels if channels is not None else self.channels
        return from_matrix(values, index, assets, channels)

    # @rebuild
    # def add_channel(self, name, values):
    #     for asset in self.assets:
    #         self.loc[:, (asset, name)] = values
    #     return self

    def split_assets(self):
        """
        Split DataBlock into assets.

        Returns:
            ``List``: List of DataBlock, each one being one asset.

        Example:

        >>> datablock
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> datablock.split_assets()
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

    def sort_assets(self):
        """
        Sort assets in alphabetical order.

        Returns:
            ``DataBlock``: Result of sorting assets.

        Example:

        >>> datablock
                        MSFT                 AAPL          
                        Open      Close      Open     Close
        Date                                                
        2005-12-21  19.577126  19.475122  2.218566  2.246069
        2005-12-22  19.460543  19.373114  2.258598  2.261960

        >>> datablock.sort_assets()
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114
        """
        return self.reindex(sorted(self.columns, key=lambda x: x[0]), axis=1)

    def sort_channels(self):
        """
        Sort channels in alphabetical order.

        Returns:
            ``DataBlock``: Result of sorting channels.

        Example:

        >>> datablock
                        MSFT                 AAPL          
                        Open      Close      Open     Close
        Date                                                
        2005-12-21  19.577126  19.475122  2.218566  2.246069
        2005-12-22  19.460543  19.373114  2.258598  2.261960

        >>> datablock.sort_channels()
                         MSFT                 AAPL          
                        Close       Open     Close      Open
        Date                                                
        2005-12-21  19.475122  19.577126  2.246069  2.218566
        2005-12-22  19.373114  19.460543  2.261960  2.258598
        """
        assets = self.assets
        channels = sorted(self.channels)
        pair = [(asset, channel) for asset in assets for channel in channels]
        return self.reindex(pair, axis=1)

    def swap_cols(self):
        """
        Swap columns levels, assets becomes channels and channels becomes assets

        Returns:
            ``DataBlock``: Result of swapping columns.

        Example:

        >>> datablock
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> datablock.swap_cols()
                       Close                 Open           
                        AAPL       MSFT      AAPL       MSFT
        Date                                                
        2005-12-21  2.246069  19.475122  2.218566  19.577126
        2005-12-22  2.261960  19.373114  2.258598  19.460543
        """
        return self.T.swaplevel(i=- 2, j=- 1, axis=0).T.sort_assets()

    def countna(self, type: str = 'asset'):
        """
        Count 'not a number' cells for each asset or channel.

        Returns:
            ``List``: List of NaN count.

        Example:

        >>> datablock
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069        NaN  19.475122
        2005-12-22  2.258598       NaN  19.460543  19.373114

        >>> datablock.countna('asset')
        AAPL    1
        MSFT    1
        dtype: int64

        >>> datablock.countna('channel')
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

        >>> type(datablock)
        wavy.block.TimeBlock

        >>> type(datablock.as_dataframe())
        pandas.core.frame.DataFrame
        """
        return pd.DataFrame(self.values, index=self.index, columns=self.columns)

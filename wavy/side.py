import functools
import operator

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .block import Block

from typing import List

# dunder_methods = ['__abs__', '__add__', '__aenter__', '__aexit__', '__aiter__', '__and__', '__anext__', '__await__', '__bool__', '__bytes__', '__call__', '__ceil__', '__class__', '__class_getitem__', '__cmp__', '__coerce__', '__complex__', '__contains__', '__del__', '__delattr__', '__delete__', '__delitem__', '__delslice__', '__dict__', '__dir__', '__div__', '__divmod__', '__enter__', '__eq__', '__exit__', '__float__', '__floor__', '__floordiv__', '__format__', '__fspath__', '__ge__', '__get__', '__getattr__', '__getattribute__', '__getitem__', '__getnewargs__', '__getslice__', '__gt__', '__hash__', '__hex__', '__iadd__', '__iand__', '__idiv__', '__ifloordiv__', '__ilshift__', '__imatmul__', '__imod__', '__import__', '__imul__', '__index__', '__init__', '__init_subclass__', '__instancecheck__', '__int__', '__invert__', '__ior__', '__ipow__', '__irshift__', '__isub__', '__iter__', '__itruediv__', '__ixor__', '__le__', '__len__', '__length_hint__', '__long__', '__lshift__', '__lt__', '__matmul__', '__metaclass__', '__missing__', '__mod__', '__mro__', '__mul__', '__ne__', '__neg__', '__new__', '__next__', '__nonzero__', '__oct__', '__or__', '__pos__', '__pow__', '__prepare__', '__radd__', '__rand__', '__rcmp__', '__rdiv__', '__rdivmod__', '__reduce__', '__reduce_ex__', '__repr__', '__reversed__', '__rfloordiv__', '__rlshift__', '__rmatmul__', '__rmod__', '__rmul__', '__ror__', '__round__', '__rpow__', '__rrshift__', '__rshift__', '__rsub__', '__rtruediv__', '__rxor__', '__set__', '__set_name__', '__setattr__', '__setitem__', '__setslice__', '__sizeof__', '__slots__', '__str__', '__sub__', '__subclasscheck__', '__subclasses__', '__truediv__', '__trunc__', '__unicode__', '__weakref__', '__xor__']
dunder_methods = ['__add__', '__sub__', '__mul__', '__ge__', '__gt__', '__le__', '__lt__', '__pow__']


# Plot
import numpy as np
import pandas as pd
import plotly as px
import plotly.graph_objects as go
import plotly.express as px
pd.set_option("multi_sparse", True)  # To see multilevel indexes
pd.options.plotting.backend = "plotly"
from plotly.subplots import make_subplots


class Side:
    def __init__(self, blocks):
        # TODO: blocks must have increasing indexes, add warning and reindex
        # TODO this check should be done when creating the panel

        class _IXIndexer:
            def __getitem__(self, item):
                return Side([i.ix[item] for i in blocks])
        class _iLocIndexer:
            def __getitem__(self, item):
                return Side([i.iloc[item] for i in blocks])
        class _LocIndexer:
            def __getitem__(self, item):
                return Side([i.loc[item] for i in blocks])
        class _AtIndexer:
            def __getitem__(self, item):
                return Side([i.at[item] for i in blocks])
        class _iAtIndexer:
            def __getitem__(self, item):
                return Side([i.iat[item] for i in blocks])

        self.blocks = blocks
        self.ix = _IXIndexer()
        self.iloc = _iLocIndexer()
        self.loc = _LocIndexer()
        self.at = _AtIndexer()
        self.iat = _iAtIndexer()

    def __getattr__(self, name):
        try:
            def wrapper(*args, **kwargs):
                return Side([getattr(block, name)(*args, **kwargs) for block in self.blocks])
            return wrapper
        except AttributeError:
            raise AttributeError(f"'Side' object has no attribute '{name}'")

    def __getitem__(self, key):
        return self.blocks.__getitem__(key)

    def __len__(self):
        return len(self.blocks)

    @property
    def first(self):
        """
        Side first DataBlock.

        Example:

        >>> side.first
                        MSFT                 AAPL
                        Open      Close      Open     Close
        Date
        2005-12-21  19.577126  19.475122  2.218566  2.246069
        2005-12-22  19.460543  19.373114  2.258598  2.261960
        """
        return self.blocks[0]

    @property
    def last(self):
        """
        Side last DataBlock.

        Example:

        >>> side.last
                        MSFT                 AAPL
                        Open      Close      Open     Close
        Date
        2005-12-22  19.460543  19.373114  2.258598  2.261960
        2005-12-23  19.322122  19.409552  2.266543  2.241485
        """
        return self.blocks[-1]

    @property
    def start(self):
        """
        Side first index.

        Example:

        >>> side.start
        Timestamp('2005-12-21 00:00:00')
        """
        return self.first.start

    @property
    def end(self):
        """
        Side last index.

        Example:

        >>> side.end
        Timestamp('2005-12-23 00:00:00')
        """
        return self.last.end

    @property
    def assets(self):
        """
        Side assets.

        Example:

        >>> side.assets
        0    AAPL
        1    MSFT
        dtype: object
        """
        return self.first.assets

    @property
    def channels(self):
        """
        Side channels.

        Example:

        >>> side.channels
        0    Open
        1    Close
        dtype: object
        """
        return self.first.channels

    @property
    def timesteps(self):
        """
        Side timesteps.

        Example:

        >>> side.timesteps
        DatetimeIndex(['2005-12-21', '2005-12-22', '2005-12-23'], dtype='datetime64[ns]', name='Date', freq=None)
        """
        # The same as the index
        return self.index

    @property
    def index(self):
        """
        Side index.

        Example:

        >>> side.index
        DatetimeIndex(['2005-12-21', '2005-12-22', '2005-12-23'], dtype='datetime64[ns]', name='Date', freq=None)
        """
        return self.as_dataframe().index

    # @property
    # def values(self):
    #     # TODO
    #     # ? In block the equivalent name is tensor
    #     """
    #     3D matrix with Side value.

    #     Example:

    #     >>> side.values
    #     array([[[19.57712554, 19.47512245,  2.21856582,  2.24606872],
    #             [19.46054323, 19.37311363,  2.25859845,  2.26195979]],
    #            [[19.46054323, 19.37311363,  2.25859845,  2.26195979],
    #             [19.32212198, 19.40955162,  2.26654326,  2.24148512]]])
    #     """
    #     return np.array(self.blocks)

    @property
    def shape(self):
        """
        Side shape.

        Example:

        >>> side.shape
        (2, 2, 2, 2)
        """
        return self.tensor4d.shape

    @property
    def tensor4d(self):
        """
        4D matrix with Side value.

        Example:

        >>> side.tensor
        array([[[[19.57712554, 19.47512245],
                 [19.46054323, 19.37311363]],
                [[ 2.21856582,  2.24606872],
                 [ 2.25859845,  2.26195979]]],
               [[[19.46054323, 19.37311363],
                 [19.32212198, 19.40955162]],
                [[ 2.25859845,  2.26195979],
                 [ 2.26654326,  2.24148512]]]])
        """
        # Could be calculate using using the block function but it is faster this way
        timesteps = self.first.index
        new_shape = (len(self), len(timesteps), len(self.assets), len(self.channels))
        values = self.tensor3d.reshape(*new_shape)
        return values.transpose(0, 2, 1, 3)
        # return np.array([block.tensor for block in tqdm(self.blocks)])

    @property
    def tensor3d(self):
        """
        3D matrix with Side value.

        Example:

        >>> side.matrix
        array([[[19.57712554, 19.47512245,  2.21856582,  2.24606872],
                [19.46054323, 19.37311363,  2.25859845,  2.26195979]],
               [[19.46054323, 19.37311363,  2.25859845,  2.26195979],
                [19.32212198, 19.40955162,  2.26654326,  2.24148512]]])
        """
        return np.array([block.matrix for block in tqdm(self.blocks)])


    def filter(self, assets: List[str] = None, channels: List[str] = None):
        """
        Side subset according to the specified assets and channels.

        Args:
            assets (list): List of assets
            channels (list): List of channels

        Returns:
            ``Side``: Filtered Side
        """
        return Side([block.filter(assets=assets, channels=channels) for block in tqdm(self.blocks)])

    def drop(self, assets=None, channels=None):
        """
        Subset of the Side columns discarding the specified assets and channels.

        Args:
            assets (list): List of assets
            channels (list): List of channels

        Returns:
            ``Side``: Filtered Side
        """
        return Side([block.drop(assets=assets, channels=channels) for block in tqdm(self.blocks)])

    def rename_assets(self, dict: dict):
        """
        Rename asset labels.

        Args:
            dict (dict): Dictionary with assets to rename

        Returns:
            ``Side``: Renamed Side
        """
        return Side([block.rename_assets(dict) for block in tqdm(self.blocks)])

    def rename_channels(self, dict: dict):
        """
        Rename channel labels.

        Args:
            dict (dict): Dictionary with channels to rename

        Returns:
            ``Side``: Renamed Side
        """
        return Side([block.rename_channels(dict) for block in tqdm(self.blocks)])

    def apply(self, func, on: str = 'timestamps'):
        """
        Apply a function along an axis of the DataBlock.

        Args:
            func (function): Function to apply to each column or row.
            on (str, default 'row'): Axis along which the function is applied:

                * 'timestamps': apply function to each timestamps.
                * 'channels': apply function to each channels.

        Returns:
            ``Side``: Result of applying `func` along the given axis of the Side.
        """
        return Side([block.apply(func, on) for block in tqdm(self.blocks)])

    def update(self, values=None, index: List = None, assets: List = None, channels: List = None):
        """
        Update function for any of Side properties.

        Args:
            values (ndarray): New values Dataframe.
            index (list): New list of index.
            assets (list): New list of assets
            channels (list): New list of channels

        Returns:
            ``Side``: Result of updated Side.
        """
        return Side([block.update(values[i], index, assets, channels) for i, block in tqdm(enumerate(self.blocks))])

    def _split_assets(self):
        # TODO RN ? Does it make sense??
        return [self.filter(asset) for asset in self.assets]
        # return [block.split_assets() for block in tqdm(self.blocks)]

    def sort_assets(self, order: List[str] = None):
        """
        Sort assets in alphabetical order.

        Args:
            order (List[str]): Asset order to be sorted.

        Returns:
            ``Side``: Result of sorting assets.
        """
        return Side([block.sort_assets(order) for block in tqdm(self.blocks)])

    def sort_channels(self, order: List[str] = None):
        """
        Sort channels in alphabetical order.

        Args:
            order (List[str]): Channel order to be sorted.

        Returns:
            ``Side``: Result of sorting channels.
        """
        return Side([block.sort_channels(order) for block in tqdm(self.blocks)])

    def swap_cols(self):
        """
        Swap columns levels, assets becomes channels and channels becomes assets

        Returns:
            ``Side``: Result of swapping columns.
        """
        return Side([block.swap_cols() for block in tqdm(self.blocks)])

    # Concept: How many blocks contain nan
    def countna(self):
        """
        Count NA/NaN cells for each Block.

        Returns:
            ``DataFrame``: NaN count for each Block.

        Example:

        >>> side.countna
           nan
        0    2
        1    2
        """
        values = [block.isnull().values.sum() for block in tqdm(self.blocks)]
        return pd.DataFrame(values, index=range(len(self.blocks)), columns=['nan'])

    def fillna(self, value=None, method: str = None):
        """
        Fill NA/NaN values using the specified method.

        Returns:
            ``Side``: Side with missing values filled.
        """
        return Side([block.fillna(value=value, method=method) for block in tqdm(self.blocks)])

    def dropna(self, x=True, y=True):
        """
        Drop pairs with missing values from the panel.

        Returns:
            ``Side``: Side with missing values dropped.
        """
        nan_values = self.findna()
        idx = {i for i in range(len(self)) if i not in nan_values}
        if not idx:
            raise ValueError("'dropna' would create empty Panel")
        return self[idx]

    # def numpy(self):
    #         new_shape = (len(self), len(self.timesteps), len(self.assets), len(self.channels))
    #         values = self.values.reshape(*new_shape)
    #         return values.transpose(0, 2, 1, 3)

    def findna(self):
        """
        Find NA/NaN values index.

        Returns:
            ``List``: List with index of missing values.
        """
        values = np.sum(self.tensor4d, axis=(3, 2, 1))
        values = pd.Series(values).isna()
        return values[values == True].index.tolist()

    # Used the function update, keep the same name as in block
    # def replace(self, data):
    #     blocks = [block.update(values=data[i]) for i, block in enumerate(self.blocks)]
    #     return Side(blocks)

    # ? Does it make sense, leave for next version
    # def add_channel(self, name, values):
    #     return [block.add_channel(name, values) for block in self.blocks]

    def as_dataframe(self):
        # Renamed from data
        """
        Reconstructs the dataframe.

        Returns:
            ``Side``: Result of sorting channels.
        """
        # Dataframe recontruction
        df = pd.concat(self.blocks)
        return df[~df.index.duplicated(keep="first")]

    def flat(self):
        """
        2D array with the flat value of each Block.

        Returns:
            ``DataFrame``: Result of flat function.

        Example:

        Side containing two Block, will present the following result.

        >>> side.first
                        MSFT                 AAPL
                        Open      Close      Open     Close
        Date
        2005-12-21  19.577126  19.475122  2.218566  2.246069
        2005-12-22  19.460543  19.373114  2.258598  2.261960

        >>> side.last
                        MSFT                 AAPL
                        Open      Close      Open     Close
        Date
        2005-12-22  19.460543  19.373114  2.258598  2.261960
        2005-12-23  19.322122  19.409552  2.266543  2.241485

        Where only the last timestep of each Block is used as index.

        >>> side.flat()
                           0         1        2        3         4         5        6        7
        2005-12-22 19.577126 19.475122 2.218566 2.246069 19.460543 19.373114 2.258598 2.261960
        2005-12-23 19.460543 19.373114 2.258598 2.261960 19.322122 19.409552 2.266543 2.241485
        """
        values = np.array([i.values.flatten() for i in self.blocks])
        index = [i.index[-1] for i in self.blocks]
        return pd.DataFrame(values, index=index)

    def flatten(self):
        # TODO return series for single column or dataframe
        """
        1D array with the flat value of all Blocks.

        Returns:
            ``array``: Result of flat function.

        Example:

        Side containing two Block, will present the following result.

        >>> side.first
                        MSFT                 AAPL
                        Open      Close      Open     Close
        Date
        2005-12-21  19.577126  19.475122  2.218566  2.246069
        2005-12-22  19.460543  19.373114  2.258598  2.261960

        >>> side.last
                        MSFT                 AAPL
                        Open      Close      Open     Close
        Date
        2005-12-22  19.460543  19.373114  2.258598  2.261960
        2005-12-23  19.322122  19.409552  2.266543  2.241485

        >>> side.flatten()
        array([19.57712554, 19.47512245,  2.21856582,  2.24606872, 19.46054323,
               19.37311363,  2.25859845,  2.26195979, 19.46054323, 19.37311363,
                2.25859845,  2.26195979, 19.32212198, 19.40955162,  2.26654326,
                2.24148512])
        """
        return self.flat().values.flatten()


    def plot(self, idx, assets: List[str] = None, channels: List[str] = None):
        """
        Side plot according to the specified assets and channels.

        Args:
            idx (int): Panel index
            assets (list): List of assets
            channels (list): List of channels

        Returns:
            ``Plot``: Plotted data
        """
        cmap = px.colors.qualitative.Plotly

        fig = make_subplots(rows=len(self.channels), cols=len(self.assets), subplot_titles=self.assets, shared_xaxes=True)

        # data = self.as_dataframe()

        for j, channel in enumerate(self.channels):
            c = cmap[j]
            for i, asset in enumerate(self.assets):

                showlegend = i <= 0
                # x_df = data.loc[:, (asset, channel)]

                x_df = self.blocks[idx].filter(assets=asset, channels=channel)
                index = x_df.index
                values = x_df.values.flatten()

                x_trace = go.Scatter(x=index, y=values,
                                line=dict(width=2, color=c), showlegend=showlegend, name=channel)

                fig.add_trace(x_trace, row=j+1, col=i+1)
                # Remove empty dates
                dt_all = pd.date_range(start=index[0],end=index[-1])
                dt_obs = [d.strftime("%Y-%m-%d") for d in index]
                dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d").tolist() if not d in dt_obs]
                fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])

        fig.update_layout(showlegend=True)
        fig.show()

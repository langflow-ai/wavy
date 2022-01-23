import functools
import operator

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import random

from .block import Block, from_matrix, from_series

from typing import List, Union

# dunder_methods = ['__abs__', '__add__', '__aenter__', '__aexit__', '__aiter__', '__and__', '__anext__', '__await__', '__bool__', '__bytes__', '__call__', '__ceil__', '__class__', '__class_getitem__', '__cmp__', '__coerce__', '__complex__', '__contains__', '__del__', '__delattr__', '__delete__', '__delitem__', '__delslice__', '__dict__', '__dir__', '__div__', '__divmod__', '__enter__', '__eq__', '__exit__', '__float__', '__floor__', '__floordiv__', '__format__', '__fspath__', '__ge__', '__get__', '__getattr__', '__getattribute__', '__getitem__', '__getnewargs__', '__getslice__', '__gt__', '__hash__', '__hex__', '__iadd__', '__iand__', '__idiv__', '__ifloordiv__', '__ilshift__', '__imatmul__', '__imod__', '__import__', '__imul__', '__index__', '__init__', '__init_subclass__', '__instancecheck__', '__int__', '__invert__', '__ior__', '__ipow__', '__irshift__', '__isub__', '__iter__', '__itruediv__', '__ixor__', '__le__', '__len__', '__length_hint__', '__long__', '__lshift__', '__lt__', '__matmul__', '__metaclass__', '__missing__', '__mod__', '__mro__', '__mul__', '__ne__', '__neg__', '__new__', '__next__', '__nonzero__', '__oct__', '__or__', '__pos__', '__pow__', '__prepare__', '__radd__', '__rand__', '__rcmp__', '__rdiv__', '__rdivmod__', '__reduce__', '__reduce_ex__', '__repr__', '__reversed__', '__rfloordiv__', '__rlshift__', '__rmatmul__', '__rmod__', '__rmul__', '__ror__', '__round__', '__rpow__', '__rrshift__', '__rshift__', '__rsub__', '__rtruediv__', '__rxor__', '__set__', '__set_name__', '__setattr__', '__setitem__', '__setslice__', '__sizeof__', '__slots__', '__str__', '__sub__', '__subclasscheck__', '__subclasses__', '__truediv__', '__trunc__', '__unicode__', '__weakref__', '__xor__']
DUNDER_METHODS = ['__add__', '__sub__', '__mul__', '__truediv__', '__ge__', '__gt__', '__le__', '__lt__', '__pow__']

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

    # TODO fix this method
    # Function to map all dunder functions
    def _one_arg(self, other, __f):
        if isinstance(other, Side):
            return Side([getattr(block, __f)(other_block) for block, other_block in zip(self.blocks, other)])
        return Side([getattr(block, __f)(other) for block in self.blocks])

    for dunder in DUNDER_METHODS:
        locals()[dunder] = lambda self, other, __f=dunder: self._one_arg(other, __f)

    def __getitem__(self, key):
        if isinstance(key, list):
            return [self.blocks[i] for i in key]
        return self.blocks.__getitem__(key)

    def __len__(self):
        return len(self.blocks)

    def __repr__(self):
        summary = pd.Series(
            {
                "size": self.__len__(),
                "num_assets": len(self.assets),
                "num_channels": len(self.channels),
                "start": self.start,
                "end": self.end,
            },
            name="Side",
        )

        print(summary)
        return f"<Side, size {self.__len__()}>"

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


    def wfilter(self, assets: List[str] = None, channels: List[str] = None):
        """
        Side subset according to the specified assets and channels.

        Similar to `Pandas filter <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.filter.html>`__

        Args:
            assets (list): List of assets
            channels (list): List of channels

        Returns:
            ``Side``: Filtered Side
        """
        return Side([block.wfilter(assets=assets, channels=channels) for block in tqdm(self.blocks)])

    def wdrop(self, assets=None, channels=None):
        """
        Subset of the Side columns discarding the specified assets and channels.

        Similar to `Pandas drop <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html>`__

        Args:
            assets (list): List of assets
            channels (list): List of channels

        Returns:
            ``Side``: Filtered Side
        """
        return Side([block.wdrop(assets=assets, channels=channels) for block in tqdm(self.blocks)])

    def rename_assets(self, dict: dict):
        """
        Rename asset labels.

        Similar to `Pandas rename <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rename.html#>`__

        Args:
            dict (dict): Dictionary with assets to rename

        Returns:
            ``Side``: Renamed Side
        """
        return Side([block.rename_assets(dict) for block in tqdm(self.blocks)])

    def rename_channels(self, dict: dict):
        """
        Rename channel labels.

        Similar to `Pandas rename <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rename.html#>`__

        Args:
            dict (dict): Dictionary with channels to rename

        Returns:
            ``Side``: Renamed Side
        """
        return Side([block.rename_channels(dict) for block in tqdm(self.blocks)])

    def wapply(self, func, on: str = 'timestamps'):
        """
        Apply a function along an axis of the DataBlock.

        Similar to `Pandas apply <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html>`__

        Args:
            func (function): Function to apply to each column or row.
            on (str, default 'row'): Axis along which the function is applied:

                * 'timestamps': apply function to each timestamps.
                * 'channels': apply function to each channels.

        Returns:
            ``Side``: Result of applying `func` along the given axis of the Side.
        """
        return Side([block.wapply(func, on) for block in tqdm(self.blocks)])

    def wupdate(self, values=None, index: List = None, assets: List = None, channels: List = None):
        """
        Update function for any of Side properties.

        Similar to `Pandas update <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.update.html>`__

        Args:
            values (ndarray): New values Dataframe.
            index (list): New list of index.
            assets (list): New list of assets
            channels (list): New list of channels

        Returns:
            ``Side``: Result of updated Side.
        """
        return Side([block.wupdate(values[i], index, assets, channels) for i, block in tqdm(enumerate(self.blocks))])

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

    # TODO add count??

    def countna(self):
        """
        Count NaN cells for each Block.

        Returns:
            ``DataFrame``: NaN count for each Block.

        Example:

        >>> side.countna()
           nan
        0    2
        1    2
        """
        values = [block.isnull().values.sum() for block in tqdm(self.blocks)]
        return pd.DataFrame(values, index=range(len(self.blocks)), columns=['nan'])

    def wfillna(self, value=None, method: str = None):
        """
        Fill NaN values using the specified method.

        Similar to `Pandas fillna <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html>`__

        Returns:
            ``Side``: Side with missing values filled.
        """
        return Side([block.wfillna(value=value, method=method) for block in tqdm(self.blocks)])

    def wdropna(self, x=True, y=True):
        """
        Drop pairs with missing values from the panel.

        Similar to `Pandas dropna <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html>`__

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
        Find NaN values index.

        Returns:
            ``List``: List with index of NaN values.
        """
        values = np.sum(self.tensor4d, axis=(3, 2, 1))
        values = pd.Series(values).isna()
        return values[values == True].index.tolist()

    def findinf(self):
        """
        Find Inf values index.

        Returns:
            ``List``: List with index of Inf values.
        """
        values = np.sum(self.tensor4d, axis=(3, 2, 1))
        values = pd.Series(np.isinf(values))
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

    def wshift(self, window: int = 1):
        """
        Shift side by desired number of blocks.

        Similar to `Pandas shift <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.shift.html>`__

        Args:
            window (int): Number of blocks to shift

        Returns:
            ``Side``: Result of side_shift function.

        Example:

        >>> side[0]
                        MSFT                 AAPL
                        Open      Close      Open     Close
        Date
        2005-12-21  19.577126  19.475122  2.218566  2.246069
        2005-12-22  19.460543  19.373114  2.258598  2.261960

        >>> side = side.wshift(window = 1)

        >>> side[0]
                MSFT       AAPL      
                Open Close Open Close
        Date                            
        2005-12-21  NaN   NaN  NaN   NaN
        2005-12-22  NaN   NaN  NaN   NaN

        >>> side[1]
                        MSFT                 AAPL
                        Open      Close      Open     Close
        Date
        2005-12-21  19.577126  19.475122  2.218566  2.246069
        2005-12-22  19.460543  19.373114  2.258598  2.261960
        """

        assert window >= 0, "Window cannot be negative!"

        new_side = []

        for i, block in enumerate(self.blocks):
            new_index = i - window
            new_index = new_index if new_index >= 0 and new_index < len(self.blocks) else None
            new_values = self.blocks[new_index] if new_index is not None else np.ones(self.blocks[0].shape) * np.nan
            new_block = from_matrix(values=new_values, index=block.index, assets=block.assets, channels=block.channels)
            new_side.append(new_block)

        return Side(new_side)

    def wdiff(self, window: int = 1):
        """
        Difference between blocks.

        Similar to `Pandas diff <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.diff.html>`__

        Args:
            window (int): Number of blocks to diff

        Returns:
            ``Side``: Result of side_diff function.

        Example:

        >>> side[0]
                        MSFT                 AAPL
                        Open      Close      Open     Close
        Date
        2005-12-21  19.577126  19.475122  2.218566  2.246069
        2005-12-22  19.460543  19.373114  2.258598  2.261960

        >>> side = side.wdiff(window = 1)

        >>> side[0]
                MSFT       AAPL      
                Open Close Open Close
        Date                            
        2005-12-21  NaN   NaN  NaN   NaN
        2005-12-22  NaN   NaN  NaN   NaN

        >>> side[1]
                        MSFT                AAPL          
                        Open     Close      Open     Close
        Date                                              
        2005-12-22 -0.116582 -0.102009  0.040033  0.015891
        2005-12-23 -0.138421  0.036438  0.007945 -0.020475
        """
        return self - self.wshift(window)

    def wpct_change(self, window: int = 1):
        """
        Percentage change between the current and a prior block.

        Similar to `Pandas pct_change <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pct_change.html>`__

        Args:
            window (int): Number of blocks to calculate percent change

        Returns:
            ``Side``: Result of side_pct_change function.

        Example:

        >>> side[0]
                        MSFT                 AAPL
                        Open      Close      Open     Close
        Date
        2005-12-21  19.577126  19.475122  2.218566  2.246069
        2005-12-22  19.460543  19.373114  2.258598  2.261960

        >>> side = side.wpct_change(window = 1)

        >>> side[0]
                MSFT       AAPL      
                Open Close Open Close
        Date                            
        2005-12-21  NaN   NaN  NaN   NaN
        2005-12-22  NaN   NaN  NaN   NaN

        >>> side[1]
                        MSFT                AAPL          
                        Open     Close      Open     Close
        Date                                              
        2005-12-22 -0.005955 -0.005238  0.018044  0.007075
        2005-12-23 -0.007113  0.001881  0.003518 -0.009052
        """
        a = self.side_shift(window)
        return (self - a) / a

    def side_sample(self, n: int = None, frac: float = None):

        # If no frac or n, default to n=1.
        if n is None and frac is None:
            n = 1
        elif frac is None and n % 1 != 0:
            raise ValueError("Only integers accepted as `n` values")
        elif n is None and frac is not None:
            n = round(frac * len(self))
        elif frac is not None:
            raise ValueError("Please enter a value for `frac` OR `n`, not both")

        # Check for negative sizes
        if n < 0:
            raise ValueError(
                "A negative number of rows requested. Please provide positive value."
            )

        locs = random.sample(range(0, len(self)), n)
        locs.sort()

        return self[locs]


    def plot_block(self, idx, assets: List[str] = None, channels: List[str] = None):
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

        fig = make_subplots(rows=len(self.channels), cols=len(self.assets), subplot_titles=self.assets)

        # data = self.as_dataframe()

        for j, channel in enumerate(self.channels):
            c = cmap[j]
            for i, asset in enumerate(self.assets):

                # showlegend = i <= 0
                # x_df = data.loc[:, (asset, channel)]

                x_df = self.blocks[idx].wfilter(assets=asset, channels=channel)
                index = x_df.index
                values = x_df.values.flatten()

                x_trace = go.Scatter(x=index, y=values,
                                line=dict(width=2, color=c), showlegend=False)

                fig.add_trace(x_trace, row=j+1, col=i+1)
                # Remove empty dates
                # dt_all = pd.date_range(start=index[0],end=index[-1])
                # dt_obs = [d.strftime("%Y-%m-%d") for d in index]
                # dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d").tolist() if not d in dt_obs]
                # fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])

        fig.update_layout(
            template='simple_white',
            showlegend=True
            )

        num_assets = len(self.assets)
        for i, channel in enumerate(self.channels):
            fig['layout'][f'yaxis{i*num_assets+1}'].update({'title':channel})

        fig.show()


    def plot_slider(self, steps: int = 100):
        """
        Make side plots with slider.

        Args:
            steps (int): Number of equally spaced blocks to plot

        Returns:
            ``Plot``: Plotted data.
        """

        if steps > 100:
            raise ValueError("Number of assets cannot be bigger than 100.")

        cmap = px.colors.qualitative.Plotly

        # Create figure
        # fig = go.Figure()
        fig = make_subplots(rows=len(self.channels), cols=len(self.assets), subplot_titles=self.assets)

        graph_number = len(self.channels) * len(self.assets)

        # Add traces, one for each slider step
        len_ = np.linspace(0,len(self.blocks), steps, dtype=int, endpoint=False)
        for step in len_: #np.arange(len(panel_.x.blocks)):

            for j, channel in enumerate(self.channels):
                c = cmap[j]
                for i, asset in enumerate(self.assets):

                    showlegend = i <= 0

                    x_df = self.blocks[step].wfilter(assets=asset, channels=channel)
                    index = x_df.index
                    values = x_df.values.flatten()

                    x_trace = go.Scatter(visible=False,
                                        x=index,
                                        y=values,
                                        line=dict(width=2, color=c), showlegend=showlegend, name=channel)

                    # x_trace = go.Scatter(x=index, y=values,
                    #                     line=dict(width=2, color=c), showlegend=showlegend, name=channel)

                    fig.add_trace(x_trace, row=j+1, col=i+1)

                    # dt_all = pd.date_range(start=index[0],end=index[-1])
                    # dt_obs = [d.strftime("%Y-%m-%d") for d in index]
                    # dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d").tolist() if not d in dt_obs]
                    # fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])

        # Make 10th trace visible
        for i in range(graph_number):
            fig.data[i].visible = True

        # Create and add slider
        steps_ = []
        for i in range(steps):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig.data)},
                    {"title": "Block " + str(len_[i])}],  # layout attribute
            )

            for g in range(graph_number):
                step["args"][0]["visible"][i*graph_number+g] = True  # Toggle i'th trace to "visible"

            steps_.append(step)


        sliders = [dict(
            active=0,
            # currentvalue={"prefix": "Block: "},
            pad={"t": 50},
            steps=steps_
        )]

        fig.update_layout(
            template='simple_white',
            sliders=sliders
        )

        # Plot y titles
        num_assets = len(self.assets)
        for i, channel in enumerate(self.channels):
            fig['layout'][f'yaxis{i*num_assets+1}'].update({'title':channel})

        fig.show()


    def wcount(self, axis: int = 0, numeric_only: bool = False):
        """
        Count non-NA cells for each column or row.

        The values None, NaN, NaT, and optionally numpy.inf (depending on pandas.options.mode.use_inf_as_na) are considered NA.

        Similar to `Pandas count <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.count.html>`__

        Args:
            axis (int): Axis for the function to be applied on.
            numeric_only (bool): Include only float, int or boolean data.

        Returns:
            ``DataBlock``: DataBlock with operation executed.

        Example:

        >>> side[0]
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> side[0].wcount()
                   AAPL       MSFT      
                   Open Close Open Close
        2005-12-21    2     2    2     2
        """
        return Side([block.wcount(axis=axis, numeric_only=numeric_only) for block in tqdm(self.blocks)])

    def wkurt(self, axis: int = None, skipna: bool = None, numeric_only=None, **kwargs):
        """
        Return unbiased kurtosis over requested axis.

        Kurtosis obtained using Fisher's definition of kurtosis (kurtosis of normal == 0.0). Normalized by N-1.
        
        Similar to `Pandas kurt <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.kurt.html>`__

        Args:
            axis (int): Axis for the function to be applied on.
            skipna (bool): Exclude NA/null values when computing the result.
            numeric_only (bool): Include only float, int or boolean data. If None, will attempt to use everything, then use only numeric data. Not implemented for Series.
            **kwargs: Additional keyword arguments to be passed to the function.

        Returns:
            ``DataBlock``: DataBlock with operation executed.

        Example:

        >>> side[0]
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> side[0].wkurt(axis=1)
                    asset
                    kurt
        Date               
        2005-12-21 -5.99944
        2005-12-22 -5.99961
        """
        return Side([block.wkurt(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs) for block in tqdm(self.blocks)])

    def wkurtosis(self, axis: int = None, skipna: bool = None, numeric_only=None, **kwargs):
        """
        Return unbiased kurtosis over requested axis.

        Kurtosis obtained using Fisher's definition of kurtosis (kurtosis of normal == 0.0). Normalized by N-1.
        
        Similar to `Pandas kurtosis <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.kurtosis.html>`__

        Args:
            axis (int): Axis for the function to be applied on.
            skipna (bool): Exclude NA/null values when computing the result.
            numeric_only (bool): Include only float, int or boolean data. If None, will attempt to use everything, then use only numeric data. Not implemented for Series.
            **kwargs: Additional keyword arguments to be passed to the function.

        Returns:
            ``DataBlock``: DataBlock with operation executed.

        Example:

        >>> side[0]
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> side[0].wkurtosis(axis=1)
                    asset
                    kurt
        Date               
        2005-12-21 -5.99944
        2005-12-22 -5.99961
        """
        return Side([block.wkurtosis(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs) for block in tqdm(self.blocks)])

    def wmad(self, axis: int = None, skipna: bool = None):
        """
        Return the mean absolute deviation of the values over the requested axis.

        Similar to `Pandas mad <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.mad.html>`__

        Args:
            axis (int): Axis for the function to be applied on.
            numeric_only (bool): Include only float, int or boolean data. If None, will attempt to use everything, then use only numeric data. Not implemented for Series.

        Returns:
            ``DataBlock``: DataBlock with operation executed.

        Example:

        >>> side[0]
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> side[0].wmad()
                        AAPL                MSFT          
                        Open     Close      Open     Close
        2005-12-21  0.020016  0.007946  0.058291  0.051004
        """
        return Side([block.wmad(axis=axis, skipna=skipna) for block in tqdm(self.blocks)])

    def wmax(self, axis: int = None, skipna: bool = None, numeric_only=None, **kwargs):
        """
        Return the maximum of the values over the requested axis.

        Similar to `Pandas max <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.max.html>`__

        Args:
            axis (int): Axis for the function to be applied on.
            skipna (bool): Exclude NA/null values when computing the result.
            numeric_only (bool): Include only float, int or boolean data. If None, will attempt to use everything, then use only numeric data. Not implemented for Series.
            **kwargs: Additional keyword arguments to be passed to the function.

        Returns:
            ``DataBlock``: DataBlock with operation executed.

        Example:

        >>> side[0]
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> side[0].wmax()
                        AAPL                MSFT           
                        Open    Close       Open      Close
        2005-12-21  2.258598  2.26196  19.577126  19.475122
        """
        return Side([block.wmax(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs) for block in tqdm(self.blocks)])

    def wmean(self, axis: int = None, skipna: bool = None, numeric_only=None, **kwargs):
        """
        Return the mean of the values over the requested axis.

        Similar to `Pandas mean <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.mean.html>`__

        Args:
            axis (int): Axis for the function to be applied on.
            skipna (bool): Exclude NA/null values when computing the result.
            numeric_only (bool): Include only float, int or boolean data. If None, will attempt to use everything, then use only numeric data. Not implemented for Series.
            **kwargs: Additional keyword arguments to be passed to the function.

        Returns:
            ``DataBlock``: DataBlock with operation executed.

        Example:

        >>> side[0]
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> side[0].wmean()
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        2005-12-21  2.238582  2.254014  19.518834  19.424118
        """
        return Side([block.wwmean(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs) for block in tqdm(self.blocks)])

    def wmedian(self, axis: int = None, skipna: bool = None, numeric_only=None, **kwargs):
        """
        Return the median of the values over the requested axis.

        Similar to `Pandas median <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.median.html>`__

        Args:
            axis (int): Axis for the function to be applied on.
            skipna (bool): Exclude NA/null values when computing the result.
            numeric_only (bool): Include only float, int or boolean data. If None, will attempt to use everything, then use only numeric data. Not implemented for Series.
            **kwargs: Additional keyword arguments to be passed to the function.

        Returns:
            ``DataBlock``: DataBlock with operation executed.

        Example:

        >>> side[0]
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> side[0].wmedian()
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        2005-12-21  2.238582  2.254014  19.518834  19.424118
        """
        return Side([block.wmedian(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs) for block in tqdm(self.blocks)])


    def wmin(self, axis: int = None, skipna: bool = None, numeric_only=None, **kwargs):
        """
        Return the minimum of the values over the requested axis.

        Similar to `Pandas min <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.min.html>`__

        Args:
            axis (int): Axis for the function to be applied on.
            skipna (bool): Exclude NA/null values when computing the result.
            numeric_only (bool): Include only float, int or boolean data. If None, will attempt to use everything, then use only numeric data. Not implemented for Series.
            **kwargs: Additional keyword arguments to be passed to the function.

        Returns:
            ``DataBlock``: DataBlock with operation executed.

        Example:

        >>> side[0]
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> side[0].wmin()
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        2005-12-21  2.218566  2.246069  19.460543  19.373114
        """
        return Side([block.wmin(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs) for block in tqdm(self.blocks)])

    def wnunique(self, axis: int = None, dropna: bool = None):
        """
        Count number of distinct elements in specified axis.

        Return Series with number of distinct elements. Can ignore NaN values.

        Similar to `Pandas nunique <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.nunique.html>`__

        Args:
            axis (int): Axis for the function to be applied on.
            dropna (bool): Don't include NaN in the counts.

        Returns:
            ``DataBlock``: DataBlock with operation executed.

        Example:

        >>> side[0]
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> side[0].wnunique()
                AAPL       MSFT      
                Open Close Open Close
        2005-12-21    2     2    2     2
        """
        return Side([block.wnunique(axis=axis, dropna=dropna) for block in tqdm(self.blocks)])

    def wprod(self, axis: int = None, skipna: bool = None, numeric_only=None, min_count: int = 0, **kwargs):
        """
        Return the product of the values over the requested axis.

        Similar to `Pandas prod <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.prod.html>`__

        Args:
            axis (int): Axis for the function to be applied on.
            skipna (bool): Exclude NA/null values when computing the result.
            numeric_only (bool): Include only float, int or boolean data. If None, will attempt to use everything, then use only numeric data. Not implemented for Series.
            min_count (int): The required number of valid values to perform the operation. If fewer than `min_count` non-NA values are present the result will be NA.
            **kwargs: Additional keyword arguments to be passed to the function.

        Returns:
            ``DataBlock``: DataBlock with operation executed.

        Example:

        >>> side[0]
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> side[0].wprod()
                        AAPL                  MSFT           
                        Open     Close        Open      Close
        2005-12-21  5.010849  5.080517  380.981498  377.29376
        """
        return Side([block.wprod(axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count, **kwargs) for block in tqdm(self.blocks)])


    def wproduct(self, axis: int = None, skipna: bool = None, numeric_only=None, min_count: int = 0, **kwargs):
        """
        Return the product of the values over the requested axis.

        Similar to `Pandas product <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.product.html>`__

        Args:
            axis (int): Axis for the function to be applied on.
            skipna (bool): Exclude NA/null values when computing the result.
            numeric_only (bool): Include only float, int or boolean data. If None, will attempt to use everything, then use only numeric data. Not implemented for Series.
            min_count (int): The required number of valid values to perform the operation. If fewer than `min_count` non-NA values are present the result will be NA.
            **kwargs: Additional keyword arguments to be passed to the function.

        Returns:
            ``DataBlock``: DataBlock with operation executed.

        Example:

        >>> side[0]
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> side[0].wprod()
                        AAPL                  MSFT           
                        Open     Close        Open      Close
        2005-12-21  5.010849  5.080517  380.981498  377.29376
        """
        return Side([block.wproduct(axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count, **kwargs) for block in tqdm(self.blocks)])


    def wquantile(self, q: Union[float, List[float]] = 0.5, interpolation: str = "linear"):
        """
        Return value at the given quantile.

        Similar to `Pandas quantile <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.quantile.html>`__

        Args:
            q (float, array): The quantile(s) to compute, which can lie in range: 0 <= q <= 1.
            interpolation (str): {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
            
                This optional parameter specifies the interpolation method to use, when the desired quantile lies between two data points `i` and `j`:

                * 'linear': `i + (j - i) * fraction`, where `fraction` is the fractional part of the index surrounded by `i` and `j`.
                * 'lower': `i`.
                * 'higher': `j`.
                * 'nearest': `i` or `j` whichever is nearest.
                * 'midpoint': (`i` + `j`) / 2.

        Returns:
            ``DataBlock``: DataBlock with operation executed.

        Example:

        >>> side[0]
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> side[0].wquantile(q=0.5, interpolation='linear')
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        2005-12-21  2.238582  2.254014  19.518834  19.424118
        """
        return Side([block.wquantile(q=q, interpolation=interpolation) for block in tqdm(self.blocks)])

    def wsem(self, axis: int = None, skipna: bool = None, ddof: int = 1, numeric_only=None, **kwargs):
        """
        Return unbiased standard error of the mean over requested axis.

        Normalized by N-1 by default. This can be changed using the ddof argument

        Similar to `Pandas sem <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sem.html>`__

        Args:
            axis (int): Axis for the function to be applied on.
            skipna (bool): Exclude NA/null values when computing the result.
            ddof (int): Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the number of elements.
            numeric_only (bool): Include only float, int or boolean data. If None, will attempt to use everything, then use only numeric data. Not implemented for Series.
            **kwargs: Additional keyword arguments to be passed to the function.

        Returns:
            ``DataBlock``: DataBlock with operation executed.

        Example:

        >>> side[0]
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> side[0].wsem()
                        AAPL                MSFT          
                        Open     Close      Open     Close
        2005-12-21  0.020016  0.007946  0.058291  0.051004
        """
        return Side([block.wsem(axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only, **kwargs) for block in tqdm(self.blocks)])


    def wskew(self, axis: int = None, skipna: bool = None, numeric_only=None, **kwargs):
        """
        Return unbiased skew over requested axis.

        Normalized by N-1.

        Similar to `Pandas skew <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.skew.html>`__

        Args:
            axis (int): Axis for the function to be applied on.
            skipna (bool): Exclude NA/null values when computing the result.
            numeric_only (bool): Include only float, int or boolean data. If None, will attempt to use everything, then use only numeric data. Not implemented for Series.
            **kwargs: Additional keyword arguments to be passed to the function.

        Returns:
            ``DataBlock``: DataBlock with operation executed.

        Example:

        >>> side[0]
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> side[0].wskew(axis=1)
                       asset
                        skew
        Date                
        2005-12-21  0.000084
        2005-12-22  0.000067
        """
        return Side([block.wskew(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs) for block in tqdm(self.blocks)])


    def wstd(self, axis: int = None, skipna: bool = None, ddof: int = 1, numeric_only=None, **kwargs):
        """
        Return sample standard deviation over requested axis.

        Normalized by N-1 by default. This can be changed using the ddof argument

        Similar to `Pandas std <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.std.html>`__

        Args:
            axis (int): Axis for the function to be applied on.
            skipna (bool): Exclude NA/null values when computing the result.
            ddof (int): Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the number of elements.
            numeric_only (bool): Include only float, int or boolean data. If None, will attempt to use everything, then use only numeric data. Not implemented for Series.
            **kwargs: Additional keyword arguments to be passed to the function.

        Returns:
            ``DataBlock``: DataBlock with operation executed.

        Example:

        >>> side[0]
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> side[0].wstd()
                        AAPL                MSFT          
                        Open     Close      Open     Close
        2005-12-21  0.028307  0.011237  0.082436  0.072131
        """
        return Side([block.wstd(axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only, **kwargs) for block in tqdm(self.blocks)])


    def wsum(self, axis: int = None, skipna: bool = None, numeric_only=None, min_count: int = 0, **kwargs):
        """
        Return the sum of the values over the requested axis.

        This is equivalent to the method `numpy.sum`.

        Similar to `Pandas sum <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sum.html>`__

        Args:
            axis (int): Axis for the function to be applied on.
            skipna (bool): Exclude NA/null values when computing the result.
            numeric_only (bool): Include only float, int or boolean data. If None, will attempt to use everything, then use only numeric data. Not implemented for Series.
            min_count (int): The required number of valid values to perform the operation. If fewer than `min_count` non-NA values are present the result will be NA.
            **kwargs: Additional keyword arguments to be passed to the function.

        Returns:
            ``DataBlock``: DataBlock with operation executed.

        Example:

        >>> side[0]
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> side[0].wsum()
                        AAPL                  MSFT           
                        Open     Close        Open      Close
        2005-12-21  5.010849  5.080517  380.981498  377.29376
        """
        return Side([block.wsum(axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count, **kwargs) for block in tqdm(self.blocks)])


    def wvar(self, axis: int = None, skipna: bool = None, ddof: int = 1, numeric_only=None, **kwargs):
        """
        Return sample variance over requested axis.

        Normalized by N-1 by default. This can be changed using the ddof argument

        Similar to `Pandas var <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.var.html>`__

        Args:
            axis (int): Axis for the function to be applied on.
            skipna (bool): Exclude NA/null values when computing the result.
            ddof (int): Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the number of elements.
            numeric_only (bool): Include only float, int or boolean data. If None, will attempt to use everything, then use only numeric data. Not implemented for Series.
            **kwargs: Additional keyword arguments to be passed to the function.

        Returns:
            ``DataBlock``: DataBlock with operation executed.

        Example:

        >>> side[0]
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> side[0].wvar()
                        AAPL                MSFT          
                        Open     Close      Open     Close
        2005-12-21  0.000801  0.000126  0.006796  0.005203
        """
        return Side([block.wvar(axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only, **kwargs) for block in tqdm(self.blocks)])

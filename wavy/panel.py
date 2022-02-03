from matplotlib.pyplot import title
from tqdm.auto import tqdm
from typing import List, Union
import math

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


def create_panel(df,
              lookback:int,
              horizon:int,
              gap:int = 0):
    """
    Create a panel from a dataframe.

    Args:
        df (DataFrame): Values DataFrame
        lookback (int): lookback size
        horizon (int): horizon size
        gap (int): gap between x and y

    Returns:
        ``Panel``: Data Panel

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

    x_timesteps = len(df.index)

    if x_timesteps - lookback - horizon - gap <= -1:
        raise ValueError("Not enough timesteps to build")

    end = x_timesteps - horizon - gap + 1

    # Convert to blocks
    x = df
    y = df

    indexes = np.arange(lookback, end)
    xblocks, yblocks = [], []

    for i in indexes:
        xblocks.append(x.iloc[i - lookback : i])
        yblocks.append(y.iloc[i + gap : i + gap + horizon])

    return Panel(xblocks), Panel(yblocks)



class Panel:
    def __init__(self, blocks):
        # TODO: blocks must have increasing indexes, add warning and reindex
        # TODO this check should be done when creating the panel

        class _IXIndexer:
            def __getitem__(self, item):
                return Panel([i.ix[item] for i in blocks])
        class _iLocIndexer:
            def __getitem__(self, item):
                return Panel([i.iloc[item] for i in blocks])
        class _LocIndexer:
            def __getitem__(self, item):
                return Panel([i.loc[item] for i in blocks])
        class _AtIndexer:
            def __getitem__(self, item):
                return Panel([i.at[item] for i in blocks])
        class _iAtIndexer:
            def __getitem__(self, item):
                return Panel([i.iat[item] for i in blocks])

        self.blocks = blocks
        self.ix = _IXIndexer()
        self.iloc = _iLocIndexer()
        self.loc = _LocIndexer()
        self.at = _AtIndexer()
        self.iat = _iAtIndexer()

    def __getattr__(self, name):
        try:
            def wrapper(*args, **kwargs):
                return Panel([getattr(block, name)(*args, **kwargs) for block in self.blocks])
            return wrapper
        except AttributeError:
            raise AttributeError(f"'Panel' object has no attribute '{name}'")

    # TODO fix this method
    # Function to map all dunder functions
    def _one_arg(self, other, __f):
        if isinstance(other, Panel):
            return Panel([getattr(block, __f)(other_block) for block, other_block in zip(self.blocks, other)])
        return Panel([getattr(block, __f)(other) for block in self.blocks])

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
                "size": len(self),
                "block": len(self[0]),
            },
            name="Panel",
        )

        print(summary)
        return f"<Panel, size {self.__len__()}>"

    @property
    def columns(self):
        """
        Panel columns.

        Example:

        >>> side.columns
        {'Level 0': {'AAPL', 'MSFT'}, 'Level 1': {'Close', 'Open'}}
        """

        # dict = {}

        # for i in range(len(self[0].columns[0])):
        #     dict[f'Level {i}'] = set([col[i] for col in self[0].columns])

        # return dict
        return self[0].columns

    @property
    def index(self):
        # Maybe describe a bit better? What is the index of the side? Is it the first date of each block?
        # TODO returning all index, does it make sense??
        """
        Panel index.

        Example:

        >>> side.index
        DatetimeIndex(['2005-12-21', '2005-12-22', '2005-12-23'], dtype='datetime64[ns]', name='Date', freq=None)
        """

        df = pd.concat(self.blocks)
        df = df[~df.index.duplicated(keep="first")]
        return df.index

    @property
    def values(self):
        """
        3D matrix with Panel value.

        Example:

        >>> side.values
        array([[[19.57712554, 19.47512245,  2.21856582,  2.24606872],
                [19.46054323, 19.37311363,  2.25859845,  2.26195979]],
               [[19.46054323, 19.37311363,  2.25859845,  2.26195979],
                [19.32212198, 19.40955162,  2.26654326,  2.24148512]]])
        """
        return np.array([block.values for block in tqdm(self.blocks)])

    @property
    def shape(self):
        """
        Panel shape.

        Example:

        >>> side.shape
        (2, 2, 4)
        """
        
        return (len(self),) + self[0].shape

    def countna(self):
        """
        Count NaN cells for each Dataframe.

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

    def dropna(self):
        """
        Drop pairs with missing values from the panel.

        Similar to `Pandas dropna <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html>`__

        Returns:
            ``Panel``: Panel with missing values dropped.
        """
        nan_values = self.findna()
        idx = {i for i in range(len(self)) if i not in nan_values}
        if not idx:
            raise ValueError("'dropna' would create empty Panel")
        return self[idx]

    def findna(self):
        """
        Find NaN values index.

        Returns:
            ``List``: List with index of NaN values.
        """

        values =  pd.Series([block.values.sum() for block in self]).isna()
        return values[values == True].index.tolist()

    def findinf(self):
        """
        Find Inf values index.

        Returns:
            ``List``: List with index of Inf values.
        """
        values = np.isinf(pd.Series([block.values.sum() for block in self]))
        return values[values == True].index.tolist()

    def flat(self):
        # Maybe the name is confusing since there's another func called flatten?
        # TODO rename function
        """
        2D array with the flat value of each Block.

        Returns:
            ``DataFrame``: Result of flat function.

        Example:

        Panel containing two Block, will present the following result.

        >>> side[0]
                        MSFT                 AAPL
                        Open      Close      Open     Close
        Date
        2005-12-21  19.577126  19.475122  2.218566  2.246069
        2005-12-22  19.460543  19.373114  2.258598  2.261960

        >>> side[-1]
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

    def shift(self, window: int = 1):
        """
        Shift side by desired number of blocks.

        Similar to `Pandas shift <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.shift.html>`__

        Args:
            window (int): Number of blocks to shift

        Returns:
            ``Panel``: Result of shift function.

        Example:

        >>> side[0]
                        MSFT                 AAPL
                        Open      Close      Open     Close
        Date
        2005-12-21  19.577126  19.475122  2.218566  2.246069
        2005-12-22  19.460543  19.373114  2.258598  2.261960

        >>> side = side.shift(window = 1)

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
            new_values = self.blocks[new_index].values if new_index is not None else np.ones(self.blocks[0].shape) * np.nan
            # new_block = from_matrix(values=new_values, index=block.index, assets=block.assets, channels=block.channels)
            new_block = pd.DataFrame(data=new_values, index=block.index, columns=block.columns)
            new_side.append(new_block)

        return Panel(new_side)

    def diff(self, window: int = 1):
        """
        Difference between blocks.

        Similar to `Pandas diff <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.diff.html>`__

        Args:
            window (int): Number of blocks to diff

        Returns:
            ``Panel``: Result of diff function.

        Example:

        >>> side[0]
                        MSFT                 AAPL
                        Open      Close      Open     Close
        Date
        2005-12-21  19.577126  19.475122  2.218566  2.246069
        2005-12-22  19.460543  19.373114  2.258598  2.261960

        >>> side = side.diff(window = 1)

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
        return self - self.shift(window)

    def pct_change(self, window: int = 1):
        """
        Percentage change between the current and a prior block.

        Similar to `Pandas pct_change <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pct_change.html>`__

        Args:
            window (int): Number of blocks to calculate percent change

        Returns:
            ``Panel``: Result of pct_change function.

        Example:

        >>> side[0]
                        MSFT                 AAPL
                        Open      Close      Open     Close
        Date
        2005-12-21  19.577126  19.475122  2.218566  2.246069
        2005-12-22  19.460543  19.373114  2.258598  2.261960

        >>> side = side.pct_change(window = 1)

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
        a = self.shift(window)
        return (self - a) / a


    def plot_block(self, index):
        """
        Dataframe plot.

        Args:
            index (int): Panel index

        Returns:
            ``Plot``: Plotted data
        """
        cmap = px.colors.qualitative.Plotly

        columns_size = len(self.columns)

        fig = make_subplots(rows=math.ceil(columns_size/2), cols=2, subplot_titles=[' '.join(column) for column in self.columns])

        for i, column in enumerate(self.columns):
            c = cmap[i]

            x_df = self.blocks[index].loc[:, column]
            idx = x_df.index
            values = x_df.values.flatten()

            x_trace = go.Scatter(x=idx, y=values, line=dict(width=2, color=c), showlegend=False)

            row = math.floor(i/2)
            col = i % 2
            fig.add_trace(x_trace, row=row+1, col=col+1)
            # Remove empty dates
            # dt_all = pd.date_range(start=index[0],end=index[-1])
            # dt_obs = [d.strftime("%Y-%m-%d") for d in index]
            # dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d").tolist() if not d in dt_obs]
            # fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])

        fig.update_layout(
            template='simple_white',
            showlegend=True
            )

        # num_assets = len(self.assets)
        # for i, channel in enumerate(self.channels):
        #     fig['layout'][f'yaxis{i*num_assets+1}'].update({'title':channel})

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
        columns_size = len(self.columns)
        fig = make_subplots(rows=math.ceil(columns_size/2), cols=2, subplot_titles=[' '.join(column) for column in self.columns])
        # fig = make_subplots(rows=len(self.channels), cols=len(self.assets), subplot_titles=self.assets)

        # Add traces, one for each slider step
        len_ = np.linspace(0,len(self.blocks), steps, dtype=int, endpoint=False)
        for step in len_: #np.arange(len(panel_.x.blocks)):

            for i, column in enumerate(self.columns):
                c = cmap[i]

                x_df = self.blocks[step].loc[:, column]
                index = x_df.index
                values = x_df.values.flatten()

                x_trace = go.Scatter(visible=False, x=index, y=values, line=dict(width=2, color=c), showlegend=False)

                # x_trace = go.Scatter(x=index, y=values,
                #                     line=dict(width=2, color=c), showlegend=showlegend, name=channel)

                row = math.floor(i/2)
                col = i % 2
                fig.add_trace(x_trace, row=row+1, col=col+1)

                # dt_all = pd.date_range(start=index[0],end=index[-1])
                # dt_obs = [d.strftime("%Y-%m-%d") for d in index]
                # dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d").tolist() if not d in dt_obs]
                # fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])

        # Make 10th trace visible
        for i in range(columns_size):
            fig.data[i].visible = True

        # Create and add slider
        steps_ = []
        for i in range(steps):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig.data)},
                    {"title": "Block " + str(len_[i])}],  # layout attribute
            )

            for g in range(columns_size):
                step["args"][0]["visible"][i*columns_size+g] = True  # Toggle i'th trace to "visible"

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
        # num_assets = len(self.assets)
        # for i, channel in enumerate(self.channels):
        #     fig['layout'][f'yaxis{i*num_assets+1}'].update({'title':channel})

        fig.show()

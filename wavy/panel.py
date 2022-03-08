import warnings
from plotly.subplots import make_subplots
from tqdm.auto import tqdm

import math
import numpy as np
import pandas as pd
import plotly as px
import plotly.express as px
import plotly.graph_objects as go

# from .plot import plot, plot_slider

# dunder_methods = ['__abs__', '__add__', '__aenter__', '__aexit__', '__aiter__', '__and__', '__anext__', '__await__', '__bool__', '__bytes__', '__call__', '__ceil__', '__class__', '__class_getitem__', '__cmp__', '__coerce__', '__complex__', '__contains__', '__del__', '__delattr__', '__delete__', '__delitem__', '__delslice__', '__dict__', '__dir__', '__div__', '__divmod__', '__enter__', '__eq__', '__exit__', '__float__', '__floor__', '__floordiv__', '__format__', '__fspath__', '__ge__', '__get__', '__getattr__', '__getattribute__', '__getitem__', '__getnewargs__', '__getslice__', '__gt__', '__hash__', '__hex__', '__iadd__', '__iand__', '__idiv__', '__ifloordiv__', '__ilshift__', '__imatmul__', '__imod__', '__import__', '__imul__', '__index__', '__init__', '__init_subclass__', '__instancecheck__', '__int__', '__invert__', '__ior__', '__ipow__', '__irshift__', '__isub__', '__iter__', '__itruediv__', '__ixor__', '__le__', '__len__', '__length_hint__', '__long__', '__lshift__', '__lt__', '__matmul__', '__metaclass__', '__missing__', '__mod__', '__mro__', '__mul__', '__ne__', '__neg__', '__new__', '__next__', '__nonzero__', '__oct__', '__or__', '__pos__', '__pow__', '__prepare__', '__radd__', '__rand__', '__rcmp__', '__rdiv__', '__rdivmod__', '__reduce__', '__reduce_ex__', '__repr__', '__reversed__', '__rfloordiv__', '__rlshift__', '__rmatmul__', '__rmod__', '__rmul__', '__ror__', '__round__', '__rpow__', '__rrshift__', '__rshift__', '__rsub__', '__rtruediv__', '__rxor__', '__set__', '__set_name__', '__setattr__', '__setitem__', '__setslice__', '__sizeof__', '__slots__', '__str__', '__sub__', '__subclasscheck__', '__subclasses__', '__truediv__', '__trunc__', '__unicode__', '__weakref__', '__xor__']

DUNDER_METHODS = ['__add__', '__sub__', '__mul__', '__truediv__', '__ge__', '__gt__', '__le__', '__lt__', '__pow__']

# Plot
pd.set_option("multi_sparse", True)  # To see multilevel indexes
pd.options.plotting.backend = "plotly"

from copy import deepcopy


def create_panels(df,
                  lookback: int,
                  horizon: int,
                  gap: int = 0):
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

    # Convert to frames
    x = df
    y = df

    indexes = np.arange(lookback, end)
    xframes, yframes = [], []

    for i in indexes:
        # ? functions that create a new panel might keep old frame indexes
        x_frame = x.iloc[i - lookback : i]
        x_frame.frame_index = i
        xframes.append(x_frame)

        y_frame = y.iloc[i + gap : i + gap + horizon]
        y_frame.frame_index = i
        yframes.append(y_frame)

    return Panel(xframes), Panel(yframes)


class Panel:
    def __init__(self, frames):
        # ? Should frames always have increasing indexes? Maybe add warning and reindex
        # ? What about duplicated indices?

        class _IXIndexer:
            def __getitem__(self, item):
                return Panel([i.ix[item] for i in frames])

        class _iLocIndexer:
            def __getitem__(self, item):
                return Panel([i.iloc[item] for i in frames])

        class _LocIndexer:
            def __getitem__(self, item):
                return Panel([i.loc[item] for i in frames])

        class _AtIndexer:
            def __getitem__(self, item):
                return Panel([i.at[item] for i in frames])

        class _iAtIndexer:
            def __getitem__(self, item):
                return Panel([i.iat[item] for i in frames])

        self.frames = frames
        self.ix = _IXIndexer()
        self.iloc = _iLocIndexer()
        self.loc = _LocIndexer()
        self.at = _AtIndexer()
        self.iat = _iAtIndexer()

        self.set_training_split()

    def __getattr__(self, name):
        try:
            def wrapper(*args, **kwargs):
                return Panel([getattr(frame, name)(*args, **kwargs) for frame in self.frames])
            return wrapper
        except AttributeError:
            raise AttributeError(f"'Panel' object has no attribute '{name}'")

    # TODO fix this method
    # Function to map all dunder functions
    def _one_arg(self, other, __f):
        if isinstance(other, Panel):
            return Panel([getattr(frame, __f)(other_frame) for frame, other_frame in zip(self.frames, other)])
        return Panel([getattr(frame, __f)(other) for frame in self.frames])

    for dunder in DUNDER_METHODS:
        locals()[dunder] = lambda self, other, __f=dunder: self._one_arg(other, __f)

    def __getitem__(self, key):
        if isinstance(key, list):
            if all(isinstance(k, str) for k in key):
                return self.loc[:, key]
            return Panel([self.frames[i] for i in key])
        elif isinstance(key, slice):
            return Panel(self.frames.__getitem__(key))
        return self.frames.__getitem__(key)

    def __len__(self):
        return len(self.frames)

    def __repr__(self):
        summary = pd.Series(
            {
                "size": len(self),
                "timesteps": self.timesteps,
                "start": self.index.iloc[0],
                "end": self.index.iloc[-1]
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

        >>> panel.columns
        {'Level 0': {'AAPL', 'MSFT'}, 'Level 1': {'Close', 'Open'}}
        """

        return self[0].columns

    @property
    def index(self):
        """
        Returns the last index of each frame in the panel.

        Example:

        >>> panel.index
        DatetimeIndex(['2005-12-21', '2005-12-22', '2005-12-23'], dtype='datetime64[ns]', name='Date', freq=None)
        """

        return pd.Series([frame.index[-1] for frame in self.frames])

    @property
    def values(self):
        """
        3D matrix with Panel value.

        Example:

        >>> panel.values
        array([[[19.57712554, 19.47512245,  2.21856582,  2.24606872],
                [19.46054323, 19.37311363,  2.25859845,  2.26195979]],
               [[19.46054323, 19.37311363,  2.25859845,  2.26195979],
                [19.32212198, 19.40955162,  2.26654326,  2.24148512]]])
        """

        return np.array([frame.values for frame in tqdm(self.frames)])

    @property
    def timesteps(self):
        return len(self[0])

    @property
    def shape(self):
        """
        Panel shape.

        Example:

        >>> panel.shape
        (2, 2, 4)
        """

        return (len(self),) + self[0].shape

    def countna(self):
        """
        Count NaN cells for each Dataframe.

        Returns:
            ``DataFrame``: NaN count for each frame.

        Example:

        >>> panel.countna()
           nan
        0    2
        1    2
        """
        values = [frame.isnull().values.sum() for frame in tqdm(self.frames)]
        return pd.DataFrame(values, index=range(len(self.frames)), columns=['nan'])

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

        values = pd.Series([frame.values.sum() for frame in self]).isna()
        return values[values == True].index.tolist()

    def findinf(self):
        """
        Find Inf values index.

        Returns:
            ``List``: List with index of Inf values.
        """
        values = np.isinf(pd.Series([frame.values.sum() for frame in self]))
        return values[values == True].index.tolist()

    def as_dataframe(self, flatten=False):
        # TODO: Add column names instead of only index
        """

        flatten=True will return one panel frame per row

        Example:

        Panel containing two frames will present the following result:

        >>> panel[0]
                        MSFT                 AAPL
                        Open      Close      Open     Close
        Date
        2005-12-21  19.577126  19.475122  2.218566  2.246069
        2005-12-22  19.460543  19.373114  2.258598  2.261960

        >>> panel.as_dataframe(flatten=True)
                           0         1        2        3         4         5        6        7
        2005-12-22 19.577126 19.475122 2.218566 2.246069 19.460543 19.373114 2.258598 2.261960
        2005-12-23 19.460543 19.373114 2.258598 2.261960 19.322122 19.409552 2.266543 2.241485
        """

        if flatten:
            values = np.array([i.values.flatten() for i in self.frames])
            index = [i.index[-1] for i in self.frames]
            return pd.DataFrame(values, index=index)

        else:
            df = self[0].copy()
            for i in self[1:]:
                df = pd.concat([df, i[self.timesteps - 1:]])
            return df

    def shift(self, window: int = 1):
        """
        Shift panel by desired number of frames.

        Similar to `Pandas shift <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.shift.html>`__

        Args:
            window (int): Number of frames to shift

        Returns:
            ``Panel``: Result of shift function.

        Example:

        >>> panel[0]
                        MSFT                 AAPL
                        Open      Close      Open     Close
        Date
        2005-12-21  19.577126  19.475122  2.218566  2.246069
        2005-12-22  19.460543  19.373114  2.258598  2.261960

        >>> panel = panel.shift(window = 1)

        >>> panel[0]
                MSFT       AAPL
                Open Close Open Close
        Date
        2005-12-21  NaN   NaN  NaN   NaN
        2005-12-22  NaN   NaN  NaN   NaN

        >>> panel[1]
                        MSFT                 AAPL
                        Open      Close      Open     Close
        Date
        2005-12-21  19.577126  19.475122  2.218566  2.246069
        2005-12-22  19.460543  19.373114  2.258598  2.261960
        """

        new_panel = []

        for i, frame in enumerate(self.frames):
            new_index = i - window
            new_index = new_index if new_index >= 0 and new_index < len(self.frames) else None
            new_values = self.frames[new_index].values if new_index is not None else np.ones(self.frames[0].shape) * np.nan
            new_frame = pd.DataFrame(data=new_values, index=frame.index, columns=frame.columns)
            new_panel.append(new_frame)

        return Panel(new_panel)

    def diff(self, window: int = 1):
        """
        Difference between frames.

        Similar to `Pandas diff <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.diff.html>`__

        Args:
            window (int): Number of frames to diff

        Returns:
            ``Panel``: Result of diff function.

        Example:

        >>> panel[0]
                        MSFT                 AAPL
                        Open      Close      Open     Close
        Date
        2005-12-21  19.577126  19.475122  2.218566  2.246069
        2005-12-22  19.460543  19.373114  2.258598  2.261960

        >>> panel = panel.diff(window = 1)

        >>> panel[0]
                MSFT       AAPL
                Open Close Open Close
        Date
        2005-12-21  NaN   NaN  NaN   NaN
        2005-12-22  NaN   NaN  NaN   NaN

        >>> panel[1]
                        MSFT                AAPL
                        Open     Close      Open     Close
        Date
        2005-12-22 -0.116582 -0.102009  0.040033  0.015891
        2005-12-23 -0.138421  0.036438  0.007945 -0.020475
        """

        return self - self.shift(window)

    def pct_change(self, window: int = 1):
        """
        Percentage change between the current and a prior frame.

        Similar to `Pandas pct_change <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pct_change.html>`__

        Args:
            window (int): Number of frames to calculate percent change

        Returns:
            ``Panel``: Result of pct_change function.

        Example:

        >>> panel[0]
                        MSFT                 AAPL
                        Open      Close      Open     Close
        Date
        2005-12-21  19.577126  19.475122  2.218566  2.246069
        2005-12-22  19.460543  19.373114  2.258598  2.261960

        >>> panel = panel.pct_change(window = 1)

        >>> panel[0]
                MSFT       AAPL
                Open Close Open Close
        Date
        2005-12-21  NaN   NaN  NaN   NaN
        2005-12-22  NaN   NaN  NaN   NaN

        >>> panel[1]
                        MSFT                AAPL
                        Open     Close      Open     Close
        Date
        2005-12-22 -0.005955 -0.005238  0.018044  0.007075
        2005-12-23 -0.007113  0.001881  0.003518 -0.009052
        """
        a = self.shift(window)
        return (self - a) / a

    def match(self, other):
        """
        Modify using values from another Panel. Aligns on indices.

        Args:
            other: (Panel)

        Returns:
            ``Panel``: Result of match function.
        """
        index = [frame.frame_index for frame in other]
        return Panel([frame for frame in tqdm(self.frames) if frame.frame_index in index])

    def set_training_split(self, val_size=0.2, test_size=0.1):
        """
        Time series split into training, validation, and test sets, avoiding data leakage.
        Splits the panel in training, validation, and test panels, accessed with the properties
        .train, .val and .test. The sum of the three sizes inserted must equals one.
        Args:
            val_size (float): Percentage of data used for the validation set.
            test_size (float): Percentage of data used for the test set.
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

    def update(self, other):
        panel = deepcopy(self)

        df = panel.as_dataframe()
        if len(df) != len(other):
            if not isinstance(other, pd.DataFrame):
                raise ValueError("If using different sizes, other must be a DataFrame")

            assert not all(other.index.duplicated())
            warnings.warn("Sizes don't match. Using dataframe indexes and columns to update.")
            other = other.loc[df.index, df.columns].values

        for i, j in zip(panel, other):
            i.iloc[:,:] = j
        return panel

    def resample(self, samples: int = 1, type: str = 'first'):
        """
        Resample panel returning a subset of frames.

        Args:
            samples (int): Number of samples to keep
            type (str): Resempling type, 'first', 'last' or 'spaced'

        Returns:
            ``Panel``: Result of resample function.
        """

        if type == 'first':
            return self[:samples]
        elif type == 'last':
            return self[-samples:]
        elif type == 'spaced':
            indexes = list(np.linspace(0, len(self.frames), samples, dtype=int, endpoint=False))
            return self[indexes]

    @property
    def train(self):
        """
        Returns the Panel with the pairs of the training set, according to
        the parameters given in the 'set_train_val_test_sets' function.
        Returns:
            ``Panel``: Panel with the pairs of the training set.
        """

        if self.train_size:
            return self[:self.train_size]

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

    # def plot(self, split_sets=False, **kwargs):
    #     return plot(self, split_sets=split_sets, **kwargs)

    # def plot_slider(self, steps):
    #     return plot_slider(self, steps)
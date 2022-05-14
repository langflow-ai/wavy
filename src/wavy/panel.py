import itertools
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from wavy.plot import plot, plot_slider

_ARG_0_METHODS = [
    "__abs__",
    "__pos__",
    "__neg__",
    "__invert__",
]
_ARG_1_METHODS = [
    "__add__",
    "__sub__",
    "__mul__",
    "__rmul__",
    "__truediv__",
    "__ge__",
    "__gt__",
    "__le__",
    "__lt__",
    "__pow__",
    "__eq__",
    "__ne__",
    "__bool__",
    "__floordiv__",
    "__rfloordiv__",
    "__matmul__",
    "__rmatmul__",
    "__rmod__",
    "__mod__",
]

# Plot
pd.set_option("multi_sparse", True)  # To see multilevel indexes
pd.options.plotting.backend = "plotly"


def create_panels(df, lookback: int, horizon: int, gap: int = 0):
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

    >>> x, y = wavy.create_panels(hist, lookback=2, horizon=1)
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
        # ? Would it make sense to have the panel as only one dataframe with index references per frame window?

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

    # TODO: add setattr, e.g. for renaming columns ()

    def __getattr__(self, name):
        try:

            def wrapper(*args, **kwargs):
                return Panel(
                    [getattr(frame, name)(*args, **kwargs) for frame in self.frames]
                )

            return wrapper
        except AttributeError:
            raise AttributeError(f"'Panel' object has no attribute '{name}'")

    # Function to map all dunder functions
    def _1_arg(self, other, __f):
        if isinstance(other, Panel):
            return Panel(
                [
                    getattr(frame, __f)(other_frame)
                    for frame, other_frame in zip(self.frames, other.frames)
                ]
            )
        return Panel([getattr(frame, __f)(other) for frame in self.frames])

    for _dunder in _ARG_1_METHODS:
        locals()[_dunder] = lambda self, other, __f=_dunder: self._1_arg(other, __f)

    def _0_arg(self, __f):
        return Panel([getattr(frame, __f)() for frame in self.frames])

    for _dunder in _ARG_0_METHODS:
        locals()[_dunder] = lambda self, __f=_dunder: self._0_arg(__f)

    def __getitem__(self, key):

        assert isinstance(
            key, (int, slice, list, str, tuple)
        ), "Panel indexing must be int, slice, list, str or tuple"
        if isinstance(key, list):
            assert all(isinstance(k, int) for k in key) or all(
                isinstance(k, str) for k in key
            ), "Panel indexing with list must be int or str"
        if isinstance(key, tuple):
            index = key[0]
            columns = key[1]
            assert isinstance(
                index, (int, slice, list)
            ), "Panel indexing rows with tuple must be int, slice or list"
            assert isinstance(
                columns, (str, list)
            ), "Panel indexing columns with tuple must be str or list"

            if isinstance(index, list):
                assert all(
                    isinstance(k, int) for k in index
                ), "Panel indexing rows with list must be int"
            if isinstance(columns, list):
                assert all(
                    isinstance(k, str) for k in columns
                ), "Panel indexing columns with list must be str"

        # Tuple
        if isinstance(key, tuple):
            index = key[0]
            columns = key[1] if isinstance(key[1], list) else [key[1]]

            if isinstance(index, int):
                return self.frames[index].loc[:, columns]
            elif isinstance(index, slice):
                return Panel(self.frames[index]).loc[:, columns]
            elif isinstance(index, list):
                return Panel([self.frames[i] for i in index]).loc[:, columns]

        # Columns
        if isinstance(key, list) and all(isinstance(k, str) for k in key):
            return self.loc[:, key]
        elif isinstance(key, str):
            return self.loc[:, [key]]

        # Index
        if isinstance(key, int):
            return self.frames[key]
        elif isinstance(key, slice):
            return Panel(self.frames[key])
        elif isinstance(key, list) and all(isinstance(k, int) for k in key):
            return Panel([self.frames[i] for i in key])

    def __len__(self):
        return len(self.frames)

    def __str__(self):
        summary = pd.Series(
            {
                "size": len(self),
                "timesteps": self.timesteps,
                "start": self.index.iloc[0],
                "end": self.index.iloc[-1],
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
        Index(['Open', 'High', 'Low', 'Close'], dtype='object')
        """

        return self[0].columns

    @property
    def index(self):
        """
        Returns the last index of each frame in the panel.

        Example:

        >>> panel.index
        0   2022-05-04
        1   2022-05-05
        2   2022-05-06
        3   2022-05-09
        dtype: datetime64[ns]
        """

        return pd.Series([frame.index[-1] for frame in self.frames])

    @property
    def values(self, verbose=False):
        """
        3D matrix with Panel value.

        Example:

        >>> panel.values
        array([[[283.95999146, 284.13000488, 280.1499939 , 281.77999878],
                [282.58999634, 290.88000488, 276.73001099, 289.98001099]],
               [[282.58999634, 290.88000488, 276.73001099, 289.98001099],
                [285.54000854, 286.3500061 , 274.33999634, 277.3500061 ]],
               [[285.54000854, 286.3500061 , 274.33999634, 277.3500061 ],
                [274.80999756, 279.25      , 271.26998901, 274.73001099]],
               [[274.80999756, 279.25      , 271.26998901, 274.73001099],
                [270.05999756, 272.35998535, 263.32000732, 264.57998657]]])
        """

        return np.array(
            [frame.values for frame in tqdm(self.frames, disable=not verbose)]
        )

    # @property
    # def timesteps(self):
    #     return len(self[0])

    @property
    def shape(self):
        """
        Panel shape.

        Example:

        >>> panel.shape
        (4, 2, 4)
        """

        return (len(self),) + self[0].shape

    def countna(self, verbose=False):
        """
        Count NaN cells for each Dataframe.

        Returns:
            ``DataFrame``: NaN count for each frame.

        Example:

        >>> panel.countna()
           nan
        0    0
        1    0
        2    0
        3    0
        """
        values = [
            frame.isnull().values.sum()
            for frame in tqdm(self.frames, disable=not verbose)
        ]
        return pd.DataFrame(values, index=range(len(self.frames)), columns=["nan"])

    def dropna(self, axis=0, how="any", thresh=None, subset=None, verbose=False):
        """
        Drop rows or columns with missing values.

        Similar to `Pandas dropna <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html>`__

        Args:
            axis (int): The axis to drop on.
            how (str): Method to use for dropping.
            thresh (int): The number of non-NA values to require.
            subset (list): List of columns to check.
            verbose (bool): Whether to print progress.

        Returns:
            ``Panel``: Dropped panel.
        """
        new_panel = Panel(
            [
                frame.dropna(axis=axis, how=how, thresh=thresh, subset=subset)
                for frame in self.frames
            ]
        )

        different = any(
            new_panel[i].shape != self[i].shape for i in range(len(new_panel))
        )

        if different:
            warnings.warn("Dropped frames have different shape")

        return new_panel

    def wdropna(self):
        """
        Drop frames with missing values from the panel.

        Returns:
            ``Panel``: Panel with missing values dropped.
        """
        nan_values = self.findna()
        idx = [i for i in range(len(self)) if i not in nan_values]
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
        return values[values].index.tolist()

    def findinf(self):
        """
        Find Inf values index.

        Returns:
            ``List``: List with index of Inf values.
        """
        values = np.isinf(pd.Series([frame.values.sum() for frame in self]))
        return values[values].index.tolist()

    def as_dataframe(self, flatten=False):
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
            columns = [
                f"{str(i)}-{col}"
                for i, col in itertools.product(range(self.shape[1]), self[0].columns)
            ]
            values = np.array([i.values.flatten() for i in self.frames])
            index = [i.index[-1] for i in self.frames]
            return pd.DataFrame(values, index=index, columns=columns)

        else:
            df = pd.DataFrame()

            for i in range(self.shape[1]):
                a = self[i].head(1).copy()
                a["frame"] = self[i].frame_index
                df = pd.concat([df, a])
            return df

    def shiftw(self, window: int = 1):
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
            new_index = (
                new_index if new_index >= 0 and new_index < len(self.frames) else None
            )
            new_values = (
                self.frames[new_index].values
                if new_index is not None
                else np.ones(self.frames[0].shape) * np.nan
            )
            new_frame = pd.DataFrame(
                data=new_values, index=frame.index, columns=frame.columns
            )
            new_panel.append(new_frame)

        return Panel(new_panel)

    def diffw(self, window: int = 1):
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

    def pct_changew(self, window: int = 1):
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

    def match(self, other, verbose=False):
        """
        Modify using values from another Panel. Aligns on indices.

        Args:
            other: (Panel)

        Returns:
            ``Panel``: Result of match function.
        """
        index = [frame.frame_index for frame in other]
        return Panel(
            [
                frame
                for frame in tqdm(self.frames, disable=not verbose)
                if frame.frame_index in index
            ]
        )

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
        """
        Update the panel with values from another panel.

        Args:
            other (Panel): Panel to update with.

        Returns:
            ``Panel``: Result of update function.
        """
        panel = deepcopy(self)

        df = panel.as_dataframe()
        if len(df) != len(other):
            if not isinstance(other, pd.DataFrame):
                raise ValueError("If using different sizes, other must be a DataFrame")

            assert not all(other.index.duplicated())
            warnings.warn(
                "Sizes don't match. Using dataframe indexes and columns to update."
            )
            other = other.loc[df.index, df.columns].values

        for i, j in zip(panel, other):
            i.iloc[:, :] = j
        return panel

    def head(self, n: int = 5):
        """
        Return the first n frames of the panel.

        Args:
            n (int): Number of frames to return.

        Returns:
            ``Panel``: Result of head function.
        """
        return Panel(self.frames[:n])

    def tail(self, n: int = 5):
        """
        Return the last n frames of the panel.

        Args:
            n (int): Number of frames to return.

        Returns:
            ``Panel``: Result of tail function.
        """
        return Panel(self.frames[-n:])

    def sample(self, samples: int = 5, type: str = "first"):
        """
        Sample panel returning a subset of frames.

        Args:
            samples (int): Number of samples to keep
            type (str): Resempling type, 'first', 'last' or 'spaced'

        Returns:
            ``Panel``: Result of sample function.
        """

        if type == "first":
            return self[:samples]
        elif type == "last":
            return self[-samples:]
        elif type == "spaced":
            indexes = list(
                np.linspace(0, len(self.frames), samples, dtype=int, endpoint=False)
            )
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

    def plot(self, split_sets=True, **kwargs):
        """
        Plot the panel.

        Args:
            split_sets (bool): If True, plot the training, validation, and test sets.
            **kwargs: Additional arguments to pass to the plot function.

        Returns:
            ``plot``: Result of plot function.
        """
        return plot(self, split_sets=split_sets, **kwargs)

    def plot_slider(self, steps):
        """
        Plot the panel using a slider.

        Args:
            steps (int): Number of steps to plot.

        Returns:
            ``plot_slider``: Result of plot_slider function.
        """
        return plot_slider(self, steps)

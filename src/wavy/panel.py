import itertools
import math
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from wavy.plot import plot

# TODO Add reset index when comparing two panels

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


def create_panels(df, lookback: int, horizon: int, gap: int = 0, verbose=False):
    """
    Create a panel from a dataframe.

    Args:
        df (DataFrame): Values DataFrame
        lookback (int): lookback size
        horizon (int): horizon size
        gap (int): gap between x and y
        verbose (bool): Whether to print progress

    Returns:
        ``Panel``: Data Panel

    Example:

    >>> x, y = wavy.create_panels(hist, lookback=2, horizon=1)
    """

    indices = df.index

    # Sort by index
    df = df.sort_index(ascending=True)

    if not all(df.index == indices):
        warnings.warn("DataFrame is being sorted!")

    x_timesteps = len(df.index)

    if x_timesteps - lookback - horizon - gap <= -1:
        raise ValueError("Not enough timesteps to build")

    end = x_timesteps - horizon - gap + 1

    indexes = np.arange(lookback, end)
    xframes, yframes = [], []

    for index, i in enumerate(tqdm(indexes, disable=not verbose)):
        # ? functions that create a new panel might keep old frame indexes
        frame = df.iloc[i - lookback : i].copy()
        frame.columns.name = index
        xframes.append(frame)

        frame = df.iloc[i + gap : i + gap + horizon].copy()
        frame.columns.name = index
        yframes.append(frame)

    a, b = Panel(xframes), Panel(yframes)
    return a, b


def shallow_copy(panel, frames=None, train_size=None, test_size=None, val_size=None):
    """
    Shallow copy of a panel.
    """
    if frames is None:
        frames = []
    elif isinstance(frames[0], pd.Series):
        frames = [pd.DataFrame(frame).T for frame in frames]
    new_panel = Panel(frames)
    new_panel.train_size = train_size if train_size is not None else panel.train_size
    new_panel.test_size = test_size if test_size is not None else panel.test_size
    new_panel.val_size = val_size if val_size is not None else panel.val_size

    return new_panel


class Panel:
    def __init__(self, frames):
        # ? What about duplicated indices?
        # ? Would it make sense to have the panel as only one dataframe with index references per frame window?

        class _IXIndexer:
            def __init__(self, outer):
                self.outer = outer

            def __getitem__(self, item):
                return shallow_copy(self.outer, [i.ix[item] for i in self.outer.frames])

        class _iLocIndexer:
            def __init__(self, outer):
                self.outer = outer

            def __getitem__(self, item):
                return shallow_copy(
                    self.outer, [i.iloc[item] for i in self.outer.frames]
                )

        class _LocIndexer:
            def __init__(self, outer):
                self.outer = outer

            def __getitem__(self, item):
                return shallow_copy(
                    self.outer, [i.loc[item] for i in self.outer.frames]
                )

        class _AtIndexer:
            def __init__(self, outer):
                self.outer = outer

            def __getitem__(self, item):
                return shallow_copy(self.outer, [i.at[item] for i in self.outer.frames])

        class _iAtIndexer:
            def __init__(self, outer):
                self.outer = outer

            def __getitem__(self, item):
                return shallow_copy(
                    self.outer, [i.iat[item] for i in self.outer.frames]
                )

        class _IXIndexer:
            def __init__(self, outer):
                self.outer = outer

            def __getitem__(self, item):
                return shallow_copy(self.outer, [i.ix[item] for i in self.outer.frames])

        class _iLocIndexer:
            def __init__(self, outer):
                self.outer = outer

            def __getitem__(self, item):
                return shallow_copy(
                    self.outer, [i.iloc[item] for i in self.outer.frames]
                )

        class _LocIndexer:
            def __init__(self, outer):
                self.outer = outer

            def __getitem__(self, item):
                return shallow_copy(
                    self.outer, [i.loc[item] for i in self.outer.frames]
                )

        class _AtIndexer:
            def __init__(self, outer):
                self.outer = outer

            def __getitem__(self, item):
                return shallow_copy(self.outer, [i.at[item] for i in self.outer.frames])

        class _iAtIndexer:
            def __init__(self, outer):
                self.outer = outer

            def __getitem__(self, item):
                return shallow_copy(
                    self.outer, [i.iat[item] for i in self.outer.frames]
                )

        self.frames = frames
        self.ix = _IXIndexer(self)
        self.iloc = _iLocIndexer(self)
        self.loc = _LocIndexer(self)
        self.at = _AtIndexer(self)
        self.iat = _iAtIndexer(self)

        self.set_training_split()

    # TODO: add setattr, e.g. for renaming columns ()

    def __getattr__(self, name):
        try:

            def wrapper(*args, **kwargs):
                frames = []
                for i, frame in enumerate(self.frames):
                    new_frame = getattr(frame.copy(), name)(*args, **kwargs)
                    if isinstance(new_frame, pd.Series):
                        new_frame = pd.DataFrame(new_frame).T
                        new_frame.index = frame.index[: len(new_frame.index)]
                    frames.append(new_frame)
                return Panel(frames)

                # new_frames = []
                # for frame in range(len(self.frames)):
                #     new_frame = getattr(frame, name)(*args, **kwargs)
                #     if isinstance(new_frame, pd.Series):
                #         new_frame = pd.DataFrame(new_frame).T
                #     self.frames[0] = new_frame
                #     new_frames.append(new_frame)
                #     return Panel(new_frames)

            return wrapper
        except AttributeError:
            raise AttributeError(f"'Panel' object has no attribute '{name}'")

    # Function to map all dunder functions
    def _1_arg(self, other, __f):
        if isinstance(other, Panel):

            # Reset index
            frames_1 = self.reset_index(drop=True)
            frames_2 = other.reset_index(drop=True)

            return shallow_copy(
                self,
                [
                    getattr(frame, __f)(other_frame)
                    for frame, other_frame in zip(frames_1, frames_2)
                ],
            )
        return shallow_copy(self, [getattr(frame, __f)(other) for frame in self.frames])

    for _dunder in _ARG_1_METHODS:
        locals()[_dunder] = lambda self, other, __f=_dunder: self._1_arg(other, __f)

    def _0_arg(self, __f):
        return shallow_copy(self, [getattr(frame, __f)() for frame in self.frames])

    for _dunder in _ARG_0_METHODS:
        locals()[_dunder] = lambda self, __f=_dunder: self._0_arg(__f)

    def __getitem__(self, key):
        # TODO: add support for dates x['2016':] (reference pandas)
        # TODO: add support for exclusion / set operation, e.g. x[x.index < 1000 & x.index > 2000] (reference pandas)
        # ? What if the panel was a np.array of frames, instead of a list of frames?

        assert isinstance(
            key, (int, slice, list, str, tuple)
        ), "Panel indexing must be int, slice, list, str or tuple"
        if isinstance(key, list):
            # TODO: check if list is a list of ints or strings. If not, try casting before raising error.
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
                return shallow_copy(self, self.frames[index]).loc[:, columns]
            elif isinstance(index, list):
                return shallow_copy(self, [self.frames[i] for i in index]).loc[
                    :, columns
                ]

        # Columns
        if isinstance(key, list) and all(isinstance(k, str) for k in key):
            return self.loc[:, key]
        elif isinstance(key, str):
            return self.loc[:, [key]]

        # Index
        if isinstance(key, int):
            return self.frames[key]
        elif isinstance(key, slice):
            return shallow_copy(self, self.frames[key])
        elif isinstance(key, list) and all(isinstance(k, int) for k in key):
            return shallow_copy(self, [self.frames[i] for i in key])

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
    def indices(self):
        """
        Return the indices of the panel.
        """
        return np.array([frame.columns.name for frame in self.frames])

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

    @property
    def timesteps(self):
        """
        Number of timesteps in the panel.

        Example:

        >>> panel.timesteps
        4
        """
        return len(self[0])

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
        new_panel = shallow_copy(self)
        new_panel.frames = [
            frame.dropna(axis=axis, how=how, thresh=thresh, subset=subset)
            for frame in self.frames
        ]
        # new_panel = Panel(
        #     [
        #         frame.dropna(axis=axis, how=how, thresh=thresh, subset=subset)
        #         for frame in self.frames
        #     ]
        # )

        different = any(
            new_panel[i].shape != self[i].shape for i in range(len(new_panel))
        )

        if different:
            warnings.warn("Dropped frames have different shape")

        return new_panel

    def dropnaw(self):
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

    def as_dataframe(self, flatten=False, frame=False):
        """
        Convert panel to DataFrame.

        Args:
            flatten (bool): Whether to flatten the panel.
            frame (bool): Whether to include frame or not.

        Returns:
            ``DataFrame``: Panel as DataFrame.

        Example:

        Panel containing two frames will present the following result:

        >>> panel[0]
                          Open        High         Low       Close
        Date
        2022-05-06  274.809998  279.250000  271.269989  274.730011
        2022-05-09  270.059998  272.359985  263.320007  264.579987

        >>> panel.as_dataframe()
                          Open        High         Low       Close  frame
        Date
        2022-05-06  274.809998  279.250000  271.269989  274.730011      2
        2022-05-09  270.059998  272.359985  263.320007  264.579987      3

        >>> panel.as_dataframe(flatten=True)
                        0-Open      0-High       0-Low     0-Close      1-Open      1-High       1-Low     1-Close
        2022-05-09  274.809998  279.250000  271.269989  274.730011  270.059998  272.359985  263.320007  264.579987
        2022-05-10  270.059998  272.359985  263.320007  264.579987  271.690002  273.750000  265.070007  269.500000
        2022-05-11  271.690002  273.750000  265.070007  269.500000  265.679993  271.359985  259.299988  260.549988
        2022-05-12  265.679993  271.359985  259.299988  260.549988  257.690002  259.880005  250.020004  255.350006
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
            list_frames = []
            for i in range(self.shape[0]):
                a = self[i].head(1).copy()
                if frame:
                    a["frame"] = self[i].columns.name
                list_frames.append(a)
            dataframe = pd.concat(list_frames)
            dataframe.drop_duplicates(inplace=True)
            return dataframe

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
                          Open        High         Low       Close
        Date
        2022-05-06  274.809998  279.250000  271.269989  274.730011
        2022-05-09  270.059998  272.359985  263.320007  264.579987

        >>> panel = panel.shift(window = 1)

        >>> panel[0]
                   Open  High  Low Close
        Date
        2022-05-06  NaN   NaN  NaN   NaN
        2022-05-09  NaN   NaN  NaN   NaN

        >>> panel[1]
                          Open        High         Low       Close
        Date
        2022-05-06  274.809998  279.250000  271.269989  274.730011
        2022-05-09  270.059998  272.359985  263.320007  264.579987
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

        return shallow_copy(self, new_panel)

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
                          Open        High         Low       Close
        Date
        2022-05-06  274.809998  279.250000  271.269989  274.730011
        2022-05-09  270.059998  272.359985  263.320007  264.579987

        >>> panel = panel.diff(window = 1)

        >>> panel[0]
                   Open  High  Low Close
        Date
        2022-05-06  NaN   NaN  NaN   NaN
        2022-05-09  NaN   NaN  NaN   NaN

        >>> panel[1]
                        Open      High       Low      Close
        Date
        2022-05-09 -4.750000 -6.890015 -7.949982 -10.150024
        2022-05-10  1.630005  1.390015  1.750000   4.920013
        """

        return self - self.shiftw(window)

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
                          Open        High         Low       Close
        Date
        2022-05-06  274.809998  279.250000  271.269989  274.730011
        2022-05-09  270.059998  272.359985  263.320007  264.579987

        >>> panel = panel.pct_change(window = 1)

        >>> panel[0]
                    Open  High  Low  Close
        Date
        2022-05-06   NaN   NaN  NaN    NaN
        2022-05-09   NaN   NaN  NaN    NaN

        >>> panel[1]
                        Open      High       Low     Close
        Date
        2022-05-09 -0.017285 -0.024673 -0.029307 -0.036945
        2022-05-10  0.006036  0.005104  0.006646  0.018596
        """
        a = self.shiftw(window)
        return (self - a) / a

    def match(self, other, verbose=False):
        """
        Modify using values from another Panel. Aligns on indices.

        Args:
            other: (Panel)

        Returns:
            ``Panel``: Result of match function.
        """
        panel = shallow_copy(
            self,
            [
                frame
                for frame in tqdm(self.frames, disable=not verbose)
                if frame.columns.name in other.indices
            ],
        )
        panel.test_size = other.test_size
        panel.val_size = other.val_size
        panel.train_size = other.train_size
        return panel

    def set_training_split(self, test_size=0.1, val_split=0.2):
        # BUG: train and test are swapped. This can be noticed when using the panel.plot() function.
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

        self.train_size = 1 - test_size
        self.test_size = test_size
        self.val_size = val_split * self.train_size
        self.train_size = self.train_size - self.val_size
        assert math.isclose(
            self.train_size + self.val_size + self.test_size, 1, abs_tol=1e-6
        )

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
        return self[:n]

    def tail(self, n: int = 5):
        """
        Return the last n frames of the panel.

        Args:
            n (int): Number of frames to return.

        Returns:
            ``Panel``: Result of tail function.
        """
        return self[-n:]

    def sample(self, samples: int = 5, how: str = "first"):
        """
        Sample panel returning a subset of frames.

        Args:
            samples (int): Number of samples to keep
            how (str): Resempling how, 'first', 'last' or 'spaced'

        Returns:
            ``Panel``: Result of sample function.
        """

        if how == "first":
            return self[:samples]
        elif how == "last":
            return self[-samples:]
        elif how == "spaced":
            indexes = np.linspace(
                0, len(self.frames), samples, dtype=int, endpoint=False
            )
            indexes = [int(i) for i in indexes]
            return self[indexes]
        # TODO: Add random sample too

    @property
    def train(self):
        """
        Returns the Panel with the pairs of the training set, according to
        the parameters given in the 'set_train_val_test_sets' function.
        Returns:
            ``Panel``: Panel with the pairs of the training set.
        """
        if self.train_size == 0:
            return None
        panel = self[: int(self.train_size * len(self))]
        return shallow_copy(
            None, frames=panel.frames, train_size=len(panel), test_size=0, val_size=0
        )

    @property
    def val(self):
        """
        Returns the Panel with the pairs of the validation set, according to
        the parameters given in the 'set_train_val_test_sets' function.
        Returns:
            ``Panel``: Panel with the pairs of the validation set.
        """
        if self.val_size == 0:
            return None
        panel = self[
            int(self.train_size * len(self)) : int(
                (self.train_size + self.val_size) * len(self)
            )
        ]
        return shallow_copy(
            None, frames=panel.frames, train_size=0, test_size=len(panel), val_size=0
        )

    @property
    def test(self):
        """
        Returns the Panel with the pairs of the testing set, according to
        the parameters given in the 'set_train_val_test_sets' function.
        Returns:
            ``Panel``: Panel with the pairs of the testing set.
        """
        if self.test_size == 0:
            return None
        panel = self[int((self.train_size + self.val_size) * len(self)) :]
        return shallow_copy(
            None, frames=panel.frames, train_size=0, test_size=0, val_size=len(panel)
        )

    def plot(self, split_sets=True, max=10_000, **kwargs):
        """
        Plot the panel.

        Args:
            split_sets (bool): If True, plot the training, validation, and test sets.
            **kwargs: Additional arguments to pass to the plot function.

        Returns:
            ``plot``: Result of plot function.
        """
        if max and len(self) > max:
            return plot(self.sample(max, how="spaced"), split_sets=split_sets, **kwargs)
        return plot(self, split_sets=split_sets, **kwargs)

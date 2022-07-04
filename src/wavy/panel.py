import itertools
import math
import random
import warnings
from copy import deepcopy
from itertools import chain
from typing import Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from wavy.plot import plot
from wavy.utils import is_dataframe, is_iterable, is_series
from wavy.validations import _validate_training_split

_ARG_0_METHODS = [
    "__abs__",
    "__pos__",
    "__neg__",
    "__invert__",
]
_ARG_1_METHODS = [
    "__add__",
    "__radd__",
    "__sub__",
    "__rsub__",
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
    "__mod__",
    "__rmod__",
]

# TODO
# 1. erro se só um por ex for series
# 5. Mudar td q usa primeiro index pra id
# 7. checar todos os exclamacao e todos e finalizar! (lol)
# 8. One function for both dunder functions
# 9. Check keras.preprocessing.timeseries_dataset_from_array


# Plot
pd.set_option("multi_sparse", True)  # To see multilevel indexes
pd.options.plotting.backend = "plotly"


def is_panel(x):
    """
    Check if x is a panel.

    Args:
        x (object): Object to check

    Returns:
        bool: True if x is a panel, False otherwise
    """
    return isinstance(x, Panel)


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
        raise ValueError("Not enough timesteps to build.")

    end = x_timesteps - horizon - gap + 1

    ids = np.arange(lookback, end)
    xframes, yframes = [], []

    for i in tqdm(ids, disable=not verbose):
        # ? functions that create a new panel might keep old frame ids
        xframes.append(df.iloc[i - lookback : i].copy())

        yframes.append(df.iloc[i + gap : i + gap + horizon].copy())

    return create_panel(xframes, reset_ids=True), create_panel(yframes, reset_ids=True)


def reset_ids(x, y):
    """
    Reset ids of a panel.

    Args:
        x (Panel): Panel to reset id of
        y (Panel): Panel to reset id of
        verbose (bool): Whether to print progress

    Returns:
        ``Panel``: Reset id of panel
    """

    # Check if id in x and y are the same
    if not np.array_equal(x.ids, y.ids):
        raise ValueError(
            "Ids for x and y are not the same. Try using match function first."
        )

    x.reset_ids()
    y.reset_ids()

    return x, y


def create_panel(
    frames: list, reset_ids=False, val_size=None, train_size=None, test_size=None
):
    """
    Create a panel from a list of dataframes.

    Args:
        frames (list): List of dataframes
        reset_ids (bool): Whether to reset ids
        val_size (int): Size of validation set
        train_size (int): Size of training set
        test_size (int): Size of test set

    Returns:
        ``Panel``: Data Panel
    """

    if frames is None:
        raise ValueError("Frames cannot be None.")

    if is_series(frames[0]):
        frames = [pd.DataFrame(frame).T for frame in frames]

    panel = Panel(frames, reset_ids=reset_ids)

    if train_size is not None and test_size is not None and val_size is not None:
        panel.set_training_split(
            train_size=train_size, val_size=val_size, test_size=test_size
        )

    return panel


def concat_(panels: list, reset_ids=False, sort=False):
    """
    Concatenate panels.

    Args:
        panels (list): List of panels
        reset_ids (bool): Whether to reset ids
        sort (bool): Whether to sort by id

    Returns:
        ``Panel``: Concatenated panels
    """

    # Get ids of all panels
    ids = list(chain(*[panel.ids for panel in panels]))

    # Check duplicated ids in list
    if len(ids) != len(set(ids)):
        raise ValueError("There are duplicated ids in the list.")

    frames = list(chain(*[panel.frames for panel in panels]))

    if sort:
        frames = sorted(frames, key=lambda x: x.columns.name)

    panel = create_panel(frames, reset_ids=reset_ids)

    return panel


class Panel:
    def __init__(self, frames, reset_ids=False):
        # ? What about duplicated indices?
        # ? Would it make sense to have the panel as just dataframe references per frame?

        class _IXIndexer:
            def __init__(self, outer):
                self.outer = outer

            def __getitem__(self, item):
                return create_panel([i.ix[item] for i in self.outer.frames])

        class _iLocIndexer:
            def __init__(self, outer):
                self.outer = outer

            def __getitem__(self, item):
                return create_panel([i.iloc[item] for i in self.outer.frames])

        class _LocIndexer:
            def __init__(self, outer):
                self.outer = outer

            def __getitem__(self, item):
                return create_panel([i.loc[item] for i in self.outer.frames])

        class _AtIndexer:
            def __init__(self, outer):
                self.outer = outer

            def __getitem__(self, item):
                return create_panel([i.at[item] for i in self.outer.frames])

        class _iAtIndexer:
            def __init__(self, outer):
                self.outer = outer

            def __getitem__(self, item):
                return create_panel([i.iat[item] for i in self.outer.frames])

        self.ix = _IXIndexer(self)
        self.iloc = _iLocIndexer(self)
        self.loc = _LocIndexer(self)
        self.at = _AtIndexer(self)
        self.iat = _iAtIndexer(self)

        self.frames = frames
        self.train_size = None
        self.test_size = None
        self.val_size = None

        if reset_ids:
            self.reset_ids()

    def __getattr__(self, name):
        try:

            def wrapper(*args, **kwargs):
                frames = []
                for frame in self:
                    new_frame = getattr(frame.copy(), name)(*args, **kwargs)
                    if isinstance(new_frame, pd.Series):
                        new_frame = pd.DataFrame(new_frame).T
                        new_frame.index = frame.index[: len(new_frame.index)]
                    frames.append(new_frame)
                return Panel(frames)

            return wrapper
        except AttributeError:
            raise AttributeError(f"'Panel' object has no attribute '{name}'")

    # Function to map all dunder functions
    def _1_arg(self, other, __f):
        def _self_vs_iterable(self, other, __f):

            if len(self) != len(other):
                raise ValueError("Length of self and other must be the same.")

            return create_panel(
                [
                    getattr(frame, __f)(other_frame)
                    for frame, other_frame in zip(self, other)
                ],
                train_size=self.train_size,
                val_size=self.val_size,
                test_size=self.test_size,
            )

        def _self_vs_scalar(self, other, __f):

            if is_iterable(other) and len(other) != len(self[0]):
                raise ValueError("Length of other must be the same as length of frame.")

            return create_panel(
                [getattr(frame, __f)(other) for frame in self],
                train_size=self.train_size,
                val_size=self.val_size,
                test_size=self.test_size,
            )

        if is_panel(other):
            return _self_vs_iterable(self, other, __f)

        return _self_vs_scalar(self, other, __f)

    for _dunder in _ARG_1_METHODS:
        locals()[_dunder] = lambda self, other, __f=_dunder: self._1_arg(other, __f)

    def _0_arg(self, __f):
        return create_panel(
            [getattr(frame, __f)() for frame in self],
            train_size=self.train_size,
            val_size=self.val_size,
            test_size=self.test_size,
        )

    for _dunder in _ARG_0_METHODS:
        locals()[_dunder] = lambda self, __f=_dunder: self._0_arg(__f)

    def __getitem__(self, key):
        # TODO: add support for dates x['2016':] (reference pandas)
        # TODO: add support for exclusion / set operation, e.g. x[x.index < 1000 & x.index > 2000] (reference pandas)
        # ? What if the panel was a np.array of frames, instead of a list of frames?

        assert isinstance(
            key, (int, np.integer, slice, list, str, tuple, np.ndarray)
        ), "Panel indexing must be int, slice, list, str tuple or ndarray"
        if isinstance(key, (list, np.ndarray)):
            assert all(isinstance(k, (int, np.integer)) for k in key) or all(
                isinstance(k, str) for k in key
            ), "Panel indexing with list must be int or str"
        if isinstance(key, tuple):
            index = key[0]
            columns = key[1]
            assert isinstance(
                index, (int, np.integer, slice, list, np.ndarray)
            ), "Panel indexing rows with tuple must be int, slice, list or ndarray"
            assert isinstance(
                columns, (str, list, np.ndarray)
            ), "Panel indexing columns with tuple must be str, list or ndarray"

            if isinstance(index, (list, np.ndarray)):
                assert all(
                    isinstance(k, (int, np.integer)) for k in index
                ), "Panel indexing rows with list must be int"
            if isinstance(columns, (list, np.ndarray)):
                assert all(
                    isinstance(k, str) for k in columns
                ), "Panel indexing columns with list must be str"

        # Tuple
        if isinstance(key, tuple):
            index = key[0]
            columns = key[1] if isinstance(key[1], list) else [key[1]]

            if isinstance(index, (int, np.integer)):
                return self.frames[index].loc[:, columns]
            elif isinstance(index, slice):
                return create_panel(self.frames[index]).loc[:, columns]
            elif isinstance(index, (list, np.ndarray)):
                return create_panel([self.frames[i] for i in index]).loc[:, columns]

        # Columns
        if isinstance(key, (list, np.ndarray)) and all(isinstance(k, str) for k in key):
            return self.loc[:, key]
        elif isinstance(key, str):
            return self.loc[:, [key]]

        # Index
        if isinstance(key, (int, np.integer)):
            return self.frames[key]
        elif isinstance(key, slice):
            return create_panel(self.frames[key])
        elif isinstance(key, (list, np.ndarray)) and all(
            isinstance(k, (int, np.integer)) for k in key
        ):
            return create_panel([self.frames[i] for i in key])

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

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        """
        Returns the next frame in the panel.
        """
        if self._index < len(self):
            result = self[self._index]
            self._index += 1
            return result

        raise StopIteration

    @property
    def ids(self):
        """
        Return the ids of the panel.
        """
        return np.array([frame.columns.name for frame in self])

    @ids.setter
    def ids(self, ids):
        """
        Set the ids of the panel.

        Args:
            ids (list): List of ids.
        """
        if len(ids) != len(self):
            raise ValueError(
                f"Length of ids must match length of panel. Got {len(ids)} and {len(self)}"
            )

        for frame, id in zip(self, ids):
            frame.columns.name = id

    def reset_ids(self):
        """
        Reset the ids of the panel.
        """
        self.ids = list(range(len(self)))

    @property
    def columns(self):
        """
        Panel columns.

        Example:

        >>> panel.columns
        Index(['Open', 'High', 'Low', 'Close'], dtype='object')
        """

        return self[0].columns

    @columns.setter
    def columns(self, columns):
        """
        Set the columns of the panel.

        Args:
            columns (list): List of columns.
        """

        if len(columns) != len(self[0].columns):
            raise ValueError(
                f"Length of columns must match length of panel. Got {len(columns)} and {len(self[0].columns)}"
            )

        for frame in self:
            frame.columns = columns

    @property
    def last_index(self):
        """
        Returns the last index of each frame in the panel.
        """

        return pd.Series([frame.index[-1] for frame in self])

    @property
    def first_index(self):
        """
        Returns the first index of each frame in the panel.
        """

        return pd.Series([frame.index[0] for frame in self])

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

        return np.array([frame.values for frame in tqdm(self, disable=not verbose)])

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

    @property
    def train(self):
        """
        Returns the Panel with the training set, according to
        the parameters given in the 'set_training_split' function.

        Returns:
            ``Panel``: Panel with the training set.
        """
        if not self.train_size:
            return None

        # panel = self[: int(self.train_size * len(self))]
        panel = self[: self.train_size]

        return create_panel(frames=panel.frames)

    @property
    def val(self):
        """
        Returns the Panel with the validation set, according to
        the parameters given in the 'set_training_split' function.

        Returns:
            ``Panel``: Panel with the validation set.
        """
        if not self.val_size:
            return None

        panel = self[self.train_size : self.train_size + self.val_size]

        return create_panel(frames=panel.frames)

    @property
    def test(self):
        """
        Returns the Panel with the testing set, according to
        the parameters given in the 'set_training_split' function.

        Returns:
            ``Panel``: Panel with the testing set.
        """

        if not self.test_size:
            return None

        panel = self[self.train_size + self.val_size :]

        return create_panel(frames=panel.frames)

    # TODO function to set_ids and only accept int or date

    def get_frame_by_id(self, id: int):
        """
        Get a frame by id.

        Args:
            id (int): Id of the frame to return.

        Returns:
            pd.DataFrame: Frame at id.

        Example:

        >>> panel.get_frame_by_id(0)
        <DataFrame>
        """
        return next((frame for frame in self if frame.columns.name == id), None)

    def rename_columns(self, columns: dict):
        """
        Rename columns.

        Args:
            columns: Dictionary of old column names to new column names.

        Returns:
            Panel with renamed columns.

        Example:

        >>> panel.rename_columns({"Open": "open", "High": "high"})
        """

        return create_panel(
            [frame.rename(columns=columns) for frame in self],
            train_size=self.train_size,
            val_size=self.val_size,
            test_size=self.test_size,
        )

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
            frame.isnull().values.sum() for frame in tqdm(self, disable=not verbose)
        ]
        return pd.DataFrame(values, index=range(len(self)), columns=["nan"])

    def dropna(self, axis=0, how="any", thresh=None, subset=None, verbose=False):
        """
        Drop rows or columns with missing values.

        Args:
            axis (int): The axis to drop on.
            how (str): Method to use for dropping.
            thresh (int): The number of non-NA values to require.
            subset (list): List of columns to check.
            verbose (bool): Whether to print progress.

        Returns:
            ``Panel``: Dropped panel.
        """

        frames = [
            frame.dropna(axis=axis, how=how, thresh=thresh, subset=subset)
            for frame in self
        ]

        if len({frame.shape for frame in frames}) > 1:
            warnings.warn("Dropped frames have different shape")

        return create_panel(
            frames,
            train_size=self.train_size,
            val_size=self.val_size,
            test_size=self.test_size,
        )

    def dropna_(self):
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
        # TODO: Remove ID from the returned dataframe.
        # It seems to be working fine, and has no id.
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

        # TODO check what to do when calculating diff() from pandas and later as_dataframe()
        # TODO check index (use last or first?) and how to handle it when flattening

        if flatten:
            columns = [
                f"{str(i)}-{col}"
                for i, col in itertools.product(range(self.shape[1]), self[0].columns)
            ]
            values = np.array([i.values.flatten() for i in self])
            index = [i.index[-1] for i in self]
            return pd.DataFrame(values, index=index, columns=columns)

        else:
            list_frames = []
            for i in range(self.shape[0]):
                a = self[i].head(1).copy()
                if frame:
                    a["frame"] = self[i].columns.name
                list_frames.append(a)
            dataframe = pd.concat(list_frames)

            # Get only rows with unique identifiers
            index = dataframe.index
            is_duplicate = index.duplicated(keep="first")
            not_duplicate = ~is_duplicate
            dataframe = dataframe[not_duplicate]

            return dataframe

    def set_index_(self, indexes):
        """
        Set index of panel.

        Args:
            indexes (list): List of indexes to set.

        Returns:
            ``Panel``: Result of set index function.
        """

        if len(indexes) != self.shape[0]:
            raise ValueError("Number of indexes must be equal to number of frames")

        frames = [frame.set_index(index) for frame, index in zip(self, indexes)]

        return create_panel(
            frames,
            train_size=self.train_size,
            val_size=self.val_size,
            test_size=self.test_size,
        )

    def shift_(self, periods: int = 1):
        """
        Shift panel by desired number of frames.

        Similar to `Pandas shift <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.shift.html>`__

        Args:
            periods (int): Number of frames to shift

        Returns:
            ``Panel``: Result of shift function.

        Example:

        >>> panel[0]
                          Open        High         Low       Close
        Date
        2022-05-06  274.809998  279.250000  271.269989  274.730011
        2022-05-09  270.059998  272.359985  263.320007  264.579987

        >>> panel = panel.shift(periods = 1)

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

        for i, frame in enumerate(self):
            new_index = i - periods
            new_index = new_index if new_index >= 0 and new_index < len(self) else None
            new_values = (
                self[new_index].values
                if new_index is not None
                else np.ones(self[0].shape) * np.nan
            )
            new_frame = pd.DataFrame(
                data=new_values, index=frame.index, columns=frame.columns
            )
            new_panel.append(new_frame)

        return create_panel(
            new_panel,
            train_size=self.train_size,
            val_size=self.val_size,
            test_size=self.test_size,
        )

    def diff_(self, periods: int = 1):
        """
        Difference between frames.

        Similar to `Pandas diff <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.diff.html>`__

        Args:
            periods (int): Number of frames to diff

        Returns:
            ``Panel``: Result of diff function.

        Example:

        >>> panel[0]
                          Open        High         Low       Close
        Date
        2022-05-06  274.809998  279.250000  271.269989  274.730011
        2022-05-09  270.059998  272.359985  263.320007  264.579987

        >>> panel = panel.diff(periods = 1)

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

        return self - self.shift_(periods)

    def pct_change_(self, periods: int = 1):
        """
        Percentage change between the current and a prior frame.

        Similar to `Pandas pct_change <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pct_change.html>`__

        Args:
            periods (int): Number of frames to calculate percent change

        Returns:
            ``Panel``: Result of pct_change function.

        Example:

        >>> panel[0]
                          Open        High         Low       Close
        Date
        2022-05-06  274.809998  279.250000  271.269989  274.730011
        2022-05-09  270.059998  272.359985  263.320007  264.579987

        >>> panel = panel.pct_change(periods = 1)

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

        a = self.shift_(periods)
        return (self - a) / a

    def match(self, other, verbose=False):
        """
        Match panel with other panel. This function will match the ids and id
        order of self based on the ids of other.

        Args:
            other: (Panel)

        Returns:
            ``Panel``: Result of match function.
        """

        if [i for i in other.ids if i not in self.ids]:
            raise ValueError("There are elements in other that are not in self.")

        other_ids = other.ids

        panel = create_panel(
            [
                frame
                for frame in tqdm(self, disable=not verbose)
                if frame.columns.name in other.ids
            ]
        )

        panel_ids = panel.ids

        indices = [int(np.where(panel_ids == id)[0]) for id in other_ids]

        return panel[indices]

    def set_training_split(
        self,
        train_size: Union[float, int],
        val_size: Union[float, int] = 0.2,
        test_size: Union[float, int] = 0.1,
    ):
        """
        Splits the panel in training, validation, and test, accessed with the
        properties .train, .val and .test.

        Args:
            train_size (float, int): Fraction of data to use for training.
            test_size (float, int): Fraction of data to use for testing.
            val_size (float, int): Fraction of data to use for validation.

        Example:
        >>> panel.set_training_split(val_size=0.2, test_size=0.1)
        """

        n_train, n_val, n_test = _validate_training_split(
            len(self), train_size=train_size, val_size=val_size, test_size=test_size
        )

        self.train_size = n_train
        self.val_size = n_val
        self.test_size = n_test

    # TODO check this function
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
            if not is_dataframe(other):
                raise ValueError("If using different sizes, other must be a DataFrame")

            assert not all(other.index.duplicated())
            warnings.warn(
                "Sizes don't match. Using dataframe indexes and columns to update."
            )
            other = other.loc[df.index, df.columns].values

        for i, j in zip(panel, other):
            i.iloc[:, :] = j
        return panel

    def head_(self, n: int = 5):
        """
        Return the first n frames of the panel.

        Args:
            n (int): Number of frames to return.

        Returns:
            ``Panel``: Result of head function.
        """
        return self[:n]

    def tail_(self, n: int = 5):
        """
        Return the last n frames of the panel.

        Args:
            n (int): Number of frames to return.

        Returns:
            ``Panel``: Result of tail function.
        """
        return self[-n:]

    def sort_(self):
        """
        Sort panel by ids.

        Returns:
            ``Panel``: Result of sort function.
        """

        self_ids = self.ids
        sorted_ids = sorted(self_ids)
        indices = [int(np.where(self_ids == id)[0]) for id in sorted_ids]

        return self[indices]

    def sample_(self, samples: int = 5, how: str = "spaced"):
        """
        Sample panel returning a subset of frames.

        Args:
            samples (int): Number of samples to keep
            how (str): Sampling method, 'spaced' or 'random'

        Returns:
            ``Panel``: Result of sample function.
        """

        if how == "random":
            warnings.warn("Random sampling can result in data leakage.")
            indexes = np.random.choice(len(self), samples, replace=False)
            indexes = sorted(indexes)
            return self[indexes]
        elif how == "spaced":
            indexes = np.linspace(0, len(self), samples, dtype=int, endpoint=False)
            return self[indexes]

    def shuffle_(self):
        """
        Shuffle the panel.

        Returns:
            ``Panel``: Result of shuffle function.
        """

        warnings.warn("Shuffling the panel can result in data leakage.")

        indexes = list(range(len(self)))
        random.shuffle(indexes)
        return self[indexes]

    def plot(self, add_annotation=True, max=10_000, use_ids=True, **kwargs):
        """
        Plot the panel.

        Args:
            add_annotation (bool): If True, plot the training, validation, and test annotation.
            **kwargs: Additional arguments to pass to the plot function.

        Returns:
            ``plot``: Result of plot function.
        """

        if max and len(self) > max:
            return plot(
                self.sample(max, how="spaced"),
                use_ids=use_ids,
                add_annotation=add_annotation,
                **kwargs,
            )
        return plot(self, use_ids=use_ids, add_annotation=add_annotation, **kwargs)

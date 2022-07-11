import warnings
from typing import Union

import numpy as np
import pandas as pd

from wavy.validations import _validate_training_split


def create_panels(df, lookback: int, horizon: int, gap: int = 0):
    """
    Creates a list of panels from a dataframe.
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

    xframes = np.zeros(shape=(len(ids) * lookback, df.shape[1]))
    xindex = np.zeros(shape=(2, len(ids) * lookback), dtype=int)

    yframes = np.zeros(shape=(len(ids) * horizon, df.shape[1]))
    yindex = np.zeros(shape=(2, len(ids) * horizon), dtype=int)

    for i in ids:
        # X
        frame = df.iloc[i - lookback : i]
        xframes[
            (i - lookback) * lookback : (i - lookback + 1) * lookback, :
        ] = frame.values
        xindex[
            :, (i - lookback) * lookback : (i - lookback + 1) * lookback
        ] = np.vstack(
            ((i - lookback) * np.ones(lookback, dtype=int), frame.index.values)
        )

        # Y
        frame = df.iloc[i + gap : i + gap + horizon]
        yframes[
            (i - lookback) * horizon : (i - lookback + 1) * horizon, :
        ] = frame.values
        yindex[:, (i - lookback) * horizon : (i - lookback + 1) * horizon] = np.vstack(
            ((i - lookback) * np.ones(horizon, dtype=int), frame.index.values)
        )

    return Panel2(
        xframes,
        columns=df.columns,
        index=pd.MultiIndex.from_arrays(xindex, names=["id", df.index.name]),
    ), Panel2(
        yframes,
        columns=df.columns,
        index=pd.MultiIndex.from_arrays(yindex, names=["id", df.index.name]),
    )


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
    pass


def concat_panel(panels: list, reset_ids=False, sort=False):
    """
    Concatenate panels.

    Args:
        panels (list): List of panels
        reset_ids (bool): Whether to reset ids
        sort (bool): Whether to sort by id

    Returns:
        ``Panel``: Concatenated panels
    """
    pass


def set_training_split(
    x,
    y,
    train_size: Union[float, int] = 0.7,
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
    >>> x,y = set_training_split(x, y, val_size=0.2, test_size=0.1)
    """

    x.set_training_split(train_size=train_size, val_size=val_size, test_size=test_size)
    y.set_training_split(train_size=train_size, val_size=val_size, test_size=test_size)


class Panel2(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.train_size = None
        self.test_size = None
        self.val_size = None

    def __getitem__(self, key):
        return type(self)(super().__getitem__(key))

    def __len__(self):
        return len(self.ids)

    # TODO add iter to frame in panel??

    @property
    def ids(self):
        return self.index.get_level_values(0).drop_duplicates()

    @property
    def panel_shape(self):
        return (len(self.ids), int(self.shape[0] / len(self.ids)), self.shape[1])

    @ids.setter
    def ids(self, ids):
        """
        Set the ids of the panel.

        Args:
            ids (list): List of ids.
        """
        self.index = self.index.set_levels(ids, level=0)

    def reset_ids(self):
        """
        Reset the ids of the panel.
        """
        self.ids = np.arange(len(self))

    @property
    def last_index(self):
        """
        Returns the last index of each frame in the panel.
        """

        return self.groupby(level=0).tail(1).index.get_level_values(1)

    @property
    def first_index(self):
        """
        Returns the first index of each frame in the panel.
        """

        return self.groupby(level=0).head(1).index.get_level_values(1)

    @property
    def values(self):
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
        return np.reshape(self.to_numpy(), self.panel_shape)

    def get_frame_by_id(self, id: Union[int, list], drop_level=True):
        """
        Get a frame by id.

        Args:
            id (int): Id of the frame to return.
            drop_level (bool): Whether to drop the id level.

        Returns:
            pd.DataFrame: Frame at id.

        Example:

        >>> panel.get_frame_by_id(0)
        <DataFrame>
        """
        return self.xs(id, level=0, axis=0, drop_level=drop_level)

    def drop_frames(self, ids: Union[list, int]):
        """
        Drop frames by id.

        Args:
            ids (list, int): List of ids to drop.

        Returns:
            ``Panel``: Panel with frames dropped.
        """
        if isinstance(ids, int):
            ids = [ids]
        return self[~self.index.get_level_values(0).isin(ids)]

    def dropna_frames(self):
        """
        Drop frames with missing values from the panel.

        Returns:
            ``Panel``: Panel with missing values dropped.
        """
        return self[~self.index.get_level_values(0).isin(self.findna_frames)]

    def findna_frames(self):
        """
        Find NaN values index.

        Returns:
            ``List``: List with index of NaN values.
        """
        return self[self.isna().any(axis=1)].index.get_level_values(0).drop_duplicates()

    # TODO fix
    # def set_index_frames(self, indexes, inplace=False):
    #     """
    #     Set index of panel.

    #     Args:
    #         indexes (list): List of indexes to set.
    #         inplace (bool): Whether to set the index inplace.

    #     Returns:
    #         ``Panel``: Result of set index function.
    #     """

    #     if len(self.index.get_level_values(1)) != len(indexes):
    #         raise ValueError("Number of indexes must be equal to number of frames")

    #     new_frame = self.reset_index(drop=True)

    #     return create_panel(
    #         frames,
    #         train_size=self.train_size,
    #         val_size=self.val_size,
    #         test_size=self.test_size,
    #     )

    def shift_panel(self, periods: int = 1):
        """
        Shift panel by desired number of frames.

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

        return self.shift(periods * self.panel_shape[1])

    def diff_panel(self, periods: int = 1):
        """
        Difference between frames.

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

        return self - self.shift_panel(periods)

    def pct_change_panel(self, periods: int = 1):
        """
        Percentage change between the current and a prior frame.

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

        a = self.shift_panel(periods)
        return (self - a) / a

    def match_frame(self, other):
        """
        Match panel with other panel. This function will match the ids and id
        order of self based on the ids of other.

        Args:
            other (``Panel``): Panel to match with.

        Returns:
            ``Panel``: Result of match function.
        """

        other_ids = set(other.ids)
        self_ids = set(self.ids)

        if [i for i in other_ids if i not in self_ids]:
            raise ValueError("There are elements in other that are not in self.")

        drop_ids = self_ids - other_ids

        return self.drop_frames(drop_ids)

    def set_training_split(
        self,
        train_size: Union[float, int] = 0.7,
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

    @property
    def train(self):
        """
        Returns the Panel with the training set, according to
        the parameters given in the 'set_training_split' function.

        Returns:
            ``Panel``: Panel with the training set.
        """

        return self[: self.train_size] if self.train_size else None

    @property
    def val(self):
        """
        Returns the Panel with the validation set, according to
        the parameters given in the 'set_training_split' function.

        Returns:
            ``Panel``: Panel with the validation set.
        """

        return (
            self[self.train_size : self.train_size + self.val_size]
            if self.val_size
            else None
        )

    @property
    def test(self):
        """
        Returns the Panel with the testing set, according to
        the parameters given in the 'set_training_split' function.

        Returns:
            ``Panel``: Panel with the testing set.
        """

        return self[self.train_size + self.val_size :] if self.test_size else None

    def head_frame(self, n: int = 5):
        """
        Return the first n frames of the panel.

        Args:
            n (int): Number of frames to return.

        Returns:
            ``Panel``: Result of head function.
        """
        return self[: n * self.panel_shape[1]]

    def tail_frame(self, n: int = 5):
        """
        Return the last n frames of the panel.

        Args:
            n (int): Number of frames to return.

        Returns:
            ``Panel``: Result of tail function.
        """
        return self[-n * self.panel_shape[1] :]

    def sort_ids(
        self,
        ascending=True,
        inplace=False,
        kind="quicksort",
        key=None,
    ):
        """
        Sort panel by ids.

        Args:
            ascending (bool or list-like of bools, default True): Sort ascending vs. descending. When the index is a MultiIndex the sort direction can be controlled for each level individually.
            inplace (bool, default False): If True, perform operation in-place.
            kind ({'quicksort', 'mergesort', 'heapsort', 'stable'}, default 'quicksort'): Choice of sorting algorithm. See also numpy.sort() for more information. mergesort and stable are the only stable algorithms. For DataFrames, this option is only applied when sorting on a single column or label.
            key (callable, optional): If not None, apply the key function to the index values before sorting. This is similar to the key argument in the builtin sorted() function, with the notable difference that this key function should be vectorized. It should expect an Index and return an Index of the same shape. For MultiIndex inputs, the key is applied per level.

        Returns:
            ``Panel or None``: The original DataFrame sorted by the labels or None if `inplace=True`.
        """

        return self.sort_index(
            level=0,
            ascending=ascending,
            inplace=inplace,
            kind=kind,
            sort_remaining=False,
            key=key,
        )

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

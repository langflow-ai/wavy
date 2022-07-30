import random
import warnings
from itertools import chain
from typing import Union

import numpy as np
import pandas as pd

from wavy.plot import plot
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
    xindex = [
        np.zeros(shape=(len(ids) * lookback), dtype=int),
        np.zeros(shape=(len(ids) * lookback), dtype=df.index.dtype),
    ]

    yframes = np.zeros(shape=(len(ids) * horizon, df.shape[1]))
    yindex = [
        np.zeros(shape=(len(ids) * horizon), dtype=int),
        np.zeros(shape=(len(ids) * horizon), dtype=df.index.dtype),
    ]

    for i in ids:
        # X
        frame = df.iloc[i - lookback : i]
        xframes[
            (i - lookback) * lookback : (i - lookback + 1) * lookback, :
        ] = frame.values
        xindex[0][(i - lookback) * lookback : (i - lookback + 1) * lookback] = (
            i - lookback
        ) * np.ones(lookback, dtype=int)
        xindex[1][
            (i - lookback) * lookback : (i - lookback + 1) * lookback
        ] = frame.index.values

        # Y
        frame = df.iloc[i + gap : i + gap + horizon]
        yframes[
            (i - lookback) * horizon : (i - lookback + 1) * horizon, :
        ] = frame.values
        yindex[0][(i - lookback) * horizon : (i - lookback + 1) * horizon] = (
            i - lookback
        ) * np.ones(horizon, dtype=int)
        yindex[1][
            (i - lookback) * horizon : (i - lookback + 1) * horizon
        ] = frame.index.values

    timesteps_name = df.index.name or "timesteps"

    return Panel(
        xframes,
        columns=df.columns,
        index=pd.MultiIndex.from_arrays(xindex, names=["id", timesteps_name]),
    ), Panel(
        yframes,
        columns=df.columns,
        index=pd.MultiIndex.from_arrays(yindex, names=["id", timesteps_name]),
    )


def reset_ids(panels, inplace=False):
    """
    Reset ids of a panel.

    Args:
        panels (list): List of panels
        inplace (bool): Whether to reset ids inplace or not.

    Returns:
        ``Panel``: Reset id of panel
    """

    # Check if id in x and y are the same
    if not all(np.array_equal(panels[0].ids, panel.ids) for panel in panels):
        raise ValueError(
            "Ids for panels are not the same. Try using match function first."
        )

    return [panel.reset_ids(inplace=inplace) for panel in panels]


def concat_panels(panels: list, reset_ids=False, sort=False):
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

    panel = Panel(pd.concat(panels, axis=0))

    if sort:
        panel = panel.sort_ids()

    if reset_ids:
        panel.reset_ids(inplace=True)

    return panel


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


class Panel(pd.DataFrame):
    def __init__(self, *args, **kw):
        super(Panel, self).__init__(*args, **kw)
        if len(args) == 1 and isinstance(args[0], Panel):
            args[0]._copy_attrs(self)

    _attributes_ = "train_size,test_size,val_size"

    def _copy_attrs(self, df):
        for attr in self._attributes_.split(","):
            df.__dict__[attr] = getattr(self, attr, None)

    @property
    def _constructor(self):
        def f(*args, **kw):

            try:
                index = [a for a in args[0].axes if isinstance(a, pd.MultiIndex)]
                if index and len(index[0]) == self.num_timesteps:
                    return pd.DataFrame(*args, **kw)
            except:
                pass

            df = Panel(*args, **kw)

            # Workaround to fix pandas bug
            if (df.index.nlevels > 1 and self.index.nlevels > 1) and len(
                df.index.levels
            ) > len(self.index.levels):
                df = df.droplevel(0, axis="index")

            if len(df) == len(self):
                self._copy_attrs(df)

            return df

        return f

    @property
    def num_frames(self):
        """Returns the number of frames in the panel."""
        return self.shape_panel[0]

    @property
    def num_timesteps(self):
        """Returns the number of timesteps in the panel."""
        return self.shape_panel[1]

    @property
    def num_columns(self):
        """Returns the number of columns in the panel."""
        return self.shape_panel[2]

    @property
    def frames(self):
        """Returns the frames in the panel."""
        return self.groupby(level=0, as_index=True)

    @property
    def ids(self):
        """
        Returns the ids of the panel.
        """
        return self.index.get_level_values(0).drop_duplicates()

    @ids.setter
    def ids(self, ids):
        """
        Set the ids of the panel.

        Args:
            ids (list): List of ids.
        """

        ids = np.repeat(np.arange(len(self)), self.shape_panel[1])
        timestep = self.index.get_level_values(1)

        index = pd.MultiIndex.from_arrays([ids, timestep], names=["id", timestep.name])

        self.index = index

    def reset_ids(self, inplace=False):
        """
        Reset the ids of the panel.

        Args:
            inplace (bool): Whether to reset ids inplace.
        """
        new_ids = np.repeat(np.arange(self.num_frames), self.num_timesteps)
        new_index = pd.MultiIndex.from_arrays(
            [new_ids, self.index.get_level_values(1)],
            names=self.index.names,
        )

        return self.set_index(new_index, inplace=inplace)

    @property
    def shape_panel(self):
        """
        Returns the shape of the panel.
        """
        return (len(self.ids), int(self.shape[0] / len(self.ids)), self.shape[1])

    def row_panel(self, n: int = 0):
        """
        Returns the nth row of each frame.
        """

        if n < -1 or n >= self.num_timesteps:
            raise ValueError("n must be -1 or between 0 and the number of timesteps")

        return self.groupby(level=0, as_index=False).nth(n)

    def get_timesteps(self, n: Union[list, int] = 0):
        """
        Returns the first timestep of each frame in the panel.

        Args:
            n (int): Timestep to return.
        """

        if isinstance(n, int):
            n = [n]

        return self.frames.take(n).index.get_level_values(2)

    @property
    def values_panel(self):
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
        return np.reshape(self.to_numpy(), self.shape_panel)

    @property
    def flatten_panel(self):
        """
        Flatten the panel.
        """
        return self.values_panel.reshape(
            self.shape_panel[0], self.shape_panel[1] * self.shape_panel[2]
        )

    def drop_ids(self, ids: Union[list, int], inplace=False):
        """
        Drop frames by id.

        Args:
            ids (list, int): List of ids to drop.
            inplace (bool): Whether to drop ids inplace.

        Returns:
            ``Panel``: Panel with frames dropped.
        """
        return self.drop(index=ids, inplace=inplace)

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

    def match_frames(self, other):
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

        return self.loc[other.ids]

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
            self.num_frames,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
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

        return self[: self.train_size * self.num_timesteps] if self.train_size else None

    @train.setter
    def train(self, value):
        """
        Set the training set.

        Args:
            value (``Panel``): Panel with the training set.
        """

        if not self.train_size:
            raise ValueError("No training set was set.")
        self[: self.train_size * self.num_timesteps] = value.values

    @property
    def val(self):
        """
        Returns the Panel with the validation set, according to
        the parameters given in the 'set_training_split' function.

        Returns:
            ``Panel``: Panel with the validation set.
        """

        return (
            self[
                self.train_size
                * self.num_timesteps : (self.train_size + self.val_size)
                * self.num_timesteps
            ]
            if self.val_size
            else None
        )

    @val.setter
    def val(self, value):
        """
        Set the validation set.

        Args:
            value (``Panel``): Panel with the validation set.
        """

        if not self.val_size:
            raise ValueError("No validation set was set.")
        self[
            self.train_size
            * self.num_timesteps : (self.train_size + self.val_size)
            * self.num_timesteps
        ] = value.values

    @property
    def test(self):
        """
        Returns the Panel with the testing set, according to
        the parameters given in the 'set_training_split' function.

        Returns:
            ``Panel``: Panel with the testing set.
        """

        return (
            self[(self.train_size + self.val_size) * self.num_timesteps :]
            if self.test_size
            else None
        )

    @test.setter
    def test(self, value):
        """
        Set the testing set.

        Args:
            value (``Panel``): Panel with the testing set.
        """

        if not self.test_size:
            raise ValueError("No testing set was set.")
        self[(self.train_size + self.val_size) * self.num_timesteps :] = value.values

    def head_panel(self, n: int = 5):
        """
        Return the first n frames of the panel.

        Args:
            n (int): Number of frames to return.

        Returns:
            ``Panel``: Result of head function.
        """
        return self[: n * self.shape_panel[1]]

    def tail_panel(self, n: int = 5):
        """
        Return the last n frames of the panel.

        Args:
            n (int): Number of frames to return.

        Returns:
            ``Panel``: Result of tail function.
        """
        return self[-n * self.shape_panel[1] :]

    def shift_panel(self, n: int = 1):
        """
        Shift the panel by n timesteps.

        Args:
            n (int): Number of timesteps to shift.

        Returns:
            ``Panel``: Result of shift function.
        """
        return self.shift(periods=n * self.num_timesteps)

    def sort_panel(
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

    def sample_panel(self, samples: int = 5, how: str = "spaced"):
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
            indexes = np.random.choice(self.ids, samples, replace=False)
            indexes = sorted(indexes)
            return self.get_frame_by_ids(indexes)
        elif how == "spaced":
            indexes = np.linspace(
                self.ids[0], self.ids[-1], samples, dtype=int, endpoint=False
            )
            return self.get_frame_by_ids(indexes)

    def shuffle_panel(self, seed: int = None):
        """
        Shuffle the panel.

        Args:
            seed (int): Random seed.

        Returns:
            ``Panel``: Result of shuffle function.
        """

        warnings.warn("Shuffling the panel can result in data leakage.")

        indexes = list(self.ids)
        random.seed(seed)
        random.shuffle(indexes)
        return self.get_frame_by_ids(indexes)

    def plot(self, add_annotation=True, max=10_000, use_timestep=False, **kwargs):
        """
        Plot the panel.

        Args:
            add_annotation (bool): If True, plot the training, validation, and test annotation.
            **kwargs: Additional arguments to pass to the plot function.

        Returns:
            ``plot``: Result of plot function.
        """

        if max and self.num_frames > max:
            return plot(
                self.sample_panel(max, how="spaced"),
                use_timestep=use_timestep,
                add_annotation=add_annotation,
                **kwargs,
            )
        return plot(
            self, use_timestep=use_timestep, add_annotation=add_annotation, **kwargs
        )

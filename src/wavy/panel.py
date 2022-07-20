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

    return Panel(
        xframes,
        columns=df.columns,
        index=pd.MultiIndex.from_arrays(xindex, names=["id", df.index.name]),
    ), Panel(
        yframes,
        columns=df.columns,
        index=pd.MultiIndex.from_arrays(yindex, names=["id", df.index.name]),
    )


def reset_ids(x, y, inplace=False):
    """
    Reset ids of a panel.

    Args:
        x (Panel): Panel to reset id of
        y (Panel): Panel to reset id of
        inplace (bool): Whether to reset ids inplace or not.

    Returns:
        ``Panel``: Reset id of panel
    """

    # Check if id in x and y are the same
    if not np.array_equal(x.ids, y.ids):
        raise ValueError(
            "Ids for x and y are not the same. Try using match function first."
        )

    x.reset_ids(inplace=inplace)
    y.reset_ids(inplace=inplace)

    return x, y


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
    # def __init__(self, *args, **kwargs):
    #     super(Panel, self).__init__(*args, **kwargs)

    #     self.train_size = None
    #     self.test_size = None
    #     self.val_size = None

    _metadata = ["train_size", "test_size", "val_size"]

    @property
    def _constructor(self):
        return Panel

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

    def frames(self):
        index = 0
        ids = self.ids
        while index < self.num_frames:
            yield self.xs(ids[index], level=0, axis=0)
            index += 1

    # Function to call pandas methods on all frames
    def __getattr__(self, name):
        try:
            name = name.replace("_panel", "")

            def wrapper(*args, **kwargs):
                panel = self.groupby(level=0).apply(name, *args, **kwargs)

                # Update ids
                ids = panel.index.get_level_values(0)
                timestamp = self.index.get_level_values(1)

                if len(ids) != len(timestamp):
                    timestamp = self.first_timestamp

                panel.index = pd.MultiIndex.from_arrays((ids, timestamp))

                return type(self)(panel)

            return wrapper
        except AttributeError:
            raise AttributeError(f"'Panel' object has no attribute '{name}'")

    @property
    def ids(self):
        return self.index.get_level_values(0).drop_duplicates()

    @ids.setter
    def ids(self, ids):
        """
        Set the ids of the panel.

        Args:
            ids (list): List of ids.
        """

        ids = np.repeat(np.arange(len(self)), self.shape_panel[1])
        timestamp = self.index.get_level_values(1)

        index = pd.MultiIndex.from_arrays(
            [ids, timestamp], names=["id", timestamp.name]
        )

        self.index = index

    def reset_ids(self, inplace=False):
        """
        Reset the ids of the panel.

        Args:
            inplace (bool): Whether to reset ids inplace.
        """
        # self.ids = np.arange(self.num_frames)
        new_ids = np.repeat(np.arange(self.num_frames), self.num_timesteps)
        new_index = pd.MultiIndex.from_arrays(
            [new_ids, self.index.get_level_values(1)],
            names=self.index.names,
        )

        return self.set_index(new_index, inplace=inplace)

    @property
    def shape_panel(self):
        return (len(self.ids), int(self.shape[0] / len(self.ids)), self.shape[1])

    def row_panel(self, n: int = 0):
        """
        Returns the nth row of each frame.
        """

        if n < 0 or n >= self.num_timesteps:
            raise ValueError("n must be between 0 and the number of timesteps")

        return self.groupby(level=0, as_index=False).nth(n)

    @property
    def first_timestamp(self):
        """
        Returns the first timestamp of each frame in the panel.
        """

        return self.groupby(level=0).head(1).index.get_level_values(1)

    @property
    def last_timestamp(self):
        """
        Returns the last timestamp of each frame in the panel.
        """

        return self.groupby(level=0).tail(1).index.get_level_values(1)

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

    def get_frame_by_ids(self, id: Union[int, list], drop_level=False):
        """
        Get a frame by id.

        Args:
            id (int): Id of the frame to return.
            drop_level (bool): Whether to drop the level of the index (only if
                the id is an int).

        Returns:
            pd.DataFrame: Frame at id.

        Example:

        >>> panel.get_frame_by_id(0)
        <DataFrame>
        """
        if isinstance(id, (int, np.integer)):
            return self.xs(id, level=0, axis=0, drop_level=drop_level)
        return self.loc[id, :]

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

        return type(self)(self.drop_frames(drop_ids))

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

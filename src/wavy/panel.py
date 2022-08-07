from __future__ import annotations

import contextlib
import random
import warnings
from itertools import chain
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib.pyplot import figure
from pandas.core.groupby import DataFrameGroupBy

warnings.simplefilter(action="ignore", category=FutureWarning)

from wavy.plot import plot
from wavy.validations import _validate_sample_panel, _validate_training_split


def create_panels(
    df: pd.DataFrame, lookback: int, horizon: int, gap: int = 0
) -> Tuple[Panel, Panel]:
    """
    Create panels from a dataframe.

    Args:
        df (pd.DataFrame): Dataframe
        lookback (int): Lookback size
        horizon (int): Horizon size
        gap (int): Gap size

    Returns:
        ``Tuple``: Tuple of panels
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


def reset_ids(panels: list[Panel], inplace: bool = False) -> list[Panel]:
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


def dropna_match(x, y):
    """
    Drop frames with NaN in both x and y and match ids.

    Args:
        x (Panel): Panel with x data
        y (Panel): Panel with y data

    Returns:
        ``Panel``: Panel with dropped frames and matched ids
    """

    x_t = x.dropnna_frames()
    y_t = y.match_frames(x_t)

    y_t = y_t.dropna_frames()
    x_t = x_t.match_frames(y_t)

    return x_t, y_t


def concat_panels(
    panels: list[Panel], reset_ids: bool = False, sort: bool = False
) -> Panel:
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
    x: Panel,
    y: Panel,
    train_size: Union[float, int] = 0.7,
    val_size: Union[float, int] = 0.2,
    test_size: Union[float, int] = 0.1,
) -> None:
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

            with contextlib.suppress(Exception):
                index = [a for a in args[0].axes if isinstance(a, pd.MultiIndex)]
                if index and len(index[0]) == self.num_timesteps:
                    return pd.DataFrame(*args, **kw)

            df = Panel(*args, **kw)

            # Workaround to fix pandas bug
            if (df.index.nlevels > 1 and self.index.nlevels > 1) and len(
                df.index.levels
            ) > len(self.index.levels):
                df = df.droplevel(0, axis="index")

            if df.num_frames == self.num_frames:
                self._copy_attrs(df)

            return df

        return f

    @property
    def num_frames(self) -> int:
        """Returns the number of frames in the panel."""
        return self.shape_panel[0]

    @property
    def num_timesteps(self) -> int:
        """Returns the number of timesteps in the panel."""
        return self.shape_panel[1]

    @property
    def num_columns(self) -> int:
        """Returns the number of columns in the panel."""
        return self.shape_panel[2]

    @property
    def frames(self) -> DataFrameGroupBy:
        """Returns the frames in the panel."""
        return self.groupby(level=0, as_index=True)

    @property
    def ids(self) -> pd.Int64Index:
        """
        Returns the ids of the panel.
        """
        return self.index.get_level_values(0).drop_duplicates()

    @ids.setter
    def ids(self, ids: list[int]) -> None:
        """
        Set the ids of the panel.

        Args:
            ids (list): List of ids.
        """

        ids = np.repeat(np.arange(len(self)), self.shape_panel[1])
        timestep = self.index.get_level_values(1)

        index = pd.MultiIndex.from_arrays([ids, timestep], names=["id", timestep.name])

        self.index = index

    def reset_ids(self, inplace: bool = False) -> Optional[Panel]:
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
    def shape_panel(self) -> Tuple[int, int, int]:
        """
        Returns the shape of the panel.
        """
        return (len(self.ids), int(self.shape[0] / len(self.ids)), self.shape[1])

    def row_panel(self, n: int = 0) -> Panel:
        """
        Returns the nth row of each frame.
        """

        if n < -1 or n >= self.num_timesteps:
            raise ValueError("n must be -1 or between 0 and the number of timesteps")

        new_panel = self.groupby(level=0, as_index=False).nth(n)
        self._copy_attrs(new_panel)
        return new_panel

    def get_timesteps(self, n: Union[list, int] = 0) -> Panel:
        """
        Returns the first timestep of each frame in the panel.

        Args:
            n (int): Timestep to return.
        """

        if isinstance(n, int):
            n = [n]

        return self.frames.take(n).index.get_level_values(2)

    @property
    def values_panel(self) -> np.ndarray:
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

    def flatten_panel(self) -> pd.DataFrame:
        """
        Flatten the panel.
        """

        new_timesteps = np.resize(
            np.arange(self.num_timesteps), self.num_timesteps * self.num_frames
        )
        new_index = pd.MultiIndex.from_arrays(
            [self.index.get_level_values(0), new_timesteps],
            names=self.index.names,
        )

        return (
            self.set_index(new_index)
            .reset_index()
            .pivot(index="id", columns=self.index.names[1])
        )

    def drop_ids(self, ids: Union[list, int], inplace: bool = False) -> Optional[Panel]:
        """
        Drop frames by id.

        Args:
            ids (list, int): List of ids to drop.
            inplace (bool): Whether to drop ids inplace.

        Returns:
            ``Panel``: Panel with frames dropped.
        """

        if self.index.nlevels == 1:
            return self.drop(index=ids, axis=0, inplace=inplace)

        return self.drop(index=ids, level=0, inplace=inplace)

    def dropna_frames(self, inplace: bool = False) -> Optional[Panel]:
        """
        Drop frames with missing values from the panel.

        Args:
            inplace (bool): Whether to drop frames inplace.

        Returns:
            ``Panel``: Panel with frames dropped.
        """
        return self.drop_ids(self.findna_frames(), inplace=inplace)

    def findna_frames(self) -> pd.Int64Index:
        """
        Find NaN values index.

        Returns:
            ``List``: List with index of NaN frames.
        """
        return self[self.isna().any(axis=1)].index.get_level_values(0).drop_duplicates()

    def match_frames(self, other: Panel, inplace: bool = False) -> Optional[Panel]:
        """
        Match panel with other panel. This function will match the ids and id
        order of self based on the ids of other.

        Args:
            other (``Panel``): Panel to match with.
            inplace (bool): Whether to match inplace.

        Returns:
            ``Panel``: Result of match function.
        """

        other_ids = set(other.ids)
        self_ids = set(self.ids)

        if [i for i in other_ids if i not in self_ids]:
            raise ValueError("There are elements in other that are not in self.")

        if inplace:
            return self.drop_ids(self_ids - other_ids, inplace=True)

        return self.loc[other.ids]

    def set_training_split(
        self,
        train_size: Union[float, int] = 0.7,
        val_size: Union[float, int] = 0.2,
        test_size: Union[float, int] = 0.1,
    ) -> None:
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
    def train(self) -> Panel:
        """
        Returns the Panel with the training set, according to
        the parameters given in the 'set_training_split' function.

        Returns:
            ``Panel``: Panel with the training set.
        """

        return self[: self.train_size * self.num_timesteps] if self.train_size else None

    @train.setter
    def train(self, value: np.ndarray) -> None:
        """
        Set the training set.

        Args:
            value (``Panel``): Panel with the training set.
        """

        if not self.train_size:
            raise ValueError("No training set was set.")
        self[: self.train_size * self.num_timesteps] = value.values

    @property
    def val(self) -> Panel:
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
    def val(self, value: np.ndarray) -> None:
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
    def test(self) -> Panel:
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
    def test(self, value: np.ndarray) -> None:
        """
        Set the testing set.

        Args:
            value (``Panel``): Panel with the testing set.
        """

        if not self.test_size:
            raise ValueError("No testing set was set.")
        self[(self.train_size + self.val_size) * self.num_timesteps :] = value.values

    def head_panel(self, n: int = 5) -> Panel:
        """
        Return the first n frames of the panel.

        Args:
            n (int): Number of frames to return.

        Returns:
            ``Panel``: Result of head function.
        """
        return self[: n * self.shape_panel[1]]

    def tail_panel(self, n: int = 5) -> Panel:
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
        ascending: bool = True,
        inplace: bool = False,
        kind: str = "quicksort",
        key: callable = None,
    ) -> Optional[Panel]:
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

    def sample_panel(
        self,
        samples: Union[int, float] = 5,
        how: str = "spaced",
        reset_ids: bool = False,
        seed: int = 42,
    ) -> Optional[Panel]:
        """
        Sample panel returning a subset of frames.

        Args:
            samples (int or float): Number or percentage of samples to return.
            how (str): Sampling method, 'spaced' or 'random'
            reset_ids (bool): If True, reset the index of the sampled panel.
            seed (int): Random seed.

        Returns:
            ``Panel``: Result of sample function.
        """

        train_size = self.train_size if hasattr(self, "train_size") else self.num_frames
        val_size = self.val_size if hasattr(self, "val_size") else 0
        test_size = self.test_size if hasattr(self, "test_size") else 0

        train_samples, val_samples, test_samples = _validate_sample_panel(
            samples=samples,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
        )

        if how == "random":
            # Set seed
            np.random.seed(seed)

            if hasattr(self, "train_size"):
                train_ids = sorted(
                    np.random.choice(self.train.ids, train_samples, replace=False)
                )
                val_ids = sorted(
                    np.random.choice(self.val.ids, val_samples, replace=False)
                )
                test_ids = sorted(
                    np.random.choice(self.test.ids, test_samples, replace=False)
                )
            else:
                train_ids = sorted(
                    np.random.choice(self.ids, train_samples, replace=False)
                )
                val_ids = []
                test_ids = []

        elif how == "spaced":
            if hasattr(self, "train_size"):
                train_ids = np.linspace(
                    self.train.ids[0],
                    self.train.ids[-1],
                    train_samples,
                    dtype=int,
                    endpoint=True,
                )
                val_ids = np.linspace(
                    self.val.ids[0],
                    self.val.ids[-1],
                    val_samples,
                    dtype=int,
                    endpoint=True,
                )
                test_ids = np.linspace(
                    self.test.ids[0],
                    self.test.ids[-1],
                    test_samples,
                    dtype=int,
                    endpoint=True,
                )
            else:
                train_ids = np.linspace(
                    self.ids[0], self.ids[-1], train_samples, dtype=int, endpoint=True
                )
                val_ids = []
                test_ids = []

        new_panel = self.loc[[*train_ids, *val_ids, *test_ids]]

        # Reset ids
        if reset_ids:
            new_panel.reset_ids(inplace=True)

        # Set new train, val, test sizes
        if hasattr(self, "train_size"):
            new_panel.train_size = train_samples
            new_panel.val_size = val_samples
            new_panel.test_size = test_samples

        # # TODO inplace not working
        # if inplace:
        #     self = new_panel
        #     return None

        return new_panel

    def shuffle_panel(
        self, seed: int = None, reset_ids: bool = False
    ) -> Optional[Panel]:
        """
        Shuffle the panel.

        Args:
            seed (int): Random seed.
            reset_ids (bool): If True, reset the index of the shuffled panel.

        Returns:
            ``Panel``: Result of shuffle function.
        """

        # warnings.warn("Shuffling the panel can result in data leakage.")

        if hasattr(self, "train_size"):
            train_ids = list(self.train.ids)
            val_ids = list(self.val.ids)
            test_ids = list(self.test.ids)
        else:
            train_ids = list(self.ids)
            val_ids = []
            test_ids = []

        random.seed(seed)
        random.shuffle(train_ids)
        random.shuffle(val_ids)
        random.shuffle(test_ids)

        new_panel = self.loc[[*train_ids, *val_ids, *test_ids]]

        # Reset ids
        if reset_ids:
            new_panel.reset_ids(inplace=True)

        # # TODO inplace not working
        # if inplace:
        #     self = new_panel
        #     return None

        return new_panel

    def plot(
        self,
        add_annotation: bool = True,
        max: int = 10_000,
        use_timestep: bool = False,
        **kwargs,
    ) -> plot.PanelFigure:
        """
        Plot the panel.

        Args:
            add_annotation (bool): If True, plot the training, validation, and test annotation.
            max (int): Maximum number of samples to plot.
            use_timestep (bool): If True, plot the timestep instead of the sample index.
            **kwargs: Additional arguments to pass to the plot function.

        Returns:
            ``plot``: Result of plot function.
        """

        panel = self.row_panel(n=0)

        if max and self.num_frames > max:
            return plot(
                panel.sample_panel(max, how="spaced"),
                use_timestep=use_timestep,
                add_annotation=add_annotation,
                **kwargs,
            )
        return plot(
            panel, use_timestep=use_timestep, add_annotation=add_annotation, **kwargs
        )

from collections import OrderedDict

import pandas as pd
from pandas import MultiIndex

from .utils import smash_array

def rebuild_from_index(
    array, index, units, channels, to_datetime=True, smash_dims=False
):
    # Needs to verify shape with dims
    if to_datetime:
        index = pd.to_datetime(index)

    # ! rebuilding may lead to unexpected column order if reproducing a dataframe
    columns = pd.MultiIndex.from_product([units, channels])
    df = pd.DataFrame(index=index, columns=columns)

    if smash_dims:
        array = smash_array(array)

    # ! Watch out for these reshapes
    # Maybe warn if two dims are the same
    df.loc[:, (slice(None), slice(None))] = array.reshape(df.shape)
    return MultiColumn(df)


def rebuild_from_array(array, df):
    # Needs to verify shape with dims
    # ! rebuilding may lead to unexpected column order if reproducing a dataframe
    df = pd.DataFrame(
        index=df.index, columns=pd.MultiIndex.from_tuples(df.columns.tolist())
    )
    df.loc[:, (slice(None), slice(None))] = array
    return MultiColumn(df)


def rebuild(df):
    # ! rebuilding may lead to unexpected column order if reproducing a dataframe
    df = pd.DataFrame(
        df.values,
        index=df.index,
        columns=pd.MultiIndex.from_tuples(df.columns.tolist()),
    )

    return MultiColumn(df)


def add_level(df, level_name="main"):
    df = pd.concat({level_name: df.T}, names=[level_name]).T
    return MultiColumn(df)


def unsmash(df, sep="_", level_name="main"):
    if sep is None:
        new = add_level(df, level_name=level_name)
    else:
        cols = list(df.columns)
        channels = [i.split(sep)[0] for i in cols]
        try:
            units = [i.split(sep)[1] for i in cols]
        except:
            units = [level_name]*len(cols)
        tuples = list(zip(units, channels))
        multicols = pd.MultiIndex.from_tuples(tuples)
        new = pd.DataFrame(df.values, index=df.index, columns=multicols)
    return rebuild(new)


class MultiColumn(pd.DataFrame):
    def __init__(self, df, sep=None, *args, **kwargs):
        super().__init__(df, *args, **kwargs)

    @property
    def _constructor(self):
        return MultiColumn

    @property
    def units(self):
        # OrderedDict to keep order
        units = [c[0] for c in self.columns]
        return list(OrderedDict.fromkeys(units))

    @property
    def channels(self):
        # OrderedDict to keep order
        channels = [c[1] for c in self.columns]
        return list(OrderedDict.fromkeys(channels))

    def filter_units(self, units):
        if not units:
            return self
        selection = self.loc[:, (units, slice(None))]
        return rebuild(selection)

    def filter_channels(self, channels):
        units = self.units
        if not channels:
            return self
        selection = self.loc[:, (slice(None), channels)][units]
        return rebuild(selection)

    def filter(self, units=None, channels=None):

        if type(units) == str:
            units = [units]
        if type(channels) == str:
            channels = [channels]

        if units is not None:
            if not all(unit in self.columns.levels[0] for unit in units):
                raise ValueError(
                    f"One of the units is not in columns.\nUnits: {units}\nColumns: {list(self.columns.levels[0])}"
                )
        if channels is not None:
            if not all(channel in self.columns.levels[1] for channel in channels):
                raise ValueError(
                    f"One of the channels is not in columns.\nChannels:{channels}\nColumns:{list(self.columns.levels[1])}"
                )
        selection = self.filter_units(units)
        selection = selection.filter_channels(channels)
        return selection

    def countna(self):
        s = pd.Series(dtype=int)
        for unit in self.units:
            s[unit] = len(self[unit]) - len(self[unit].dropna())
        return s

    def remove_channels(self, channels):
        if isinstance(channels, str):
            channels = [channels]
        new_channels = [c for c in self.channels if c not in channels]
        return self.filter_channels(new_channels)

    def remove_units(self, units):
        if isinstance(units, str):
            units = [units]
        new_units = [u for u in self.units if u not in units]
        return self.filter_units(new_units)

    def inspect_units(self):
        # Better if prints number of channels per unit
        # Or a summary of them
        inconsistent = []
        for unit in self.units:
            udf = self[unit]
            if len(udf.dropna()) == 0:
                inconsistent.append(unit)
        return inconsistent

    def channel_apply(self, func, name):
        units = self.units
        result = pd.DataFrame()
        for unit in units:
            temp = self[unit]
            temp_tranform = func(temp)
            if not isinstance(temp_tranform, pd.Series):
                temp_tranform = pd.Series(
                    data=temp_tranform, index=[str(temp.index[0])]
                )
            result[unit] = temp_tranform
        result.columns = pd.MultiIndex.from_product([result.columns, [name]])
        result = result[units]
        return MultiColumn(result)

    def channel_set(self, channels, values):
        """
        Runs inplace
        """
        new = self.copy()
        if isinstance(values, pd.core.frame.DataFrame):
            values = values.values
        new.loc[:, (slice(None), channels)] = values
        return MultiColumn(new)

    def add_level(self, level_name="unit_0"):
        channels = list(self.columns)
        units = [level_name] * len(channels)
        tuples = list(zip(units, channels))
        multicols = pd.MultiIndex.from_tuples(tuples)
        new = pd.DataFrame(self.values, index=self.index, columns=multicols)
        return rebuild(new)

    def channel_count(self):
        count = {c: 0 for c in self.channels}

        for unit in self.units:
            cols = self[unit].columns
            for col in cols:
                count[col] += 1
        return count

    def rename_subindex(self, df, sep="_"):
        tuples = []
        for tup in df.index:
            tuples.append((tup[0], tup[0] + sep + tup[1]))
        df.index = MultiIndex.from_tuples(tuples)
        return df

    def smash(self, sep="_"):
        new = self.T
        new = self.rename_subindex(new, sep=sep)
        new = new.droplevel(0)
        new = new.T
        return MultiColumn(new)

    def pandas(self):
        return pd.DataFrame(self.values, index=self.index, columns=self.columns)

    def copy(self):
        return MultiColumn(self.pandas())


# def multi_plot(mdf, unit, channels=None, return_traces=False, prefix='', dash='solid', cmap=cmap1, mode="lines"):
#     mdf = mdf.filter(unit, channels)[unit]
#     return line_plot(mdf, return_traces, prefix, dash, cmap, mode)




        # def as_multicol(function):
        #     def wrapper(x):
        #         if isinstance(function(x), pd.DataFrame):
        #             return MultiColumn(function(x, *args, **kwargs))
        #         else:
        #             return function(x, *args, **kwargs)
        #     return wrapper

        # METHODS = [func for func in dir(self) if callable(getattr(self, func)) and not func.startswith("_")]

        # for name in METHODS:
        #     try:
        #         function = getattr(self, name)
        #         setattr(self, name, as_multicol(function))
        #     except:
        #         print(name)
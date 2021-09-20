import random
import warnings
from collections import OrderedDict
from copy import copy
from itertools import groupby

import numpy as np
import pandas as pd
import plotly as px
import plotly.graph_objects as go
from pandas import MultiIndex

pd.set_option('multi_sparse', True)  # To see multilevel indexes
pd.options.plotting.backend = "plotly"

cmap1 = px.colors.qualitative.Plotly
cmap2 = cmap1[::-1]


def line_plot(df, return_traces=False, prefix='', dash='solid', cmap=cmap1, mode="lines"):
    fig = go.Figure()
    for idx, col in enumerate(df.columns):
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=prefix + col, mode=mode,
                                 line=dict(color=cmap[idx], width=2, dash=dash)))
    fig.update_layout(title='', xaxis_title='Timestamps', yaxis_title='Values')
    if return_traces:
        return fig
    else:
        fig.show()


def pair_plot(pair, unit, channels=None):
    if not channels:
        channels = pair.xframe[unit].columns
    x = pair.xframe[unit][channels]
    y = pair.yframe[unit][channels]

    fig = go.Figure()

    for _, channel in enumerate(channels):
        c = random.choice(cmap1)
        fig.add_trace(go.Scatter(x=x.index, y=x[channel], name="x_" + channel,
                                 line=dict(width=2, color=c)))

        fig.add_trace(go.Scatter(x=y.index, y=y[channel], name="y_" + channel,
                                 line=dict(width=2, dash='dot', color=c)))

    fig.update_layout(title='', xaxis_title='Timestamps', yaxis_title='Values')
    fig.show()


def multi_plot(mdf, unit, channels=None, return_traces=False, prefix='', dash='solid', cmap=cmap1, mode="lines"):
    mdf = mdf.sel(unit, channels)[unit]
    return line_plot(mdf, return_traces, prefix, dash, cmap, mode)


def pred_plot(y_test, y_pred, unit, channels=None, mode="lines"):
    test_trace = multi_plot(y_test, unit, channels, prefix='test_', return_traces=True, cmap=cmap1, mode=mode)
    pred_trace = multi_plot(y_pred, unit, channels, prefix='pred_', return_traces=True, dash='dot', cmap=cmap2,
                            mode=mode)
    fig = copy(test_trace)
    for trace in pred_trace.data:
        fig.add_trace(trace)
    fig.show()


def all_equal(iterable):
    """
    Check if all the iterables are equals.

    Parameters
    ----------
    iterable : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def drop_zero_rows(df):
    """
    Drops rows with zero values.

    """
    df = df.replace(0, np.nan)
    df = df.dropna(how='all')
    df = df.fillna(0)
    return df


def entangle(a, b):
    units = b.columns.levels[0].tolist()
    channels = b.columns.levels[1].tolist()
    a = a.sel(units, channels)
    a = a.loc[b.index]
    return a, b


def revert_pct_change(original, changed, gap=0):
    return (original.shift(gap) * changed) + original.shift(gap)


# Arrays

# def smash_2D(array, dim=0):
#     assert dim in [0, 1]
#     if dim == 0:
#         array = array.reshape(array.shape[0] * array.shape[1], array.shape[2])
#     elif dim == 1:
#         array = array.reshape(array.shape[0], array.shape[1] * array.shape[2])
#     return array


# def smash_3D(array, dim=1):
#     assert dim in [0, 1, 2]
#     if dim == 0:
#         array = array.reshape(array.shape[0] * array.shape[1], array.shape[2], array.shape[3])
#     elif dim == 1:
#         array = array.reshape(array.shape[0], array.shape[1] * array.shape[2], array.shape[3])
#     elif dim == 2:
#         array = array.reshape(array.shape[0], array.shape[1], array.shape[2] * array.shape[3])
#     return array


# def smash(array):
#     if len(array.shape) == 4:
#         array = array.reshape(array.shape[0], array.shape[2], array.shape[1] * array.shape[3])
#     elif len(array.shape) == 3:
#         array = array.reshape(array.shape[1], array.shape[0] * array.shape[2])
#     else:
#         raise("Array must have 3 or 4 dimensions")
#     return array


def smash_array(array):
    """
    Transforms an 3D or 4D array into a D-1 dimension array.

    Parameters
    ----------
    array : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if array.ndim == 4:
        arr_list = list()
        for arr in array:
            arr = np.hstack([arr[i] for i in range(len(arr))])
            arr_list.append(arr)
        return np.array(arr_list)

    elif array.ndim == 3:
        array = np.hstack([array[i] for i in range(len(array))])
        return array
    else:
        raise("Array must have 3 or 4 dimensions")


def shift(array, n):
    """
    Equivalent to pandas shift.
    """
    shifted = np.full(array.shape, np.nan)
    if n == 0:
        shifted = copy(array)
    elif n > 0:
        shifted[n:] = array[:-n]
    else:
        shifted[:n] = array[-n:]
    return shifted


# Multilevel dataframe


# Create class MultiLevelDataFrame, extending DataFrame, which will have properties
# like units, channels, and all of its functions will perform the rebuild func in the end


# def rebuild(df):
#     # ! rebuilding may lead to unexpected column order if reproducing a dataframe
#     df = pd.DataFrame(df.values, index=df.index,
#                         columns=pd.MultiIndex.from_tuples(df.columns.tolist()))
#     return df


# def rebuild_from_array(array, df):
#     # Needs to verify shape with dims
#     # ! rebuilding may lead to unexpected column order if reproducing a dataframe
#     df = pd.DataFrame(index=df.index, columns=pd.MultiIndex.from_tuples(df.columns.tolist()))
#     df.loc[:, (slice(None), slice(None))] = array
#     return df


# def rebuild_from_index(array, index, units, channels, to_datetime=True, smash_dims=False):
#     # Needs to verify shape with dims
#     if to_datetime:
#         index = pd.to_datetime(index)

#     # ! rebuilding may lead to unexpected column order if reproducing a dataframe
#     columns = pd.MultiIndex.from_product([units, channels])
#     df = pd.DataFrame(index=index, columns=columns)

#     if smash_dims:
#         array = smash(array)

#     # ! Watch out for these reshapes
#     # Maybe warn if two dims are the same
#     df.loc[:, (slice(None), slice(None))] = array.reshape(df.shape)
#     return df


# def add_level(df, level_name='main'):
#     if df.columns.nlevels == 1:
#         df = pd.concat({level_name: df.T}, names=[level_name]).T
#     return df


# def rename_subindex(df):
#     tuples = []
#     for tup in df.index:
#         tuples.append((tup[0], tup[0] + '_' + tup[1]))
#     df.index = MultiIndex.from_tuples(tuples)
#     return df


# def get_units(df):
#     # OrderedDict to keep order
#     units = [c[0] for c in df.columns]
#     return list(OrderedDict.fromkeys(units))


# def get_channels(df):
#     # OrderedDict to keep order
#     channels = [c[1] for c in df.columns]
#     return list(OrderedDict.fromkeys(channels))


# def count_channels(df):
#     units = get_units(df)
#     channels = get_channels(df)

#     count = {c:0 for c in channels}

#     for unit in units:
#         cols = df[unit].columns
#         for col in cols:
#             count[col]+=1
#     return count


# def select_units(df, units):
#     if not units:
#         return df
#     selection = df.loc[:, (units, slice(None))]
#     return rebuild(selection)


# def select_channels(df, channels):
#     units = get_units(df)
#     if not channels:
#         return df
#     selection = df.loc[:, (slice(None), channels)][units]
#     return rebuild(selection)


# def channel_apply(df, func, name):
#     cols = get_units(df)
#     result = pd.DataFrame()
#     for col in cols:
#         temp = df[col]
#         temp = temp.apply(func, axis=1)
#         result[col] = temp
#     result.columns = pd.MultiIndex.from_product([result.columns, [name]])
#     result = result[cols]
#     return result


# def channel_set(df, channels, values):
#     if type(values) == pd.core.frame.DataFrame:
#         values = values.values
#     df.loc[:, (slice(None), channels)] = values


# def select(df, units, channels):
    
#     if type(units) == str:
#         units = [units]
#     if type(channels) == str:
#         channels = [channels]
        
#     if units is not None:
#         if not all(unit in df.columns.levels[0] for unit in units):
#             raise ValueError(
#                 f'One of the units is not in columns.\nUnits: {units}\nColumns: {list(df.columns.levels[0])}')
#     if channels is not None:
#         if not all(channel in df.columns.levels[1] for channel in channels):
#             raise ValueError(
#                 f'One of the channels is not in columns.\nChannels:{channels}\nColumns:{list(df.columns.levels[1])}')
#     selection = select_units(df, units)
#     selection = select_channels(selection, channels)
#     return selection


# def remove_channels(df, channels):
#     if isinstance(channels, str):
#         channels = [channels]
#     new_channels = [c for c in get_channels(df) if c not in channels]
#     return select_channels(df, new_channels)


# def remove_units(df, units):
#     if isinstance(units, str):
#         units = [units]
#     new_units = [u for u in get_units(df) if u not in units]
#     return select_units(df, new_units)


# def inspect_units(df):
#     units = get_units(df)
#     inconsistent = []
#     for unit in units:
#         udf = df[unit]
#         if len(udf.dropna()) == 0:
#             inconsistent.append(unit)
#     return inconsistent


# def is_multilevel(index):
#     return index.nlevels > 1


# def fix_multilevel(xframe, yframe):
#     if not is_multilevel(xframe.columns):
#         xframe = add_level(xframe, 'main')
#         warnings.warn(f"Index of 'xframe' is not multi-level. A unit called 'main' was added.")
#     if not is_multilevel(yframe.columns):
#         yframe = add_level(yframe, 'main')
#         warnings.warn(f"Index of 'yframe' is not multi-level. A unit called 'main' was added.")
#     return xframe, yframe


# def smash_cols(df):
#     df = df.T
#     df = rename_subindex(df)
#     df = df.droplevel(0)
#     df = df.T
#     return df


# def unsmash(df, sep='_'):

#     cols = list(df.columns)
#     units = [i.split(sep)[0] for i in cols]
#     channels = [i.split(sep)[1] for i in cols]
#     tuples = list(zip(units, channels))
#     multicols = pd.MultiIndex.from_tuples(tuples)

#     return pd.DataFrame(df.values, index=df.index, columns=multicols)


# Rolling Functions

def last_max(x):
    """
    Return True if last element is the biggest one
    """
    return x[-1] > np.max(x[:-1])


def last_min(x):
    """
    Return True if last element is the smallest one
    """
    return x[-1] < np.min(x[:-1])

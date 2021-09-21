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

def ffill(arr, axis):
    arr = arr.astype(float)
    idx_shape = tuple([slice(None)] + [np.newaxis] * (len(arr.shape) - axis - 1))
    idx = np.where(~np.isnan(arr), np.arange(arr.shape[axis])[idx_shape], 0)
    np.maximum.accumulate(idx, axis=axis, out=idx)
    slc = [
        np.arange(k)[
            tuple(
                slice(None) if dim == i else np.newaxis
                for dim in range(len(arr.shape))
            )
        ]
        for i, k in enumerate(arr.shape)
    ]

    slc[axis] = idx
    return arr[tuple(slc)]

def bfill(arr, axis):
    arr = arr.astype(float)
    idx_shape = tuple([slice(None)] + [np.newaxis] * (len(arr.shape) - axis - 1))
    idx = np.where(~np.isnan(arr), np.arange(arr.shape[axis])[idx_shape], arr.shape[axis] - 1)
    slc = [
        np.arange(k)[
            tuple(
                slice(None) if dim == i else np.newaxis
                for dim in range(len(arr.shape))
            )
        ]
        for i, k in enumerate(arr.shape)
    ]

    slc[axis] = idx
    return arr[tuple(slc)]


def get_null_indexes(x):
    s = np.sum(x, axis=3)
    s = np.sum(s, axis=2)
    s = np.sum(s, axis=1)
    s = pd.Series(s).isna()
    return s[s==True].index.tolist()


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
    a = a.filter(units, channels)
    a = a.loc[b.index]
    return a, b


def revert_pct_change(original, changed, gap=0):
    return (original.shift(gap) * changed) + original.shift(gap)



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

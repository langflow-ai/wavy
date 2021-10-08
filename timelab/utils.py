from copy import copy
from itertools import groupby

import numpy as np
import pandas as pd
import plotly as px

pd.set_option("multi_sparse", True)  # To see multilevel indexes
pd.options.plotting.backend = "plotly"

cmap1 = px.colors.qualitative.Plotly
cmap2 = cmap1[::-1]


def ffill(arr, axis):
    arr = arr.astype(float)
    idx_shape = tuple([slice(None)] + [np.newaxis] * (len(arr.shape) - axis - 1))
    idx = np.where(~np.isnan(arr), np.arange(arr.shape[axis])[idx_shape], 0)
    np.maximum.accumulate(idx, axis=axis, out=idx)
    slc = [
        np.arange(k)[tuple(slice(None) if dim == i else np.newaxis for dim in range(len(arr.shape)))]
        for i, k in enumerate(arr.shape)
    ]

    slc[axis] = idx
    return arr[tuple(slc)]


def bfill(arr, axis):
    arr = arr.astype(float)
    idx_shape = tuple([slice(None)] + [np.newaxis] * (len(arr.shape) - axis - 1))
    idx = np.where(~np.isnan(arr), np.arange(arr.shape[axis])[idx_shape], arr.shape[axis] - 1)
    slc = [
        np.arange(k)[tuple(slice(None) if dim == i else np.newaxis for dim in range(len(arr.shape)))]
        for i, k in enumerate(arr.shape)
    ]

    slc[axis] = idx
    return arr[tuple(slc)]


def get_null_indexes(x):
    s = np.sum(x, axis=3)
    s = np.sum(s, axis=2)
    s = np.sum(s, axis=1)
    s = pd.Series(s).isna()
    return s[s == True].index.tolist()

# TODO: Check how this works
def get_all_unique(array):
    all_ = [i for i in array[0]]
    for i in array[1:]:
        all_.append(i[-1])
    return np.array(all_)


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
    df = df.dropna(how="all")
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
        arr_list = []
        for arr in array:
            arr = np.hstack([arr[i] for i in range(len(arr))])
            arr_list.append(arr)
        return np.array(arr_list)

    elif array.ndim == 3:
        array = np.hstack([array[i] for i in range(len(array))])
        return array
    else:
        raise ValueError("Array must have 3 or 4 dimensions")


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

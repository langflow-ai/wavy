from copy import copy
from itertools import groupby

import numpy as np
import pandas as pd


def reverse_pct_change(panel, df):
    df = df.shift() * (1 + panel.as_dataframe())
    return panel.update(df)



# def add_level(df, level_name):
#     """
#     Add asset level to DataFrame

#     Args:
#         df (DataFrame): Pandas DataFrame
#         level_name (string): Asset level name

#     Returns:
#         ``DataFrame``: DataFrame with asset level added
#     """

#     # return pd.concat({level_name: df.T}, names=[level_name]).T
#     df.columns = pd.MultiIndex.from_product([[level_name], df.columns])
#     return df


# def replace(ls, value, new_value):
#     ls = copy(ls)
#     idx = ls.index(value)
#     ls[idx] = new_value
#     return ls


# def ffill(arr, axis):
#     arr = arr.astype(float)
#     idx_shape = tuple([slice(None)] + [np.newaxis] * (len(arr.shape) - axis - 1))
#     idx = np.where(~np.isnan(arr), np.arange(arr.shape[axis])[idx_shape], 0)
#     np.maximum.accumulate(idx, axis=axis, out=idx)
#     slc = [
#         np.arange(k)[tuple(slice(None) if dim == i else np.newaxis for dim in range(len(arr.shape)))]
#         for i, k in enumerate(arr.shape)
#     ]

#     slc[axis] = idx
#     return arr[tuple(slc)]


# def bfill(arr, axis):
#     arr = arr.astype(float)
#     idx_shape = tuple([slice(None)] + [np.newaxis] * (len(arr.shape) - axis - 1))
#     idx = np.where(~np.isnan(arr), np.arange(arr.shape[axis])[idx_shape], arr.shape[axis] - 1)
#     slc = [
#         np.arange(k)[tuple(slice(None) if dim == i else np.newaxis for dim in range(len(arr.shape)))]
#         for i, k in enumerate(arr.shape)
#     ]

#     slc[axis] = idx
#     return arr[tuple(slc)]


# def all_equal(iterable):
#     """
#     Check if all the iterables are equals.

#     Parameters
#     ----------
#     iterable : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     TYPE
#         DESCRIPTION.

#     """
#     g = groupby(iterable)
#     return next(g, True) and not next(g, False)


# def add_dim(x, n=1):
#     for _ in range(n):
#         x = np.array([x])
#     return x


# # Rolling Functions


# def last_max(x):
#     """
#     Return True if last element is the biggest one
#     """
#     return x[-1] > np.max(x[:-1])


# def last_min(x):
#     """
#     Return True if last element is the smallest one
#     """
#     return x[-1] < np.min(x[:-1])
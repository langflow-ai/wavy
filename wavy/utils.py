import numpy as np
import pandas as pd


def reverse_pct_change(panel, df):
    df = df.shift() * (1 + panel.as_dataframe())
    return panel.update(df)


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
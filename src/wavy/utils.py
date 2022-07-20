from collections.abc import Iterable

import pandas as pd


def reverse_pct_change(change_df, original_df, periods=1):
    """
    Reverse the pct_change function.

    Args:
        change_df (pd.DataFrame): Dataframe to reverse
        original_df (pd.DataFrame): Reference Dataframe
        periods (int): Number of periods used on pct_change operation

    Returns:
        pd.DataFrame: Reversed dataframe
    """

    return original_df.shift(periods) * (change_df + 1)


def reverse_diff(diff_df, original_df, periods=1):
    """
    Reverse the pct_diff function.

    Args:
        diff_df (pd.DataFrame): Dataframe to reverse
        original_df (pd.DataFrame): Reference Dataframe
        periods (int): Number of periods used on diff operation

    Returns:
        pd.DataFrame: Reversed dataframe
    """

    return original_df.shift(periods) + diff_df

from collections.abc import Iterable

import pandas as pd

# from wavy.panel import Panel

# -------------------
# DataFrame utils


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


# -------------------
# Panel utils

# def reverse_pct_change(changed_panel, initial_panel, periods=1):
#     # ! Work in progress
#     """
#     Reverse the pct_change function.

#     Args:
#         panel_change (wavy.Panel): Panel to reverse
#         ref_panel (wavy.Panel): Reference Panel

#     Returns:
#         pd.DataFrame: Reversed dataframe
#     """

#     # TODO: Reference must come from columns.name
#     # Just shifting the panel is dangerous.
#     # e.g. say that panel skips 10 and 11 indices, and just has 9, 12, 13, 14...
#     # Shifting will skip from 9 to 12, unless there's a parameter to shift based on the indices (columns.name).
#     initial_shifted = initial_panel.shift_(periods)
#     return initial_shifted * (changed_panel + 1)


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


# -------------------
# Other utils


# def is_dataframe(x):
#     """
#     Check if x is a dataframe.

#     Args:
#         x (object): Object to check

#     Returns:
#         bool: True if x is a dataframe, False otherwise
#     """
#     return isinstance(x, pd.DataFrame)


# def is_series(x):
#     """
#     Check if x is a pd.Series.

#     Args:
#         x (object): Object to check

#     Returns:
#         bool: True if x is pd.Series, False otherwise
#     """
#     return isinstance(x, pd.Series)


# def is_iterable(x):
#     """
#     Check if x is iterable.

#     Args:
#         x (object): Object to check

#     Returns:
#         bool: True if x is iterable, False otherwise
#     """
#     return isinstance(x, Iterable)


# def is_panel(x):
#     """
#     Check if x is a panel.

#     Args:
#         x (object): Object to check

#     Returns:
#         bool: True if x is a panel, False otherwise
#     """
#     return isinstance(x, Panel)

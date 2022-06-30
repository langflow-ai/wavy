import pandas as pd


def reverse_pct_change(changed_panel, initial_panel, periods=1):
    """
    Reverse the pct_change function.

    Args:
        panel_change (wavy.Panel): Panel to reverse
        ref_panel (wavy.Panel): Reference Panel

    Returns:
        pd.DataFrame: Reversed dataframe
    """

    # TODO: Reference must come from columns.name
    # Just shifting the panel is dangerous.
    # e.g. say that panel skips 10 and 11 indices, and just has 9, 12, 13, 14...
    # Shifting will skip from 9 to 12, unless there's a parameter to shift based on the indices (columns.name).
    initial_shifted = initial_panel.shift_(periods)
    return initial_shifted * (changed_panel + 1)


def is_dataframe(x):
    """
    Check if x is a dataframe.

    Args:
        x (object): Object to check

    Returns:
        bool: True if x is a dataframe, False otherwise
    """
    return isinstance(x, pd.DataFrame)


def is_series(x):
    """
    Check if x is a pd.Series.

    Args:
        x (object): Object to check

    Returns:
        bool: True if x is pd.Series, False otherwise
    """
    return isinstance(x, pd.Series)


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

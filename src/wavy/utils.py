import pandas as pd


def reverse_pct_change(panel, df):
    """
    Reverse the pct_change function.

    Args:
        panel (wavy.Panel): Panel to reverse
        df (pd.DataFrame): Dataframe to reverse

    Returns:
        pd.DataFrame: Reversed dataframe
    """
    df = df.shift() * (1 + panel.as_dataframe())
    return panel.update(df)


def is_dataframe(x):
    """
    Check if x is a dataframe.

    Args:
        x (object): Object to check

    Returns:
        bool: True if x is a dataframe, False otherwise
    """
    return isinstance(x, pd.DataFrame)


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

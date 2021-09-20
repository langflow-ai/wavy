import numpy as np
import pandas as pd
from tqdm import tqdm


def min_frequency(data):
    return np.diff(data.index.values).min().astype(int)


def max_frequency(data):
    return np.diff(data.index.values).max().astype(int)


def constant_freq(data):
    if min_frequency(data) != max_frequency(data):
        return False
    return True


def positive_freq(data):
    if min_frequency(data) < 0:
        return False
    return True


def check_frequency(data):
    print(f"First: {data.index[0]}")
    print(f"Last: {data.index[-1]}")
    diff = pd.Series(data.index).diff()
    print(f"Min Frequency: {diff.min()}")
    print(f"Max Frequency: {diff.max()}")

    if infer_frequency(data):
        return True
    else:
        return False


def inspect_freq(data):
    """
    Inspect if the data frequency is constant.

    Parameters
    ----------
    data : dataframe

    """

    if check_frequency(data):
        print(f"Frequency is constant: {infer_frequency(data)}")
        return

    print(f"Min Frequency: {min_frequency(data)} ns")
    print(f"Max Frequency: {max_frequency(data)} ns")

    # Show duplicated indexes
    duplicated_index = data.index.duplicated()
    duplicated_data = data.loc[duplicated_index]
    print(f"Duplicated Date Times found:\n{duplicated_data.index}")

    # Remove duplicated indexes
    non_duplicated_index = list(~np.array(duplicated_index))
    data = data.loc[non_duplicated_index]

    while True:
        # for index, row in tqdm(data.iterrows()):
        for i, (index, row) in tqdm(enumerate(data.iterrows())):
            if index == data.index[0] or index == data.index[1]:
                continue
            if not check_frequency(data.loc[:index]):
                print(f"Frequency breaks at {index}")
                data = data.iloc[i:]
                break
        if check_frequency(data.loc[:index]):
            break
    return


def infer_frequency(data):
    return data.index.inferred_freq


def resample_datetimes(data, rule='1H0min'):

    # Remove duplicated datetimes
    duplicated_index = data.index.duplicated()
    non_duplicated_index = list(~np.array(duplicated_index))
    data = data.loc[non_duplicated_index]

    # Resample
    data = data.resample(rule=rule).backfill()

    return data

import numpy as np


def minfreq(data):
    return np.diff(data.index.values).min().astype(int)


def maxfreq(data):
    return np.diff(data.index.values).max().astype(int)


def is_constant(data):
    return minfreq(data) == maxfreq(data)


def is_positive(data):
    return minfreq(data) > 0


def infer(data):
    return data.index.inferred_freq


def get_duplicated(data):
    return data.loc[data.index.duplicated()].index.tolist()


def remove_duplicated(data):
    duplicated = get_duplicated(data)
    non_duplicated = list(~np.array(duplicated))
    return data.loc[non_duplicated]


def inspect(data, verbose=1):
    """
    Inspect if the data frequency is inferable at each timestep.

    Parameters
    ----------
    data : dataframe

    """

    if verbose > 0:
        print(f"First: {data.index[0]}")
        print(f"Last: {data.index[-1]}\n")

        print(f"Min Frequency: {minfreq(data)} ns")
        print(f"Max Frequency: {maxfreq(data)} ns\n")

    duplicated = get_duplicated(data)
    if duplicated and verbose > 0:
            print(f"Duplicated Date Times found:\n{duplicated}")

    breaks = []

    while True:
        for i, index in enumerate(data.index):
            if index in (data.index[0], data.index[1]):
                continue
            if not infer(data.loc[:index]):
                if verbose > 1:
                    print(f"Frequency breaks at {index}")
                breaks.append(index)
                data = data.iloc[i:]
                break
        if infer(data.loc[:index]):
            break
    return duplicated, breaks

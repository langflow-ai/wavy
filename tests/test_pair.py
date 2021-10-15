import os
import unittest

import pandas as pd

# from sklearn.metrics import mean_squared_error

from timelab.pair import from_frames
from timelab.utils import all_equal

TEST_DATA = os.path.join(os.path.dirname(__file__), "test_data/multi_asset.pkl")


def assert_all(panel):
    # Change to type error
    # Add more checks
    # All gaps remain constant
    # All ys come after xs
    # All indexes are timestamps
    # All xs have same number of channels and units
    # All ys have same number of channels and units
    # lookback, horizon and gap must be int
    # Check when each assert must be done
    # check all: xframe1.iloc[0].index + gap == yframe0.iloc[-1].index or smth like that
    # Assert data is sorted
    # Assert all xunits and xchannels are equal, same for ys
    # No xstart date should repeat, I think
    # y freq should be the same as x freq, I think
    # All xindex must be smaller than the next xindex
    # All yindex must be smaller than the next yindex
    # Must check if y is always a future x

    assert all_equal([i.xunits for i in panel.pairs]), "Pairs must have the same units."
    assert all_equal([i.xchannels for i in panel.pairs]), "Pairs must have the same channels."
    assert all_equal([i.horizon for i in panel.pairs]), "Pairs must have the same horizon."
    assert all_equal([i.lookback for i in panel.pairs]), "Pairs must have the same lookback."
    assert all_equal([i.X.shape for i in panel.pairs]), "Pairs must have the same shape."
    assert all_equal([i.y.shape for i in panel.pairs]), "Pairs must have the same shape."
    assert panel.lookback is not None, "lookback was not defined."
    assert panel.horizon is not None, "horizon was not defined."
    assert panel.gap is not None, "gap was not defined."
    assert panel.gap >= 0
    assert panel.horizon > 0
    assert panel.lookback > 0

    # TODO: improve / speed up this check
    # if not all(isinstance(pair, TimePair) for pair in panel.pairs):
    #     raise AttributeError(
    #         "Attribute pairs does not consist of a list of TimePairs."
    #     )


class TestPair:
    df = pd.read_pickle(TEST_DATA)
    df = df[["LNC", "MAS", "CSX"]]

    xframe = df.iloc[:50]
    yframe = df.iloc[50:]

    xframe_stock = xframe["LNC"]
    yframe_stock = yframe["LNC"]

    def test_from_frames(self):
        pair_stock = from_frames(self.xframe_stock, self.yframe_stock)

        # Test xframe and yframe
        x_mse = mean_squared_error(pair_stock.xframe.sum()["main"], self.xframe_stock.sum())
        y_mse = mean_squared_error(pair_stock.yframe.sum()["main"], self.yframe_stock.sum())
        assert x_mse == 0
        assert y_mse == 0

        # Test X & y
        xframe_row = self.xframe_stock.iloc[12]
        X_row = pd.DataFrame(pair_stock.X[0]).iloc[12]
        yframe_row = self.yframe_stock.iloc[12]
        y_row = pd.DataFrame(pair_stock.y[0]).iloc[12]
        assert xframe_row.sum() == X_row.sum()
        assert yframe_row.sum() == y_row.sum()

        # Test frequencies
        assert pair_stock.yframe.index[0] > pair_stock.xframe.index[-1] == True
        assert pair_stock.xframe.index[-1] > pair_stock.xframe.index[0] == True
        assert pair_stock.yframe.index[-1] > pair_stock.yframe.index[0] == True

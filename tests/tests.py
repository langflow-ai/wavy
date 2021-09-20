import os
import unittest
import pandas as pd
from sklearn.metrics import mean_squared_error

from timelab.pair import from_frames


TEST_DATA = os.path.join(os.path.dirname(__file__), 'test_data/multi_asset.pkl')


class TestPair(unittest.TestCase):
    df = pd.read_pickle(TEST_DATA)
    df = df[['LNC', 'MAS', 'CSX']]

    xframe = df.iloc[:50]
    yframe = df.iloc[50:]

    xframe_stock = xframe['LNC']
    yframe_stock = yframe['LNC']

    def test_from_frames(self):
        pair_stock = from_frames(self.xframe_stock, self.yframe_stock)

        # Test xframe and yframe
        x_mse = mean_squared_error(pair_stock.xframe.sum()['main'], self.xframe_stock.sum())
        y_mse = mean_squared_error(pair_stock.yframe.sum()['main'], self.yframe_stock.sum())
        self.assertEqual(x_mse, 0)
        self.assertEqual(y_mse, 0)

        # Test X & y
        xframe_row = self.xframe_stock.iloc[12]
        X_row = pd.DataFrame(pair_stock.X[0]).iloc[12]
        yframe_row = self.yframe_stock.iloc[12]
        y_row = pd.DataFrame(pair_stock.y[0]).iloc[12]
        self.assertEqual(xframe_row.sum(), X_row.sum())
        self.assertEqual(yframe_row.sum(), y_row.sum())

        # Test frequencies
        self.assertEqual(pair_stock.yframe.index[0] > pair_stock.xframe.index[-1], True)
        self.assertEqual(pair_stock.xframe.index[-1] > pair_stock.xframe.index[0], True)
        self.assertEqual(pair_stock.yframe.index[-1] > pair_stock.yframe.index[0], True)
        return


if __name__ == '__main__':
    unittest.main()

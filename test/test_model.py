import numpy as np
import pandas as pd
import pytest
import wavy
from wavy import models


@pytest.fixture
def bigger_panel():
    columns = ["A", "B", "C", "D"]
    index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    data = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
        [17, 18, 19, 20],
        [21, 22, 23, 24],
        [25, 26, 27, 28],
        [29, 30, 31, 32],
        [33, 34, 35, 36],
        [37, 38, 39, 40],
        [41, 42, 43, 44],
        [45, 46, 47, 48],
    ]
    df = pd.DataFrame(columns=columns, index=index, data=data)
    x, y = wavy.create_panels(df, lookback=2, horizon=1, gap=0)
    y = y[["D"]]
    return x, y


def test_bool_conversion(bigger_panel):
    x, y = bigger_panel
    y = y % 8 == 0
    wavy.set_training_split(x, y, train_size=0.3, test_size=0.3, val_size=0.4)
    _ = models.DenseModel(x, y, model_type="classification")

    assert y["D"].dtype == "int", "y should be converted to int"

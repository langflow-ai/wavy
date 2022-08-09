import numpy as np
import pandas as pd
import pytest
import wavy

columns = ["A", "B"]
index = [1, 2, 3, 4, 5, 6]
data = [[np.nan, 2], [np.inf, 4], [5, 6], [7, 8], [9, 10], [11, 12]]


@pytest.fixture
def panel():
    df = pd.DataFrame(columns=columns, index=index, data=data)
    x, y = wavy.create_panels(df, lookback=3, horizon=1, gap=0)
    return x, y


def test_panel_creation():
    df = pd.DataFrame(columns=columns, index=index, data=data)
    try:
        _ = wavy.create_panels(df, lookback=1, horizon=1, gap=0)
    except Exception as e:
        assert False, f"'create_panels' raised an exception {e}"


def test_columns(panel):
    x, y = panel
    assert x.columns.to_list() == columns, "Columns of x are not correct"
    assert y.columns.to_list() == columns, "Columns of y are not correct"


def test_shape(panel):
    x, y = panel
    assert x.shape_panel == (3, 3, 2), "Shape panel x is not correct"
    assert y.shape_panel == (3, 1, 2), "Shape panel y is not correct"
    assert x.shape == (9, 2), "Shape x is not correct"
    assert y.shape == (3, 2), "Shape y is not correct"


def test_timesteps(panel):
    x, y = panel
    assert x.num_timesteps == 3, "Timesteps x is not correct"
    assert y.num_timesteps == 1, "Timesteps y is not correct"


def test_frames(panel):
    x, y = panel
    assert x.num_frames == 3, "Frames x is not correct"
    assert y.num_frames == 3, "Frames y is not correct"


def test_findna(panel):
    x, _ = panel
    assert x.findna_frames() == [0], "Nan x indexes are not correct"


def test_ids(panel):
    x, y = panel
    assert x.ids.to_list() == [0, 1, 2], "IDs x are not correct"
    assert y.ids.to_list() == [0, 1, 2], "IDs y are not correct"


def test_dropna(panel):
    x, _ = panel
    assert set(x.ids) - set(x.dropna_frames().ids) == {0}, "Nan indexes are not correct"
    x.dropna_frames(inplace=True)
    assert x.ids.to_list() == [1, 2], "IDs x are not correct"


def test_reset_ids(panel):
    x, _ = panel
    assert x.dropna_frames().reset_ids().ids.to_list() == [
        0,
        1,
    ], "IDs x are not correct"
    x.dropna_frames(inplace=True)
    x.reset_ids(inplace=True)
    assert x.ids.to_list() == [0, 1], "IDs x are not correct"


if __name__ == "__main__":
    pytest.main()

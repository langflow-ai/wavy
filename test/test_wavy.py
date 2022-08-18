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


# TODO test function reset_ids

# TODO this test is not correct
def test_dropna_match(panel):
    x, y = panel
    x = x.dropna_frames()
    y = y.match_frames(x)
    assert x.ids.to_list() == [1, 2], "IDs y are not correct"
    assert y.ids.to_list() == [1, 2], "IDs y are not correct"


def test_concat_panel(panel):
    x, y = panel
    with pytest.raises(ValueError):
        wavy.concat_panels([x, x])
    a = x.copy()
    a.ids = [3, 4, 5]
    assert wavy.concat_panels([x, a]).ids.to_list() == [
        0,
        1,
        2,
        3,
        4,
        5,
    ], "IDs x are not correct"
    assert wavy.concat_panels([a, x], reset_ids=True).ids.to_list() == [
        0,
        1,
        2,
        3,
        4,
        5,
    ], "IDs x are not correct"
    assert wavy.concat_panels([a, x], sort=True).ids.to_list() == [
        0,
        1,
        2,
        3,
        4,
        5,
    ], "IDs x are not correct"


# TODO test set_training_split


def test_num_frames(panel):
    x, y = panel
    assert x.num_frames == 3, "Frames x is not correct"
    assert y.num_frames == 3, "Frames y is not correct"


def test_num_timesteps(panel):
    x, y = panel
    assert x.num_timesteps == 3, "Timesteps x is not correct"
    assert y.num_timesteps == 1, "Timesteps y is not correct"


# TODO test num_columns


def test_columns(panel):
    x, y = panel
    assert x.columns.to_list() == columns, "Columns of x are not correct"
    assert y.columns.to_list() == columns, "Columns of y are not correct"


def test_ids(panel):
    x, y = panel
    assert x.ids.to_list() == [0, 1, 2], "IDs x are not correct"
    assert y.ids.to_list() == [0, 1, 2], "IDs y are not correct"


def test_reset_ids(panel):
    x, _ = panel
    assert x.dropna_frames().reset_ids().ids.to_list() == [
        0,
        1,
    ], "IDs x are not correct"
    x.dropna_frames(inplace=True)
    x.reset_ids(inplace=True)
    assert x.ids.to_list() == [0, 1], "IDs x are not correct"


def test_shape(panel):
    x, y = panel
    assert x.shape_panel == (3, 3, 2), "Shape panel x is not correct"
    assert y.shape_panel == (3, 1, 2), "Shape panel y is not correct"
    assert x.shape == (9, 2), "Shape x is not correct"
    assert y.shape == (3, 2), "Shape y is not correct"


# TODO test row_panel with a list
# TODO check if values are correct, not only shape
def test_row_panel(panel):
    x, y = panel
    assert x.row_panel().shape[0] == 3, "Row panel x is not correct"
    assert y.row_panel().shape[0] == 3, "Row panel y is not correct"


# TODO test get_timesteps with a list
def test_get_timesteps(panel):
    x, _ = panel
    assert x.get_timesteps().to_list() == [1, 2, 3], "Timesteps x are not correct"
    assert x.get_timesteps(1).to_list() == [2, 3, 4], "Timesteps x are not correct"
    assert x.get_timesteps(2).to_list() == [3, 4, 5], "Timesteps x are not correct"


# TODO check if values are correct, not only shape
def test_values_panel(panel):
    x, y = panel
    assert x.values_panel.shape == (3, 3, 2), "Values panel x is not correct"
    assert y.values_panel.shape == (3, 1, 2), "Values panel y is not correct"


# TODO check if values are correct, not only shape
def test_flatten_panel(panel):
    x, y = panel
    assert x.flatten_panel().shape == (3, 6), "Flatten panel x is not correct"
    assert y.flatten_panel().shape == (3, 2), "Flatten panel y is not correct"


def test_drop_ids(panel):
    x, y = panel
    assert x.drop_ids(ids=[1]).ids.to_list() == [0, 2], "IDs x are not correct"
    assert wavy.Panel(x.drop_ids(ids=[0, 2])).ids.to_list() == [
        1
    ], "IDs x are not correct"
    assert y.drop_ids(ids=1).ids.to_list() == [0, 2], "IDs y are not correct"
    assert wavy.Panel(y.drop_ids(ids=[0, 1])).ids.to_list() == [
        2
    ], "IDs y are not correct"
    x.drop_ids(ids=[1], inplace=True)
    assert x.ids.to_list() == [0, 2], "IDs x are not correct"
    wavy.Panel(x.drop_ids(ids=[1, 2], inplace=True))
    assert x.ids.to_list() == [0], "IDs x are not correct"
    y.drop_ids(ids=1, inplace=True)
    assert y.ids.to_list() == [0, 2], "IDs y are not correct"
    wavy.Panel(y.drop_ids(ids=[0, 1], inplace=True))
    assert y.ids.to_list() == [2], "IDs y are not correct"


def test_findna_frames(panel):
    x, _ = panel
    assert x.findna_frames() == [0], "Nan x indexes are not correct"


def test_dropna_frames(panel):
    x, _ = panel
    assert set(x.ids) - set(x.dropna_frames().ids) == {0}, "Nan indexes are not correct"
    x.dropna_frames(inplace=True)
    assert x.ids.to_list() == [1, 2], "IDs x are not correct"


# TODO this test is not correct, should remove a frame and then match to test
def test_match_frames(panel):
    x, y = panel
    assert x.match_frames(y).ids.to_list() == [0, 1, 2], "IDs x are not matching"
    assert y.match_frames(x).ids.to_list() == [0, 1, 2], "IDs y are not matching"
    x.match_frames(y, inplace=True)
    assert x.ids.to_list() == [0, 1, 2], "IDs x are not matching"
    y.match_frames(x, inplace=True)
    assert y.ids.to_list() == [0, 1, 2], "IDs y are not matching"


# TODO test set_training_split


def test_to_dataframe(panel):
    x, _ = panel
    assert isinstance(x.to_dataframe(), pd.DataFrame), "Dataframe x is not correct"


# TODO test train/val/test

# TODO this test is wrong, use x.head_panel(2) and check values
def test_head_panel(panel):
    x, y = panel
    assert x.head_panel(3).shape_panel == (3, 3, 2), "Head panel x is not correct"
    assert y.head_panel(3).shape_panel == (3, 1, 2), "Head panel y is not correct"


# TODO this test is wrong, use x.tail_panel(2) and check values
def test_tail_panel(panel):
    x, y = panel
    assert x.tail_panel(2).shape_panel == (2, 3, 2), "Tail panel x is not correct"
    assert y.tail_panel(2).shape_panel == (2, 1, 2), "Tail panel y is not correct"


def test_sort_panel(panel):
    x, _ = panel
    x.sort_panel(ascending=False, inplace=True)
    assert x.ids.to_list() == [2, 1, 0], "IDs x are not correct"
    x.sort_panel(ascending=True, inplace=True)
    assert x.ids.to_list() == [0, 1, 2], "IDs x are not correct"
    a = x.sort_panel(ascending=False, inplace=False)
    assert a.ids.to_list() == [2, 1, 0], "IDs x are not correct"


# TODO test sample_panel, set seed and check values
# TODO test shuffle_panel


# TODO test models in a new file

if __name__ == "__main__":
    pytest.main()

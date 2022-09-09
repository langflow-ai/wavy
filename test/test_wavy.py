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


@pytest.fixture
def bigger_panel():
    index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    data = [
        [np.nan, 2],
        [np.inf, 4],
        [5, 6],
        [7, 8],
        [9, 10],
        [11, 12],
        [13, 14],
        [15, 16],
        [17, 18],
        [19, 20],
        [21, 22],
        [23, 24],
    ]
    df = pd.DataFrame(columns=columns, index=index, data=data)
    x, y = wavy.create_panels(df, lookback=2, horizon=1, gap=0)
    return x, y


def test_panel_creation():
    df = pd.DataFrame(columns=columns, index=index, data=data)
    try:
        _ = wavy.create_panels(df, lookback=1, horizon=1, gap=0)
    except Exception as e:
        assert False, f"'create_panels' raised an exception {e}"


def test_reset_ids(panel):
    x, _ = panel
    assert x.dropna_frames().reset_ids().ids.to_list() == [
        0,
        1,
    ], "IDs x are not correct"
    x.dropna_frames(inplace=True)
    x.reset_ids(inplace=True)
    assert x.ids.to_list() == [0, 1], "IDs x are not correct"


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


def test_num_frames(panel):
    x, y = panel
    assert x.num_frames == 3, "Frames x is not correct"
    assert y.num_frames == 3, "Frames y is not correct"


def test_num_timesteps(panel):
    x, y = panel
    assert x.num_timesteps == 3, "Timesteps x is not correct"
    assert y.num_timesteps == 1, "Timesteps y is not correct"


def test_num_columns(panel):
    x, y = panel
    assert x.num_columns == 2, "Columns x are not correct"
    assert y.num_columns == 2, "Columns y are not correct"


def test_columns(panel):
    x, y = panel
    assert x.columns.to_list() == columns, "Columns of x are not correct"
    assert y.columns.to_list() == columns, "Columns of y are not correct"


def test_ids(panel):
    x, y = panel
    assert x.ids.to_list() == [0, 1, 2], "IDs x are not correct"
    assert y.ids.to_list() == [0, 1, 2], "IDs y are not correct"


def test_shape(panel):
    x, y = panel
    assert x.shape_panel == (3, 3, 2), "Shape panel x is not correct"
    assert y.shape_panel == (3, 1, 2), "Shape panel y is not correct"
    assert x.shape == (9, 2), "Shape x is not correct"
    assert y.shape == (3, 2), "Shape y is not correct"


def test_row_panel(panel):
    x, y = panel
    assert x.row_panel().get_timesteps().to_list() == [
        1,
        2,
        3,
    ], "Row panel x are not correct"
    assert y.row_panel().get_timesteps().to_list() == [
        4,
        5,
        6,
    ], "Row panel y are not correct"


def test_get_timesteps(panel):
    x, y = panel
    assert x.get_timesteps().to_list() == [1, 2, 3], "Timesteps x are not correct"
    assert x.get_timesteps([1, 2]).to_list() == [
        2,
        3,
        3,
        4,
        4,
        5,
    ], "Timesteps x are not correct"
    assert y.get_timesteps().to_list() == [4, 5, 6], "Timesteps y are not correct"


def test_values_panel(panel):
    x, y = panel
    assert x.values_panel.shape == (3, 3, 2), "Values panel x is not correct"
    assert y.values_panel.shape == (3, 1, 2), "Values panel y is not correct"
    assert np.alltrue(
        x.values_panel[0][2] == [5.0, 6.0]
    ), "Values panel x is not correct"
    assert np.alltrue(
        y.values_panel[0][0] == [7.0, 8.0]
    ), "Values panel y is not correct"


def test_flatten_panel(panel):
    x, y = panel
    assert x.flatten_panel().shape == (3, 6), "Flatten panel x is not correct"
    assert y.flatten_panel().shape == (3, 2), "Flatten panel y is not correct"
    assert np.alltrue(
        x.flatten_panel().values[2] == [5.0, 7.0, 9.0, 6.0, 8.0, 10.0]
    ), "Flatten panel x is not correct"
    assert np.alltrue(
        y.flatten_panel().values[2] == [11.0, 12.0]
    ), "Flatten panel y is not correct"


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


def test_match_frames(panel):
    x, y = panel
    assert x.match_frames(y).ids.to_list() == [0, 1, 2], "IDs x are not matching"
    assert y.match_frames(x).ids.to_list() == [0, 1, 2], "IDs y are not matching"
    x.drop_ids([0], inplace=True)
    y.drop_ids([0], inplace=True)
    x.match_frames(y, inplace=True)
    assert x.ids.to_list() == [1, 2], "IDs x are not matching"
    y.match_frames(x, inplace=True)
    assert y.ids.to_list() == [1, 2], "IDs y are not matching"


def test_to_dataframe(panel):
    x, _ = panel
    assert isinstance(x.to_dataframe(), pd.DataFrame), "Dataframe x is not correct"


def test_head_panel(panel):
    x, y = panel
    assert x.head_panel(2).ids.to_list() == [0, 1], "Head panel x is not correct"
    assert y.head_panel(2).ids.to_list() == [0, 1], "Head panel y is not correct"


def test_tail_panel(panel):
    x, y = panel
    assert x.tail_panel(2).ids.to_list() == [1, 2], "Tail panel x is not correct"
    assert y.tail_panel(2).ids.to_list() == [1, 2], "Tail panel y is not correct"


def test_sort_panel(panel):
    x, _ = panel
    x.sort_panel(ascending=False, inplace=True)
    assert x.ids.to_list() == [2, 1, 0], "IDs x are not correct"
    x.sort_panel(ascending=True, inplace=True)
    assert x.ids.to_list() == [0, 1, 2], "IDs x are not correct"
    a = x.sort_panel(ascending=False, inplace=False)
    assert a.ids.to_list() == [2, 1, 0], "IDs x are not correct"


def test_set_training_split(bigger_panel):
    # Using only float values
    x, y = bigger_panel
    wavy.set_training_split(x, y, train_size=0.4, val_size=0.2, test_size=0.4)
    assert x.train.num_frames == 4, "Train frames are not correct"
    assert x.val.num_frames == 1, "Val frames are not correct"
    assert x.test.num_frames == 3, "Test frames are not correct"

    # Using only int values
    wavy.set_training_split(x, y, train_size=5, test_size=3)
    assert x.train.num_frames == 5, "Train frames are not correct"
    assert x.test.num_frames == 2, "Test frames are not correct"
    assert x.val.num_frames == 1, "Val frames are not correct"


def test_train_val_test(bigger_panel):
    # Using only float values
    x, y = bigger_panel
    wavy.set_training_split(x, y, train_size=0.6, val_size=0.2)
    a = x.train.values
    b = np.array(
        [
            [np.nan, 2.0],
            [np.inf, 4.0],
            [np.inf, 4.0],
            [5.0, 6.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [9.0, 10.0],
            [11.0, 12.0],
            [11.0, 12.0],
            [13.0, 14.0],
        ]
    )
    assert np.alltrue(
        (a == b) | (np.isnan(a) & np.isnan(b))
    ), "Train values are not correct"
    assert np.alltrue(
        x.val.values == np.array([[15.0, 16.0], [17.0, 18.0]])
    ), "Val values are not correct"
    assert np.alltrue(
        x.test.values == np.array([[19.0, 20.0], [21.0, 22.0]])
    ), "Test values are not correct"

    # Using only int values
    wavy.set_training_split(x, y, train_size=5, test_size=3)
    a = x.train.values
    b = np.array(
        [
            [np.nan, 2.0],
            [np.inf, 4.0],
            [np.inf, 4.0],
            [5.0, 6.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [9.0, 10.0],
            [11.0, 12.0],
        ]
    )
    assert np.alltrue(
        (a == b) | (np.isnan(a) & np.isnan(b))
    ), "Train values are not correct"
    assert np.alltrue(
        x.test.values
        == np.array([[17.0, 18.0], [19.0, 20.0], [19.0, 20.0], [21.0, 22.0]])
    ), "Test values are not correct"
    assert np.alltrue(
        x.val.values == np.array([[13.0, 14.0], [15.0, 16.0]])
    ), "Val values are not correct"


def test_sample_panel(bigger_panel):
    x, y = bigger_panel
    wavy.set_training_split(x, y, train_size=5, test_size=3)
    a = x.sample_panel(samples=4, how="spaced", reset_ids=True, seed=101)
    b = np.array(
        [
            [np.nan, 2.0],
            [np.inf, 4.0],
            [9.0, 10.0],
            [11.0, 12.0],
            [17.0, 18.0],
            [19.0, 20.0],
        ]
    )
    assert np.alltrue(
        (a == b) | (np.isnan(a) & np.isnan(b))
    ), "Sample panel are not correct"
    assert a.ids.to_list() == [0, 1, 2], "Sample panel ids are not correct"

    # Using reset_ids = False
    k = y.sample_panel(samples=7, how="random", reset_ids=False, seed=101)
    np.array(
        [
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [13.0, 14.0],
            [15.0, 16.0],
            [19.0, 20.0],
            [23.0, 24.0],
        ]
    )
    assert np.alltrue(
        k.values
        == np.array(
            [
                [5.0, 6.0],
                [7.0, 8.0],
                [9.0, 10.0],
                [13.0, 14.0],
                [15.0, 16.0],
                [19.0, 20.0],
                [23.0, 24.0],
            ]
        )
    ), "Sample panel are not correct"
    assert k.ids.to_list() == [0, 1, 2, 4, 5, 7, 9], "Sample panel ids are not correct"


def test_shuffle_panel(bigger_panel):
    x, y = bigger_panel
    wavy.set_training_split(x, y, train_size=0.5, test_size=3, val_size=0.2)
    a = x.shuffle_panel(seed=101, reset_ids=True)
    b = np.array(
        [
            [np.nan, 2.0],
            [np.inf, 4.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [np.inf, 4.0],
            [5.0, 6.0],
            [9.0, 10.0],
            [11.0, 12.0],
            [13.0, 14.0],
            [15.0, 16.0],
            [17.0, 18.0],
            [19.0, 20.0],
            [19.0, 20.0],
            [21.0, 22.0],
        ]
    )
    assert np.alltrue(
        (a == b) | (np.isnan(a) & np.isnan(b))
    ), "Shuffle panel are not correct"
    assert a.num_frames == 8, "Shuffle panel frames are not correct"
    assert a.ids.to_list() == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    ], "Shuffle panel ids are not correct"
    assert a.shape[0] == 16, "Shuffle panel are not correct"

    # Using reset_ids = False
    k = x.shuffle_panel(seed=101, reset_ids=False)
    assert k.ids.to_list() == [
        0,
        3,
        2,
        1,
        4,
        6,
        8,
        9,
    ], "Shuffle panel ids are not correct"


if __name__ == "__main__":
    pytest.main()

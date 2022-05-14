import numpy as np
import pandas as pd
import pytest
import wavy

columns = ["A", "B"]
index = [1, 2, 3, 4]
values = [[np.nan, 2], [np.inf, 4], [5, 6], [7, 8]]


@pytest.fixture
def panel():
    df = pd.DataFrame(columns=columns, index=index, data=values)
    x, _ = wavy.create_panels(df, lookback=1, horizon=1, gap=0)
    return x


def test_panel_creation():
    df = pd.DataFrame(
        columns=["A", "B"], index=[1, 2, 3], data=[[1, 2], [3, 4], [5, 6]]
    )
    try:
        _ = wavy.create_panels(df, lookback=1, horizon=1, gap=0)
    except Exception as e:
        assert False, f"'create_panels' raised an exception {e}"


def test_columns(panel):
    assert panel.columns.to_list() == columns, "Columns are not correct"


def test_index(panel):
    assert (
        panel.index.all() in index and panel[1].index.all() in index
    ), "Indexes are not correct"


# def test_data(panel):
#     assert panel[0].values == values, "Indexes are not correct"


def test_shape(panel):
    assert panel.shape == (3, 1, 2), "Shape is not correct"


def test_countna(panel):
    assert panel.countna().sum().sum() == 1, "Nan sum is not correct"


def test_dropna(panel):
    assert panel.dropnaw().shape[0] == 2, "Nan indexes are not correct"


def test_findna(panel):
    assert panel.findna() == [0], "Nan indexes are not correct"


def test_findinf(panel):
    assert panel.findinf() == [1], "Inf indexes are not correct"


def test_get_data(panel):
    assert isinstance(panel[0], pd.DataFrame), "Error getting panel with integer index"
    assert isinstance(panel[:1], wavy.Panel), "Error getting panel with slice index"
    assert isinstance(panel[[0, 1, 2]], wavy.Panel), "Error getting panel with list"
    assert isinstance(panel["B"], wavy.Panel), "Error getting panel with string"
    assert isinstance(
        panel[["B"]], wavy.Panel
    ), "Error getting panel with list of strings"
    assert isinstance(
        panel[0, "B"], pd.DataFrame
    ), "Error getting panel with tuple of int and string"
    assert isinstance(
        panel[:2, "B"], wavy.Panel
    ), "Error getting panel with slice and strings"
    assert isinstance(
        panel[[0, 1, 2], "B"], wavy.Panel
    ), "Error getting panel with list of int and strings"
    assert isinstance(
        panel[0, ["B"]], pd.DataFrame
    ), "Error getting panel with int and list of strings"
    assert isinstance(
        panel[:2, ["B"]], wavy.Panel
    ), "Error getting panel with slice and list of strings"
    assert isinstance(
        panel[[0, 1, 2], ["B"]], wavy.Panel
    ), "Error getting panel with list of int and list of strings"


def test_asdataframe(panel):
    assert isinstance(
        panel.as_dataframe(), pd.DataFrame
    ), "Error getting panel as dataframe"


def test_shift(panel):
    assert panel.shift(1).shape == (3, 1, 2), "Shift is not correct"


def test_diff(panel):
    assert panel.diff().shape == (3, 1, 2), "Diff is not correct"


def test_pct_change(panel):
    assert panel.pct_change().shape == (3, 1, 2), "Pct change is not correct"

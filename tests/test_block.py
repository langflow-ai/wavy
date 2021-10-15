import numpy as np
import pandas as pd
import os
from timelab import block

TEST_DATA = os.path.join(os.path.dirname(__file__), "test_data/multi_asset.pkl")


class TestTimeBlock:
    df = pd.read_pickle(TEST_DATA)
    assets = ["LNC", "MAS", "CSX"]
    df = df[assets]

    def test_from_dataframe(self):
        time_block = block.from_dataframe(self.df)
        assert isinstance(time_block, block.TimeBlock)

    def test_assets(self):
        time_block = block.from_dataframe(self.df)
        assert len(time_block.assets) != len(self.df.columns.levels[0])

    def test_channels(self):
        time_block = block.from_dataframe(self.df)
        assert time_block.channels.isin(["Volume", "Low", "Close", "Open", "High"]).all()

    def test_values(self):
        time_block = block.from_dataframe(self.df)
        assert (time_block.values == self.df.values).all()

    def test_filter(self):
        time_block = block.from_dataframe(self.df)
        time_block = time_block.filter(assets=self.assets[:1], channels="High")
        assert (time_block.assets.isin(self.assets[:1])).all()
        assert (time_block.channels.isin(["High"])).all()

    def test_drop(self):
        time_block = block.from_dataframe(self.df)
        time_block = time_block.drop(assets=self.assets[:1], channels="High")
        assert not (time_block.assets.isin(self.assets[:1])).all()
        assert not (time_block.channels.isin(["High"])).all()

    def test_rename_assets(self):
        time_block = block.from_dataframe(self.df)
        new_values = ["a", "b", "c"]
        time_block = time_block.rename_assets(values=self.assets, new_values=new_values)
        assert (time_block.assets.isin(new_values)).all()

    def test_rename_channels(self):
        time_block = block.from_dataframe(self.df)
        values = ["High", "Open"]
        new_values = ["a", "b"]
        time_block = time_block.rename_channels(values=values, new_values=new_values)
        assert time_block.channels.isin(["Volume", "Low", "Close", "b", "a"]).all()

    def test_apply(self):
        time_block = block.from_dataframe(self.df)
        assert (time_block.apply("mean", axis=0).values == self.df.apply("mean", axis=0).values).all()
        assert (time_block.apply("mean", axis=1).values == self.df.apply("mean", axis=1).values).all()

    def test_add_channel(self):
        time_block = block.from_dataframe(self.df)
        values = np.arange(len(time_block.values))
        new_channel = "test_channel"
        time_block = time_block.add_channel(new_channel, values)
        assert new_channel in time_block.channels.values

    def test_pandas(self):
        time_block = block.from_dataframe(self.df)
        assert isinstance(time_block.pandas(), pd.DataFrame)

    def test_start_end(self):
        time_block = block.from_dataframe(self.df)
        assert time_block.start == self.df.index[0]
        assert time_block.end == self.df.index[-1]

    def test_countna(self):
        time_block = block.from_dataframe(self.df)
        s = pd.Series(dtype=int)
        for asset in self.assets:
            s[asset] = len(self.df[asset]) - len(self.df[asset].dropna())
        assert (time_block.countna() == s).all()
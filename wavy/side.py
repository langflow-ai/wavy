import functools
import operator

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .block import TimeBlock


class PanelSide:
    def __init__(self, blocks):
        # TODO: blocks must have increasing indexes, add warning and reindex
        self.blocks = blocks

    def wrap_block(self, func):
        @functools.wraps(func)
        def newfunc(*fargs, **fkeywords):
            return PanelSide([getattr(block, func.__name__)(*fargs, **fkeywords) for block in self.blocks])

        return newfunc

    # # TODO: Implement
    # def __repr__(self):
    #     return "<PanelSide>"

    # # TODO: Implement
    # def _repr_html_(self):
    #     return "<p>PanelSide</p>"

    def __getattr__(self, name):
        # Temporary fix
        if "repr" in name:
            return getattr(self.values, name)
        try:
            block_func = getattr(TimeBlock, name)
            if callable(block_func):
                return self.wrap_block(block_func)
            return PanelSide([getattr(block, name) for block in self.blocks])
        except AttributeError:
            raise AttributeError(f"'PanelSide' object has no attribute '{name}'")

    def __abs__(self):
        return PanelSide([block.__abs__() for block in self.blocks])

    def __add__(self, other):
        if isinstance(other, PanelSide):
            return PanelSide([block.__add__(other_block) for block, other_block in zip(self.blocks, other)])
        return PanelSide([block.__add__(other) for block in self.blocks])

    def __sub__(self, other):
        if isinstance(other, PanelSide):
            return PanelSide([block.__sub__(other_block) for block, other_block in zip(self.blocks, other)])
        return PanelSide([block.__sub__(other) for block in self.blocks])

    def __mul__(self, other):
        if isinstance(other, PanelSide):
            return PanelSide([block.__mul__(other_block) for block, other_block in zip(self.blocks, other)])
        return PanelSide([block.__mul__(other) for block in self.blocks])

    def __ge__(self, other):
        if isinstance(other, PanelSide):
            return PanelSide([block.__ge__(other_block) for block, other_block in zip(self.blocks, other)])
        return PanelSide([block.__ge__(other) for block in self.blocks])

    def __gt__(self, other):
        if isinstance(other, PanelSide):
            return PanelSide([block.__gt__(other_block) for block, other_block in zip(self.blocks, other)])
        return PanelSide([block.__gt__(other) for block in self.blocks])

    def __le__(self, other):
        if isinstance(other, PanelSide):
            return PanelSide([block.__le__(other_block) for block, other_block in zip(self.blocks, other)])
        return PanelSide([block.__le__(other) for block in self.blocks])

    def __lt__(self, other):
        if isinstance(other, PanelSide):
            return PanelSide([block.__lt__(other_block) for block, other_block in zip(self.blocks, other)])
        return PanelSide([block.__lt__(other) for block in self.blocks])

    def __pow__(self, other):
        if isinstance(other, PanelSide):
            return PanelSide([block.__pow__(other_block) for block, other_block in zip(self.blocks, other)])
        return PanelSide([block.__pow__(other) for block in self.blocks])

    # def __rpow__(self, other):
    #     if isinstance(other, PanelSide):
    #         return PanelSide([block.__rpow__(other_block) for block, other_block in zip(self.blocks, other)])
    #     return PanelSide([block.__rpow__(other) for block in self.blocks])

    # def __radd__(self, other):
    #     return self.__add__(other)

    # def __repr__(self):
    #     return str(self.blocks)

    # def __str__(self):
    #     return str(self.blocks)

    def __getitem__(self, key):
        return self.blocks.__getitem__(key)

    def __len__(self):
        return len(self.blocks)

    @property
    def values(self):
        return np.array(self.blocks)

    @property
    def first(self):
        return self.blocks[0]

    @property
    def last(self):
        return self.blocks[-1]

    @property
    def assets(self):
        return self.first.assets

    @property
    def channels(self):
        return self.first.channels

    @property
    def start(self):
        return self.first.start

    @property
    def end(self):
        return self.last.end

    @property
    def timesteps(self):
        return self.first.index

    @property
    def index(self):
        return self.data().index

    @property
    def shape(self):
        return self.numpy().shape

    def apply(self, func, axis=0):
        return PanelSide([block.apply(func, axis) for block in tqdm(self.blocks)])

    def filter(self, assets=None, channels=None):
        return PanelSide([block.filter(assets, channels) for block in tqdm(self.blocks)])

    def fillna(self, value=None, method=None):
        return PanelSide([block.fillna(value=value, method=method) for block in (self.blocks)])

    def findna(self):
        values = np.sum(self.numpy(), axis=(3, 2, 1))
        values = pd.Series(values).isna()
        return values[values == True].index.tolist()

    def drop(self, assets=None, channels=None):
        return PanelSide([block.drop(assets=assets, channels=channels) for block in (self.blocks)])

    def rename_assets(self, values, new_values):
        return PanelSide([block.rename_assets(values=values, new_values=new_values) for block in (self.blocks)])

    def rename_channels(self, values, new_values):
        return PanelSide([block.rename_channels(values=values, new_values=new_values) for block in (self.blocks)])

    def split_assets(self):
        return [self.filter(asset) for asset in self.assets]

    def replace(self, data):
        blocks = [block.update(values=data[i]) for i, block in enumerate(self.blocks)]
        return PanelSide(blocks)

    def add_channel(self, name, values):
        return [block.add_channel(name, values) for block in self.blocks]

    def data(self):
        df = pd.concat(self.blocks)
        return df[~df.index.duplicated(keep="first")]

    def numpy(self):
        new_shape = (len(self), len(self.timesteps), len(self.assets), len(self.channels))
        values = self.values.reshape(*new_shape)
        return values.transpose(0, 2, 1, 3)

    def flat(self):
        values = np.array([i.values.flatten() for i in self.blocks])
        index = [i.index[-1] for i in self.blocks]
        return pd.DataFrame(values, index=index)

    def flatten(self):
        return self.flat().values.flatten()

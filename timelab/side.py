import functools
import operator

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .block import TimeBlock

# Panel Side extende Panel block e ja tem funcoes todas como rename_channels, ...
class PanelSide:

    def __init__(self, blocks):
        self.blocks = blocks

    #     for func in dir(TimeBlock):
    #         try:
    #             block_func = getattr(TimeBlock, func)
    #             new_func = self.wrap_block(block_func)
    #             setattr(self, func, new_func)
    #             print(func)
    #         except:
    #             pass

    # def wrap_block(self, func):
    #     @functools.wraps(func)
    #     def newfunc(*fargs, **fkeywords):
    #         return PanelSide([getattr(block, func.__name__)(*fargs, **fkeywords) for block in self.blocks])
    #     return newfunc

    def __getitem__(self, key):
        return self.blocks.__getitem__(key)

    def __len__(self):
        return len(self.blocks)

    @property
    def values(self):
        return np.array([block.values for block in self.blocks])

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
    def index(self):
        indexes = np.array([block.index.values for block in self.blocks]).flatten()
        return list(set(indexes))

    @property
    def shape(self):
        return self.numpy().shape

    def apply(self, func, axis=0):
        return PanelSide([block.apply(func, axis) for block in tqdm(self.blocks)])

    def filter(self, assets=None, channels=None):
        return PanelSide([block.filter(assets, channels) for block in tqdm(self.blocks)])

    def fillna(self, value=None, method=None):
        return PanelSide([block.fillna(value=value,
                                        method=method) for block in (self.blocks)])

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
        blocks = [
            block.update(values=data[i])
            for i, block in enumerate(self.blocks)
        ]
        return PanelSide(blocks)

    def add_channel(self, name, values):
        return [block.add_channel(name, values) for block in self.blocks]

    def data(self):
        df = pd.concat(self.blocks)
        return df[~df.index.duplicated(keep='first')]

    def numpy(self):
        return np.array([block.numpy() for block in self.blocks])


# def operate(func):
#     def inner(self, other):
#         return PanelSide([func(a, b) for a, b in zip(self.blocks, other.blocks)])
#     return inner

# for name in ['__add__','__mul__','__sub__']:
#     setattr(PanelSide, name, operate(getattr(operator, name)))



# def blockfunc(func):
#     def inner(self):
#         return PanelSide([getatrr(block, func.__name__)() for block in self.blocks])
#     return inner

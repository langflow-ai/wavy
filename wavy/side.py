import functools
import operator

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .block import TimeBlock

from typing import List

# dunder_methods = ['__abs__', '__add__', '__aenter__', '__aexit__', '__aiter__', '__and__', '__anext__', '__await__', '__bool__', '__bytes__', '__call__', '__ceil__', '__class__', '__class_getitem__', '__cmp__', '__coerce__', '__complex__', '__contains__', '__del__', '__delattr__', '__delete__', '__delitem__', '__delslice__', '__dict__', '__dir__', '__div__', '__divmod__', '__enter__', '__eq__', '__exit__', '__float__', '__floor__', '__floordiv__', '__format__', '__fspath__', '__ge__', '__get__', '__getattr__', '__getattribute__', '__getitem__', '__getnewargs__', '__getslice__', '__gt__', '__hash__', '__hex__', '__iadd__', '__iand__', '__idiv__', '__ifloordiv__', '__ilshift__', '__imatmul__', '__imod__', '__import__', '__imul__', '__index__', '__init__', '__init_subclass__', '__instancecheck__', '__int__', '__invert__', '__ior__', '__ipow__', '__irshift__', '__isub__', '__iter__', '__itruediv__', '__ixor__', '__le__', '__len__', '__length_hint__', '__long__', '__lshift__', '__lt__', '__matmul__', '__metaclass__', '__missing__', '__mod__', '__mro__', '__mul__', '__ne__', '__neg__', '__new__', '__next__', '__nonzero__', '__oct__', '__or__', '__pos__', '__pow__', '__prepare__', '__radd__', '__rand__', '__rcmp__', '__rdiv__', '__rdivmod__', '__reduce__', '__reduce_ex__', '__repr__', '__reversed__', '__rfloordiv__', '__rlshift__', '__rmatmul__', '__rmod__', '__rmul__', '__ror__', '__round__', '__rpow__', '__rrshift__', '__rshift__', '__rsub__', '__rtruediv__', '__rxor__', '__set__', '__set_name__', '__setattr__', '__setitem__', '__setslice__', '__sizeof__', '__slots__', '__str__', '__sub__', '__subclasscheck__', '__subclasses__', '__truediv__', '__trunc__', '__unicode__', '__weakref__', '__xor__']
dunder_methods = ['__add__', '__sub__', '__mul__', '__ge__', '__gt__', '__le__', '__lt__', '__pow__']

class PanelSide:
    def __init__(self, blocks):
        # TODO: blocks must have increasing indexes, add warning and reindex
        self.blocks = blocks

    # TODO understand function
    def wrap_block(self, func):
        @functools.wraps(func)
        def newfunc(*fargs, **fkeywords):
            return PanelSide([getattr(block, func.__name__)(*fargs, **fkeywords) for block in self.blocks])

        return newfunc

    # Function to map all dunder functions
    def _one_arg(self, other, __f):
        if isinstance(other, PanelSide):
            return PanelSide([getattr(block, __f)(other_block) for block, other_block in zip(self.blocks, other)])
        return PanelSide([getattr(block, __f)(other) for block in self.blocks])

    for dunder in dunder_methods:
        locals()[dunder] = lambda self, other, __f=dunder: self._one_arg(other, __f)


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

    # TODO add dunder for these other functions
    def __abs__(self):
        return PanelSide([block.__abs__() for block in self.blocks])

    def __getitem__(self, key):
        return self.blocks.__getitem__(key)

    def __len__(self):
        return len(self.blocks)





    @property
    def first(self):
        """
        PanelSide first DataBlock.

        Example:

        >>> panelside.first
                        MSFT                 AAPL          
                        Open      Close      Open     Close
        Date                                                
        2005-12-21  19.577126  19.475122  2.218566  2.246069
        2005-12-22  19.460543  19.373114  2.258598  2.261960
        """
        return self.blocks[0]

    @property
    def last(self):
        """
        PanelSide last DataBlock.

        Example:

        >>> panelside.last
                        MSFT                 AAPL          
                        Open      Close      Open     Close
        Date                                                
        2005-12-22  19.460543  19.373114  2.258598  2.261960
        2005-12-23  19.322122  19.409552  2.266543  2.241485
        """
        return self.blocks[-1]

    @property
    def start(self):
        """
        PanelSide first index.

        Example:

        >>> panelside.start
        Timestamp('2005-12-21 00:00:00')
        """
        return self.first.start

    @property
    def end(self):
        """
        PanelSide last index.

        Example:

        >>> panelside.end
        Timestamp('2005-12-23 00:00:00')
        """
        return self.last.end

    @property
    def assets(self):
        """
        PanelSide assets.

        Example:

        >>> panelside.assets
        0    AAPL
        1    MSFT
        dtype: object
        """
        return self.first.assets

    @property
    def channels(self):
        """
        PanelSide channels.

        Example:

        >>> panelside.channels
        0    Open
        1    Close
        dtype: object
        """
        return self.first.channels

    @property
    def timesteps(self):
        #? Didn't make sense
        return self.first.index

    @property
    def index(self):
        return self.data().index

    @property
    def values(self):
        # ? In block the equivalent name is tensor
        """
        3D matrix with PanelSide value.

        Example:

        >>> panelside.values
        array([[[19.57712554, 19.47512245,  2.21856582,  2.24606872],
                [19.46054323, 19.37311363,  2.25859845,  2.26195979]],
               [[19.46054323, 19.37311363,  2.25859845,  2.26195979],
                [19.32212198, 19.40955162,  2.26654326,  2.24148512]]])
        """
        return np.array(self.blocks)

    @property
    def shape(self):
        return self.numpy().shape

    def filter(self, assets: List[str] = None, channels: List[str] = None):
        """
        Subset of the PanelSide columns according to the specified assets and channels.

        Args:
            assets (list): List of assets
            channels (list): List of channels

        Returns:
            ``PanelSide``: Filtered PanelSide
        """
        return PanelSide([block.filter(assets=assets, channels=channels) for block in tqdm(self.blocks)])

    def drop(self, assets=None, channels=None):
        """
        Subset of the PanelSide columns discarding the specified assets and channels.

        Args:
            assets (list): List of assets
            channels (list): List of channels

        Returns:
            ``PanelSide``: Filtered PanelSide
        """
        return PanelSide([block.drop(assets=assets, channels=channels) for block in tqdm(self.blocks)])

    def rename_assets(self, dict: dict):
        """
        Rename asset labels.

        Args:
            dict (dict): Dictionary with assets to rename

        Returns:
            ``PanelSide``: Renamed PanelSide
        """
        return PanelSide([block.rename_assets(dict) for block in tqdm(self.blocks)])

    def rename_channels(self, dict: dict):
        """
        Rename channel labels.

        Args:
            dict (dict): Dictionary with channels to rename

        Returns:
            ``PanelSide``: Renamed PanelSide
        """
        return PanelSide([block.rename_channels(dict) for block in tqdm(self.blocks)])

    def apply(self, func, on: str = 'timestamps'):
        """
        Apply a function along an axis of the DataBlock.

        Args:
            func (function): Function to apply to each column or row.
            on (str, default 'row'): Axis along which the function is applied:

                * 'timestamps': apply function to each timestamps.
                * 'channels': apply function to each channels.

        Returns:
            ``PanelSide``: Result of applying `func` along the given axis of the PanelSide.
        """
        return PanelSide([block.apply(func, on) for block in tqdm(self.blocks)])

    # TODO add update???

    def split_assets(self):
        # ? Does it make sense??
        return [self.filter(asset) for asset in self.assets]

    # TODO add sort_assets???
    # TODO add sort_channels???
    # TODO add swap_cols???
    # TODO add countna???

    def fillna(self, value=None, method=None):
        return PanelSide([block.fillna(value=value, method=method) for block in (self.blocks)])

    def findna(self):
        values = np.sum(self.numpy(), axis=(3, 2, 1))
        values = pd.Series(values).isna()
        return values[values == True].index.tolist()

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

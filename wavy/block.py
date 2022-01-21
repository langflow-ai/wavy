import functools
from collections import OrderedDict
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from typing import List, Union
from .utils import add_dim, add_level

# Plot
import numpy as np
import pandas as pd
import plotly as px
import plotly.graph_objects as go
import plotly.express as px
pd.set_option("multi_sparse", True)  # To see multilevel indexes
pd.options.plotting.backend = "plotly"
from plotly.subplots import make_subplots

FUNCTIONS = ['T', '_AXIS_LEN', '_AXIS_NAMES', '_AXIS_NUMBERS', '_AXIS_ORDERS', '_AXIS_REVERSED', '_AXIS_TO_AXIS_NUMBER', '_HANDLED_TYPES', '__abs__', '__add__', '__and__', '__annotations__', '__array__', '__array_priority__', '__array_ufunc__', '__array_wrap__', '__bool__', '__class__', '__contains__', '__copy__', '__deepcopy__', '__delattr__', '__delitem__', '__dict__', '__dir__', '__divmod__', '__doc__', '__eq__', '__finalize__', '__floordiv__', '__format__', '__ge__', '__getattr__', '__getattribute__', '__getitem__', '__getstate__', '__gt__', '__hash__', '__iadd__', '__iand__', '__ifloordiv__', '__imod__', '__imul__', '__init__', '__init_subclass__', '__invert__', '__ior__', '__ipow__', '__isub__', '__iter__', '__itruediv__', '__ixor__', '__le__', '__len__', '__lt__', '__matmul__', '__mod__', '__module__', '__mul__', '__ne__', '__neg__', '__new__', '__nonzero__', '__or__', '__pos__', '__pow__', '__radd__', '__rand__', '__rdivmod__', '__reduce__', '__reduce_ex__', '__repr__', '__rfloordiv__', '__rmatmul__', '__rmod__', '__rmul__', '__ror__', '__round__', '__rpow__', '__rsub__', '__rtruediv__', '__rxor__', '__setattr__', '__setitem__', '__setstate__', '__sizeof__', '__str__', '__sub__', '__subclasshook__', '__truediv__', '__weakref__', '__xor__', '_accessors', '_accum_func', '_add_numeric_operations', '_agg_by_level', '_agg_examples_doc', '_agg_summary_and_see_also_doc', '_align_frame', '_align_series', '_arith_method', '_as_manager', '_box_col_values', '_can_fast_transpose', '_check_inplace_and_allows_duplicate_labels', '_check_inplace_setting', '_check_is_chained_assignment_possible', '_check_label_or_level_ambiguity', '_check_setitem_copy', '_clear_item_cache', '_clip_with_one_bound', '_clip_with_scalar', '_cmp_method', '_combine_frame', '_consolidate', '_consolidate_inplace', '_construct_axes_dict', '_construct_axes_from_arguments', '_construct_result', '_constructor', '_constructor_sliced', '_convert', '_count_level', '_data', '_dir_additions', '_dir_deletions', '_dispatch_frame_op', '_drop_axis', '_drop_labels_or_levels', '_ensure_valid_index', '_find_valid_index', '_from_arrays', '_from_mgr', '_get_agg_axis', '_get_axis', '_get_axis_name', '_get_axis_number', '_get_axis_resolvers', '_get_block_manager_axis', '_get_bool_data', '_get_cleaned_column_resolvers', '_get_column_array', '_get_index_resolvers', '_get_item_cache', '_get_label_or_level_values', '_get_numeric_data', '_get_value', '_getitem_bool_array', '_getitem_multilevel', '_gotitem', '_hidden_attrs', '_indexed_same', '_info_axis', '_info_axis_name', '_info_axis_number', '_info_repr', '_init_mgr', '_inplace_method', '_internal_names', '_internal_names_set', '_is_copy', '_is_homogeneous_type', '_is_label_or_level_reference', '_is_label_reference', '_is_level_reference', '_is_mixed_type', '_is_view', '_iset_item', '_iset_item_mgr', '_iset_not_inplace', '_iter_column_arrays', '_ixs', '_join_compat', '_logical_func', '_logical_method', '_maybe_cache_changed', '_maybe_update_cacher', '_metadata', '_min_count_stat_function', '_needs_reindex_multi', '_protect_consolidate', '_reduce', '_reindex_axes', '_reindex_columns', '_reindex_index', '_reindex_multi', '_reindex_with_indexers', '_replace_columnwise', '_repr_data_resource_', '_repr_fits_horizontal_', '_repr_fits_vertical_', '_repr_html_', '_repr_latex_', '_reset_cache', '_reset_cacher', '_sanitize_column', '_series', '_set_axis', '_set_axis_name', '_set_axis_nocheck', '_set_is_copy', '_set_item', '_set_item_frame_value', '_set_item_mgr', '_set_value', '_setitem_array', '_setitem_frame', '_setitem_slice', '_slice', '_stat_axis', '_stat_axis_name', '_stat_axis_number', '_stat_function', '_stat_function_ddof', '_take_with_is_copy', '_to_dict_of_blocks', '_typ', '_update_inplace', '_validate_dtype', '_values', '_where', 'abs', 'add', 'add_prefix', 'add_suffix', 'agg', 'aggregate', 'align', 'all', 'any', 'append', 'apply', 'applymap', 'asfreq', 'asof', 'assign', 'astype', 'at', 'at_time', 'attrs', 'axes', 'backfill', 'between_time', 'bfill', 'bool', 'boxplot', 'clip', 'columns', 'combine', 'combine_first', 'compare', 'convert_dtypes', 'copy', 'corr', 'corrwith', 'count', 'cov', 'cummax', 'cummin', 'cumprod', 'cumsum', 'describe', 'diff', 'div', 'divide', 'dot', 'drop', 'drop_duplicates', 'droplevel', 'dropna', 'dtypes', 'duplicated', 'empty', 'eq', 'equals', 'eval', 'ewm', 'expanding', 'explode', 'ffill', 'fillna', 'filter', 'first', 'first_valid_index', 'flags', 'floordiv', 'from_dict', 'from_records', 'ge', 'get', 'groupby', 'gt', 'head', 'hist', 'iat', 'idxmax', 'idxmin', 'iloc', 'index', 'infer_objects', 'info', 'insert', 'interpolate', 'isin', 'isna', 'isnull', 'items', 'iteritems', 'iterrows', 'itertuples', 'join', 'keys', 'kurt', 'kurtosis', 'last', 'last_valid_index', 'le', 'loc', 'lookup', 'lt', 'mad', 'mask', 'max', 'mean', 'median', 'melt', 'memory_usage', 'merge', 'min', 'mod', 'mode', 'mul', 'multiply', 'ndim', 'ne', 'nlargest', 'notna', 'notnull', 'nsmallest', 'nunique', 'pad', 'pct_change', 'pipe', 'pivot', 'pivot_table', 'plot', 'pop', 'pow', 'prod', 'product', 'quantile', 'query', 'radd', 'rank', 'rdiv', 'reindex', 'reindex_like', 'rename', 'rename_axis', 'reorder_levels', 'replace', 'resample', 'reset_index', 'rfloordiv', 'rmod', 'rmul', 'rolling', 'round', 'rpow', 'rsub', 'rtruediv', 'sample', 'select_dtypes', 'sem', 'set_axis', 'set_flags', 'set_index', 'shape', 'shift', 'size', 'skew', 'slice_shift', 'sort_index', 'sort_values', 'sparse', 'squeeze', 'stack', 'std', 'style', 'sub', 'subtract', 'sum', 'swapaxes', 'swaplevel', 'tail', 'take', 'to_clipboard', 'to_csv', 'to_dict', 'to_excel', 'to_feather', 'to_gbq', 'to_hdf', 'to_html', 'to_json', 'to_latex', 'to_markdown', 'to_numpy', 'to_parquet', 'to_period', 'to_pickle', 'to_records', 'to_sql', 'to_stata', 'to_string', 'to_timestamp', 'to_xarray', 'to_xml', 'transform', 'transpose', 'truediv', 'truncate', 'tshift', 'tz_convert', 'tz_localize', 'unstack', 'update', 'value_counts', 'values', 'var', 'where', 'xs']

def from_dataframe(df: DataFrame, asset: str = 'asset'):
    """
    Generate Block from DataFrame

    Args:
        df (DataFrame): Pandas DataFrame
        asset (string): Asset name

    Returns:
        ``Block``: Constructed Block

    Example:

    >>> data
                    Open     Close
    Date
    2005-12-21  2.218566  2.246069
    2005-12-22  2.258598  2.261960

    >>> block = from_dataframe(data, 'AAPL')
                    AAPL
                    Open     Close
    Date
    2005-12-21  2.218566  2.246069
    2005-12-22  2.258598  2.261960

    >>> type(block)
    wavy.block.Block

    """

    # Add level if level equals to 1
    if df.T.index.nlevels == 1:
        df = add_level(df, asset)

    # Create Block
    tb = Block(
            pd.DataFrame(
                df.values,
                index=df.index,
                columns=df.columns,
                )
            )

    return tb

def from_series(data, axis=0, index=0, name='none'):
    if axis == 1:
        block = from_dataframe(data.to_frame(name=name))
    else:
        block = from_dataframe(data.to_frame().T)
        block.index = [index]
        
    return block

def from_dict(data: dict):
    """
    Generate Block from dictionary

    Args:
        data ({str: DataFrame}): Dictionary containing asset name and DataFrame

    Returns:
        ``Block``: Constructed Block

    Example:

    >>> dict
    {'AAPL':                 Open     Close
    Date
    2005-12-21  2.218566  2.246069
    2005-12-22  2.258598  2.261960,
    'MSFT':                  Open      Close
    Date
    2005-12-21  19.577126  19.475122
    2005-12-22  19.460543  19.373114}

    >>> from_dict(dict)
                    AAPL                 MSFT
                    Open     Close       Open      Close
    Date
    2005-12-21  2.218566  2.246069  19.577126  19.475122
    2005-12-22  2.258598  2.261960  19.460543  19.373114
    """

    previous_channels = None
    for _, value in data.items():
        assert isinstance(value, DataFrame), 'Data must be a DataFrame'

        # ? Remove upper level if data is multilevel
        assert value.T.index.nlevels == 1, 'Data cannot be multilevel'

        channels_flag = value.shape[1] == previous_channels if previous_channels else True
        assert channels_flag, 'Data with different number of channels'
        previous_channels = value.shape[1]

    return Block(pd.concat(data.values(), axis=1, keys=data.keys()))


def from_dataframes(data: List[DataFrame], assets: List[str] = None):
    """
    Generate a Block from a list of dataframes. Each dataframe becomes one asset.

    Args:
        data (list): List of dataframes
        assets (list): List of assets

    Returns:
        ``Block``: Constructed Block

    Example:

    >>> aapl
                    Open     Close
    Date
    2005-12-21  2.218566  2.246069
    2005-12-22  2.258598  2.261960

    >>> msft
                     Open      Close
    Date
    2005-12-21  19.577126  19.475122
    2005-12-22  19.460543  19.373114

    Generating Block with list of dataframes

    >>> from_dataframes([aapl, msft])
             asset_0              asset_1
                    Open     Close       Open      Close
    Date
    2005-12-21  2.218566  2.246069  19.577126  19.475122
    2005-12-22  2.258598  2.261960  19.460543  19.373114


    Generating Block with list of dataframes and assets

    >>> from_dataframes([aapl, msft], ['AAPL', 'MSFT'])
                AAPL                 MSFT
                    Open     Close       Open      Close
    Date
    2005-12-21  2.218566  2.246069  19.577126  19.475122
    2005-12-22  2.258598  2.261960  19.460543  19.373114
    """
    if assets:
        dict = {assets[k]: v for k, v in enumerate(data)}
    else:
        dict = {"asset_" + str(k): v for k, v in enumerate(data)}

    return from_dict(dict)

def from_tensor(values, index: List = None, assets: List[str] = None, channels: List[str] = None):
    """
    Generate a Block from list of attributes.

    Args:
        values (ndarray): Dataframes of size (assets x index x channels)
        index (list): List of index
        assets (list): List of assets
        channels (list): List of channels

    Returns:
        ``Block``: Constructed Block

    Example:

    >>> values
    array([[[ 2.21856582,  2.24606872],
            [ 2.25859845,  2.26195979]],
           [[19.57712554, 19.47512245],
            [19.46054323, 19.37311363]]])

    >>> index
    DatetimeIndex(['2005-12-21', '2005-12-22'], dtype='datetime64[ns]', name='Date', freq=None)

    >>> assets
    Index(['AAPL', 'MSFT'], dtype='object')

    >>> from_tensor(values, index=index, assets=assets, channels=channels)
                    AAPL                 MSFT
                    Open     Close       Open      Close
    Date
    2005-12-21  2.218566  2.246069  19.577126  19.475122
    2005-12-22  2.258598  2.261960  19.460543  19.373114
    """

    values = np.concatenate(values, axis=1)

    return from_matrix(values, index=index, assets=assets, channels=channels)


def from_matrix(values, index: List = None, assets: List[str] = None, channels: List[str] = None):
    """
    Generate a Block from list of attributes.

    Args:
        values (ndarray): Dataframes of size (index x [assets * channels])
        index (list): List of index
        assets (list): List of assets
        channels (list): List of channels

    Returns:
        ``Block``: Constructed Block

    Example:

    >>> values
    array([[ 2.21856582,  2.24606872, 19.57712554, 19.47512245],
           [ 2.25859845,  2.26195979, 19.46054323, 19.37311363]])

    >>> index
    DatetimeIndex(['2005-12-21', '2005-12-22'], dtype='datetime64[ns]', name='Date', freq=None)

    >>> assets
    Index(['AAPL', 'MSFT'], dtype='object')

    >>> from_matrix(values, index=index, assets=assets, channels=channels)
                    AAPL                 MSFT
                    Open     Close       Open      Close
    Date
    2005-12-21  2.218566  2.246069  19.577126  19.475122
    2005-12-22  2.258598  2.261960  19.460543  19.373114
    """

    values = add_dim(values, n = 3 - len(values.shape))
    if assets is None:
        assets = range(values.shape[0])
    if index is None:
        index = range(values.shape[1])
    if channels is None:
        channels = range(values.shape[2])

    columns = pd.MultiIndex.from_product([assets, channels])
    df = pd.DataFrame(index=index, columns=columns)
    df.loc[:, (slice(None), slice(None))] = values.reshape(df.shape)
    return Block(df)



def _rebuild(func):
    # Avoid problem with frozen list from pandas
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        df = func(*args, **kwargs)
        return from_matrix(df.values, df.index, df.assets, df.channels)

    return wrapper

# def improve_function(func):
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         df = func(*args, **kwargs)

#         if isinstance(df, _AssetSeries):
#             return df.to_frame().T
#         else:
#             return df

#     return wrapper

class _AssetSeries(pd.Series):
    def __init__(self, df, *args, **kwargs):
        super().__init__(df, *args, **kwargs)

    @property
    def _constructor_expanddim(self):
        return Block

    @property
    def _constructor(self):
        return _AssetSeries


class Block(pd.DataFrame):

    def __init__(self, df, *args, **kwargs):
        super().__init__(df, *args, **kwargs)


    # def override_series(self, func, *args, **kwargs):
    #     df = self.as_dataframe().__getattr__(func)(*args, **kwargs)

    #     if isinstance(df, pd.Series):
    #         return from_series(df, self.index[0])
    #     else:
    #         return df

    # # dir(pd.DataFrame)
    # # ['min', 'max', 'mean', 'mad']
    # for func in dir(pd.DataFrame):
    #     # if not func.startswith('_') and callable(getattr(pd.DataFrame, func)):
    #     locals()[func] = lambda self, func=func, *args, **kwargs: self.override_series(func, *args, **kwargs)



    def _check_method(self, method):
        if method not in self.__class__.__dict__:  # Not defined in method : inherited
            # return 'inherited'
            raise NotImplementedError
        # elif hasattr(super(), method):  # Present in parent : overloaded
        #     return 'overloaded'
        # else:  # Not present in parent : newly defined
        #     return 'newly defined'


    # def mean(self):
    #     return super().mean().to_frame().T

    # def __getattr__(self, name):
    #     try:
    #         def wrapper(*args, **kwargs):

    #             df = getattr(self, name)(*args, **kwargs)

    #             if isinstance(df, _AssetSeries):
    #                 return df.to_frame().T
    #             else:
    #                 return df
    #         return wrapper
    #     except AttributeError:
    #         raise AttributeError(f"'Block' object has no attribute '{name}'")

    # def __repr__(self):
    #     if not self.is_valid:
    #         return "<empty block>"
    #     else:
    #         return super().__repr__()

    # # TODO check why panel calls __repr__ and block calls __str__
    # def __str__(self):
    #     if not self.is_valid:
    #         return "<empty block>"
    #     else:
    #         return super().__repr__()

    # @property
    # def is_valid(self):
    #     return False if any(self.index.isna()) else True

    # TODO convert return of dataframe functions to block

    # def _rebuild(func):
    #     # Avoid problem with frozen list from pandas
    #     @functools.wraps(func)
    #     def wrapper(*args, **kwargs):
    #         df = func(*args, **kwargs)
    #         return from_matrix(df.values, df.index, df.assets, df.channels)

    #     return wrapper

    @property
    def _constructor(self):
        return Block

    @property
    def _constructor_sliced(self):
        return _AssetSeries

    @property
    def start(self):
        """
        Block first index.

        Example:

        >>> block
                        AAPL                 MSFT
                        Open     Close       Open      Close
        Date
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> block.start
        Timestamp('2005-12-21 00:00:00')
        """
        return self.index[0]

    @property
    def end(self):
        """
        Block last index.

        Example:

        >>> block
                        AAPL                 MSFT
                        Open     Close       Open      Close
        Date
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> block.end
        Timestamp('2005-12-22 00:00:00')
        """
        return self.index[-1]

    @property
    def assets(self):
        """
        Block assets.

        Example:

        >>> block
                        AAPL                 MSFT
                        Open     Close       Open      Close
        Date
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> block.assets
        0    AAPL
        1    MSFT
        dtype: object
        """
        assets = [col[0] for col in self.columns]
        # OrderedDict to keep order
        # ? Is it correct to  order, what happens in case the user wants to _rebuild the block?
        return pd.Series(tuple(OrderedDict.fromkeys(assets)))

    @property
    def channels(self):
        """
        Block channels.

        Example:

        >>> block
                        AAPL                 MSFT
                        Open     Close       Open      Close
        Date
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> block.channels
        0    Open
        1    Close
        dtype: object
        """
        channels = [col[1] for col in self.columns]
        # OrderedDict to keep order
        # ? Is it correct to  order, what happens in case the user wants to _rebuild the block?
        return pd.Series(list(OrderedDict.fromkeys(channels)))

    @property
    def timesteps(self):
        """
        Block timesteps.

        Example:

        >>> block.timesteps
        DatetimeIndex(['2005-12-21', '2005-12-22', '2005-12-23'], dtype='datetime64[ns]', name='Date', freq=None)
        """
        # The same as the index
        return self.index

    # Causing error, overwriting shape function
    # @property
    # def shape(self):
    #     """
    #     Block shape.

    #     Example:

    #     >>> block.shape
    #     (2, 2, 2)
    #     """
    #     return self.tensor.shape

    @property
    def tensor(self):
        """
        3D matrix with DataBlock value.

        Example:

        >>> block
                        AAPL                 MSFT
                        Open     Close       Open      Close
        Date
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> block.tensor
        array([[[ 2.21856582,  2.24606872],
                [ 2.25859845,  2.26195979]],
               [[19.57712554, 19.47512245],
                [19.46054323, 19.37311363]]])
        """
        new_shape = (len(self), len(self.assets), len(self.channels))
        values = self.values.reshape(*new_shape)
        return values.transpose(1, 0, 2)

    @property
    def matrix(self):
        """
        2D matrix with DataBlock value.

        Example:

        >>> block
                        AAPL                 MSFT
                        Open     Close       Open      Close
        Date
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> block.matrix
        array([[ 2.21856582,  2.24606872, 19.57712554, 19.47512245],
            [ 2.25859845,  2.26195979, 19.46054323, 19.37311363]])
        """
        return self.values

    def filter(self, assets: List[str] = None, channels: List[str] = None):
        """
        Block subset according to the specified assets and channels.

        Similar to `Pandas Rename <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.filter.html>`__

        Args:
            assets (list): List of assets
            channels (list): List of channels

        Returns:
            ``Block``: Filtered Block

        Example:

        >>> block
                        AAPL                 MSFT
                        Open     Close       Open      Close
        Date
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> block.filter(assets=['AAPL'], channels=['Open'])
                        AAPL
                        Open
        Date
        2005-12-21  2.218566
        2005-12-22  2.258598
        """
        filtered = self._filter_assets(assets)
        filtered = filtered._filter_channels(channels)
        return filtered

    @_rebuild
    def _filter_assets(self, assets):
        if type(assets) == str:
            assets = [assets]

        # TODO improve speed

        if assets is not None and any(asset not in list(self.assets) for asset in assets):
            raise ValueError(f"{assets} not found in columns. Columns: {list(self.assets)}")

        return self.loc[:, (assets, slice(None))] if assets else self

    @_rebuild
    def _filter_channels(self, channels):
        if type(channels) == str:
            channels = [channels]

        # TODO improve speed

        if channels is not None and any(channel not in list(self.channels) for channel in channels):
            raise ValueError(f"{channels} not found in columns. Columns: {list(self.channels)}")

        return self.loc[:, (slice(None), channels)][self.assets] if channels else self


    def drop(self, assets: List[str] = None, channels: List[str] = None):
        """
        Subset of the dataframe columns discarding the specified assets and channels.

        Similar to `Pandas Rename <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html>`__

        Args:
            assets (list): List of assets
            channels (list): List of channels

        Returns:
            ``Block``: Filtered Block

        Example:

        >>> block
                        AAPL                 MSFT
                        Open     Close       Open      Close
        Date
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> block.drop(assets=['AAPL'], channels=['Open'])
                         MSFT
                        Close
        Date
        2005-12-21  19.475122
        2005-12-22  19.373114
        """
        filtered = self._drop_assets(assets)
        filtered = filtered._drop_channels(channels)
        return filtered

    @_rebuild
    def _drop_assets(self, assets):
        if isinstance(assets, str):
            assets = [assets]
        new_assets = [u for u in self.assets if u not in assets]
        return self._filter_assets(new_assets)

    @_rebuild
    def _drop_channels(self, channels):
        if isinstance(channels, str):
            channels = [channels]
        new_channels = [c for c in self.channels if c not in channels]
        return self._filter_channels(new_channels)

    def rename_assets(self, dict: dict):
        """
        Rename asset labels.

        Similar to `Pandas Rename <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rename.html#>`__

        Args:
            dict (dict): Dictionary with assets to rename

        Returns:
            ``Block``: Renamed Block

        Example:

        >>> block
                        AAPL                 MSFT
                        Open     Close       Open      Close
        Date
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> block.rename_assets({'AAPL': 'Apple', 'MSFT': 'Microsoft'})
                       Apple            Microsoft
                        Open     Close       Open      Close
        Date
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114
        """

        values = dict.keys()
        new_values = dict.values()

        assets = self.assets.replace(to_replace=values, value=new_values)
        return self.update(assets=assets.values)

    def rename_channels(self, dict: dict):
        """
        Rename channel labels.

        Similar to `Pandas Rename <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rename.html#>`__

        Args:
            dict (dict): Dictionary with channels to rename

        Returns:
            ``Block``: Renamed Block

        Example:

        >>> block
                        AAPL                 MSFT
                        Open     Close       Open      Close
        Date
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> block.rename_channels({'Open': 'Op', 'Close': 'Cl'})
                        AAPL                 MSFT
                        Op        Cl         Op         Cl
        Date
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114
        """
        values = dict.keys()
        new_values = dict.values()

        channels = self.channels.replace(to_replace=values, value=new_values)
        return self.update(channels=channels.values)

    def wapply(self, func, on: str = 'timestamps'):
        """
        Apply a function along an axis of the DataBlock.

        Similar to `Pandas Apply <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html>`__

        Args:
            func (function): Function to apply to each column or row.
            on (str, default 'row'): Axis along which the function is applied:

                * 'timestamps': apply function to each timestamps.
                * 'channels': apply function to each channels.

        Returns:
            ``Block``: Result of applying `func` along the given axis of the Block.

        Example:

        >>> block
                        AAPL                 MSFT
                        Open     Close       Open      Close
        Date
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> block.apply(np.max, on='rows')
            AAPL                MSFT
            Open    Close       Open      Close
        0  2.258598  2.26196  19.577126  19.475122

        >>> block.apply(np.max, on='columns')
                        AAPL       MSFT
                        amax       amax
        Date
        2005-12-21  2.246069  19.577126
        2005-12-22  2.261960  19.460543
        """

        if on == 'timestamps ':
            return self._timestamp_apply(func)
        elif on == 'channels':
            return self._channel_apply(func)

        raise ValueError(f"{on} not acceptable for 'axis'. Available values are [0, 1]")

    def _timestamp_apply(self, func):
        df = self.as_dataframe().apply(func, axis=0)
        if isinstance(df, pd.Series):
            return df.to_frame().T
        return df.T

    def _channel_apply(self, func):
        splits = self.split_assets()
        return from_matrix(
            np.swapaxes(
                np.array(
                    [
                        asset.as_dataframe().apply(func, axis=1).values
                        for asset in splits
                    ]
                ),
                0,
                1,
            ),
            index=self.index,
            assets=self.assets,
            channels=[func.__name__],
        )

    def update(self, values=None, index: List = None, assets: List = None, channels: List = None):
        """
        Update function for any of DataBlock properties.

        Similar to `Pandas Update <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.update.html>`__

        Args:
            values (ndarray): New values Dataframe.
            index (list): New list of index.
            assets (list): New list of assets
            channels (list): New list of channels

        Returns:
            ``Block``: Result of updated Block.

        Example:

        >>> block
                        AAPL                 MSFT
                        Open     Close       Open      Close
        Date
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> block.update(assets=['Microsoft', 'Apple'], channels=['Op', 'Cl'])
                 Microsoft                 Apple
                        Op         Cl         Op         Cl
        Date
        2005-12-21  19.577126  19.475122  19.460543  19.373114
        2005-12-22   2.218566   2.246069   2.258598   2.261960
        """
        assets = assets if assets is not None else self.assets
        index = index if index is not None else self.index
        channels = channels if channels is not None else self.channels
        values = values if values is not None else self.matrix

        if values is not None:
            if len(values.shape) == 3:
                db = from_tensor(values, index, assets, channels)
            elif len(values.shape) == 2:
                db = from_matrix(values, index, assets, channels)
        return db

    def split_assets(self):
        """
        Split DataBlock into assets.

        Returns:
            ``List``: List of DataBlock, each one being one asset.

        Example:

        >>> block
                        AAPL                 MSFT
                        Open     Close       Open      Close
        Date
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> block.split_assets()
        [               AAPL
                        Open     Close
        Date
        2005-12-21  2.218566  2.246069
        2005-12-22  2.258598  2.261960,
                        MSFT
                        Open      Close
        Date
        2005-12-21  19.577126  19.475122
        2005-12-22  19.460543  19.373114]
        """
        return [self.filter(asset) for asset in self.assets]

    def sort_assets(self, order: List[str] = None):
        """
        Sort assets in alphabetical order.

        Args:
            order (List[str]): Asset order to be sorted.

        Returns:
            ``DataBlock``: Result of sorting assets.

        Example:

        >>> block
                        MSFT                 AAPL
                        Open      Close      Open     Close
        Date
        2005-12-21  19.577126  19.475122  2.218566  2.246069
        2005-12-22  19.460543  19.373114  2.258598  2.261960

        >>> block.sort_assets()
                        AAPL                 MSFT
                        Open     Close       Open      Close
        Date
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114
        """
        assets = sorted(self.assets) if order is None else order
        channels = self.channels
        pair = [(asset, channel) for asset in assets for channel in channels]
        return self.reindex(pair, axis=1)

    def sort_channels(self, order: List[str] = None):
        """
        Sort channels in alphabetical order.

        Args:
            order (List[str]): Channel order to be sorted.

        Returns:
            ``DataBlock``: Result of sorting channels.

        Example:

        >>> block
                        MSFT                 AAPL
                        Open      Close      Open     Close
        Date
        2005-12-21  19.577126  19.475122  2.218566  2.246069
        2005-12-22  19.460543  19.373114  2.258598  2.261960

        >>> block.sort_channels()
                         MSFT                 AAPL
                        Close       Open     Close      Open
        Date
        2005-12-21  19.475122  19.577126  2.246069  2.218566
        2005-12-22  19.373114  19.460543  2.261960  2.258598
        """
        assets = self.assets
        channels = sorted(self.channels) if order is None else order
        pair = [(asset, channel) for asset in assets for channel in channels]
        return self.reindex(pair, axis=1)

    def swap_cols(self):
        """
        Swap columns levels, assets becomes channels and channels becomes assets

        Returns:
            ``DataBlock``: Result of swapping columns.

        Example:

        >>> block
                        AAPL                 MSFT
                        Open     Close       Open      Close
        Date
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> block.swap_cols()
                       Close                 Open
                        AAPL       MSFT      AAPL       MSFT
        Date
        2005-12-21  2.246069  19.475122  2.218566  19.577126
        2005-12-22  2.261960  19.373114  2.258598  19.460543
        """
        channels = self.channels
        return self.T.swaplevel(i=- 2, j=- 1, axis=0).T.sort_assets(channels)


    def smash(self, sep: str = '_'):
        # TODO: Document
        """
        Removes hierarchical columns using a separatora and returns a single level DataFrame.
        """
        columns = [sep.join(tup).rstrip(sep) for tup in self.columns.values]
        index = self.index
        values = self.values
        return pd.DataFrame(values, index=index, columns=columns)


    def countna(self, type: str = 'asset'):
        """
        Count NaN cells for each asset or channel.

        Returns:
            ``List``: List of NaN count.

        Example:

        >>> block
                        AAPL                 MSFT
                        Open     Close       Open      Close
        Date
        2005-12-21  2.218566  2.246069        NaN  19.475122
        2005-12-22  2.258598       NaN  19.460543  19.373114

        >>> block.countna('asset')
        AAPL    1
        MSFT    1
        dtype: int64

        >>> block.countna('channel')
        AAPL  Open     0
              Close    1
        MSFT  Open     1
              Close    0
        dtype: int64
        """
        if type == 'asset':
            s = pd.Series(dtype=int)
            for asset in self.assets:
                s[asset] = len(self[asset]) - len(self[asset].dropna())
        elif type == 'channel':
            s = self.isnull().sum(axis = 0)
        return s

    def as_dataframe(self):
        """
        Generate DataFrame from Block

        Returns:
            ``DataFrame``: Constructed DataFrame

        Example:

        >>> type(block)
        wavy.block.Block

        >>> type(block.as_dataframe())
        pandas.core.frame.DataFrame
        """
        return pd.DataFrame(self.values, index=self.index, columns=self.columns)

    def wfillna(self, value=None, method=None, axis=None, inplace=False, limit=None, downcast=None):
        """
        Fill NaN values using the specified method.

        Similar to `Pandas Fillna <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html>`__

        Returns:
            ``DataBlock``: DataBlock with missing values filled.

        Example:

        >>> block
                        AAPL                 MSFT
                        Open     Close       Open      Close
        Date
        2005-12-21  2.218566  2.246069        NaN  19.475122
        2005-12-22  2.258598       NaN  19.460543  19.373114

        >>> block.fillna(0)
                        MSFT                 AAPL
                        Open      Close      Open     Close
        Date
        2005-12-21  19.577126  19.475122  0.000000  2.246069
        2005-12-22  19.460543   0.000000  2.258598  2.261960
        """
        return super().fillna(value, method, axis, inplace, limit, downcast)


    def wcount(self, axis: int = 0, numeric_only: bool = False):
        """
        Count non-NA cells for each column or row.

        The values None, NaN, NaT, and optionally numpy.inf (depending on pandas.options.mode.use_inf_as_na) are considered NA.

        Similar to `Pandas count <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.count.html>`__

        Args:
            axis (int): Axis for the function to be applied on.
            numeric_only (bool): Include only float, int or boolean data.

        Returns:
            ``DataBlock``: DataBlock with operation executed.

        Example:

        >>> block
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> block.wcount()
                   AAPL       MSFT      
                   Open Close Open Close
        2005-12-21    2     2    2     2
        """
        return from_series(super().count(axis=axis, numeric_only=numeric_only), axis, self.index[0], 'count')

    def wkurt(self, axis: int = None, skipna: bool = None, numeric_only=None, **kwargs):
        """
        Return unbiased kurtosis over requested axis.

        Kurtosis obtained using Fisher's definition of kurtosis (kurtosis of normal == 0.0). Normalized by N-1.
        
        Similar to `Pandas kurt <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.kurt.html>`__

        Args:
            axis (int): Axis for the function to be applied on.
            skipna (bool): Exclude NA/null values when computing the result.
            numeric_only (bool): Include only float, int or boolean data. If None, will attempt to use everything, then use only numeric data. Not implemented for Series.
            **kwargs: Additional keyword arguments to be passed to the function.

        Returns:
            ``DataBlock``: DataBlock with operation executed.

        Example:

        >>> block
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> block.wkurt(axis=1)
                    asset
                    kurt
        Date               
        2005-12-21 -5.99944
        2005-12-22 -5.99961
        """
        return from_series(super().kurt(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs), axis, self.index[0], 'kurt')

    def wkurtosis(self, axis: int = None, skipna: bool = None, numeric_only=None, **kwargs):
        """
        Return unbiased kurtosis over requested axis.

        Kurtosis obtained using Fisher's definition of kurtosis (kurtosis of normal == 0.0). Normalized by N-1.
        
        Similar to `Pandas kurtosis <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.kurtosis.html>`__

        Args:
            axis (int): Axis for the function to be applied on.
            skipna (bool): Exclude NA/null values when computing the result.
            numeric_only (bool): Include only float, int or boolean data. If None, will attempt to use everything, then use only numeric data. Not implemented for Series.
            **kwargs: Additional keyword arguments to be passed to the function.

        Returns:
            ``DataBlock``: DataBlock with operation executed.

        Example:

        >>> block
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> block.wkurtosis(axis=1)
                    asset
                    kurt
        Date               
        2005-12-21 -5.99944
        2005-12-22 -5.99961
        """
        return from_series(super().kurtosis(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs), axis, self.index[0], 'kurtosis')

    def wmad(self, axis: int = None, skipna: bool = None):
        """
        Return the mean absolute deviation of the values over the requested axis.

        Similar to `Pandas mad <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.mad.html>`__

        Args:
            axis (int): Axis for the function to be applied on.
            numeric_only (bool): Include only float, int or boolean data. If None, will attempt to use everything, then use only numeric data. Not implemented for Series.

        Returns:
            ``DataBlock``: DataBlock with operation executed.

        Example:

        >>> block
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> block.wmad()
                        AAPL                MSFT          
                        Open     Close      Open     Close
        2005-12-21  0.020016  0.007946  0.058291  0.051004
        """
        return from_series(super().mad(axis=axis, skipna=skipna), axis, self.index[0], 'mad')

    def wmax(self, axis: int = None, skipna: bool = None, numeric_only=None, **kwargs):
        """
        Return the maximum of the values over the requested axis.

        Similar to `Pandas max <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.max.html>`__

        Args:
            axis (int): Axis for the function to be applied on.
            skipna (bool): Exclude NA/null values when computing the result.
            numeric_only (bool): Include only float, int or boolean data. If None, will attempt to use everything, then use only numeric data. Not implemented for Series.
            **kwargs: Additional keyword arguments to be passed to the function.

        Returns:
            ``DataBlock``: DataBlock with operation executed.

        Example:

        >>> block
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> block.wmax()
                        AAPL                MSFT           
                        Open    Close       Open      Close
        2005-12-21  2.258598  2.26196  19.577126  19.475122
        """
        return from_series(super().max(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs), axis, self.index[0], 'max')

    def wmean(self, axis: int = None, skipna: bool = None, numeric_only=None, **kwargs):
        """
        Return the mean of the values over the requested axis.

        Similar to `Pandas mean <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.mean.html>`__

        Args:
            axis (int): Axis for the function to be applied on.
            skipna (bool): Exclude NA/null values when computing the result.
            numeric_only (bool): Include only float, int or boolean data. If None, will attempt to use everything, then use only numeric data. Not implemented for Series.
            **kwargs: Additional keyword arguments to be passed to the function.

        Returns:
            ``DataBlock``: DataBlock with operation executed.

        Example:

        >>> block
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> block.wmean()
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        2005-12-21  2.238582  2.254014  19.518834  19.424118
        """
        return from_series(super().mean(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs), axis, self.index[0], 'mean')

    def wmedian(self, axis: int = None, skipna: bool = None, numeric_only=None, **kwargs):
        """
        Return the median of the values over the requested axis.

        Similar to `Pandas median <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.median.html>`__

        Args:
            axis (int): Axis for the function to be applied on.
            skipna (bool): Exclude NA/null values when computing the result.
            numeric_only (bool): Include only float, int or boolean data. If None, will attempt to use everything, then use only numeric data. Not implemented for Series.
            **kwargs: Additional keyword arguments to be passed to the function.

        Returns:
            ``DataBlock``: DataBlock with operation executed.

        Example:

        >>> block
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> block.wmedian()
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        2005-12-21  2.238582  2.254014  19.518834  19.424118
        """
        return from_series(super().median(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs), axis, self.index[0], 'median')


    def wmin(self, axis: int = None, skipna: bool = None, numeric_only=None, **kwargs):
        """
        Return the minimum of the values over the requested axis.

        Similar to `Pandas min <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.min.html>`__

        Args:
            axis (int): Axis for the function to be applied on.
            skipna (bool): Exclude NA/null values when computing the result.
            numeric_only (bool): Include only float, int or boolean data. If None, will attempt to use everything, then use only numeric data. Not implemented for Series.
            **kwargs: Additional keyword arguments to be passed to the function.

        Returns:
            ``DataBlock``: DataBlock with operation executed.

        Example:

        >>> block
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> block.wmin()
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        2005-12-21  2.218566  2.246069  19.460543  19.373114
        """
        return from_series(super().min(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs), axis, self.index[0], 'min')

    def wnunique(self, axis: int = None, dropna: bool = None):
        """
        Count number of distinct elements in specified axis.

        Return Series with number of distinct elements. Can ignore NaN values.

        Similar to `Pandas nunique <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.nunique.html>`__

        Args:
            axis (int): Axis for the function to be applied on.
            dropna (bool): Don't include NaN in the counts.

        Returns:
            ``DataBlock``: DataBlock with operation executed.

        Example:

        >>> block
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> block.wnunique()
                AAPL       MSFT      
                Open Close Open Close
        2005-12-21    2     2    2     2
        """
        return from_series(super().nunique(axis=axis, dropna=dropna), axis, self.index[0], 'nunique')

    def wprod(self, axis: int = None, skipna: bool = None, numeric_only=None, min_count: int = 0, **kwargs):
        """
        Return the product of the values over the requested axis.

        Similar to `Pandas prod <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.prod.html>`__

        Args:
            axis (int): Axis for the function to be applied on.
            skipna (bool): Exclude NA/null values when computing the result.
            numeric_only (bool): Include only float, int or boolean data. If None, will attempt to use everything, then use only numeric data. Not implemented for Series.
            min_count (int): The required number of valid values to perform the operation. If fewer than `min_count` non-NA values are present the result will be NA.
            **kwargs: Additional keyword arguments to be passed to the function.

        Returns:
            ``DataBlock``: DataBlock with operation executed.

        Example:

        >>> block
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> block.wprod()
                        AAPL                  MSFT           
                        Open     Close        Open      Close
        2005-12-21  5.010849  5.080517  380.981498  377.29376
        """
        return from_series(super().prod(axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count, **kwargs), axis, self.index[0], 'prod')


    def wproduct(self, axis: int = None, skipna: bool = None, numeric_only=None, min_count: int = 0, **kwargs):
        """
        Return the product of the values over the requested axis.

        Similar to `Pandas product <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.product.html>`__

        Args:
            axis (int): Axis for the function to be applied on.
            skipna (bool): Exclude NA/null values when computing the result.
            numeric_only (bool): Include only float, int or boolean data. If None, will attempt to use everything, then use only numeric data. Not implemented for Series.
            min_count (int): The required number of valid values to perform the operation. If fewer than `min_count` non-NA values are present the result will be NA.
            **kwargs: Additional keyword arguments to be passed to the function.

        Returns:
            ``DataBlock``: DataBlock with operation executed.

        Example:

        >>> block
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> block.wprod()
                        AAPL                  MSFT           
                        Open     Close        Open      Close
        2005-12-21  5.010849  5.080517  380.981498  377.29376
        """
        return from_series(super().product(axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count, **kwargs), axis, self.index[0], 'product')


    def wquantile(self, q: Union[float, List[float]] = 0.5, interpolation: str = "linear"):
        """
        Return value at the given quantile.

        Similar to `Pandas quantile <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.quantile.html>`__

        Args:
            q (float, array): The quantile(s) to compute, which can lie in range: 0 <= q <= 1.
            interpolation (str): {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
            
                This optional parameter specifies the interpolation method to use, when the desired quantile lies between two data points `i` and `j`:

                * 'linear': `i + (j - i) * fraction`, where `fraction` is the fractional part of the index surrounded by `i` and `j`.
                * 'lower': `i`.
                * 'higher': `j`.
                * 'nearest': `i` or `j` whichever is nearest.
                * 'midpoint': (`i` + `j`) / 2.

        Returns:
            ``DataBlock``: DataBlock with operation executed.

        Example:

        >>> block
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> block.wquantile(q=0.5, interpolation='linear')
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        2005-12-21  2.238582  2.254014  19.518834  19.424118
        """
        return from_series(super().quantile(q=q, interpolation=interpolation), index=self.index[0], name='quantile')

    def wsem(self, axis: int = None, skipna: bool = None, ddof: int = 1, numeric_only=None, **kwargs):
        """
        Return unbiased standard error of the mean over requested axis.

        Normalized by N-1 by default. This can be changed using the ddof argument

        Similar to `Pandas sem <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sem.html>`__

        Args:
            axis (int): Axis for the function to be applied on.
            skipna (bool): Exclude NA/null values when computing the result.
            ddof (int): Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the number of elements.
            numeric_only (bool): Include only float, int or boolean data. If None, will attempt to use everything, then use only numeric data. Not implemented for Series.
            **kwargs: Additional keyword arguments to be passed to the function.

        Returns:
            ``DataBlock``: DataBlock with operation executed.

        Example:

        >>> block
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> block.wsem()
                        AAPL                MSFT          
                        Open     Close      Open     Close
        2005-12-21  0.020016  0.007946  0.058291  0.051004
        """
        return from_series(super().sem(axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only, **kwargs), axis, self.index[0], 'sem')


    def wskew(self, axis: int = None, skipna: bool = None, numeric_only=None, **kwargs):
        """
        Return unbiased skew over requested axis.

        Normalized by N-1.

        Similar to `Pandas skew <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.skew.html>`__

        Args:
            axis (int): Axis for the function to be applied on.
            skipna (bool): Exclude NA/null values when computing the result.
            numeric_only (bool): Include only float, int or boolean data. If None, will attempt to use everything, then use only numeric data. Not implemented for Series.
            **kwargs: Additional keyword arguments to be passed to the function.

        Returns:
            ``DataBlock``: DataBlock with operation executed.

        Example:

        >>> block
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> block.wskew(axis=1)
                       asset
                        skew
        Date                
        2005-12-21  0.000084
        2005-12-22  0.000067
        """
        return from_series(super().skew(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs), axis, self.index[0], 'skew')


    def wstd(self, axis: int = None, skipna: bool = None, ddof: int = 1, numeric_only=None, **kwargs):
        """
        Return sample standard deviation over requested axis.

        Normalized by N-1 by default. This can be changed using the ddof argument

        Similar to `Pandas std <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.std.html>`__

        Args:
            axis (int): Axis for the function to be applied on.
            skipna (bool): Exclude NA/null values when computing the result.
            ddof (int): Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the number of elements.
            numeric_only (bool): Include only float, int or boolean data. If None, will attempt to use everything, then use only numeric data. Not implemented for Series.
            **kwargs: Additional keyword arguments to be passed to the function.

        Returns:
            ``DataBlock``: DataBlock with operation executed.

        Example:

        >>> block
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> block.wstd()
                        AAPL                MSFT          
                        Open     Close      Open     Close
        2005-12-21  0.028307  0.011237  0.082436  0.072131
        """
        return from_series(super().std(axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only, **kwargs), axis, self.index[0], 'std')


    def wsum(self, axis: int = None, skipna: bool = None, numeric_only=None, min_count: int = 0, **kwargs):
        """
        Return the sum of the values over the requested axis.

        This is equivalent to the method `numpy.sum`.

        Similar to `Pandas sum <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sum.html>`__

        Args:
            axis (int): Axis for the function to be applied on.
            skipna (bool): Exclude NA/null values when computing the result.
            numeric_only (bool): Include only float, int or boolean data. If None, will attempt to use everything, then use only numeric data. Not implemented for Series.
            min_count (int): The required number of valid values to perform the operation. If fewer than `min_count` non-NA values are present the result will be NA.
            **kwargs: Additional keyword arguments to be passed to the function.

        Returns:
            ``DataBlock``: DataBlock with operation executed.

        Example:

        >>> block
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> block.wsum()
                        AAPL                  MSFT           
                        Open     Close        Open      Close
        2005-12-21  5.010849  5.080517  380.981498  377.29376
        """
        return from_series(super().sum(axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count, **kwargs), axis, self.index[0], 'sum')


    def wvar(self, axis: int = None, skipna: bool = None, ddof: int = 1, numeric_only=None, **kwargs):
        """
        Return sample variance over requested axis.

        Normalized by N-1 by default. This can be changed using the ddof argument

        Similar to `Pandas var <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.var.html>`__

        Args:
            axis (int): Axis for the function to be applied on.
            skipna (bool): Exclude NA/null values when computing the result.
            ddof (int): Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the number of elements.
            numeric_only (bool): Include only float, int or boolean data. If None, will attempt to use everything, then use only numeric data. Not implemented for Series.
            **kwargs: Additional keyword arguments to be passed to the function.

        Returns:
            ``DataBlock``: DataBlock with operation executed.

        Example:

        >>> block
                        AAPL                 MSFT           
                        Open     Close       Open      Close
        Date                                                
        2005-12-21  2.218566  2.246069  19.577126  19.475122
        2005-12-22  2.258598  2.261960  19.460543  19.373114

        >>> block.wvar()
                        AAPL                MSFT          
                        Open     Close      Open     Close
        2005-12-21  0.000801  0.000126  0.006796  0.005203
        """
        return from_series(super().var(axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only, **kwargs), axis, self.index[0], 'var')


    def plot(self, assets: List[str] = None, channels: List[str] = None):
        """
        Block plot according to the specified assets and channels.

        Args:
            assets (list): List of assets
            channels (list): List of channels

        Returns:
            ``Plot``: Plotted data
        """
        cmap = px.colors.qualitative.Plotly

        fig = make_subplots(rows=len(self.channels), cols=len(self.assets), subplot_titles=self.assets)

        # data = self.as_dataframe()

        for j, channel in enumerate(self.channels):
            c = cmap[j]
            for i, asset in enumerate(self.assets):

                # showlegend = i <= 0
                # x_df = data.loc[:, (asset, channel)]

                x_df = self.filter(assets=asset, channels=channel)
                index = x_df.index
                values = x_df.values.flatten()

                x_trace = go.Scatter(x=index, y=values,
                                line=dict(width=2, color=c), showlegend=False)

                fig.add_trace(x_trace, row=j+1, col=i+1)
                # Remove empty dates
                # dt_all = pd.date_range(start=index[0],end=index[-1])
                # dt_obs = [d.strftime("%Y-%m-%d") for d in index]
                # dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d").tolist() if not d in dt_obs]
                # fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])

        fig.update_layout(
            template='simple_white',
            showlegend=True
            )

        # Plot y titles
        num_assets = len(self.assets)
        for i, channel in enumerate(self.channels):
            fig['layout'][f'yaxis{i*num_assets+1}'].update({'title':channel})

        fig.show()



    # TODO Not implemented error for all pandas functions not used in wavy

    # TODO dropna

    # TODO add findna???
    # TODO add findinf???
    # TODO add_channel???
    # TODO flat???
    # TODO flatten???
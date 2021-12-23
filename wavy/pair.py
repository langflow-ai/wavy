import numpy as np
import pandas as pd
from typing import List

class TimePair:

    DIMS = ("assets", "timesteps", "channels")

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        x_equals = self.x.equals(other.x)
        y_equals = self.y.equals(other.y)
        return x_equals and y_equals

    def __repr__(self):
        return f"<TimePair, lookback {self.lookback}, horizon {self.horizon}>"

    @property
    def lookback(self):
        """
        TimePair lookback.
        """
        return len(self.x)

    @property
    def horizon(self):
        """
        TimePair horizon.
        """
        return len(self.y)

    @property
    def start(self):
        """
        TimePair start.

        Example:

        >>> timepair.start
        Timestamp('2005-12-27 00:00:00')
        """
        return self.x.start

    @property
    def end(self):
        """
        TimePair end.

        Example:

        >>> timepair.end
        Timestamp('2005-12-30 00:00:00')
        """
        return self.y.end

    @property
    def assets(self):
        """
        TimePair assets.

        Example:

        >>> timepair.assets
        0    AAPL
        1    MSFT
        dtype: object
        """
        return self.x.assets

    @property
    def channels(self):
        """
        TimePair channels.

        Example:

        >>> timepair.channels
        0    Open
        1    Close
        dtype: object
        """
        return self.x.channels

    # TODO include timesteps ???
    # timepair.x.timesteps
    # timepair.y.timesteps
    # @property
    # def timesteps(self):
    #     """
    #     TimePair index.

    #     Example:

    #     >>> timepair.timesteps
    #     DatetimeIndex(['2005-12-21', '2005-12-22', '2005-12-23'], dtype='datetime64[ns]', name='Date', freq=None)
    #     """
    #     # The same as the index
    #     return self.x.timesteps

    # TODO tensor
    # TODO matrix

    def filter(self, assets: List[str] = None, channels: List[str] = None):
        """
        TimePair subset according to the specified assets and channels.

        Args:
            assets (list): List of assets
            channels (list): List of channels

        Returns:
            ``TimeBlock``: Filtered TimeBlock

        Example:

        >>> timepair.x
                        MSFT                 AAPL          
                        Open      Close      Open     Close
        Date                                                
        2005-12-27  19.438692  19.278402  2.261349  2.268378
        2005-12-28  19.314840  19.227409  2.275712  2.248209

        >>> timepair.y
                        MSFT                 AAPL          
                        Open      Close      Open     Close
        Date                                                
        2005-12-29  19.241974  19.139973  2.254626  2.183424
        2005-12-30  19.052540  19.052540  2.166923  2.196870

        >>> new_timepair = timepair.filter(assets=['AAPL'], channels=['Open'])

        >>> new_timepair.x
                        AAPL
                        Open
        Date                
        2005-12-27  2.261349
        2005-12-28  2.275712

        >>> new_timepair.y
                        AAPL
                        Open
        Date                
        2005-12-29  2.254626
        2005-12-30  2.166923
        """
        x = self.x.filter(assets=assets, channels=channels)
        y = self.y.filter(assets=assets, channels=channels)
        return TimePair(x, y)

    # TODO drop
    # TODO rename_assets
    # TODO rename_channels
    # TODO apply
    # TODO update
    # TODO split_assets
    # TODO sort_assets
    # TODO sort_channels
    # TODO swap_cols
    # TODO countna
    # TODO as_dataframe
    # TODO fillna

    @property
    def shape(self):
        """
        TimePair shape.

        Example:

        >>> timepair.shape
           assets  timesteps  channels
        x       2          2         2
        y       2          2         2
        """
        return pd.DataFrame([self.x.tensor.shape, self.y.tensor.shape], index=["x", "y"], columns=self.DIMS)

    # TODO add findna???
    # TODO add_channel???
    # TODO flat???
    # TODO flatten???

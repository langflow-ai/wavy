import random
from copy import copy
from itertools import groupby

import numpy as np
import pandas as pd
import plotly as px
import plotly.graph_objects as go
import plotly.express as px

pd.set_option("multi_sparse", True)  # To see multilevel indexes
pd.options.plotting.backend = "plotly"

cmap1 = px.colors.qualitative.Plotly
cmap2 = cmap1[::-1]

# def predict_plot(df):
#     fig = px.line(df, x=df.index, y=df.columns)
#     fig.show()

# def line_plot(df, return_traces=False, prefix="", dash="solid", cmap=cmap1, mode="lines"):
#     fig = go.Figure()
#     for idx, col in enumerate(df.columns):
#         fig.add_trace(
#             go.Scatter(
#                 x=df.index, y=df[col], name=prefix + col, mode=mode, line=dict(color=cmap[idx], width=2, dash=dash)
#             )
#         )
#     fig.update_layout(title="", xaxis_title="Timestamps", yaxis_title="Values")
#     if return_traces:
#         return fig
#     else:
#         fig.show()


# def pair_plot(pair, unit, channels=None):
#     if not channels:
#         channels = pair.xframe[unit].columns
#     x = pair.xframe[unit][channels]
#     y = pair.yframe[unit][channels]

#     fig = go.Figure()

#     for channel in channels:
#         c = random.choice(cmap1)
#         fig.add_trace(go.Scatter(x=x.index, y=x[channel], name="x_" + channel, line=dict(width=2, color=c)))

#         fig.add_trace(
#             go.Scatter(x=y.index, y=y[channel], name="y_" + channel, line=dict(width=2, dash="dot", color=c))
#         )

#     fig.update_layout(title="", xaxis_title="Timestamps", yaxis_title="Values")
#     fig.show()


# # def pred_plot(y_test, y_pred, unit, channels=None, mode="lines"):
# #     test_trace = multi_plot(y_test, unit, channels, prefix="test_", return_traces=True, cmap=cmap1, mode=mode)
# #     pred_trace = multi_plot(
# #         y_pred, unit, channels, prefix="pred_", return_traces=True, dash="dot", cmap=cmap2, mode=mode
# #     )
# #     fig = copy(test_trace)
# #     for trace in pred_trace.data:
# #         fig.add_trace(trace)
# #     fig.show()


# def plot_data(self, start=None, end=None, channels=None, units=None, on="xdata"):
#     """
#     Plots the data according to the given parameters.

#     Parameters
#     ----------
#     start : Timestamp
#         First date to be plotted.
#     end : Timestamp
#         Last date to be plotted.
#     channels : str or list of strs
#         Channels to be plotted. If none is inserted, all the channels will be plotted.
#     units : str or list of strs
#         Units to be plotted. If none is inserted, all the units will be plotted.
#     on : str
#         Data to be plotted, options:
#             "xdata",
#             "ydata"

#     Raises
#     ------
#     ValueError
#         When no start or end date is inserted.

#     Returns
#     -------
#     None.

#     """

#     if start is None:
#         raise ValueError("Must enter the start date!")
#     if end is None:
#         raise ValueError("Must enter the start date!")

#     if channels is None:
#         channels = self.channels
#     elif isinstance(channels, str):
#         channels = [channels]

#     if units is None:
#         units = self.units
#     elif isinstance(units, str):
#         units = [units]

#     if on == "xdata":
#         data = self.xdata[start:end]
#     elif on == "ydata":
#         data = self.ydata[start:end]
#     else:
#         raise ValueError("Please select 'xdata' or 'ydata'")

#     for unit in units:
#         for channel in channels:
#             data_aux = data

#             data_aux = data_aux[unit][channel]
#             indexes = data_aux.index

#             fig = px.line(data_aux, x=indexes, y=channel, title=unit)
#             fig.show()
import math
from turtle import filling
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# TODO: Set plotting configs and add kwargs to functions
# TODO: Check if kwargs would overwrite fig.add_trace if same params are used

class Figure:
    """ Higher level wrapper for plotly, especially focused on time series plots.

    1. Lower level functions
    - Given a series, create the trace
    - e.g. add_line, add_bar, add_scatter
    - Contain all the logic for creating the trace
    - Don't carry **kwargs, keep simple parameters for simplicity

    2. Higher level functions
    - Group lower level functions into a single function

    """

    def __init__(self):
        self.fig = go.Figure()
        # TODO: Add fill area between 2 series

    def line(self, series, dash=None, **kwargs):
        # TODO: Add trace names for hovering

        self.fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series,
                mode="lines",
                line=dict(width=1.5, dash=dash, **kwargs),
            )
        )

    def area(self, series, **kwargs):
        self.fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series,
                mode="lines",
                line=dict(width=1.5, **kwargs),
                fill='tozeroy',
            )
        )

    def bar(self, series, **kwargs):
        self.fig.add_trace(
            go.Bar(
                x=series.index,
                y=series,
                **kwargs,
            )
        )

    def linebar(self, series, color="gray", opacity=0.5, **kwargs):
        self.fig.add_trace(
            go.Bar(
                x=series.index,
                y=series,
                width=1,
                marker=dict(line=dict(width=0.6, color=color), opacity=opacity, color=color, **kwargs,),
            )
        )

    def scatter(self, series, **kwargs):
        self.fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series,
                mode="markers",
                **kwargs,
            )
        )

    def dotline(self, series, color="gray", opacity=0.5):
        # ? Name might be confusing
        self.linebar(series, color=color, opacity=opacity)
        self.scatter(series)

    def threshline(self, series, up_thresh, down_thresh, up_color, down_color):

        up_df = series[series > up_thresh].index
        down_df = series[series < down_thresh].index

        print(len(down_df), len(up_df))

        for i in up_df:
            self.fig.add_vline(x=i, line_dash="dot", line_color=up_color,)

        for i in down_df:
            self.fig.add_vline(x=i, line_dash="dot", line_color=down_color,)

    def background(self, series, text, color, opacity):
        self.fig.add_vrect(x0=series.index.min(), x1=series.index.max(),
                           annotation_text=text, annotation_position="top left",
                           fillcolor=color, opacity=opacity, line_width=0, layer="below")

    def show(self):
        return self.fig


class PanelFigure(Figure):
    def __init__(self):
        # TODO: Add dynamic color changing once new traces are added
        # TODO: Add candlestick plot
        super().__init__()

    def split_sets(self, panel, color="gray", opacity=1):
        # BUG: Seems to break if using "ggplot2"
        # ! Won't take effect until next trace is added (no axis was added)
        # TODO: Functions could accept both dataframe or panel

        data = {'train': panel.train.as_dataframe(),
                'val': panel.val.as_dataframe(),
                'test': panel.test.as_dataframe()
                }

        xtrain_min = data['train'].index[0]
        xval_min = data['val'].index[0]
        xtest_min = data['test'].index[0]

        ymax = max(data['train'].max().max(), data['val'].max().max(), data['test'].max().max())

        self.fig.add_vline(x=xtrain_min, line_dash="dot", line_color=color, opacity=opacity)
        self.fig.add_vline(x=xval_min, line_dash="dot", line_color=color, opacity=opacity)
        self.fig.add_vline(x=xtest_min, line_dash="dot", line_color=color, opacity=opacity)

        self.fig.add_annotation(x=xtrain_min, y=ymax, text="Train", showarrow=False, xshift=20)
        self.fig.add_annotation(x=xval_min, y=ymax, text="Validation", showarrow=False, xshift=35)
        self.fig.add_annotation(x=xtest_min, y=ymax, text="Test", showarrow=False, xshift=18)

    def add_line(self, panel, **kwargs):
        for col in panel.columns:
            self.line(panel.as_dataframe()[col], **kwargs)

    def add_area(self, panel, **kwargs):
        for col in panel.columns:
            self.area(panel.as_dataframe()[col], **kwargs)

    def add_bar(self, panel, **kwargs):
        for col in panel.columns:
            self.bar(panel.as_dataframe()[col], **kwargs)

    def add_scatter(self, panel, **kwargs):
        for col in panel.columns:
            self.scatter(panel.as_dataframe()[col], **kwargs)

    def add_dotline(self, panel, color="gray", opacity=0.5):
        # BUG: Dots and lines look displaced if zoomed in
        for col in panel.columns:
            self.dotline(panel.as_dataframe()[col], color=color, opacity=opacity)

    def add_threshline(self, panel, up_thresh, down_thresh, up_color="green", down_color="red", col=None):
        col = self._colcheck(panel, col)
        self.threshline(panel.as_dataframe()[col], up_thresh, down_thresh, up_color, down_color)

    def _colcheck(self, panel, col):
        if col is None:
            if panel.columns.size == 1:
                return panel.columns[0]
            else:
                raise ValueError("Must specify column to plot")



# def plot_dataframes(dfs, **kwargs):
#     """
#     Plot dataframes.

#     Args:
#         dfs (list): List of dataframes

#     Returns:
#         ``Plot``: Plotted data
#     """

#     return pd.concat(dfs, axis=1).plot(**kwargs)


def plot(panel, split=False, **kwargs):
    """
    Plot a panel.

    Args:
        panel (Panel): Panel object
        split (bool): If True, plot vertical lines showing train, val, and test periods

    Returns:
        ``Plot``: Plotted data
    """
    fig = PanelFigure()

    if split:
        fig.split_sets(panel)

    fig.add_line(panel, **kwargs)
    return fig.show()


def plot_frame(panel, index):
    """
    Dataframe plot.

    Args:
        index (int): Panel index

    Returns:
        ``Plot``: Plotted data
    """
    cmap = px.colors.qualitative.Plotly

    columns_size = len(panel.columns)

    fig = make_subplots(rows=math.ceil(columns_size / 2), cols=2, subplot_titles=[' '.join(column) for column in panel.columns])

    for i, column in enumerate(panel.columns):
        c = cmap[i]

        x_df = panel.frames[index].loc[:, column]
        idx = x_df.index
        values = x_df.values.flatten()

        x_trace = go.Scatter(x=idx, y=values, line=dict(width=2, color=c), showlegend=False)

        row = math.floor(i / 2)
        col = i % 2
        fig.add_trace(x_trace, row=row + 1, col=col + 1)
        # Remove empty dates
        # dt_all = pd.date_range(start=index[0],end=index[-1])
        # dt_obs = [d.strftime("%Y-%m-%d") for d in index]
        # dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d").tolist() if not d in dt_obs]
        # fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])

    fig.update_layout(
        template='simple_white',
        showlegend=True
    )

    # num_assets = len(panel.assets)
    # for i, channel in enumerate(panel.channels):
    #     fig['layout'][f'yaxis{i*num_assets+1}'].update({'title':channel})

    fig.show()


def plot_slider(panel, steps=100):
    """
    Make panel plots with slider.

    Args:
        steps (int): Number of equally spaced frames to plot

    Returns:
        ``Plot``: Plotted data.
    """

    if steps > 100:
        raise ValueError("Number of steps cannot be bigger than 100.")

    cmap = px.colors.qualitative.Plotly

    # Create figure
    columns_size = len(panel.columns)
    fig = make_subplots(rows=math.ceil(columns_size / 2), cols=2, subplot_titles=[' '.join(column) for column in panel.columns])
    # fig = make_subplots(rows=len(panel.channels), cols=len(panel.assets), subplot_titles=panel.assets)

    # Add traces, one for each slider step
    len_ = np.linspace(0, len(panel.frames), steps, dtype=int, endpoint=False)
    for step in len_:  # np.arange(len(panel_.x.frames)):

        for i, column in enumerate(panel.columns):
            c = cmap[i]

            x_df = panel.frames[step].loc[:, column]
            index = x_df.index
            values = x_df.values.flatten()

            x_trace = go.Scatter(visible=False, x=index, y=values, line=dict(width=2, color=c), showlegend=False)

            # x_trace = go.Scatter(x=index, y=values,
            #                     line=dict(width=2, color=c), showlegend=showlegend, name=channel)

            row = math.floor(i / 2)
            col = i % 2
            fig.add_trace(x_trace, row=row + 1, col=col + 1)

            # dt_all = pd.date_range(start=index[0],end=index[-1])
            # dt_obs = [d.strftime("%Y-%m-%d") for d in index]
            # dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d").tolist() if not d in dt_obs]
            # fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])

    # Make 10th trace visible
    for i in range(columns_size):
        fig.data[i].visible = True

    # Create and add slider
    steps_ = []
    for i in range(steps):
        step = dict(
            method="update",
            args=[
                {"visible": [False] * len(fig.data)},
                {"title": f"frame {str(len_[i])}"},
            ],
        )

        for g in range(columns_size):
            step["args"][0]["visible"][i * columns_size + g] = True  # Toggle i'th trace to "visible"

        steps_.append(step)

    sliders = [dict(
        active=0,
        # currentvalue={"prefix": "frame: "},
        pad={"t": 50},
        steps=steps_
    )]

    fig.update_layout(
        template='simple_white',
        sliders=sliders
    )

    # Plot y titles
    # num_assets = len(panel.assets)
    # for i, channel in enumerate(panel.channels):
    #     fig['layout'][f'yaxis{i*num_assets+1}'].update({'title':channel})

    fig.show()


# def plot_prediction(panel, x):
#     cmap = px.colors.qualitative.Plotly

#     columns_size = len(panel.columns)

#     fig = px.line(x)

#     # fig = make_subplots(rows=math.ceil(columns_size / 2), cols=2, subplot_titles=[' '.join(column) for column in panel.columns])

#     # for i, column in enumerate(panel.columns):
#     #     c = cmap[i]

#     #     x_df = panel.frames[index].loc[:, column]
#     #     idx = x_df.index
#     #     values = x_df.values.flatten()

#     #     x_trace = go.Scatter(x=idx, y=values, line=dict(width=2, color=c), showlegend=False)

#     #     row = math.floor(i / 2)
#     #     col = i % 2
#     #     fig.add_trace(x_trace, row=row + 1, col=col + 1)
#     #     # Remove empty dates
#     #     # dt_all = pd.date_range(start=index[0],end=index[-1])
#     #     # dt_obs = [d.strftime("%Y-%m-%d") for d in index]
#     #     # dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d").tolist() if not d in dt_obs]
#     #     # fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])

#     fig.update_layout(
#         template='simple_white',
#         showlegend=True
#     )

#     # num_assets = len(panel.assets)
#     # for i, channel in enumerate(panel.channels):
#     #     fig['layout'][f'yaxis{i*num_assets+1}'].update({'title':channel})

#     fig.show()

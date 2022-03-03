import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

pio.templates.default = "plotly_white"

# TODO: Set plotting configs and add kwargs to functions


def add_line_trace(fig, df, col, color, add_markers=False, dash=None, area=False, opacity=1):

    mode = "lines+markers" if add_markers else "lines"
    fill = 'tozeroy' if area else None

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[col],
            mode=mode,
            line=dict(width=1.5, dash=dash, color=color),
            fill=fill,
            opacity=opacity,
        )
    )


def add_bar_trace(fig, df, col,):
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df[col],
        )
    )


def add_scatter_trace(fig, df, col, mode="markers"):
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[col],
            mode=mode,
        )
    )


def add_vline_trace(fig, df, col, up_thresh, down_thresh):

    up_df = df[df[col] > up_thresh].index
    down_df = df[df[col] < down_thresh].index

    print(len(down_df), len(up_df))

    for i in up_df:
        fig.add_vline(x=i, line_dash="dot", line_color="green",)

    for i in down_df:
        fig.add_vline(x=i, line_dash="dot", line_color="red",)


def add_background_trace(fig, df, text, color, opacity):
    fig.add_vrect(x0=df.index.min(), x1=df.index.max(),
                  annotation_text=text, annotation_position="top left",
                  fillcolor=color, opacity=opacity, line_width=0, layer="below")




def split_background(fig, data, color="gray"):
    add_background_trace(fig, data['train'], text="Train", color=color, opacity=0.6)
    add_background_trace(fig, data['val'], text="Validation", color=color, opacity=0.4)
    add_background_trace(fig, data['test'], text="Test", color=color, opacity=0.2)


def line_plot(fig, data, col, color='#c94f4f'):
    # TODO: Add trace names for hovering
    add_line_trace(fig, data['train'], col, color=color, opacity=1)
    add_line_trace(fig, data['val'], col, color=color, opacity=1)
    add_line_trace(fig, data['test'], col, color=color, opacity=1)


def compile_plot(fig):
    fig.update_layout(
        plot_bgcolor="white",
        showlegend=False,
        title="Time Series",
        xaxis=dict(title="Periods", showgrid=False, zeroline=False),
        yaxis=dict(
            title="Value",
            showgrid=False,
            zerolinewidth=2,
            zerolinecolor='gray',
        ),
    )


def panel_plot(panel, col):
    data = {'train': panel.train.as_dataframe(),
            'val': panel.val.as_dataframe(),
            'test': panel.test.as_dataframe()
            }

    fig = go.Figure()
    line_plot(fig, data, col)
    split_background(fig, data)
    compile_plot(fig)

    return fig




# def plot_many(panels, **kwargs):
#     """
#     Plot many panels.

#     Args:
#         panels (list): List of panels

#     Returns:
#         fig (plotly.graph_objects.Figure): Figure object
#     """

#     dfs = [panel.as_dataframe() for panel in panels]
#     return plot_dataframes(dfs, **kwargs)


def plot_dataframes(dfs, **kwargs):
    """
    Plot dataframes.

    Args:
        dfs (list): List of dataframes

    Returns:
        ``Plot``: Plotted data
    """

    return pd.concat(dfs, axis=1).plot(**kwargs)


def plot(panel, **kwargs):
    """
    Plot a panel.

    Args:
        panel (Panel): Panel object

    Returns:
        ``Plot``: Plotted data
    """
    return panel.as_dataframe().plot(**kwargs)


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
        raise ValueError("Number of assets cannot be bigger than 100.")

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
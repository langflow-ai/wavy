from __future__ import annotations

from collections.abc import Iterable
from functools import wraps

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Figure:
    """
    Higher level wrapper for plotly, especially focused on time series plots.
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
        self.cmap = px.colors.qualitative.Plotly

    def __call__(self):
        return self.fig

    def xaxes_setting(self, **kwargs):
        self.fig.update_xaxes(**kwargs)

    def set_plot_size(self, width=1350, height=650, **kwargs):
        # ? Add this to init?
        self.fig.update_layout(
            width=width,
            height=height,
            **kwargs,
        )

    def layout(self, **kwargs):
        self.fig.update_layout(
            **kwargs,
        )

    def set_labels_size(self, titlefont_size=20, tickfont_size=17, **kwargs):
        self.fig.update_layout(
            xaxis=dict(
                titlefont_size=titlefont_size,
                tickfont_size=tickfont_size,
            ),
            yaxis=dict(
                titlefont_size=titlefont_size,
                tickfont_size=tickfont_size,
            ),
            **kwargs,
        )

    def annotation(self, x_value, y_value, text, **kwargs):
        self.fig.add_annotation(
            x=x_value,
            y=y_value,
            xref="x",
            yref="y",
            text=text,
            showarrow=True,
            font=dict(size=14, color="#ffffff"),
            align="center",
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#636363",
            ax=5,
            ay=-50,
            bordercolor="#c7c7c7",
            borderwidth=2,
            borderpad=4,
            bgcolor="#3C4470",
            opacity=0.8,
            **kwargs,
        )

    def hbar(self, series, **kwargs):
        self.fig.add_trace(
            go.Bar(
                x=series,
                y=series.index,
                orientation="h",
                name=series.name,
                **kwargs,
            )
        )

    def line(self, series, dash=None, **kwargs):
        self.fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series,
                mode="lines",
                line=dict(width=1.5, dash=dash, **kwargs),
                name=series.name,
            )
        )

    def area(self, series, **kwargs):
        self.fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series,
                mode="lines",
                line=dict(width=1.5),
                fill="tozeroy",
                name=series.name,
                **kwargs,
            )
        )

    def bar(self, series, **kwargs):
        self.fig.add_trace(
            go.Bar(
                x=series.index,
                y=series,
                name=series.name,
                **kwargs,
            )
        )

    def linebar(self, series, color="gray", opacity=0.5, **kwargs):
        # BUG: Fix width (might need to be relative)
        self.fig.add_trace(
            go.Bar(
                x=series.index,
                y=series,
                width=1,
                name=series.name,
                marker=dict(
                    line=dict(width=0.00001, color=color),
                    opacity=opacity,
                    color=color,

                    **kwargs,
                ),
            )
        )

    def scatter(self, series, mode="markers", **kwargs):
        self.fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series,
                mode=mode,
                name=series.name,
                **kwargs,
            )
        )

    def dotline(self, series, color="gray", opacity=0.5):
        self.linebar(series, color=color, opacity=opacity)
        self.scatter(series)

    def waterfall(self, df, measure, x_values, text, y_values, orientation="v"):
        #TODO: Document
        """
        DATAFRAME REQUIRES PRE-PROCESSING FOR PLOT
            df: dataframe
            measure(str): realitve or total
            x_values: values in x axis
            y_values(int or float): values in y axis
            text(str): text in the bar
        """
        self.fig.add_trace(
            go.Waterfall(
                name="20",
                orientation=orientation,
                measure=list(df[measure]),
                x=list(df[x_values]),
                textposition="outside",
                text=list(df[text]),
                y=list(df[y_values]),
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            )
        )


    def hline(self, y_value, **kwargs):
        # TODO: Allow list as input or other types
        self.fig.add_hline(y=y_value, **kwargs)

    def vline(self, x_value, **kwargs):
        # check if x_value is iterable:
        if isinstance(x_value, Iterable):
            for x in x_value:
                self.fig.add_vline(x=x, **kwargs)
        self.fig.add_vline(x=x_value, **kwargs)

    def clear(self):
        self.fig = go.Figure()

    def show(self):
        return self()



# TODO: Set plotting configs and add kwargs to functions
# TODO: Check if kwargs would overwrite fig.add_trace if same params are used


class PanelFigure(Figure):
    """
    PanelFigure class.
    """

    def __init__(self):
        # TODO: Add dynamic color changing once new traces are added
        # TODO: Add candlestick plot
        # TODO: Add trace names as panel cols
        super().__init__()
        self.colors = px.colors.qualitative.Plotly
        self.color_index = 0

    def add_annotation(self, panel, color: str = "gray", opacity: float = 1.0) -> None:
        """
        Plot vertical lines showing train, val, and test periods.

        Args:
            panel (``Panel``): Panel to split.
            color (``str``): Color of the sets.
            opacity (``float``): Opacity of the sets.
        """
        # BUG: Seems to break if using "ggplot2"
        # ! Won't take effect until next trace is added (no axis was added)

        ymax = panel.max().max() if panel.train_size else 0

        if hasattr(panel, "train_size") and panel.train_size:
            xtrain_min = panel.train.index.min()
            self.fig.add_vline(
                x=xtrain_min, line_dash="dot", line_color=color, opacity=opacity
            )
            self.fig.add_annotation(
                x=xtrain_min, y=ymax, text="Train", showarrow=False, xshift=20
            )

        if hasattr(panel, "val_size") and panel.val_size:
            xval_min = panel.val.index.min()
            self.fig.add_vline(
                x=xval_min, line_dash="dot", line_color=color, opacity=opacity
            )
            self.fig.add_annotation(
                x=xval_min, y=ymax, text="Validation", showarrow=False, xshift=35
            )

        if hasattr(panel, "test_size") and panel.test_size:
            xtest_min = panel.test.index.min()
            self.fig.add_vline(
                x=xtest_min, line_dash="dot", line_color=color, opacity=opacity
            )
            self.fig.add_annotation(
                x=xtest_min, y=ymax, text="Test", showarrow=False, xshift=18
            )

    # Add decorator for instance check and for loop
    def _iterator(func):
        @wraps(func)
        def inner(self, *args, **kwargs):

            args = list(args)
            df = args.pop(0)

            for col in df.columns:
                kwargs["color"] = self.colors[self.color_index]
                self.color_index = (self.color_index + 1) % len(self.colors)
                if col != "frame":
                    func(self, df[col], *tuple(args), **kwargs)

        return inner

    @_iterator
    def add_line(self, col: str, *args, **kwargs) -> None:
        """
        Add a line to the figure.

        Args:
            col (``str``): Column to plot
        """
        self.line(col, *args, **kwargs)

    @_iterator
    def add_area(self, col: str, *args, **kwargs) -> None:
        """
        Add an area to the figure.

        Args:
            col (``str``): Column to plot
        """
        self.area(col, *args, **kwargs)

    @_iterator
    def add_bar(self, col: str, *args, **kwargs) -> None:
        """
        Add a bar to the figure.

        Args:
            col (``str``): Column to plot
        """
        self.bar(col, *args, **kwargs)

    @_iterator
    def add_scatter(self, col: str, *args, **kwargs) -> None:
        """
        Add a scatter to the figure.

        Args:
            col (``str``): Column to plot.
        """
        self.scatter(col, *args, **kwargs)

    @_iterator
    def add_dotline(self, col: str, *args, **kwargs) -> None:
        """
        Add a dotline to the figure.

        Args:
            col (``str``): Column to plot.
        """
        self.dotline(col, *args, **kwargs)


def plot(
    panel, use_timestep: bool = False, add_annotation: bool = False, **kwargs
) -> PanelFigure:
    """
    Plot panel.

    Args:
        panel (``Panel``): Panel object.
        use_timestep (``bool``): Use timestep instead of id.
        add_annotation (``bool``): If True, plot vertical lines showing train, val, and test periods.

    Returns:
        ``Plot``: Plotted data
    """
    fig = PanelFigure()

    for col in panel.columns:
        if panel[col].dtype == bool:
            panel[col] = panel[col].astype(int)

    if use_timestep:
        panel = panel.droplevel(0, axis=0) if panel.index.nlevels > 1 else panel
    else:
        panel = panel.droplevel(1, axis=0) if panel.index.nlevels > 1 else panel

    fig.add_line(panel, **kwargs)

    if add_annotation and isinstance(panel.train_size, int):
        fig.add_annotation(panel)

    return fig()

import plotly.graph_objects as go

class Logplot:
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
                y=series.values,
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

    # def make_subplots(self, rows:int, cols:int, subplot_titles:List[str]):
    #     self.fig.mak

    def show(self):
        return self.fig
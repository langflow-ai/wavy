from __future__ import annotations

import plotly.express as px
from plotlab import Figure

# TODO: Set plotting configs and add kwargs to functions
# TODO: Check if kwargs would overwrite fig.add_trace if same params are used


class PanelFigure(Figure):
    def __init__(self):
        # TODO: Add dynamic color changing once new traces are added
        # TODO: Add candlestick plot
        # TODO: Add trace names as panel cols
        super().__init__()
        self.colors = px.colors.qualitative.Plotly
        self.color_index = 0

    def add_annotation(self, panel, color="gray", opacity=1):
        """
        Split panel into sets.

        Args:
            panel (wavy.Panel): Panel to split
            color (str): Color of the sets
            opacity (float): Opacity of the sets
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
    def iterator(func):
        def inner(self, *args, **kwargs):

            args = list(args)
            df = args.pop(0)

            for col in df.columns:
                kwargs["color"] = self.colors[self.color_index]
                self.color_index = (self.color_index + 1) % len(self.colors)
                if col != "frame":
                    func(self, df[col], *tuple(args), **kwargs)

        return inner

    @iterator
    def add_line(self, col: str, *args, **kwargs):
        """
        Add a line to the figure.

        Args:
            col (str): Column to plot
        """
        self.line(col, *args, **kwargs)

    @iterator
    def add_area(self, col: str, *args, **kwargs):
        """
        Add an area to the figure.

        Args:
            col (str): Column to plot
        """
        self.area(col, *args, **kwargs)

    @iterator
    def add_bar(self, col: str, *args, **kwargs):
        """
        Add a bar to the figure.

        Args:
            col (str): Column to plot
        """
        self.bar(col, *args, **kwargs)

    @iterator
    def add_scatter(self, col: str, *args, **kwargs):
        """
        Add a scatter to the figure.

        Args:
            col (str): Column to plot
        """
        self.scatter(col, *args, **kwargs)

    @iterator
    def add_dotline(self, col: str, *args, **kwargs):
        """
        Add a dotline to the figure.

        Args:
            col (str): Column to plot
        """
        self.dotline(col, *args, **kwargs)


def plot(
    panel, use_timestep: bool = False, add_annotation: bool = False, **kwargs
) -> PanelFigure:
    """
    Plot a panel.

    Args:
        panel (Panel): Panel object
        use_timestep (bool): Use timestep instead of id
        add_annotation (bool): If True, plot vertical lines showing train, val, and test periods

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

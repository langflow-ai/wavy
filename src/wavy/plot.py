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

        ymax = max(
            panel.train.max().max() if panel.train_size else 0,
            panel.val.max().max() if panel.val_size else 0,
            panel.test.max().max() if panel.test_size else 0,
        )

        if panel.train_size:
            xtrain_min = panel.train.index[0]
            self.fig.add_vline(
                x=xtrain_min, line_dash="dot", line_color=color, opacity=opacity
            )
            self.fig.add_annotation(
                x=xtrain_min, y=ymax, text="Train", showarrow=False, xshift=20
            )

        if panel.val_size:
            xval_min = panel.val.index[0]
            self.fig.add_vline(
                x=xval_min, line_dash="dot", line_color=color, opacity=opacity
            )
            self.fig.add_annotation(
                x=xval_min, y=ymax, text="Validation", showarrow=False, xshift=35
            )

        if panel.test_size:
            xtest_min = panel.test.index[0]
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
    def add_line(self, col, *args, **kwargs):
        """
        Add a line to the figure.

        Args:
            col (str): Column to plot
        """
        self.line(col, *args, **kwargs)

    @iterator
    def add_area(self, col, *args, **kwargs):
        """
        Add an area to the figure.

        Args:
            col (str): Column to plot
        """
        self.area(col, *args, **kwargs)

    @iterator
    def add_bar(self, col, *args, **kwargs):
        """
        Add a bar to the figure.

        Args:
            col (str): Column to plot
        """
        self.bar(col, *args, **kwargs)

    @iterator
    def add_scatter(self, col, *args, **kwargs):
        """
        Add a scatter to the figure.

        Args:
            col (str): Column to plot
        """
        self.scatter(col, *args, **kwargs)

    @iterator
    def add_dotline(self, col, *args, **kwargs):  # color="gray", opacity=0.5):
        """
        Add a dotline to the figure.

        Args:
            col (str): Column to plot
        """
        # BUG: Dots and lines look displaced if zoomed in
        self.dotline(col, *args, **kwargs)  # color=color, opacity=opacity)


def plot(panel, use_timestep=False, add_annotation=False, **kwargs):
    # TODO: Add "kind" parameter to chose between plot types
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

    panel = panel.row_panel(n=0)

    if use_timestep:
        panel = panel.droplevel(0, axis=0)
    else:
        panel = panel.droplevel(1, axis=0)

    fig.add_line(panel, **kwargs)

    if add_annotation and isinstance(panel.train_size, int):
        fig.add_annotation(panel)

    return fig()


# def plot_dataframes(dfs, **kwargs):
#     """
#     Plot dataframes.

#     Args:
#         dfs (list): List of dataframes

#     Returns:
#         ``Plot``: Plotted data
#     """

#     return pd.concat(dfs, axis=1).plot(**kwargs)


# def add_threshline(
#     # self,
#     panel,
#     up_thresh,
#     down_thresh,
#     up_color="green",
#     down_color="red",
#     col=None,
# ):
#     """
#     Add a threshold line to the figure.

#     Args:
#         panel (wavy.Panel): Panel to plot
#         up_thresh (float): Upper threshold
#         down_thresh (float): Lower threshold
#         up_color (str): Color of the upper threshold
#         down_color (str): Color of the lower threshold
#         col (str): Column to plot
#     """

#     def _colcheck(panel, col):
#         if col is None:
#             if panel.columns.size == 1:
#                 return panel.columns[0]
#             else:
#                 raise ValueError("Must specify column to plot")

#     col = _colcheck(panel, col)

#     fig = PanelFigure()

#     fig.threshline(
#         panel.as_dataframe()[col], up_thresh, down_thresh, up_color, down_color
#     )

#     return fig


# def plot_frame(x, y, index=None):
#     """
#     Plot a dataframe.

#     Args:

#     """

#     # TODO: Code below is confusing
#     # TODO: split into two functions?
#     # if not(is_panel(x) and is_panel(y) and index is not None) or (is_dataframe(x) and is_dataframe(y)):
#     #     raise Exception("x and y should be either Panel or DataFrame")

#     # if is_panel(x):
#     #     x = x[index]
#     #     y = y[index]

#     fig = PanelFigure()
#     fig.add_line(x)
#     fig.add_scatter(x)
#     fig.add_line(y)
#     fig.add_scatter(y)

#     return fig()


# def plot_slider(panel, steps):
#     """
#     Make panel plots with slider.

#     Args:
#         steps (int): Number of equally spaced frames to plot

#     Returns:
#         ``Plot``: Plotted data.
#     """

#     # assert len(x) == len(
#     #     y
#     # ), "Length of frame should be equal, try using match function!"
#     # assert len(x) < 100, "Number of steps cannot be bigger than 100."

#     # cmap = px.colors.qualitative.Plotly

#     # Create figure
#     # columns_size = len(panel.columns)
#     # fig = make_subplots(rows=columns_size, cols=1, subplot_titles=[' '.join(column) for column in panel.columns])
#     # fig = make_subplots(rows=len(panel.channels), cols=len(panel.assets), subplot_titles=panel.assets)

#     # fig =

#     # Add traces, one for each slider step
#     # len_ = np.linspace(0, len(x.frames), steps, dtype=int, endpoint=False)
#     # for step in len_:  # np.arange(len(panel_.x.frames)):

#     fig = go.Figure()

#     len_ = list(range(len(x)))

#     for i in range(len(x)):
#         frame = plot_frame(x[i], y[i]).data

#         for f in frame:
#             fig.add_trace(f)

#         # for i, column in enumerate(panel.columns):
#         #     # c = cmap[i]

#         #     x_df = panel.frames[step].loc[:, column]
#         #     index = x_df.index
#         #     values = x_df.values.flatten()

#         #     fig =

#         # x_trace = go.Scatter(visible=False, x=index, y=values, line=dict(width=2, color=c), showlegend=False)

#         # x_trace = go.Scatter(x=index, y=values,
#         #                     line=dict(width=2, color=c), showlegend=showlegend, name=channel)

#         # row = math.floor(i / 2)
#         # col = i % 2
#         # fig.add_trace(x_trace, row = row + 1, col = 1)

#     # Make 10th trace visible
#     for i in range(len(fig.data)):
#         fig.data[i].visible = i < len(frame)
#     # Create and add slider
#     steps = []
#     for i in range(len(x)):
#         step = dict(
#             method="update",
#             args=[
#                 {"visible": [False] * len(fig.data)},
#                 {"title": f"frame {str(i)}"},
#             ],
#         )
#         # step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"

#         for g in range(len(frame)):
#             step["args"][0]["visible"][
#                 i * len(frame) + g
#             ] = True  # Toggle i'th trace to "visible"

#         steps.append(step)

#     sliders = [
#         dict(
#             active=0,
#             # currentvalue={"prefix": "Frequency: "},
#             pad={"t": 50},
#             steps=steps,
#         )
#     ]

#     fig.update_layout(sliders=sliders)

#     # # Create and add slider
#     # steps_ = []
#     # for i in range(steps):
#     #     step = dict(
#     #         method="update",
#     #         args=[
#     #             {"visible": [False] * len(fig.data)},
#     #             {"title": f"frame {str(len_[i])}"},
#     #         ],
#     #     )

#     #     for g in range(columns_size):
#     #         step["args"][0]["visible"][i * columns_size + g] = True  # Toggle i'th trace to "visible"

#     #     steps_.append(step)

#     # sliders = [dict(
#     #     active=0,
#     #     # currentvalue={"prefix": "frame: "},
#     #     pad={"t": 50},
#     #     steps=steps_
#     # )]

#     # fig.update_layout(
#     #     template='simple_white',
#     #     sliders=sliders
#     # )

#     return fig.show()

# Plot y titles
# num_assets = len(panel.assets)
# for i, channel in enumerate(panel.channels):
#     fig['layout'][f'yaxis{i*num_assets+1}'].update({'title':channel})

# fig.show()


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

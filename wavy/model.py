import warnings
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import (Conv1D, Dense, Flatten, Input,
                                     MaxPooling1D, Reshape, SeparableConv1D,
                                     concatenate)
from .panel import Panel


from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import plotly as px
from matplotlib.pyplot import title
import math


class _ConstantKerasModel(tf.keras.Model):
    """ A Keras model that returns the input values as outputs. """

    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return inputs


class _BaseModel:
    """Base class for panel models."""

    # TODO: Add warning when panel has nan values
    # TODO: Auto convert boolean to int

    def __init__(self, x, y, model_type: str = None, loss: str = None, optimizer: str = None, metrics: List[str] = None, last_activation: str = None):

        PARAMS = {
            'regression': {
                'loss': 'MSE',
                'optimizer': 'adam',
                'metrics': ['MAE'],
                'last_activation': 'linear'
            },
            'classifier': {
                'loss': 'binary_crossentropy',
                'optimizer': 'adam',
                'metrics': ['AUC', 'accuracy'],
                'last_activation': 'sigmoid'
            },
            'multi_classifier': {
                'loss': 'categorical_crossentropy',
                'optimizer': 'adam',
                'metrics': ['AUC', 'accuracy'],
                'last_activation': 'softmax'
            }
        }

        self.x = x
        self.y = y

        self.loss = loss or PARAMS[model_type]['loss']
        self.optimizer = optimizer or PARAMS[model_type]['optimizer']
        self.metrics = metrics or PARAMS[model_type]['metrics']
        self.last_activation = last_activation or PARAMS[model_type]['last_activation']

        self.set_arrays()
        self.build()
        self.compile()

    def set_arrays(self):
        self.x_train = self.x.train.values
        self.x_val = self.x.val.values
        self.x_test = self.x.test.values

        self.y_train = self.y.train.values
        self.y_val = self.y.val.values
        self.y_test = self.y.test.values

    def fit(self, **kwargs):
        """Fit the model."""
        self.model.fit(self.x_train, self.y_train, validation_data=(self.x_val, self.y_val), **kwargs)
        # return self

    def compile(self):
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

    def build(self):
        # raise NotImplementedError
        pass

    def predict(self, x):
        # TODO: Output prediction as a panel - needs something similar to from_matrix method
        return self.model.predict(x.values)

    def plot_prediction(self, x):
        cmap = px.colors.qualitative.Plotly

        columns_size = len(self.columns)

        fig = px.line(x)

        # fig = make_subplots(rows=math.ceil(columns_size / 2), cols=2, subplot_titles=[' '.join(column) for column in self.columns])

        # for i, column in enumerate(self.columns):
        #     c = cmap[i]

        #     x_df = self.frames[index].loc[:, column]
        #     idx = x_df.index
        #     values = x_df.values.flatten()

        #     x_trace = go.Scatter(x=idx, y=values, line=dict(width=2, color=c), showlegend=False)

        #     row = math.floor(i / 2)
        #     col = i % 2
        #     fig.add_trace(x_trace, row=row + 1, col=col + 1)
        #     # Remove empty dates
        #     # dt_all = pd.date_range(start=index[0],end=index[-1])
        #     # dt_obs = [d.strftime("%Y-%m-%d") for d in index]
        #     # dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d").tolist() if not d in dt_obs]
        #     # fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])

        fig.update_layout(
            template='simple_white',
            showlegend=True
        )

        # num_assets = len(self.assets)
        # for i, channel in enumerate(self.channels):
        #     fig['layout'][f'yaxis{i*num_assets+1}'].update({'title':channel})

        fig.show()

class BaselineModel(_BaseModel):
    def __init__(
        self,
        x,
        y,
        model_type: str,
        loss: str = None,
        metrics: List[str] = None,
    ):

        super().__init__(x=x, y=y, model_type=model_type, loss=loss, metrics=metrics)

    def set_arrays(self):
        self.x_train = self.y.train.shift(1).values[1:]
        self.x_val = self.y.val.shift(1).values[1:]
        self.x_test = self.y.test.shift(1).values[1:]

        self.y_train = self.y.train.values[1:]
        self.y_val = self.y.val.values[1:]
        self.y_test = self.y.test.values[1:]

    def build(self):
        self.model = _ConstantKerasModel()


class DenseModel(_BaseModel):
    def __init__(
        self,
        x,
        y,
        model_type: str,
        dense_layers: int = 1,
        dense_units: int = 32,
        activation: str = 'relu',
        loss: str = None,
        optimizer: str = None,
        metrics: List[str] = None,
        last_activation: str = None
    ):
        """
        Dense Model.
        Args:
            panel (Panel): Panel with data
            model_type (str): Model type (regression, classifier, multi_classifier)
            dense_layers (int): Number of dense layers
            dense_units (int): Number of neurons in each dense layer
            activation (str): Activation type of each dense layer
            loss (str): Loss name
            optimizer (str): Optimizer name
            metrics (List[str]): Metrics list
            last_activation (str): Activation type of the last layer
        Returns:
            ``DenseModel``: Constructed DenseModel
        """

        self.dense_layers = dense_layers
        self.dense_units = dense_units
        self.activation = activation

        super().__init__(x=x, y=y, model_type=model_type, loss=loss, optimizer=optimizer, metrics=metrics, last_activation=last_activation)

    def build(self):
        dense = Dense(units=self.dense_units, activation=self.activation)
        layers = [Flatten()]  # (time, features) => (time*features)
        layers += [dense for _ in range(self.dense_layers)]
        layers += [Dense(units=self.y.timesteps * len(self.x.columns), activation=self.last_activation), Reshape(self.y_train.shape[1:])]

        self.model = Sequential(layers)


class ConvModel(_BaseModel):
    def __init__(
        self,
        x,
        y,
        model_type: str,
        conv_layers: int = 1,
        conv_filters: int = 32,
        kernel_size: int = 3,
        dense_layers: int = 1,
        dense_units: int = 32,
        activation: str = 'relu',
        loss: str = None,
        optimizer: str = None,
        metrics: List[str] = None,
        last_activation: str = None
    ):
        """
        Convolution Model.
        Args:
            panel (Panel): Panel with data
            model_type (str): Model type (regression, classifier, multi_classifier)
            conv_layers (int): Number of convolution layers
            conv_filters (int): Number of convolution filters
            kernel_size (int): Kernel size of convolution layer
            dense_layers (int): Number of dense layers
            dense_units (int): Number of neurons in each dense layer
            activation (str): Activation type of each dense layer
            loss (str): Loss name
            optimizer (str): Optimizer name
            metrics (List[str]): Metrics list
            last_activation (str): Activation type of the last layer
        Returns:
            ``DenseModel``: Constructed DenseModel
        """

        self.conv_layers = conv_layers
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.dense_layers = dense_layers
        self.dense_units = dense_units
        self.activation = activation

        super().__init__(x=x, y=y, model_type=model_type, loss=loss, optimizer=optimizer, metrics=metrics, last_activation=last_activation)

    def build(self):
        if self.x.timesteps % self.kernel_size != 0:
            warnings.warn("Kernel size is not a divisor of lookback.")

        conv = Conv1D(filters=self.conv_filters, kernel_size=self.kernel_size, activation=self.activation)

        dense = Dense(units=self.dense_units, activation=self.activation)

        layers = [conv for _ in range(self.conv_layers)]
        layers += [Flatten()]
        layers += [conv for _ in range(self.conv_layers)]
        layers += [dense for _ in range(self.dense_layers)]
        layers += [Dense(units=self.y.timesteps * len(self.x.columns), activation=self.last_activation), Reshape(self.y_train.shape[1:])]

        self.model = Sequential(layers)


class LinearRegression(DenseModel):
    def __init__(self, x, y):
        super().__init__(x=x, y=y, model_type="regression", dense_layers=0)

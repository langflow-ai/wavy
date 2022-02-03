import warnings

import tensorflow as tf
from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers import Dense, Flatten, Input, Reshape, concatenate
from tensorflow.keras.layers import Conv1D, SeparableConv1D, MaxPooling1D
import pandas as pd
import numpy as np
# from .block import from_matrix
from .panel import Panel
from sklearn.metrics import mean_squared_error

from typing import List


class _ConstantKerasModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return inputs


class KerasBaseline:
    """Baseline model to predict values by using last (horizon shifted) y values."""

    def __init__(self, panel):
        self.panel = panel

        self.set_arrays()
        self.build_model()

    def fit(self, **kwargs):
        """Fit the model."""
        self.model.fit(self.x_train, self.y_train, validation_data=(self.x_val, self.y_val), **kwargs)
        # return self

    def _predict(self, type: str):
        if type == 'test':
            predicted = self.model.predict(self.x_test)
            y = self.panel.test.y
        else:
            predicted = self.model.predict(self.x_val)
            y = self.panel.val.y

        assets = self.panel.assets
        channels = self.panel.channels

        blocks = []

        for i, block_data in enumerate(predicted):
            blocks.append(from_matrix(block_data, index = y[i].index, assets=assets, channels=channels))

        return Panel(blocks)

    def predict(self):
        """Predict the test set."""
        return self._predict(type='test')

    def predict_val(self):
        """Predict the val set."""
        return self._predict(type='val')

    def evaluate(self, type: str = 'test'):
        return True

    def build_model(self):
        input = Input(shape=(1,))
        self.model = tf.roll(input, shift=1, axis=0)


    def set_arrays(self):
        if self.use_assets:
            self.x_train = self.panel.train.x.tensor4d
            self.x_val = self.panel.val.x.tensor4d
            self.x_test = self.panel.test.x.tensor4d

            self.y_train = self.panel.train.y.tensor4d
            self.y_val = self.panel.val.y.tensor4d
            self.y_test = self.panel.test.y.tensor4d

        else:
            self.x_train = self.panel.train.x.tensor3d
            self.x_val = self.panel.val.x.tensor3d
            self.x_test = self.panel.test.x.tensor3d

            self.y_train = self.panel.train.y.tensor3d
            self.y_val = self.panel.val.y.tensor3d
            self.y_test = self.panel.test.y.tensor3d



class _BaseModel:
    """Base class for panel models."""

    # TODO: Add warning when panel has nan values
    # TODO: Auto convert boolean to int

    def __init__(self, panel, model_type: str = None, loss: str = None, optimizer: str = None, metrics: List[str] = None, last_activation: str = None):

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

        self.panel = panel

        self.loss = loss or PARAMS[model_type]['loss']
        self.optimizer = optimizer or PARAMS[model_type]['optimizer']
        self.metrics = metrics or PARAMS[model_type]['metrics']
        self.last_activation = last_activation or PARAMS[model_type]['last_activation']

        self.set_arrays()
        self.build_model()
        self.compile_model()

    def fit(self, **kwargs):
        """Fit the model."""
        self.model.fit(self.x_train, self.y_train, validation_data=(self.x_val, self.y_val), **kwargs)
        # return self

    def _predict(self, type: str):
        if type == 'test':
            predicted = self.model.predict(self.x_test)
            y = self.panel.test.y
        else:
            predicted = self.model.predict(self.x_val)
            y = self.panel.val.y

        assets = self.panel.assets
        channels = self.panel.channels

        blocks = [
            from_matrix(
                block_data, index=y[i].index, assets=assets, channels=channels
            )
            for i, block_data in enumerate(predicted)
        ]

        return Panel(blocks)

    def predict(self):
        """Predict the test set."""
        return self._predict(type='test')

    def predict_val(self):
        """Predict the val set."""
        return self._predict(type='val')

    def set_arrays(self):
        self.x_train = self.panel.train.x.tensor3d
        self.x_val = self.panel.val.x.tensor3d
        self.x_test = self.panel.test.x.tensor3d

        self.y_train = self.panel.train.y.tensor3d
        self.y_val = self.panel.val.y.tensor3d
        self.y_test = self.panel.test.y.tensor3d

    def compile_model(self):
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

    def build_model(self):
        # raise NotImplementedError
        pass

    def evaluate(self, type: str = 'test'):
        if type == 'test':
            return self.model.evaluate(self.x_test, self.y_test)
        elif type == 'val':
            return self.model.evaluate(self.x_val, self.y_val)
        else:
            return self.model.evaluate(self.x_train, self.y_train)


class BaselineModel(_BaseModel):
    def __init__(
        self,
        panel,
        model_type: str,
        loss: str = None,
        metrics: List[str] = None,
    ):

        super().__init__(panel=panel, model_type=model_type, loss=loss, metrics=metrics)

    def set_arrays(self):
        self.x_train = self.panel.train.y.wshift(1).tensor3d[1:]
        self.x_val = self.panel.val.y.wshift(1).tensor3d[1:]
        self.x_test = self.panel.test.y.wshift(1).tensor3d[1:]

        self.y_train = self.panel.train.y.tensor3d[1:]
        self.y_val = self.panel.val.y.tensor3d[1:]
        self.y_test = self.panel.test.y.tensor3d[1:]

    def build_model(self):
        self.model = _ConstantKerasModel()


class DenseModel(_BaseModel):
    def __init__(
        self,
        panel,
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

        super().__init__(panel=panel, model_type=model_type, loss=loss, optimizer=optimizer, metrics=metrics, last_activation=last_activation)

    def build_model(self):
        dense = Dense(units=self.dense_units, activation=self.activation)
        layers = [Flatten()]  # (time, features) => (time*features)
        layers += [dense for _ in range(self.dense_layers)]
        layers += [Dense(units=self.panel.horizon * len(self.panel.assets) * len(self.panel.channels), activation=self.last_activation), Reshape(self.y_train.shape[1:])]

        self.model = Sequential(layers)


class ConvModel(_BaseModel):
    def __init__(
        self,
        panel,
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

        super().__init__(panel=panel, model_type=model_type, loss=loss, optimizer=optimizer, metrics=metrics, last_activation=last_activation)

    def build_model(self):
        if self.panel.lookback % self.kernel_size != 0:
            warnings.warn("Kernel size is not a divisor of lookback.")

        conv = Conv1D(filters=self.conv_filters, kernel_size=self.kernel_size, activation=self.activation)

        dense = Dense(units=self.dense_units, activation=self.activation)

        layers = [conv for _ in range(self.conv_layers)]
        layers += [Flatten()]
        layers += [conv for _ in range(self.conv_layers)]
        layers += [dense for _ in range(self.dense_layers)]
        layers += [Dense(units=self.panel.horizon * len(self.panel.assets) * len(self.panel.channels), activation=self.last_activation), Reshape(self.y_train.shape[1:])]

        self.model = Sequential(layers)


class LinearRegression(DenseModel):
    def __init__(self, panel):
        super().__init__(panel, model_type="regression", dense_layers=0)


class SeparateAssetModel(ConvModel):
    def __init__(
        self,
        panel,
        model_type: str = None,
        conv_layers: int = 1,
        conv_filters: int = 32,
        kernel_size: int = 3,
        dense_layers: int = 1,
        dense_units: int = 32,
        activation: str = 'relu',
        loss: str = None,
        optimizer: str = None,
        metrics: List[str] = None,
        last_activation: str = None,
        **kwargs
    ):
        """
        Separate Asset Model.

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

        super().__init__(panel=panel,
                         model_type=model_type,
                         conv_layers=conv_layers,
                         conv_filters=conv_filters,
                         kernel_size=kernel_size,
                         dense_layers=dense_layers,
                         dense_units=dense_units,
                         activation=activation,
                         loss=loss,
                         optimizer=optimizer,
                         metrics=metrics,
                         last_activation=last_activation,
                         )

        # self.hidden_activation = activation
        # self.last_activation = last_activation

    # def set_arrays(self):
    #     train_splits = self.panel.train.x._split_assets()
    #     val_splits = self.panel.val.x._split_assets()
    #     test_splits = self.panel.test.x._split_assets()

    #     self.x_train = [x.values for x in train_splits]
    #     self.x_val = [x.values for x in val_splits]
    #     self.x_test = [x.values for x in test_splits]

    #     # shape 3d (2, 673, 5, 2)
    #     # shape 4d (673, 2, 5, 2)

    #     self.input_info = [dict(shape=x.shape[2:], name=x.assets[0]) for x in train_splits]

    def create_hidden_layers(self, inputs):
        raise NotImplementedError


class SeparateAssetConvModel(SeparateAssetModel):
    def __init__(
        self,
        panel,
        loss: str = None,
        optimizer: str = None,
        metrics: List[str] = None,
        hidden_activation: str = None,
        last_activation: str = None,
        hidden_size: int = 10,
        filters: int = 10,
        **kwargs

    ):
        super().__init__(
            panel=panel,
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
            hidden_activation=hidden_activation,
            last_activation=last_activation,
            **kwargs
        )

        self.hidden_size = hidden_size
        self.filters = filters

    def build_asset_hidden(self, input_):

        hidden_name = "hidden." + input_.name
        flatten_name = "flatten." + input_.name
        dense_name = "dense." + input_.name

        hidden = SeparableConv1D(self.filters, self.panel.lookback, name=hidden_name,
                                 activation=self.hidden_activation)(input_)
        hidden = Flatten(name=flatten_name)(hidden)
        hidden = Dense(self.hidden_size, activation=self.hidden_activation,
                       name=dense_name)(hidden)
        return hidden

    def create_hidden_layers(self, inputs):
        return [self.build_asset_hidden(input_) for input_ in inputs]


# class SeparateAssetModel(BaseModel):
#     def __init__(
#         self,
#         panel,
#         optimizer="Adam",
#         loss="binary_crossentropy",
#         metrics=["binary_crossentropy", "AUC"],
#         hidden_size=10,
#         filters=10,
#     ):

#         self.hidden_size = hidden_size
#         self.filters = filters
#         super().__init__(panel=panel)
#         self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

#     def build_asset_hidden(self, input_):
#         # TODO: Add to hidden_size and filter
#         # M1 = 1  # Multiplier for the channel representation. Increases CONV filters.
#         # M2 = 1  # Multiplier for the asset representation before concat. Nonsense if higher than [lookback]?

#         # Convoluting on the time dimension
#         # [lookback] timesteps reduced to [filters] nodes
#         name = input_.name
#         hidden = SeparableConv1D(self.filters, self.panel.lookback, name="hidden." + name, activation="relu")(input_)
#         hidden = Flatten(name="flatten." + name)(hidden)
#         hidden = Dense(self.hidden_size, activation="relu", name="dense." + name)(hidden)
#         return hidden

#     def set_arrays(self):
#         pass

#     def build_model(self):

#         x_train_assets = self.panel.train.x.split_assets()

#         inputs, x_ = [], []
#         for side in x_train_assets:
#             inputs.append(Input(shape=side.shape[2:], name=side.assets[0]))
#             x_.append(side.values)

#         self.x_train = [side.values for side in x_train_assets]
#         self.x_val = [side.values for side in self.panel.val.x.split_assets()]
#         self.x_test = [side.values for side in self.panel.test.x.split_assets()]

#         # self.y_train = self.panel.train.y.numpy()
#         # self.y_val = self.panel.val.y.numpy()
#         # self.y_test = self.panel.test.y.numpy()

#         hidden = [self.build_asset_hidden(input_) for input_ in inputs]

#         x = concatenate(hidden)
#         x = Dense(self.panel.y.shape[1], activation="sigmoid")(x)
#         outputs = Reshape(self.panel.y.shape[1:])(x)

#         self.model = Model(inputs=inputs, outputs=outputs)

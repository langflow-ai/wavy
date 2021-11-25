import warnings

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (Conv1D, Dense, Flatten, Input, Reshape,
                                     SeparableConv1D, concatenate)
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError as MAE
from tensorflow.keras.optimizers import Adam
from tensorflow.nn import relu, sigmoid


class Baseline:
    """Baseline model to predict values by using last (horizon shifted) y values."""

    def __init__(self, panel):
        self.panel = panel

    def build_model(self):
        raise NotImplementedError

    def fit(self, **kwargs):
        raise NotImplementedError

    def predict(self):
        return self.panel.test.y.data().shift(self.panel.horizon).values


class BaseModel:
    """Base class for panel models."""

    def __init__(self, panel, loss=MeanSquaredError(), optimizer=Adam(),
    metrics=[MAE()], use_assets=False, **kwargs):

        self.panel = panel
        self.use_assets = use_assets
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

        self.set_arrays()
        self.build_model()
        self.compile_model(**kwargs)

    def fit(self, **kwargs):
        """Fit the model."""
        self.model.fit(self.x_train, self.y_train, validation_data=(self.x_val, self.y_val), **kwargs)
        return self

    def predict(self):
        """Predict the test set."""
        return self.model.predict(self.x_test)

    def set_arrays(self):
        if self.use_assets:
            self.x_train = self.panel.train.x.numpy()
            self.x_val = self.panel.val.x.numpy()
            self.x_test = self.panel.test.x.numpy()

            self.y_train = self.panel.train.y.numpy()
            self.y_val = self.panel.val.y.numpy()
            self.y_test = self.panel.test.y.numpy()

        else:
            self.x_train = self.panel.train.x.values
            self.x_val = self.panel.val.x.values
            self.x_test = self.panel.test.x.values

            self.y_train = self.panel.train.y.values
            self.y_val = self.panel.val.y.values
            self.y_test = self.panel.test.y.values

    def compile_model(self, **kwargs):
        self.model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=self.metrics,
            **kwargs
        )

    def build_model(self):
        raise NotImplementedError


class LinearModel(BaseModel):
    def __init__(self, panel):
        super().__init__(panel, use_assets=False)

    def build_model(self):
        self.model = Sequential([Flatten(),
        Dense(units=self.panel.horizon),
        Reshape(self.y_train.shape[1:])])


class DenseModel(BaseModel):
    def __init__(self, panel, dense_layers=1, dense_units=32, activation="relu"):
        self.dense_layers = dense_layers
        self.dense_units = dense_units
        self.activation = activation
        super().__init__(panel, use_assets=False)

    def build_model(self):
        dense = Dense(units=self.dense_units, activation=self.activation)

        layers = [Flatten()]  # (time, features) => (time*features)
        layers += [dense for _ in range(self.dense_layers)]
        layers += [Dense(units=self.panel.horizon), Reshape(self.y_train.shape[1:])]

        self.model = Sequential(layers)


class ConvModel(BaseModel):
    def __init__(
        self, panel, conv_layers=1, conv_filters=32, kernel_size=3, dense_layers=1, dense_units=32, activation="relu"
    ):

        self.conv_layers = conv_layers
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.dense_layers = dense_layers
        self.dense_units = dense_units
        self.activation = activation
        super().__init__(panel, use_assets=False)

    def build_model(self):
        if self.panel.lookback % self.kernel_size != 0:
            warnings.warn("Kernel size is not a divisor of lookback.")

        conv = Conv1D(filters=self.conv_filters, kernel_size=self.kernel_size, activation=self.activation)

        dense = Dense(units=self.dense_units, activation=self.activation)

        layers = [conv for _ in range(self.conv_layers)]
        layers += [Flatten()]
        layers += [conv for _ in range(self.conv_layers)]
        layers += [dense for _ in range(self.dense_layers)]
        layers += [Dense(units=self.panel.horizon), Reshape(self.y_train.shape[1:])]

        self.model = tf.keras.Sequential(layers)


class SeparateAssetModel(BaseModel):
    def __init__(
        self,
        panel,
        optimizer="Adam",
        loss="binary_crossentropy",
        metrics=["binary_crossentropy", "AUC"],
        hidden_size=10,
        filters=10,
    ):

        self.hidden_size = hidden_size
        self.filters = filters
        super().__init__(panel=panel, use_assets=True)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def build_asset_input(self, asset_side):
        input_shape = asset_side.shape[2:]
        return Input(shape=input_shape, name=asset_side.assets[0])

    def build_asset_hidden(self, input_):
        # TODO: Add to hidden_size and filter
        # M1 = 1  # Multiplier for the channel representation. Increases CONV filters.
        # M2 = 1  # Multiplier for the asset representation before concat. Nonsense if higher than [lookback]?

        # Convoluting on the time dimension
        # [lookback] timesteps reduced to [filters] nodes
        name = input_.name
        hidden = SeparableConv1D(self.filters, self.panel.lookback, name="hidden." + name, activation=relu)(input_)
        hidden = Flatten(name="flatten." + name)(hidden)
        hidden = Dense(self.hidden_size, activation=relu, name="dense." + name)(hidden)
        return hidden

    def set_arrays(self):
        pass

    def build_model(self):

        x_train_assets = self.panel.train.x.split_assets()

        inputs, x_ = [], []
        for side in x_train_assets:
            inputs.append(self.build_asset_input(side))
            x_.append(side.values)

        self.x_train = [side.values for side in x_train_assets]
        self.x_val = [side.values for side in self.panel.val.x.split_assets()]
        self.x_test = [side.values for side in self.panel.test.x.split_assets()]

        # self.y_train = self.panel.train.y.numpy()
        # self.y_val = self.panel.val.y.numpy()
        # self.y_test = self.panel.test.y.numpy()

        hidden = [self.build_asset_hidden(input_) for input_ in inputs]

        x = concatenate(hidden)
        x = Dense(self.panel.y.shape[1], activation=sigmoid)(x)
        outputs = Reshape(self.panel.y.shape[1:])(x)

        self.model = Model(inputs=inputs, outputs=outputs)

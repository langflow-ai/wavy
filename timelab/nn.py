import warnings

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Input, Reshape, concatenate
from tensorflow.keras.layers import Conv1D, SeparableConv1D, MaxPooling1D


class Baseline:
    """Baseline model to predict values by using last (horizon shifted) y values."""

    def __init__(self, panel):
        self.panel = panel

    def build_model(self):
        raise NotImplementedError

    def fit(self, **kwargs):
        raise NotImplementedError

    def predict_val(self):
        return self.panel.val.y.data().shift(self.panel.horizon).values

    def predict(self):
        return self.panel.test.y.data().shift(self.panel.horizon).values


class BaseModel:
    """Base class for panel models."""

    def __init__(self, panel, loss, optimizer, metrics, use_assets, **kwargs):

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

    def predict_val(self):
        """Predict the val set."""
        return self.model.predict(self.x_val)

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
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics, **kwargs)

    def build_model(self):
        raise NotImplementedError


class BaseRegressor:
    """Base class for regressor models."""

    def __init__(self):
        self.loss="MSE"
        self.optimizer="adam"
        self.metrics=["MAE"]
        self.last_activation = "linear"

class BaseClassifier:
    """Base class for regressor models."""

    def __init__(self):
        self.loss="binary_crossentropy"
        self.optimizer="adam"
        self.metrics=["AUC"]
        self.last_activation = "sigmoid"

class BaseMultiClassifier:
    """Base class for multi classifier models."""

    def __init__(self):
        self.loss="binary_crossentropy"
        self.optimizer="adam"
        self.metrics=["AUC"]
        self.last_activation = "softmax"

class DenseModel(BaseModel):
    def __init__(
        self, panel, dense_layers, dense_units, activation, last_activation, loss=None, optimizer=None, metrics=None, **kwargs
    ):
        self.dense_layers = dense_layers
        self.dense_units = dense_units
        self.activation = activation
        self.last_activation = last_activation

        self.loss = loss or self.loss
        self.optimizer = optimizer or self.optimizer
        self.metrics = metrics or self.metrics

        super().__init__(panel=panel, loss=loss, optimizer=optimizer, metrics=metrics, use_assets=False, **kwargs)

    def build_model(self):
        dense = Dense(units=self.dense_units, activation=self.activation)

        layers = [Flatten()]  # (time, features) => (time*features)
        layers += [dense for _ in range(self.dense_layers)]
        layers += [Dense(units=self.panel.horizon, activation=self.last_activation), Reshape(self.y_train.shape[1:])]

        self.model = Sequential(layers)

class DenseRegressor(BaseRegressor, DenseModel):
    def __init__(self, panel, dense_layers=1, dense_units=32):
        BaseRegressor.__init__(self)
        super().__init__(
            panel=panel,
            dense_layers=dense_layers,
            dense_units=dense_units,
            activation="relu",
            last_activation="linear",
        )


class DenseClassifier(DenseModel):
    def __init__(self, panel, dense_layers=1, dense_units=32):
        super().__init__(
            panel=panel,
            loss="binary_crossentropy",
            optimizer="adam",
            metrics=["AUC"],
            dense_layers=dense_layers,
            dense_units=dense_units,
            activation="relu",
            last_activation="sigmoid",
        )


class DenseMultiClassifier(DenseModel):
    def __init__(self, panel, dense_layers=1, dense_units=32):
        super().__init__(
            panel=panel,
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["AUC"],
            dense_layers=dense_layers,
            dense_units=dense_units,
            activation="relu",
            last_activation="softmax",
        )


class LinearRegressor(DenseRegressor):
    def __init__(self, panel):
        super().__init__(panel, dense_layers=0)


class ConvModel(BaseModel):
    def __init__(
        self,
        panel,
        loss,
        optimizer,
        metrics,
        conv_layers,
        conv_filters,
        kernel_size,
        dense_layers,
        dense_units,
        activation,
        last_activation,
    ):

        self.conv_layers = conv_layers
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.dense_layers = dense_layers
        self.dense_units = dense_units
        self.activation = activation
        self.last_activation = last_activation
        super().__init__(panel=panel, loss=loss, optimizer=optimizer, metrics=metrics, use_assets=False)

    def build_model(self):
        if self.panel.lookback % self.kernel_size != 0:
            warnings.warn("Kernel size is not a divisor of lookback.")

        conv = Conv1D(filters=self.conv_filters, kernel_size=self.kernel_size, activation=self.activation)

        dense = Dense(units=self.dense_units, activation=self.activation)

        layers = [conv for _ in range(self.conv_layers)]
        layers += [Flatten()]
        layers += [conv for _ in range(self.conv_layers)]
        layers += [dense for _ in range(self.dense_layers)]
        layers += [Dense(units=self.panel.horizon, activation=self.last_activation), Reshape(self.y_train.shape[1:])]

        self.model = tf.keras.Sequential(layers)


class ConvRegressor(ConvModel):
    def __init__(self, panel, conv_layers=1, conv_filters=32, kernel_size=3, dense_layers=1, dense_units=32):
        super().__init__(
            panel=panel,
            loss="MSE",
            optimizer="adam",
            metrics=["MAE"],
            conv_layers=conv_layers,
            conv_filters=conv_filters,
            kernel_size=kernel_size,
            dense_layers=dense_layers,
            dense_units=dense_units,
            activation="relu",
            last_activation="linear",
        )


class ConvClassifier(ConvModel):
    def __init__(self, panel, conv_layers=1, conv_filters=32, kernel_size=3, dense_layers=1, dense_units=32):
        super().__init__(
            panel=panel,
            loss="binary_crossentropy",
            optimizer="adam",
            metrics=["AUC"],
            conv_layers=conv_layers,
            conv_filters=conv_filters,
            kernel_size=kernel_size,
            dense_layers=dense_layers,
            dense_units=dense_units,
            activation="relu",
            last_activation="sigmoid",
        )


class ConvMultiClassifier(ConvModel):
    def __init__(self, panel, conv_layers=1, conv_filters=32, kernel_size=3, dense_layers=1, dense_units=32):
        super().__init__(
            panel=panel,
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["AUC"],
            conv_layers=conv_layers,
            conv_filters=conv_filters,
            kernel_size=kernel_size,
            dense_layers=dense_layers,
            dense_units=dense_units,
            activation="relu",
            last_activation="softmax",
        )


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
#         super().__init__(panel=panel, use_assets=True)
#         self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

#     def build_asset_input(self, asset_side):
#         input_shape = asset_side.shape[2:]
#         return Input(shape=input_shape, name=asset_side.assets[0])

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
#             inputs.append(self.build_asset_input(side))
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

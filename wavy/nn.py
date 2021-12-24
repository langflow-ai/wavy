import warnings

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Input, Reshape, concatenate
from tensorflow.keras.layers import Conv1D, SeparableConv1D, MaxPooling1D
import pandas as pd
import numpy as np

class Baseline:
    """Baseline model to predict values by using last (horizon shifted) y values."""

    def __init__(self, panel):
        self.panel = panel

    def build_model(self):
        raise NotImplementedError

    def fit(self, **kwargs):
        raise NotImplementedError

    def _predict(self, type):
        columns = pd.MultiIndex.from_product([self.panel.assets, self.panel.channels])
        if type == 'test':
            predicted = np.squeeze(self.panel.test.y.as_dataframe().shift(self.panel.horizon).values)
        else:
            predicted = np.squeeze(self.panel.val.y.as_dataframe().shift(self.panel.horizon).values)
        timesteps = self.panel.test.y.timesteps
        if len(predicted.shape) > 1:
            dataframe = pd.DataFrame(predicted, columns=columns, index=timesteps)
        else:
            dataframe = pd.Series(predicted, index=timesteps)
        return dataframe[self.panel.horizon:]

    def predict_val(self):
        return self._predict('val')

    def predict(self):
        return self._predict('test')


class _BaseModel:
    """Base class for panel models."""

    def __init__(self, panel, model_type=None, use_assets=False, loss=None, optimizer=None, metrics=None, last_activation=None):

        PARAMS = {
            'regression': {
                'loss': "MSE",
                'optimizer': "adam",
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
        self.use_assets = use_assets

        # if model_type:
        self.loss = loss if loss else PARAMS[model_type]['loss']
        self.optimizer = optimizer if optimizer else PARAMS[model_type]['optimizer']
        self.metrics = metrics if metrics else PARAMS[model_type]['metrics']
        self.last_activation = last_activation if last_activation else PARAMS[model_type]['last_activation']

        self.set_arrays()
        self.build_model()
        self.compile_model()

    def fit(self, **kwargs):
        """Fit the model."""
        self.model.fit(self.x_train, self.y_train, validation_data=(self.x_val, self.y_val), **kwargs)
        # return self

    def predict(self):
        """Predict the test set."""
        return self.model.predict(self.x_test)

    def predict_val(self):
        """Predict the val set."""
        return self.model.predict(self.x_val)

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
            # TODO check squeeze

    def compile_model(self):
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

    def build_model(self):
        # raise NotImplementedError
        pass


class DenseModel(_BaseModel):
    def __init__(
        self,
        panel,
        model_type,
        dense_layers=1,
        dense_units=32,
        activation="relu",
        loss=None,
        optimizer=None,
        metrics=None,
        last_activation=None
    ):

        self.dense_layers = dense_layers
        self.dense_units = dense_units
        self.activation = activation

        super().__init__(panel=panel, model_type=model_type, loss=loss, optimizer=optimizer, metrics=metrics, last_activation=last_activation, use_assets=False)

    def build_model(self):
        dense = Dense(units=self.dense_units, activation=self.activation)
        layers = [Flatten()]  # (time, features) => (time*features)
        layers += [dense for _ in range(self.dense_layers)]
        layers += [Dense(units=self.panel.horizon * len(self.panel.assets) * len(self.panel.channels), activation=self.last_activation), Reshape(self.y_train.shape[1:])]

        self.model = Sequential(layers)

# END SHAPE - (horizon x assets x channels)

class ConvModel(_BaseModel):
    def __init__(
        self,
        panel,
        model_type,
        conv_layers=1,
        conv_filters=32,
        kernel_size=3,
        dense_layers=1,
        dense_units=32,
        activation="relu",
        loss=None,
        optimizer=None,
        metrics=None,
        last_activation=None,
    ):

        self.conv_layers = conv_layers
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.dense_layers = dense_layers
        self.dense_units = dense_units
        self.activation = activation

        super().__init__(panel=panel, model_type=model_type, loss=loss, optimizer=optimizer, metrics=metrics, last_activation=last_activation, use_assets=False)

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

        self.model = tf.keras.Sequential(layers)

class LinearRegression(DenseModel):
    def __init__(self, panel):
        super().__init__(panel, model_type="regression", dense_layers=0)


class SeparateAssetModel(_BaseModel):
    def __init__(
        self,
        panel,
        loss,
        optimizer,
        metrics,
        hidden_activation,
        last_activation,
        use_assets=True,
        **kwargs
    ):
        super().__init__(panel=panel, loss=loss, optimizer=optimizer, metrics=metrics, use_assets=use_assets, **kwargs)

        self.hidden_activation = hidden_activation
        self.last_activation = last_activation

    def set_arrays(self):
        train_splits = self.panel.train.x.split_assets()
        val_splits = self.panel.val.x.split_assets()
        test_splits = self.panel.test.x.split_assets()

        self.x_train = [x.values for x in train_splits]
        self.x_val = [x.values for x in val_splits]
        self.x_test = [x.values for x in test_splits]

        self.input_info = [dict(shape=x.shape[2:], name=x.assets[0]) for x in train_splits]

    def create_hidden_layers(self, inputs):
        raise NotImplementedError

class SeparateAssetConvModel(SeparateAssetModel):
    def __init__(
        self,
        panel,
        loss,
        optimizer,
        metrics,
        hidden_activation,
        last_activation,
        hidden_size=10,
        filters=10,
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
#         super().__init__(panel=panel, use_assets=True)
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
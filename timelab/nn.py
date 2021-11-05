import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Input,
    Reshape,
    SeparableConv1D,
    concatenate,
)

from tensorflow.nn import relu, sigmoid

class BaseModel:
    def __init__(self, panel):
        self.panel = panel
        self.set_arrays()
        self.build_model()

    def fit(self, **kwargs):
        self.model.fit(self.x_train, self.y_train,
                       validation_data=(self.x_val, self.y_val),
                       **kwargs)
        return self

    def predict(self):
        return self.model.predict(self.x_test)

    def set_arrays(self):
        self.x_train = self.panel.train.x.numpy()
        self.x_val = self.panel.val.x.numpy()
        self.x_test = self.panel.test.x.numpy()

        self.y_train = self.panel.train.y.numpy()
        self.y_val = self.panel.val.y.numpy()
        self.y_test = self.panel.test.y.numpy()

    def build_model(self):
        raise NotImplementedError


class Baseline(BaseModel):
    def __init__(self, panel):
        super().__init__(panel)

    def build_model(self):
        pass

    def fit(self, **kwargs):
        pass

    def predict(self):
        empty = np.zeros(self.panel.y.numpy().shape)[0:self.panel.horizon]
        empty[empty == 0] = np.nan
        return np.concatenate([empty, self.panel.y.numpy()])[:-self.panel.horizon]

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
        super().__init__(panel=panel)
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

        self.y_train = self.panel.train.y.numpy()
        self.y_val = self.panel.val.y.numpy()
        self.y_test = self.panel.test.y.numpy()

        hidden = [self.build_asset_hidden(input_) for input_ in inputs]

        x = concatenate(hidden)
        x = Dense(self.panel.y.shape[1], activation=sigmoid)(x)
        outputs = Reshape(self.panel.y.shape[1:])(x)

        self.model = Model(inputs=inputs, outputs=outputs)

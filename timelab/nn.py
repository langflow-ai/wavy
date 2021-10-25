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

from .panel import TimePanel
from .utils import smash_array


class SeparateAssetModel:
    def __init__(self, panel, hidden_size=10, filters=10):
        self.model = self.build_model(panel, hidden_size, filters)
        self.panel = panel

    def fit(self, train: TimePanel = None, val: TimePanel = None, **kwargs):
        if not train:
            train = self.panel.train
        if not val:
            val = self.panel.val

        if train is None:
            raise ValueError("Train panel must not be None. Try set panel training split before fitting.")

        X_train = [asset_train.values for asset_train in train.x.split_assets()]
        y_train = train.y.numpy()

        X_val = [asset_val.values for asset_val in val.x.split_assets()]
        y_val = val.y.numpy()

        self.model.fit(X_train, y_train, validation_data=(X_val, y_val), **kwargs)
        return self

    def predict(self, test: TimePanel = None):
        if not test:
            test = self.panel.test
        X_test = [asset_test.values for asset_test in test.x.split_assets()]
        return self.model.predict(X_test)

    def build_asset_input(self, asset_side):
        assert len(asset_side.assets) == 1
        input_shape = asset_side.shape[2:]
        print(input_shape)
        return Input(shape=input_shape, name=asset_side.assets[0])

    def build_asset_hidden(self, input_, lookback, hidden_size, filters):
        # TODO: Add to hidden_size and filter
        # M1 = 1  # Multiplier to for the channel representation. Increases CONV filters.
        # M2 = 1  # Multiplier to for the asset representation before concat. Nonsense if higher than [lookback]?

        # Convoluting on the time dimension
        # [lookback] timesteps reduced to [filters] nodes
        name = input_.name
        hidden = SeparableConv1D(filters, lookback, name="hidden." + name, activation=relu)(input_)
        hidden = Flatten(name="flatten." + name)(hidden)
        hidden = Dense(hidden_size, activation=relu, name="dense." + name)(hidden)
        return hidden

    def build_model(self, panel, hidden_size, filters):
        inputs = [self.build_asset_input(asset_side) for asset_side in panel.x.split_assets()]
        hidden = [self.build_asset_hidden(input_, panel.lookback, hidden_size, filters) for input_ in inputs]
        x = concatenate(hidden)
        x = Dense(panel.y.shape[1], activation=sigmoid)(x)
        outputs = Reshape(panel.y.shape[1:])(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["binary_crossentropy", "AUC"])
        return model

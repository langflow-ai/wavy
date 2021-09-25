from tensorflow.keras.layers import SeparableConv1D, Flatten, Dense, Reshape, Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras import Model
from tensorflow.nn import relu, sigmoid

from .panel import TimePanel
from .utils import smash_array


class SeparateUnitModel:

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

        X_train = [smash_array(unit_train.X) for unit_train in train.split_units()]
        y_train = train.y

        X_val = [smash_array(unit_val.X) for unit_val in val.split_units()]
        y_val = val.y

        self.model.fit(X_train, y_train, validation_data=(X_val, y_val), **kwargs)
        return self

    def predict(self, test: TimePanel = None):
        if not test:
            test = self.panel.test
        X_test = [smash_array(unit_test.X) for unit_test in test.split_units()]
        return self.model.predict(X_test)

    def build_unit_input(self, unit_panel):
        assert len(unit_panel.xunits) == 1
        input_shape = unit_panel.X.shape[2:]
        input_ = Input(shape=input_shape, name=unit_panel.xunits[0])
        return input_

    def build_unit_hidden(self, input_, lookback, hidden_size, filters):
        # TODO: Add to hidden_size and filter
        # M1 = 1  # Multiplier to for the channel representation. Increases CONV filters.
        # M2 = 1  # Multiplier to for the unit representation before concat. Nonsense if higher than [lookback]?

        # Convoluting on the time dimension
        # [lookback] timesteps reduced to [filters] nodes
        name = input_.name
        hidden = SeparableConv1D(
            filters, lookback, name='hidden.' + name, activation=relu)(input_)
        hidden = Flatten(name='flatten.' + name)(hidden)
        hidden = Dense(hidden_size, activation=relu,
                       name='dense.' + name)(hidden)
        return hidden

    def build_model(self, panel, hidden_size, filters):
        inputs = [self.build_unit_input(unit_panel)
                  for unit_panel in panel.split_units()]
        hidden = [self.build_unit_hidden(input_, panel.lookback, hidden_size, filters)
                  for input_ in inputs]
        x = concatenate(hidden)
        x = Dense(panel.y.shape[1], activation=sigmoid)(x)
        outputs = Reshape(panel.y.shape[1:])(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="Adam", loss="binary_crossentropy",
                      metrics=["binary_crossentropy", 'AUC'])
        return model

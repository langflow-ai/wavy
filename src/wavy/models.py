import warnings
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.base import is_classifier
from sklearn.metrics import auc, roc_curve
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Reshape

from .panel import Panel

# ? Maybe we get rid of model_type and add e.g. DenseRegressor / DenseClassifier.


class _ConstantKerasModel(tf.keras.Model):
    """A Keras model that returns the input values as outputs."""

    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return inputs


class _BaseModel:
    """Base class for panel models."""

    def __init__(
        self,
        x,
        y,
        model_type: str = None,
        loss: str = None,
        optimizer: str = None,
        metrics: List[str] = None,
        last_activation: str = None,
    ):

        PARAMS = {
            "regression": {
                "loss": "MSE",
                "optimizer": "adam",
                "metrics": ["mae"],
                "last_activation": "linear",
            },
            "classification": {
                "loss": "binary_crossentropy",
                "optimizer": "adam",
                "metrics": ["AUC", "accuracy"],
                "last_activation": "sigmoid",
            },
            "multi_classification": {
                "loss": "categorical_crossentropy",
                "optimizer": "adam",
                "metrics": ["AUC", "accuracy"],
                "last_activation": "softmax",
            },
        }

        # Raise error when panel has nan values
        if x.findna_frames().any():
            raise ValueError("Panel x has NaN values.")
        if y.findna_frames().any():
            raise ValueError("Panel y has NaN values.")

        # Convert boolean in x and y to int
        for col in x.columns:
            if x[col].dtype == bool:
                x[col] = x[col].astype(int)
        for col in y.columns:
            if y[col].dtype == bool:
                y[col] = y[col].astype(int)

        # Raise error if column is not numeric
        for sample in [x, y]:
            for col in sample.columns:
                if sample[col].dtype not in [np.float64, np.int64]:
                    raise ValueError(f"Column {col} is not numeric.")

        self.x = x
        self.y = y

        self.model_type = model_type
        self.loss = loss or PARAMS[model_type]["loss"]
        self.optimizer = optimizer or PARAMS[model_type]["optimizer"]
        self.metrics = metrics or PARAMS[model_type]["metrics"]
        self.last_activation = last_activation or PARAMS[model_type]["last_activation"]

        self.set_arrays()
        self.build()
        self.compile()
        self.model._name = self.__class__.__name__

    def set_arrays(self):
        """Set the arrays."""
        self.x_train = self.x.train.values_panel
        self.x_val = self.x.val.values_panel
        self.x_test = self.x.test.values_panel

        self.y_train = self.y.train.values_panel.squeeze(axis=2)
        self.y_val = self.y.val.values_panel.squeeze(axis=2)
        self.y_test = self.y.test.values_panel.squeeze(axis=2)

    def get_auc(self):
        """Get the AUC score."""
        y = self.y_test.squeeze()
        prediction = self.model.predict(self.x_test).squeeze()
        fpr, tpr, _ = roc_curve(y, prediction)
        fpr, tpr, _ = roc_curve(y, prediction)
        return auc(fpr, tpr)

    def fit(self, **kwargs):
        """Fit the model.

        Args:
            **kwargs: Additional arguments to pass to the fit method.
        """
        self.model.fit(
            self.x_train,
            self.y_train,
            validation_data=(self.x_val, self.y_val),
            **kwargs,
        )

    def compile(self, **kwargs):
        """Compile the model.

        Args:
            **kwargs: Additional arguments to pass to the compile method.
        """
        self.model.compile(
            loss=self.loss, optimizer=self.optimizer, metrics=self.metrics, **kwargs
        )

    def build(self):
        """Build the model."""
        pass

    def predict_proba(self, data: Panel = None, **kwargs):
        """Predict probabilities.

        Args:
            data: Panel of data to predict.
            **kwargs: Additional arguments to pass to the predict method.

        Returns:
            Panel of predicted probabilities.
        """

        if data is not None:
            x = data.values_panel
            index = pd.MultiIndex.from_arrays([data.ids, data.get_timesteps(0)])
        else:
            x = np.concatenate([self.x_train, self.x_val, self.x_test], axis=0)
            index = pd.MultiIndex.from_tuples(
                np.concatenate(
                    [self.y.train.index, self.y.val.index, self.y.test.index]
                ),
                names=self.y.index.names,
            )

        return Panel(
            self.model.predict(x),
            columns=self.y.columns,
            index=index,
        )

    def predict(self, data: Panel = None, **kwargs):
        """Predict.

        Args:
            data: Panel of data to predict.
            **kwargs: Additional arguments to pass to the predict method.

        Returns:
            Panel of predicted values.
        """

        threshold = self.get_auc() if self.model_type == "classification" else None

        panel = self.predict_proba(data=data, **kwargs)

        return (
            panel if threshold is None else panel.apply(lambda x: (x > threshold) + 0)
        )

    def score(self, on=None, **kwargs):
        """Score the model.

        Args:
            on: Columns to score on.
            **kwargs: Additional arguments to pass to the score method.

        Returns:
            Panel of scores.
        """
        on = [on] if on else ["train", "val", "test"]

        dic = {}
        if "train" in on:
            dic["train"] = self.model.evaluate(
                self.x_train, self.y_train, verbose=0, **kwargs
            )
        if "test" in on:
            dic["test"] = self.model.evaluate(
                self.x_test, self.y_test, verbose=0, **kwargs
            )
        if "val" in on:
            dic["val"] = self.model.evaluate(
                self.x_val, self.y_val, verbose=0, **kwargs
            )

        indexes = [
            self.model.metrics_names.index(metric.lower()) for metric in self.metrics
        ]

        return pd.DataFrame(
            {key: [value[index] for index in indexes] for key, value in dic.items()},
            index=self.metrics,
        )

    def residuals(self):
        """Residuals.

        Returns:
            Panel of residuals.
        """
        return self.predict() - self.y


class _Baseline(_BaseModel):
    def __init__(
        self,
        x,
        y,
        model_type: str,
        loss: str = None,
        metrics: List[str] = None,
    ):

        super().__init__(x=x, y=y, model_type=model_type, loss=loss, metrics=metrics)

    def build(self):
        """Build the model."""
        self.model = _ConstantKerasModel()


class BaselineShift(_Baseline):
    # ! Maybe shift should be y.horizon by default, to avoid leakage
    # TODO test with different gap and horizon values

    def __init__(
        self,
        x,
        y,
        model_type: str,
        loss: str = None,
        metrics: List[str] = None,
        fillna=0,
        shift=1,
    ):

        self.fillna = fillna
        self.shift = shift
        super().__init__(x=x, y=y, model_type=model_type, loss=loss, metrics=metrics)

    def set_arrays(self):
        """Set the arrays."""
        self.x_train = self.y.train.shift_panel(self.shift).fillna(self.fillna).values
        self.x_val = self.y.val.shift_panel(self.shift).fillna(self.fillna).values
        self.x_test = self.y.test.shift_panel(self.shift).fillna(self.fillna).values

        self.y_train = self.y.train.values
        self.y_val = self.y.val.values
        self.y_test = self.y.test.values

    def build(self):
        """Build the model."""
        self.model = _ConstantKerasModel()


class BaselineConstant(_Baseline):
    # TODO BUG: Not working when model_type="classification"
    def __init__(
        self,
        x,
        y,
        model_type: str,
        loss: str = None,
        metrics: List[str] = None,
        constant: float = 0,
    ):

        self.constant = constant if model_type == "regression" else int(constant)
        super().__init__(x=x, y=y, model_type=model_type, loss=loss, metrics=metrics)

    def set_arrays(self):
        """Set the arrays."""
        self.x_train = np.full(self.y.train.shape, self.constant)
        self.x_val = np.full(self.y.val.shape, self.constant)
        self.x_test = np.full(self.y.test.shape, self.constant)

        self.y_train = self.y.train.values
        self.y_val = self.y.val.values
        self.y_test = self.y.test.values


class DenseModel(_BaseModel):
    def __init__(
        self,
        x,
        y,
        model_type: str,
        dense_layers: int = 1,
        dense_units: int = 32,
        activation: str = "relu",
        loss: str = None,
        optimizer: str = None,
        metrics: List[str] = None,
        last_activation: str = None,
    ):
        """
        Dense Model.

        Args:
            panel (Panel): Panel with data
            model_type (str): Model type (regression, classification, multi_classification)
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

        super().__init__(
            x=x,
            y=y,
            model_type=model_type,
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
            last_activation=last_activation,
        )

    def build(self):
        """Build the model."""
        dense = Dense(units=self.dense_units, activation=self.activation)
        layers = [Flatten()]  # (time, features) => (time*features)
        layers += [dense for _ in range(self.dense_layers)]
        layers += [
            Dense(
                units=self.y.num_timesteps * self.y.num_columns,
                activation=self.last_activation,
            ),
            Reshape((self.y.num_columns,)),
        ]

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
        activation: str = "relu",
        loss: str = None,
        optimizer: str = None,
        metrics: List[str] = None,
        last_activation: str = None,
    ):
        """
        Convolution Model.

        Args:
            panel (Panel): Panel with data
            model_type (str): Model type (regression, classification, multi_classification)
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

        if x.shape_panel[1] < kernel_size:
            raise ValueError(
                f"Lookback ({x.shape_panel[1]}) must be greater or equal to kernel_size ({kernel_size})"
            )

        self.conv_layers = conv_layers
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.dense_layers = dense_layers
        self.dense_units = dense_units
        self.activation = activation

        super().__init__(
            x=x,
            y=y,
            model_type=model_type,
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
            last_activation=last_activation,
        )

    def build(self):
        """Build the model."""

        if self.x.num_timesteps % self.kernel_size != 0:
            warnings.warn("Kernel size is not a divisor of lookback.")

        conv = Conv1D(
            filters=self.conv_filters,
            kernel_size=self.kernel_size,
            activation=self.activation,
        )

        dense = Dense(units=self.dense_units, activation=self.activation)

        layers = [conv for _ in range(self.conv_layers)]
        layers += [Flatten()]
        layers += [conv for _ in range(self.conv_layers)]
        layers += [dense for _ in range(self.dense_layers)]
        layers += [
            Dense(
                units=self.y.num_timesteps * self.y.num_columns,
                activation=self.last_activation,
            ),
            Reshape((self.y.num_columns,)),
        ]

        self.model = Sequential(layers)


class LinearRegression(DenseModel):
    def __init__(self, x, y, **kwargs):
        super().__init__(x=x, y=y, model_type="regression", dense_layers=0, **kwargs)


class LogisticRegression(DenseModel):
    def __init__(self, x, y, **kwargs):
        super().__init__(
            x=x, y=y, model_type="classification", dense_layers=0, **kwargs
        )


class ShallowModel:
    def __init__(self, x, y, model, metrics, **kwargs):
        """Shallow Model.

        Args:
            x (Panel): Panel with data
            y (Panel): Panel with data
            model (str): Model (regression, classification, multi_classification)
            metrics (List[str]): Metrics list
            **kwargs: Additional arguments

        Returns:
            ``ShallowModel``: Constructed ShallowModel
        """

        self.x = x
        self.y = y

        if len(self.y.columns) > 1:
            raise ValueError("ShallowModel can only be used for single-output models.")

        self.model = model(**kwargs)
        self.metrics = metrics
        self.set_arrays()

    def set_arrays(self):
        """
        Sets arrays for training, testing, and validation.
        """

        self.x_train = self.x.train.flatten_panel
        self.y_train = self.y.train.values

        self.x_val = self.x.val.flatten_panel
        self.y_val = self.y.val.values

        self.x_test = self.x.test.flatten_panel
        self.y_test = self.y.test.values

    def fit(self, **kwargs):
        """Fit the model.

        Args:
            **kwargs: Keyword arguments for the fit method of the model.

        Returns:
            ``ShallowModel``: The fitted model.
        """
        return self.model.fit(X=self.x_train, y=self.y_train, **kwargs)

    def get_auc(self):
        """Get the AUC score."""

        y = self.y_test.squeeze()
        prediction = self.model.predict(self.x_test).squeeze()
        fpr, tpr, _ = roc_curve(y, prediction)
        return auc(fpr, tpr)

    def predict_proba(self, data: Panel = None):
        """Predict probabilities.

        Args:
            data (Panel): Panel with data

        Returns:
            ``ShallowModel``: The predicted probabilities.
        """

        if data is not None:
            x = data.flatten_panel
            index = pd.MultiIndex.from_arrays([data.ids, data.first_timestamp])
        else:
            x = np.concatenate([self.x_train, self.x_val, self.x_test], axis=0)
            index = pd.MultiIndex.from_tuples(
                np.concatenate(
                    [self.y.train.index, self.y.val.index, self.y.test.index]
                ),
                names=self.y.index.names,
            )

        output = self.model.predict_proba(x)

        if output.shape[1] == 2:
            output = output[:, 1]

            return Panel(
                output,
                columns=self.y.columns,
                index=index,
            )

        return Panel(
            output,
            columns=[f"{i}_prob" for i in range(output.shape[1])],
            index=index,
        )

    def predict(self, data: Panel = None):
        """Predict on data.

        Args:
            data (Panel, optional): Data to predict on. Defaults to None.

        Returns:
            Panel: Predicted data
        """
        if is_classifier(self.model):
            threshold = self.get_auc()
            panel = self.predict_proba(data)
            return panel.apply(lambda x: (x > threshold) + 0)

        else:
            if data is not None:
                x = data.flatten_panel
                index = pd.MultiIndex.from_arrays([data.ids, data.first_timestamp])
            else:
                x = np.concatenate([self.x_train, self.x_val, self.x_test], axis=0)
                index = pd.MultiIndex.from_tuples(
                    np.concatenate(
                        [self.y.train.index, self.y.val.index, self.y.test.index]
                    ),
                    names=self.y.index.names,
                )

            return Panel(
                self.model.predict(x),
                columns=self.y.columns,
                index=index,
            )

    def score(self, on=None):
        """Score the model.

        Args:
            on (str): Data to use for scoring

        Returns:
            pd.Series: Score
        """
        on = [on] if on else ["train", "val", "test"]

        dic = {}

        if "train" in on:
            metrics_dict = {
                a.__name__: a(
                    self.y.train.values.squeeze(),
                    self.predict(self.x.train).values.squeeze(),
                )
                for a in self.metrics
            }

            dic["train"] = metrics_dict
        if "test" in on:
            metrics_dict = {
                a.__name__: a(
                    self.y.test.values.squeeze(),
                    self.predict(self.x.test).values.squeeze(),
                )
                for a in self.metrics
            }

            dic["test"] = metrics_dict
        if "val" in on:
            metrics_dict = {
                a.__name__: a(
                    self.y.val.values.squeeze(),
                    self.predict(self.x.val).values.squeeze(),
                )
                for a in self.metrics
            }

            dic["val"] = metrics_dict
        return pd.DataFrame(dic, index=[a.__name__ for a in self.metrics])

    def residuals(self):
        """Residuals.

        Returns:
            Panel: Residuals
        """
        return self.predict() - self.y


def compute_score_per_model(*models, on="val"):
    # BUG
    """
    Compute score per model

    Args:
        *models: Models to score
        on (str, optional): Data to use for scoring. Defaults to "val".

    Returns:
        pd.DataFrame: Scores
    """

    return pd.DataFrame(
        [model.score(on=on) for model in models],
        index=[model.model.name for model in models],
    )


def compute_default_scores(x, y, model_type, epochs=10, verbose=0, **kwargs):
    # BUG
    """
    Compute default scores for a model.

    Args:
        x (Panel): X data
        y (Panel): Y data
        model_type (str): Model type
        epochs (int, optional): Number of epochs. Defaults to 10.
        verbose (int, optional): Verbosity. Defaults to 0.
        **kwargs: Keyword arguments for the model.

    Returns:
        pd.DataFrame: Scores
    """
    models = [BaselineConstant, BaselineShift, DenseModel]
    models = [model(x=x, y=y, model_type=model_type) for model in models]
    for model in models:
        model.fit(epochs=epochs, verbose=verbose, **kwargs)
    return compute_score_per_model(*models)

import warnings
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import auc, mean_squared_error, roc_curve
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Flatten,
    Input,
    MaxPooling1D,
    Reshape,
    SeparableConv1D,
    concatenate,
)

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
                "metrics": ["MAE"],
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
        if x.findna():
            raise ValueError("Panel x has NaN values.")
        if y.findna():
            raise ValueError("Panel x has NaN values.")

        # Convert boolean in x and y to int
        for col in x[0].columns:
            if x[0][col].dtype == bool:
                for i in range(len(x)):
                    x[i][col] = x[i][col].astype(int)
        for col in y[0].columns:
            if y[0][col].dtype == bool:
                for i in range(len(y)):
                    y[i][col] = y[i][col].astype(int)

        # Raise error if column is not numeric
        for sample in [x[0], y[0]]:
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
        self.x_train = self.x.train.values
        self.x_val = self.x.val.values
        self.x_test = self.x.test.values

        self.y_train = self.y.train.values
        self.y_val = self.y.val.values
        self.y_test = self.y.test.values

    def get_roc(self):
        y = self.y_test.squeeze()
        prediction = self.model.predict(self.x_test).squeeze()

        # prediction = self.predict(data=self.x_test).values.squeeze()

        fpr, tpr, thresholds = roc_curve(y, prediction)
        # roc_auc = auc(fpr, tpr)
        # print("ROC_AUC Score : ",roc_auc)
        # print("Function for ROC_AUC Score : ",roc_auc_score(y_test, predictions)) # Function present
        optimal_idx = np.argmax(tpr - fpr)
        return thresholds[optimal_idx]
        # print("Threshold value is:", optimal_threshold)
        # plot_roc_curve(fpr, tpr)

    def fit(self, **kwargs):
        """Fit the model."""
        return self.model.fit(
            self.x_train,
            self.y_train,
            validation_data=(self.x_val, self.y_val),
            **kwargs,
        )

    def compile(self, **kwargs):
        self.model.compile(
            loss=self.loss, optimizer=self.optimizer, metrics=self.metrics, **kwargs
        )

    def build(self):
        pass

    def predict(self, data: Panel = None, **kwargs):

        threshold = self.get_roc() if self.model_type == "classification" else None

        if data is not None:
            panel = Panel(
                [
                    pd.DataFrame(b, columns=self.y[0].columns)
                    for b in self.model.predict(data.values, **kwargs)
                ]
            )

            return (
                panel
                if threshold is None
                else panel.apply(lambda x: (x > threshold) + 0)
            )

        pred_train = [
            pd.DataFrame(a, columns=self.y[0].columns, index=b.index)
            for a, b in zip(self.model.predict(self.x_train, **kwargs), self.y.train)
        ]
        pred_val = [
            pd.DataFrame(a, columns=self.y[0].columns, index=b.index)
            for a, b in zip(self.model.predict(self.x_val, **kwargs), self.y.val)
        ]
        pred_test = [
            pd.DataFrame(a, columns=self.y[0].columns, index=b.index)
            for a, b in zip(self.model.predict(self.x_test, **kwargs), self.y.test)
        ]

        panel = Panel(pred_train + pred_val + pred_test)
        return (
            panel if threshold is None else panel.apply(lambda x: (x > threshold) + 0)
        )

    def score(self, on=None, **kwargs):
        on = [on] if on else ["train", "val", "test"]

        metrics_names = self.model.metrics_names

        dic = {}
        if "train" in on:
            train_metrics = self.model.evaluate(
                self.x_train, self.y_train, verbose=0, **kwargs
            )
            dic["train"] = train_metrics
        if "test" in on:
            test_metrics = self.model.evaluate(
                self.x_test, self.y_test, verbose=0, **kwargs
            )
            dic["test"] = test_metrics
        if "val" in on:
            val_metrics = self.model.evaluate(
                self.x_val, self.y_val, verbose=0, **kwargs
            )
            dic["val"] = val_metrics

        return pd.DataFrame(
            dic,
            index=metrics_names,
        )

    def residuals(self):
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
        self.model = _ConstantKerasModel()


class BaselineShift(_Baseline):
    # ! VERY SLOW
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
        # TODO: Replace as_dataframe w/ quicker function
        self.x_train = (
            self.y.train.as_dataframe().shift(self.shift).fillna(self.fillna).values
        )
        self.x_val = (
            self.y.val.as_dataframe().shift(self.shift).fillna(self.fillna).values
        )
        self.x_test = (
            self.y.test.as_dataframe().shift(self.shift).fillna(self.fillna).values
        )

        self.y_train = self.y.train.as_dataframe().values
        self.y_val = self.y.val.as_dataframe().values
        self.y_test = self.y.test.as_dataframe().values

    def build(self):
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
        dense = Dense(units=self.dense_units, activation=self.activation)
        layers = [Flatten()]  # (time, features) => (time*features)
        layers += [dense for _ in range(self.dense_layers)]
        layers += [
            Dense(
                units=self.y.timesteps * len(self.y.columns),
                activation=self.last_activation,
            ),
            Reshape(self.y_train.shape[1:]),
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

        if x.shape[1] < kernel_size:
            raise ValueError(
                f"Lookback ({x.shape[1]}) must be greater or equal to kernel_size ({kernel_size})"
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
        if self.x.timesteps % self.kernel_size != 0:
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
                units=self.y.timesteps * len(self.y.columns),
                activation=self.last_activation,
            ),
            Reshape(self.y_train.shape[1:]),
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

        if x.shape[1] < kernel_size:
            raise ValueError(
                f"Lookback ({x.shape[1]}) must be greater or equal to kernel_size ({kernel_size})"
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
        if self.x.timesteps % self.kernel_size != 0:
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
                units=self.y.timesteps * len(self.y.columns),
                activation=self.last_activation,
            ),
            Reshape(self.y_train.shape[1:]),
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
        # TODO: Fix shape issues
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

        self.x_train = self.x.train.as_dataframe(flatten=True)
        self.y_train = self.y.train.as_dataframe(flatten=True)

        self.x_val = self.x.val.as_dataframe(flatten=True)
        self.y_val = self.y.val.as_dataframe(flatten=True)

        self.x_test = self.x.test.as_dataframe(flatten=True)
        self.y_test = self.y.test.as_dataframe(flatten=True)

    def fit(self, **kwargs):
        """Fit the model.

        Args:
            **kwargs: Keyword arguments for the fit method of the model.

        Returns:
            ``ShallowModel``: The fitted model.
        """
        return self.model.fit(X=self.x_train, y=self.y_train, **kwargs)

    def predict(self, data: Panel = None):
        """Predict on data.

        Args:
            data (Panel, optional): Data to predict on. Defaults to None.

        Returns:
            Panel: Predicted data
        """

        if data is not None:
            return Panel(
                [
                    pd.DataFrame(b, columns=self.y[0].columns)
                    for b in self.model.predict(data.values)
                ]
            )
        pred_train = [
            pd.DataFrame(a, columns=self.y[0].columns, index=b.index)
            for a, b in zip(self.model.predict(self.x_train), self.y.train)
        ]
        pred_val = [
            pd.DataFrame(a, columns=self.y[0].columns, index=b.index)
            for a, b in zip(self.model.predict(self.x_val), self.y.val)
        ]
        pred_test = [
            pd.DataFrame(a, columns=self.y[0].columns, index=b.index)
            for a, b in zip(self.model.predict(self.x_test), self.y.test)
        ]

        return Panel(pred_train + pred_val + pred_test)

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
            metrics_dict = {}
            for a in self.metrics:
                metrics_dict[a.__name__] = a(
                    self.y_train.values.squeeze(),
                    self.predict(self.x_train).values.squeeze(),
                )
            dic["train"] = metrics_dict
        if "test" in on:
            metrics_dict = {}
            for a in self.metrics:
                metrics_dict[a.__name__] = a(
                    self.y_test.values.squeeze(),
                    self.predict(self.x_test).values.squeeze(),
                )
            dic["test"] = metrics_dict
        if "val" in on:
            metrics_dict = {}
            for a in self.metrics:
                metrics_dict[a.__name__] = a(
                    self.y_val.values.squeeze(),
                    self.predict(self.x_val).values.squeeze(),
                )
            dic["val"] = metrics_dict
        return pd.DataFrame(dic, index=[a.__name__ for a in self.metrics])


def compute_score_per_model(*models, on="val"):
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
    models = [BaselineShift, DenseModel, ConvModel]
    models = [model(x=x, y=y, model_type=model_type) for model in models]
    for model in models:
        model.fit(epochs=epochs, verbose=verbose, **kwargs)
    return compute_score_per_model(*models)

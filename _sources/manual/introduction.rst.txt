Introduction
============

`wavy` is a library for time series analysis in Python. It is a wrapper around the `pandas` package.

The goal is to provide a simple and intuitive interface for time series analysis, while providing a high level of flexibility and extensibility.

The processing starts with a `DataFrame` with the data to be analyzed, where the index is the time and the columns are the different variables.

When creating the `Panel` object, the user needs to specify three things:

* lookback - the number of time steps to look back
* gap - the number of time steps to skip
* horizon - the number of time steps to look ahead

.. image:: ../_images/panel.png
    :scale: 30 %
    :align: center

For example, lets suppose that we have a `DataFrame` with 12 time indexes and we want to create a `Panel` with ``lookback=5``, ``gap=1`` and ``horizon=2``.

.. image:: ../_images/panel_divided.png
    :scale: 50 %
    :align: center


At the end we get a `Panel` for x and y with 3 frames each.
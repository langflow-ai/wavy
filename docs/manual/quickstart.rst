Quickstart
==========

This is a short introduction to `wavy`, geared mainly for new users. You can see more complex examples in :ref:`tutorials`.

.. code-block:: python

    # Import libraries
    import numpy as np
    import pandas as pd
    import wavy

    # Start with any time series dataframe
    df = pd.DataFrame({'price': np.random.randn(1000)}, index=range(1000))

    # Create panels. Each panel is composed of a list of frames
    # x and y are contain the past and corresponding future data
    x, y = wavy.create_panels(df, lookback=10, horizon=1)

    # Lookback and horizon are the number of timesteps
    print("Lookback:", len(x[0]), "Horizon:", len(y[0]))

    # Plot the target.
    y.plot()


.. code-block:: python

    # Convert to numpy arrays. Panels contain a train-val-test split by default
    x_train, y_train = x.train.values, y.train.values
    x_test, y_test = x.test.values, y.test.values
    print(x_train.shape, y_train.shape)

    # Or just instantiate a model
    model = wavy.LinearRegression(x, y)
    model.score()
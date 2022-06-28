.. _tutorials:

Tutorials
=========

Stock Price Analysis
--------------------

All the processing starts with the data to be analyzed. Here we are getting the Microsoft stock price from 2005 on.

.. code-block:: python

    # Import libraries
    import yfinance as yf
    import wavy
    import pandas as pd

    # Read in the data
    msft = yf.Ticker("MSFT")

    # Get historical market data from 2005 on
    hist = msft.history(period="max", start="2005-01-01")

With the data we can filter the interest columns and .

.. code-block:: python

    # Select the interest columns
    hist = hist[['Open', 'High', 'Low', 'Close']]

    # Plot the data
    hist.plot()

.. image:: ../_images/newplot.png
    :scale: 50 %

Calculate the percent change and drop rows with NaN.

.. code-block:: python

    # Calculate percent change
    hist = hist.pct_change()

    # Drop rows with NaN
    hist.dropna(inplace=True)


We will create a model to predict if the stock in the next day will close higher or lower based on the information of the last 5 days.
For this, we will create the panel using the following configuration.

.. code-block:: python

    # Create panels
    x, y = wavy.create_panels(hist, lookback=5, gap=0, horizon=1)

    # y will be the Close price of the next day
    y = y[['Close']]

    # Convert y to boolean
    y = y['Close'] > 0

    # Plot x
    x.plot()

.. image:: ../_images/x.png
    :scale: 50 %

Now we can create the model.

.. code-block:: python

    # Create model
    densemodel = wavy.DenseModel(x, y, model_type="classification")

With the model we can fit, and see the score for the validation set.

.. code-block:: python

    # Fit model
    densemodel.fit()

    # Score
    densemodel.score(on='val')

If we want, we can also predict on another dataset.

.. code-block:: python

    # Predict
    predicted = densemodel.predict(data=x.val)
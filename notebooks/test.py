import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

import wavy
from wavy.model import compute_default_scores

# from wavy.plot import plot_dataframes
from wavy.utils import reverse_pct_change

tickers = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[
    0
].Symbol

msft = yf.Ticker("MSFT")

hist = msft.history(period="max", start="2005-01-01")

hist = hist[["Open", "High", "Low", "Close"]]

df = hist.pct_change()

df.dropna(inplace=True)

x, y = wavy.create_panels(df, lookback=10, horizon=1)

# frame0 = x[0]
# frame1 = x[1:3, 'High']
# frame2 = x[3:5, 'High']
# frame3 = x[[1, 2, 3, 4], 'High']

a = x[0:3, "High"]

for i in a:
    print(i)

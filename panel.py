# Add to path
import sys

sys.path.append("src")

import numpy as np
import pandas as pd

import wavy

columns = ["A", "B"]
index = [1, 2, 3, 4]
values = [[np.nan, np.nan], [np.inf, 4], [5, 6], [7, 8]]

df = pd.DataFrame(columns=columns, index=index, data=values)
x, y = wavy.create_panels(df, lookback=1, horizon=1, gap=0)

print(x.columns)


x[0]
x[:2]
x[[0, 1, 2]]
x["B"]
x[["B"]]
x[0, "B"]
x[0:2, "B"]
x[[0, 1, 2], "B"]
x[0, ["B"]]
x[0:2, ["B"]]
x[[0, 1, 2], ["B"]]

# a = x[0, "A"]
# a = x.values
# a = x.countna()
# a = x.findna()
# a = x.dropna()
# a = x.findinf()
# print(a)
# print("ibis")

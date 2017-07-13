import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = "..."

df = pd.read_csv(path + "\\train.csv").fillna("")
df.head()

df.info()
df.shape
df.groupby("is_duplicate")['id'].count().plot.bar()
